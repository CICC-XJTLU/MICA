# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Copyright 2026 MICA authors.
# Licensed under the Apache License, Version 2.0.
# The attention forward is adapted from Transformers v4.57.6 for MICA writeback.
# https://www.apache.org/licenses/LICENSE-2.0
"""Native-mass coalition interaction before the attention output projection."""
from __future__ import annotations
import math
import types
import torch
from torch import nn
from torch.nn import functional as F


def coalition_terms(function, text, visual):
    zero = torch.zeros_like(text)
    # Shared evaluations preserve the mixed-difference null identity.
    f0, ft, fv, fj = function(
        torch.stack([zero, text, visual, text + visual], 0)
    ).unbind(0)
    return fj - ft - fv + f0, ft + fv - 2 * f0


def native_messages(q, k, v, queries, text, visual, valid, scale, extra_mask=None):
    """Single sample H,S,D. Preserve the all-token causal denominator."""
    scores = q[:, queries].float() @ k.float().transpose(-1, -2) * scale
    causal = torch.arange(k.shape[1], device=q.device)[None, :] <= queries[:, None]
    allowed = causal & valid[None, :]
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    if extra_mask is not None:
        mask = extra_mask[..., queries, : k.shape[1]]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.float()
    a = torch.nan_to_num(scores.softmax(-1))
    at, av = a * text[None, None, :], a * visual[None, None, :]
    mt, mv = at @ v.float(), av @ v.float()
    pt, pv = at.sum(-1, keepdim=True), av.sum(-1, keepdim=True)

    def entropy(am, p, m):
        prob = am / p.clamp_min(1e-9)
        h = -(prob * prob.clamp_min(1e-9).log()).sum(-1, keepdim=True)
        count = (allowed & m[None, :]).sum(-1).clamp_min(2).float().log()
        return h / count[None, :, None]

    info = torch.cat([pt, pv, entropy(at, pt, text), entropy(av, pv, visual)], -1)
    return mt, mv, info, a


class CoalitionAdapter(nn.Module):
    def __init__(self, heads=32, head_dim=128, rank=32, levels=3, variant="mica"):
        super().__init__()
        self.variant = variant
        self.intervention = "none"
        self.masks = None
        self.traces = {}
        self.down = nn.ParameterList(
            [
                nn.Parameter(torch.randn(heads, head_dim, rank) / math.sqrt(head_dim))
                for _ in range(levels)
            ]
        )
        self.up = nn.ParameterList(
            [
                nn.Parameter(torch.randn(heads, rank, head_dim) * 0.03)
                for _ in range(levels)
            ]
        )
        self.functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(rank, 2 * rank), nn.GELU(), nn.Linear(2 * rank, rank)
                )
                for _ in range(levels)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * rank + 4, 32), nn.GELU(), nn.Linear(32, 1))
                for _ in range(levels)
            ]
        )
        self.route_keys = nn.ModuleList(
            [nn.Linear(rank, 16, bias=False) for _ in range(levels)]
        )
        self.route_values = nn.ModuleList(
            [nn.Linear(rank, rank, bias=False) for _ in range(levels)]
        )
        self.strength = nn.Parameter(torch.full((levels,), -1.0))
        if variant == "no_route":
            for group in (self.route_keys, self.route_values):
                for p in group.parameters():
                    p.requires_grad_(False)

    def edit(self, q, k, v, out, level, attention_mask, scale):
        # out B,S,H,D. Read only original projection values, edit only caption/readout queries.
        if self.intervention == "bypass":
            self.traces[level] = {"energy": out.float().sum() * 0}
            return out
        text, visual, valid, readout = self.masks
        changed = []
        records = []
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, 1)
        v = v.repeat_interleave(repeat, 1)
        for b in range(len(q)):
            qi = (text[b] | readout[b]).nonzero().flatten()
            extra = None if attention_mask is None else attention_mask[b]
            mt, mv, info, a = native_messages(
                q[b], k[b], v[b], qi, text[b], visual[b], valid[b], scale, extra
            )
            if self.intervention == "zero_visual_message":
                mv = mv * 0
            if self.intervention == "zero_text_message":
                mt = mt * 0
            zt = torch.einsum("hnd,hdr->hnr", mt, self.down[level])
            zv = torch.einsum("hnd,hdr->hnr", mv, self.down[level])
            # Fixed smooth bounding shared by both sources; no per-coalition normalization.
            zt, zv = torch.tanh(zt), torch.tanh(zv)
            joint, unary = coalition_terms(self.functions[level], zt, zv)
            evidence = unary if self.variant == "additive" else joint
            route_energy = evidence.sum() * 0
            if self.variant != "no_route" and self.intervention != "no_route":
                keys = F.normalize(
                    self.route_keys[level](evidence), dim=-1, eps=1e-6
                ).transpose(0, 1)
                affinity = keys @ keys.transpose(-1, -2) / 0.5
                affinity = affinity.masked_fill(
                    torch.eye(q.shape[1], device=q.device, dtype=torch.bool)[None], -1e4
                )
                routing = affinity.softmax(-1)
                routed = (
                    routing @ self.route_values[level](evidence).transpose(0, 1)
                ).transpose(0, 1)
                route_energy = routed.square().mean()
                evidence = evidence + routed
            if self.intervention == "reverse_joint":
                evidence = -evidence
            gate = self.gates[level](torch.cat([zt, zv, info], -1)).sigmoid()
            pt, pv = info[..., :1], info[..., 1:2]
            budget = 4 * pt * pv / (pt + pv).square().clamp_min(1e-8)
            # If either source is absent the non-additive branch remains identically zero.
            bounded = (
                evidence / (evidence.square().mean(-1, keepdim=True) + 1e-4).sqrt()
            )
            delta = torch.einsum("hnr,hrd->hnd", bounded, self.up[level])
            delta = delta * gate * budget * 0.15 * self.strength[level].sigmoid()
            old = out[b, qi].transpose(0, 1).float()
            limit = 0.15 * old.norm(dim=-1, keepdim=True).detach().clamp_min(1e-3)
            delta = delta * (
                limit / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            ).clamp(max=1)
            new = out[b].float().index_add(0, qi, delta.transpose(0, 1))
            changed.append(new.to(out.dtype))
            records.append(
                dict(
                    energy=delta.square().mean(),
                    joint_rms=joint.square().mean().sqrt().detach(),
                    write_rms=delta.square().mean().sqrt().detach(),
                    budget=budget.mean().detach(),
                    route_rms=route_energy.sqrt().detach(),
                    gate=gate.mean().detach(),
                )
            )
        self.traces[level] = {
            key: torch.stack([r[key] for r in records]).mean() for key in records[0]
        }
        return torch.stack(changed)


def install_attention(attn, adapter, level):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_values is not None or kwargs.get("past_key_value") is not None:
            raise ValueError("MICA is a full-input classifier; KV cache unsupported")
        shape = hidden_states.shape[:-1]
        hshape = (*shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hshape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hshape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        interface = (
            eager_attention_forward
            if self.config._attn_implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        )
        out, weights = interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        out = adapter.edit(q, k, v, out, level, attention_mask, self.scaling)
        return self.o_proj(out.reshape(*shape, -1).contiguous()), weights

    attn.forward = types.MethodType(forward, attn)


class MICAModel(nn.Module):
    def __init__(self, backbone, layers=(11, 23, 31), rank=32, variant="mica", seed=42):
        super().__init__()
        self.backbone = backbone
        self.layers = tuple(layers)
        self.variant = variant
        base = backbone.get_base_model()
        c = base.config.text_config
        torch.manual_seed(seed + 900)
        self.interactor = (
            None
            if variant == "lora"
            else CoalitionAdapter(
                c.num_attention_heads, c.head_dim, rank, len(layers), variant
            )
        )
        torch.manual_seed(
            seed + 901
        )  # Identical readout initialization across all variants.
        self.head = nn.Sequential(
            nn.LayerNorm(c.hidden_size * 2),
            nn.Linear(c.hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )
        if self.interactor is not None:
            for level, layer in enumerate(layers):
                install_attention(
                    base.model.language_model.layers[layer].self_attn,
                    self.interactor,
                    level,
                )

    def forward(self, text_mask, visual_mask, **inputs):
        if self.interactor is not None:
            valid = inputs["attention_mask"].bool()
            readout = torch.zeros_like(valid)
            readout[
                torch.arange(len(valid), device=valid.device), valid.sum(-1) - 1
            ] = True
            self.interactor.masks = (text_mask, visual_mask, valid, readout)
            self.interactor.traces = {}
        h = (
            self.backbone.get_base_model()
            .model(**inputs, use_cache=False, return_dict=True)
            .last_hidden_state
        )
        mask = text_mask.to(h.dtype)[..., None]
        mean = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        last = h[
            torch.arange(len(h), device=h.device), inputs["attention_mask"].sum(-1) - 1
        ]
        logits = self.head(torch.cat([mean, last], -1).float())
        traces = {} if self.interactor is None else self.interactor.traces
        energy = (
            torch.stack([t["energy"] for t in traces.values()]).mean()
            if traces
            else logits.sum() * 0
        )
        stats = {
            key: torch.stack([t[key] for t in traces.values() if key in t]).mean()
            for key in ("joint_rms", "write_rms", "budget", "route_rms", "gate")
            if any(key in t for t in traces.values())
        }
        return dict(logits=logits, energy=energy, stats=stats)

    def custom_state(self):
        return {
            n: t.detach().cpu()
            for n, t in self.state_dict().items()
            if not n.startswith("backbone.")
        }


def build_model(config):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    base = AutoModelForImageTextToText.from_pretrained(
        config["base_model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    backbone = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            exclude_modules=".*visual.*",
        ),
    )
    return MICAModel(
        backbone, config["layers"], config["rank"], config["variant"], config["seed"]
    )
