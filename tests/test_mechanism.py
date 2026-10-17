"""Unit tests for coalition extraction and native attention message updates."""

import math
import unittest

import torch

from mica.model import (
    CoalitionAdapter,
    coalition_terms,
    install_attention,
    native_messages,
)


class CoalitionTermsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.text, self.visual = torch.randn(2, 5, 4), torch.randn(2, 5, 4)
        self.function = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4)
        )

    def test_affine_function_has_no_mixed_term(self):
        mixed, _ = coalition_terms(torch.nn.Linear(4, 4), self.text, self.visual)
        self.assertLess(mixed.abs().max().item(), 1e-6)

    def test_absent_source_has_no_mixed_term(self):
        for text, visual in (
            (self.text, self.visual * 0),
            (self.text * 0, self.visual),
        ):
            with self.subTest(zero_text=not bool(text.any())):
                mixed, _ = coalition_terms(self.function, text, visual)
                self.assertLess(mixed.abs().max().item(), 1e-6)

    def test_nonlinear_mixed_term_is_nonzero(self):
        mixed, _ = coalition_terms(self.function, self.text, self.visual)
        self.assertGreater(mixed.abs().max().item(), 1e-4)

    def test_additive_term_matches_shared_function(self):
        _, unary = coalition_terms(self.function, self.text, self.visual)
        expected = self.function(self.text) + self.function(self.visual)
        expected = expected - 2 * self.function(torch.zeros_like(self.text))
        torch.testing.assert_close(unary, expected)


class NativeMessageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.q, self.k, self.v = [torch.randn(2, 6, 8) for _ in range(3)]
        self.queries = torch.tensor([3, 4, 5])
        self.visual = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
        self.text = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
        self.valid = torch.ones(6, dtype=torch.bool)
        self.scale = 1 / math.sqrt(8)

    def messages(self, values=None, valid=None, extra_mask=None):
        return native_messages(
            self.q,
            self.k,
            self.v if values is None else values,
            self.queries,
            self.text,
            self.visual,
            self.valid if valid is None else valid,
            self.scale,
            extra_mask,
        )

    def test_messages_reconstruct_native_attention(self):
        text, visual, _, attention = self.messages()
        other = (attention * (~(self.text | self.visual))[None, None, :]) @ self.v
        torch.testing.assert_close(
            text + visual + other, attention @ self.v, atol=1e-6, rtol=1e-6
        )

    def test_causal_and_padding_masks(self):
        _, _, _, attention = self.messages()
        self.assertEqual(attention[:, 0, 4:].abs().sum().item(), 0)
        valid = self.valid.clone()
        valid[5] = False
        _, _, _, attention = self.messages(valid=valid)
        self.assertEqual(attention[..., 5].abs().sum().item(), 0)

    def test_unselected_value_does_not_change_modality_messages(self):
        text, visual, _, _ = self.messages()
        values = self.v.clone()
        values[:, 5] += 100
        changed_text, changed_visual, _, _ = self.messages(values=values)
        self.assertTrue(torch.equal(text, changed_text))
        self.assertTrue(torch.equal(visual, changed_visual))

    def test_extra_boolean_and_additive_masks_agree(self):
        boolean = torch.ones(1, 6, 6, dtype=torch.bool)
        boolean[..., 0] = False
        additive = torch.zeros(1, 6, 6).masked_fill(~boolean, float("-inf"))
        bool_result = self.messages(extra_mask=boolean)
        float_result = self.messages(extra_mask=additive)
        for left, right in zip(bool_result, float_result):
            torch.testing.assert_close(left, right)
        self.assertEqual(bool_result[-1][..., 0].abs().sum().item(), 0)

    def adapter_inputs(self, variant="mica"):
        output = torch.nn.functional.scaled_dot_product_attention(
            self.q[None], self.k[None], self.v[None], is_causal=True
        ).transpose(1, 2)
        adapter = CoalitionAdapter(2, 8, 4, 1, variant=variant)
        readout = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
        adapter.masks = (
            self.text[None],
            self.visual[None],
            self.valid[None],
            readout[None],
        )
        return adapter, output

    def edit(self, adapter, output):
        return adapter.edit(
            self.q[None], self.k[None], self.v[None], output, 0, None, self.scale
        )

    def test_selected_queries_and_projection_gradients(self):
        adapter, original = self.adapter_inputs()
        edited = self.edit(adapter, original)
        self.assertTrue(torch.equal(edited[:, :3], original[:, :3]))
        self.assertGreater((edited - original).abs().max().item(), 0)
        edited.square().mean().backward()
        self.assertGreater(adapter.down[0].grad.abs().sum().item(), 0)
        self.assertGreater(adapter.up[0].grad.abs().sum().item(), 0)

    def test_source_interventions_remove_update(self):
        adapter, original = self.adapter_inputs()
        for mode in ("zero_visual_message", "zero_text_message", "bypass"):
            with self.subTest(mode=mode):
                adapter.intervention = mode
                edited = self.edit(adapter, original)
                torch.testing.assert_close(edited, original, atol=1e-6, rtol=0)

    def test_update_obeys_norm_bound(self):
        adapter, original = self.adapter_inputs()
        edited = self.edit(adapter, original)
        difference = (edited - original).norm(dim=-1)
        limit = 0.15 * original.norm(dim=-1).clamp_min(1e-3)
        self.assertTrue(bool((difference <= limit + 1e-6).all()))

    def test_no_route_freezes_routing_parameters(self):
        adapter, _ = self.adapter_inputs(variant="no_route")
        for group in (adapter.route_keys, adapter.route_values):
            self.assertTrue(
                all(not parameter.requires_grad for parameter in group.parameters())
            )


class QwenAttentionTests(unittest.TestCase):
    def setUp(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        torch.manual_seed(42)
        torch.set_num_threads(1)
        config = Qwen3VLTextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        self.attention = Qwen3VLTextAttention(config, layer_idx=0).eval()
        self.hidden = torch.randn(2, 6, 64)
        self.positions = (torch.ones(2, 6, 16), torch.zeros(2, 6, 16))
        causal = torch.ones(6, 6, dtype=torch.bool).tril()
        self.mask = torch.zeros(2, 1, 6, 6).masked_fill(~causal, float("-inf"))
        self.adapter = CoalitionAdapter(4, 16, 8, 1).eval()
        text = torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.bool).expand(2, -1)
        visual = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool).expand(2, -1)
        valid = torch.ones(2, 6, dtype=torch.bool)
        readout = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.bool).expand(2, -1)
        self.adapter.masks = (text, visual, valid, readout)

    def forward(self):
        return self.attention(
            self.hidden, position_embeddings=self.positions, attention_mask=self.mask
        )[0]

    def test_bypass_matches_native_grouped_query_attention(self):
        native = self.forward()
        install_attention(self.attention, self.adapter, 0)
        self.adapter.intervention = "bypass"
        bypass = self.forward()
        torch.testing.assert_close(bypass, native, atol=0, rtol=0)

    def test_coalition_update_changes_selected_queries_only(self):
        native = self.forward()
        install_attention(self.attention, self.adapter, 0)
        changed = self.forward()
        torch.testing.assert_close(changed[:, :3], native[:, :3], atol=0, rtol=0)
        self.assertGreater((changed[:, 3:] - native[:, 3:]).abs().max().item(), 1e-7)
        changed.square().mean().backward()
        self.assertGreater(self.adapter.down[0].grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
