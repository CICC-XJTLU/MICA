"""Train MICA with validation checkpoint selection and resumable optimizer state."""

import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup
from mica.model import build_model
from mica.data import read_rows, subset, Collator, device_batch, metrics

ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def emit(out, name, obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)
    with (out / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def trainable_state(model):
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["cpu"])
    torch.cuda.set_rng_state_all(s["cuda"])


def load_trainable(model, state):
    params = dict(model.named_parameters())
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(state) != expected:
        raise RuntimeError("Trainable parameter keys do not match checkpoint")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t.to(params[n].device))


def save_resume(path, model, opt, sched, identity, cursor, best, elapsed):
    obj = dict(
        format_version=1,
        identity=identity,
        cursor=cursor,
        best=best,
        elapsed=elapsed,
        trainable=trainable_state(model),
        optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        rng=rng_state(),
    )
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.inference_mode()
def evaluate(model, loader, device, path=None):
    model.eval()
    ys = []
    ps = []
    records = []
    for batch in loader:
        b = device_batch(batch, device)
        r = model(text_mask=b["text_mask"], visual_mask=b["visual_mask"], **b["inputs"])
        probs = r["logits"].softmax(-1).cpu().tolist()
        pred = r["logits"].argmax(-1).cpu().tolist()
        labels = b["labels"].cpu().tolist()
        records.extend(
            dict(id=i, label=y, prediction=p, probabilities=prob)
            for i, y, p, prob in zip(b["ids"], labels, pred, probs)
        )
        ys.extend(labels)
        ps.extend(pred)
    result = metrics(ys, ps)
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        tmp = path / "predictions.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        os.replace(tmp, path / "predictions.jsonl")
        write(path / "metrics.json", result)
    return result


def save_best(model, c, path, step, epoch, metric):
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "trainable.pt.tmp"
    torch.save(trainable_state(model), tmp)
    os.replace(tmp, path / "trainable.pt")
    write(
        path / "config.json",
        dict(c, selected_step=step, selected_epoch=epoch, validation=metric),
    )


def loader(rows, collator, batch_size):
    # Private generator prevents iterator construction from perturbing model dropout RNG.
    return DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(9371),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text())
    out = Path(c["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists() and not args.smoke:
        done = json.loads((out / "completed.json").read_text())
        expected = c.get("expected_test_samples")
        if expected is not None and done["test"]["samples"] != expected:
            raise ValueError("Completed test size does not match configuration")
        print("REUSE_COMPLETED", flush=True)
        return
    if (out / "train.jsonl").exists() and not args.resume and not args.smoke:
        raise RuntimeError(
            "Unfinished output: explicit --resume required; refusing overwrite"
        )
    if args.resume and not (out / "resume.pt").exists():
        raise RuntimeError("No full-state resume checkpoint")
    seed = c["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(c["base_model"], local_files_only=True)
    collator = Collator(processor)
    splits = {s: read_rows(c[s + "_data"]) for s in ("train", "val", "test")}
    ids = {s: {str(r["id"]) for r in rows} for s, rows in splits.items()}
    assert all(len(ids[s]) == len(splits[s]) for s in splits)
    assert not (
        ids["train"] & ids["val"]
        or ids["train"] & ids["test"]
        or ids["val"] & ids["test"]
    )
    if c.get("expected_split_sizes") is not None:
        actual = {s: len(x) for s, x in splits.items()}
        if actual != c["expected_split_sizes"]:
            raise ValueError(f"Split sizes {actual} do not match expected_split_sizes")
    identity = dict(
        config=c,
        source_hashes={
            str(p.relative_to(ROOT)): sha(p)
            for p in [ROOT / "mica/model.py", ROOT / "mica/data.py", Path(__file__)]
        },
        data_hashes={s: sha(c[s + "_data"]) for s in splits},
    )
    write(
        out / "manifest.json",
        dict(
            identity=identity,
            counts={s: len(x) for s, x in splits.items()},
            no_split_id_overlap=True,
            selection="validation macro_f1 once per epoch; test not used for selection",
            smoke=args.smoke,
        ),
    )
    model = build_model(c).to(device)
    lora = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("backbone.")
    ]
    custom = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("backbone.")
    ]
    parameters = dict(
        trainable=sum(p.numel() for p in lora + custom),
        custom=sum(p.numel() for p in custom),
        total=sum(p.numel() for p in model.parameters()),
        added_vocabulary_tokens=0,
    )
    write(out / "parameters.json", parameters)
    print("PARAMETERS", parameters, flush=True)
    opt = torch.optim.AdamW(
        [
            {"params": lora, "lr": c["lora_lr"]},
            {"params": custom, "lr": c["module_lr"]},
        ],
        weight_decay=0.01,
    )
    n_batches = math.ceil(len(splits["train"]) / c["batch_size"])
    acc = c["accumulation"]
    steps_per = math.ceil(n_batches / acc)
    total = steps_per * c["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total * 0.05)), total)
    step = 0
    start_epoch = 0
    next_batch = 0
    best = -1.0
    offset = 0.0
    start = time.time()
    opt.zero_grad(set_to_none=True)
    if args.resume:
        ck = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if ck["identity"] != identity:
            raise RuntimeError("Resume identity/source/data mismatch; inspect manually")
        load_trainable(model, ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        restore_rng(ck["rng"])
        step = ck["cursor"]["step"]
        start_epoch = ck["cursor"]["epoch"]
        next_batch = ck["cursor"]["next_batch"]
        best = ck["best"]
        offset = ck["elapsed"]
        # Preserve original event log, discard only rows from unsaved optimizer steps.
        for name in ("train.jsonl", "validation.jsonl"):
            p = out / name
            if p.exists():
                original = p.read_text()
                (out / (name + ".before_resume_" + str(time.time_ns()))).write_text(
                    original
                )
                rows = [
                    json.loads(line) for line in original.splitlines() if line.strip()
                ]
                p.write_text(
                    "".join(
                        json.dumps(r) + "\n" for r in rows if r.get("step", 0) <= step
                    )
                )
        del ck
        emit(
            out,
            "events.jsonl",
            dict(
                event="resume_full_state",
                step=step,
                epoch=start_epoch,
                next_batch=next_batch,
            ),
        )
    if args.smoke:
        model.train()
        b = device_batch(next(iter(loader(splits["train"][:2], collator, 2))), device)
        r = model(text_mask=b["text_mask"], visual_mask=b["visual_mask"], **b["inputs"])
        cls = torch.nn.functional.cross_entropy(r["logits"], b["labels"])
        cls.backward()
        norms = {
            n: float(p.grad.norm())
            for n, p in model.named_parameters()
            if p.requires_grad
            and p.grad is not None
            and ("interactor" in n or "head" in n)
        }
        assert torch.isfinite(cls) and any(
            v > 0 for n, v in norms.items() if "head" in n
        )
        if model.interactor is not None:
            for needle in ("down", "up", "functions", "gates"):
                assert any(v > 0 for n, v in norms.items() if needle in n), (
                    needle,
                    norms,
                )
            if c["variant"] != "no_route":
                assert any(v > 0 for n, v in norms.items() if "route_values" in n)
        # A real optimizer update, then round-trip the full model/optimizer/RNG state.
        # Scheduler warmup would make update zero at step 0; use target LRs for smoke only.
        for group, lr in zip(opt.param_groups, [c["lora_lr"], c["module_lr"]]):
            group["lr"] = lr
        before = model.head[-1].weight.detach().clone()
        opt.step()
        opt.zero_grad(set_to_none=True)
        assert not torch.equal(before, model.head[-1].weight)
        checkpoint = out / "smoke_resume.pt"
        save_resume(
            checkpoint,
            model,
            opt,
            sched,
            identity,
            dict(step=1, epoch=0, next_batch=1),
            -1,
            0,
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reference = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        with torch.no_grad():
            model.head[-1].weight.add_(1)
        load_trainable(model, saved["trainable"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        assert all(
            torch.equal(p, reference[n])
            for n, p in model.named_parameters()
            if p.requires_grad
        )
        # Isolate actual module influence and native attention equivalence with bypass.
        model.eval()
        differences = {}
        with torch.no_grad():
            native = model(
                text_mask=b["text_mask"], visual_mask=b["visual_mask"], **b["inputs"]
            )["logits"]
            if model.interactor is not None:
                bypass_logits = None
                for mode in ("bypass", "no_route", "zero_visual_message"):
                    model.interactor.intervention = mode
                    changed = model(
                        text_mask=b["text_mask"],
                        visual_mask=b["visual_mask"],
                        **b["inputs"],
                    )["logits"]
                    differences[mode] = float((native - changed).abs().max())
                    if mode == "bypass":
                        bypass_logits = changed.clone()
                # Undo only the three forward replacements; native Qwen implementation
                # must match our bypass, without touching any learned weights.
                import types

                attentions = [
                    model.backbone.get_base_model()
                    .model.language_model.layers[i]
                    .self_attn
                    for i in model.layers
                ]
                patched = [a.forward for a in attentions]
                for a in attentions:
                    a.forward = types.MethodType(type(a).forward, a)
                pristine = model(
                    text_mask=b["text_mask"],
                    visual_mask=b["visual_mask"],
                    **b["inputs"],
                )["logits"]
                differences["bypass_vs_native_qwen"] = float(
                    (bypass_logits - pristine).abs().max()
                )
                assert torch.allclose(
                    bypass_logits, pristine, atol=1e-5, rtol=1e-5
                ), differences
                for a, forward in zip(attentions, patched):
                    a.forward = forward
                model.interactor.intervention = "none"
        emit(
            out,
            "smoke.jsonl",
            dict(
                event="forward_backward_optimizer_checkpoint_pass",
                variant=c["variant"],
                loss=float(cls),
                gradient_norms=norms,
                mechanism_stats={k: float(v) for k, v in r["stats"].items()},
                logit_differences=differences,
                peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
                elapsed_sec=time.time() - start,
            ),
        )
        checkpoint.unlink()
        return
    val = loader(splits["val"], collator, c["eval_batch_size"])
    emit(
        out,
        "events.jsonl",
        dict(
            event="training_started",
            variant=c["variant"],
            backbone=c["backbone_name"],
            total_steps=total,
            resume=args.resume,
        ),
    )
    for epoch in range(start_epoch, c["epochs"]):
        order = torch.randperm(
            len(splits["train"]), generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        first_batch = next_batch if epoch == start_epoch else 0
        rows = [splits["train"][i] for i in order[first_batch * c["batch_size"] :]]
        model.train()
        running = []
        for bi, batch in enumerate(
            loader(rows, collator, c["batch_size"]), start=first_batch
        ):
            b = device_batch(batch, device)
            r = model(
                text_mask=b["text_mask"], visual_mask=b["visual_mask"], **b["inputs"]
            )
            cls = torch.nn.functional.cross_entropy(r["logits"], b["labels"])
            loss = cls + c["energy_weight"] * r["energy"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            group_size = min(acc, n_batches - (bi // acc) * acc)
            (loss / group_size).backward()
            running.append([float(cls.detach()), float(r["energy"].detach())])
            if (bi + 1) % acc and bi + 1 < n_batches:
                continue
            norm = torch.nn.utils.clip_grad_norm_(lora + custom, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite gradient norm")
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step <= 3 or step % 10 == 0 or bi + 1 == n_batches:
                emit(
                    out,
                    "train.jsonl",
                    dict(
                        step=step,
                        total_steps=total,
                        epoch=epoch + (bi + 1) / n_batches,
                        cls=float(np.mean(running, axis=0)[0]),
                        energy=float(np.mean(running, axis=0)[1]),
                        grad_norm=float(norm),
                        elapsed_sec=offset + time.time() - start,
                        peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
                        lr=sched.get_last_lr(),
                        mechanism_stats={k: float(v) for k, v in r["stats"].items()},
                    ),
                )
                running = []
            if step % c["checkpoint_every"] == 0:
                save_resume(
                    out / "resume.pt",
                    model,
                    opt,
                    sched,
                    identity,
                    dict(step=step, epoch=epoch, next_batch=bi + 1),
                    best,
                    offset + time.time() - start,
                )
        vm = evaluate(model, val, device)
        emit(out, "validation.jsonl", dict(step=step, epoch=epoch + 1, **vm))
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            save_best(model, c, out / "checkpoint-best", step, epoch + 1, vm)
        save_resume(
            out / "resume.pt",
            model,
            opt,
            sched,
            identity,
            dict(step=step, epoch=epoch + 1, next_batch=0),
            best,
            offset + time.time() - start,
        )
    write(
        out / "training_done.json",
        dict(
            step=step,
            total_steps=total,
            best_val_macro_f1=best,
            elapsed_sec=offset + time.time() - start,
        ),
    )
    load_trainable(
        model,
        torch.load(
            out / "checkpoint-best/trainable.pt", map_location="cpu", weights_only=True
        ),
    )
    tm = evaluate(
        model,
        loader(splits["test"], collator, c["eval_batch_size"]),
        device,
        out / "test",
    )
    interventions = {}
    if c["variant"] == "mica":
        dl = loader(
            subset(splits["val"], 128, seed=3907), collator, c["eval_batch_size"]
        )
        for mode in (
            "none",
            "bypass",
            "no_route",
            "zero_visual_message",
            "reverse_joint",
        ):
            model.interactor.intervention = mode
            interventions[mode] = evaluate(
                model, dl, device, out / "diagnostics_val" / mode
            )
        model.interactor.intervention = "none"
    selected = json.loads((out / "checkpoint-best/config.json").read_text())
    write(
        out / "completed.json",
        dict(
            test=tm,
            selected_epoch=selected["selected_epoch"],
            selected_step=selected["selected_step"],
            diagnostic_split="validation_subset_128",
            interventions=interventions,
            elapsed_sec=offset + time.time() - start,
            identity=identity,
        ),
    )
    print("COMPLETED", json.dumps(tm), flush=True)


if __name__ == "__main__":
    main()
