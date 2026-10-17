"""Evaluate a saved MICA classifier on labeled image-text pairs."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from mica.data import Collator, read_rows
from mica.model import build_model
from scripts.train import evaluate, load_trainable, loader


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory containing config.json and trainable.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Labeled JSONL file with id, text, image, and label fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory; defaults to the checkpoint parent/evaluation.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "trainable.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Checkpoint must contain config.json and trainable.pt.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = read_rows(args.data)
    if not rows:
        raise ValueError("Evaluation data must contain at least one labeled pair.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; choose an available device with --device."
        )
    processor = AutoProcessor.from_pretrained(
        config["base_model"], local_files_only=config.get("local_files_only", True)
    )
    model = build_model(config).to(device)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    load_trainable(model, state)
    output = (
        args.output if args.output is not None else checkpoint.parent / "evaluation"
    )
    result = evaluate(
        model,
        loader(rows, Collator(processor), config.get("eval_batch_size", 2)),
        device,
        output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
