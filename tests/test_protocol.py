"""Configuration, data, and checkpoint evaluation interface tests."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

from mica.data import Collator, metrics, read_rows, subset
from scripts.evaluate import main, parse_args


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_all_backbone_variant_configurations(self):
        variants = {"mica", "lora", "additive", "no_route"}
        for size in ("4b", "8b"):
            directory = ROOT / "configs" / ("qwen3vl" + size)
            self.assertEqual({p.stem for p in directory.glob("*.json")}, variants)
            configs = []
            for variant in sorted(variants):
                with self.subTest(size=size, variant=variant):
                    config = json.loads(
                        (directory / (variant + ".json")).read_text(encoding="utf-8")
                    )
                    configs.append(config)
                    self.assertEqual(config["variant"], variant)
                    self.assertIn(size.upper(), config["backbone_name"])
                    self.assertEqual(config["layers"], [11, 23, 31])
                    self.assertEqual(config["rank"], 32)
                    self.assertEqual(config["expected_test_samples"], 2409)
                    self.assertEqual(config["batch_size"] * config["accumulation"], 16)
                    for key in (
                        "output_dir",
                        "train_data",
                        "val_data",
                        "test_data",
                        "base_model",
                    ):
                        self.assertFalse(Path(config[key]).is_absolute())
                        self.assertNotIn("..", Path(config[key]).parts)
            for key in (
                "seed",
                "epochs",
                "batch_size",
                "accumulation",
                "lora_lr",
                "module_lr",
            ):
                self.assertEqual(len({config[key] for config in configs}), 1)


class DataTests(unittest.TestCase):
    def read(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            return read_rows(path)

    @staticmethod
    def row(label=0):
        return {
            "id": "post-1",
            "text": "A caption.",
            "image": "images/post-1.jpg",
            "label": label,
        }

    def test_supported_labels(self):
        for label, expected in (
            (0, 0),
            (1, 1),
            ("0", 0),
            ("1", 1),
            ("sarcastic", 1),
            ("not_sarcastic", 0),
            ("non-sarcastic", 0),
        ):
            with self.subTest(label=label):
                self.assertEqual(self.read([self.row(label)])[0]["label"], expected)

    def test_unknown_label_is_rejected(self):
        for label in ("unknown", 2, -1):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.read([self.row(label)])

    def test_required_fields(self):
        for key in ("id", "text", "image", "label"):
            row = self.row()
            del row[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read([row])

    def test_text_and_image_types(self):
        for key, value in (
            ("text", None),
            ("text", 17),
            ("image", None),
            ("image", 17),
            ("image", ""),
        ):
            row = self.row()
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.read([row])

    def test_paths_are_kept_relative_to_working_directory(self):
        row = self.read([self.row()])[0]
        self.assertEqual(row["image"], "images/post-1.jpg")

    def test_stratified_subset_is_reproducible(self):
        rows = [dict(self.row(index % 2), id=str(index)) for index in range(20)]
        first, second = subset(rows, 8, seed=42), subset(rows, 8, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(sum(row["label"] for row in first), 4)
        self.assertEqual(subset(rows, 100), rows)

    def test_binary_metrics(self):
        result = metrics([0, 1, 0, 1], [0, 1, 1, 1])
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 5) / 2)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])


class SmallProcessor:
    def __init__(self):
        self.tokenizer = self
        self.padding_side = "left"

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 10, "<|vision_end|>": 11, "<|im_end|>": 12}[token]

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"][1]["text"]

    def __call__(self, text, images, **kwargs):
        ids = torch.tensor([[1, 10, 10, 11, 20, 21, 12, 30]] * len(text))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class CollatorTests(unittest.TestCase):
    def test_caption_visual_and_readout_positions(self):
        processor = SmallProcessor()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (8, 8), "white").save(image)
            row = {"id": "post", "text": "A caption.", "image": str(image), "label": 1}
            batch = Collator(processor)([row])
        self.assertEqual(processor.padding_side, "right")
        self.assertEqual(batch["visual_mask"].nonzero()[:, 1].tolist(), [1, 2])
        self.assertEqual(batch["text_mask"].nonzero()[:, 1].tolist(), [4, 5])
        self.assertFalse(bool(batch["text_mask"][0, -1]))
        self.assertEqual(batch["ids"], ["post"])
        self.assertEqual(batch["labels"].tolist(), [1])


class SmallClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, text_mask, visual_mask, features):
        return {"logits": features + self.bias}


class EvaluationTests(unittest.TestCase):
    def test_argument_defaults(self):
        args = parse_args(
            ["--checkpoint", "outputs/run/checkpoint-best", "--data", "data/test.jsonl"]
        )
        self.assertEqual(args.device, "cuda:0")
        self.assertIsNone(args.output)

    def test_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(
            FileNotFoundError
        ):
            main(
                ["--checkpoint", directory, "--data", "pairs.jsonl", "--device", "cpu"]
            )

    def test_checkpoint_restoration_and_prediction_outputs(self):
        def collate(rows):
            indices = [int(row["id"]) for row in rows]
            return {
                "inputs": {"features": torch.eye(2)[indices] * 4},
                "text_mask": torch.ones(len(rows), 1, dtype=torch.bool),
                "visual_mask": torch.ones(len(rows), 1, dtype=torch.bool),
                "labels": torch.tensor([row["label"] for row in rows]),
                "ids": [row["id"] for row in rows],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-best"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                json.dumps({"base_model": "models/qwen3vl", "eval_batch_size": 2}),
                encoding="utf-8",
            )
            torch.save({"bias": torch.tensor([0.1, -0.1])}, checkpoint / "trainable.pt")
            data = root / "pairs.jsonl"
            data.write_text(
                "\n".join(
                    json.dumps(dict(DataTests.row(index), id=str(index)))
                    for index in (0, 1)
                ),
                encoding="utf-8",
            )
            model = SmallClassifier()
            with patch(
                "scripts.evaluate.AutoProcessor.from_pretrained"
            ) as processor, patch(
                "scripts.evaluate.build_model", return_value=model
            ), patch(
                "scripts.evaluate.Collator", return_value=collate
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                result = main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--data",
                        str(data),
                        "--device",
                        "cpu",
                    ]
                )
            processor.assert_called_once_with("models/qwen3vl", local_files_only=True)
            torch.testing.assert_close(model.bias, torch.tensor([0.1, -0.1]))
            self.assertEqual(result["accuracy"], 1)
            saved = json.loads(
                (root / "evaluation" / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["samples"], 2)
            predictions = (
                (root / "evaluation" / "predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(
                [json.loads(row)["prediction"] for row in predictions], [0, 1]
            )


if __name__ == "__main__":
    unittest.main()
