import json
import random
from pathlib import Path
import torch
from PIL import Image


def read_rows(path):
    rows = []
    labels = {"sarcastic": 1, "not_sarcastic": 0, "non-sarcastic": 0, "0": 0, "1": 1}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not {"id", "text", "image", "label"} <= row.keys()
        ):
            raise ValueError(f"Line {line_number}: expected id, text, image and label")
        if (
            row["id"] is None
            or not isinstance(row["text"], str)
            or not isinstance(row["image"], str)
            or not row["image"]
        ):
            raise ValueError(f"Line {line_number}: invalid id, text or image")
        label = row["label"]
        if isinstance(label, str):
            if label not in labels:
                raise ValueError(f"Line {line_number}: unknown label {label!r}")
            label = labels[label]
        if label not in (0, 1):
            raise ValueError(f"Line {line_number}: label must be 0 or 1")
        row["label"] = int(label)
        rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def subset(rows, n, seed=2718):
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    buckets = [[r for r in rows if r["label"] == k] for k in (0, 1)]
    for b in buckets:
        rng.shuffle(b)
    n0 = round(n * len(buckets[0]) / len(rows))
    selected = buckets[0][:n0] + buckets[1][: n - n0]
    rng.shuffle(selected)
    return selected


class Collator:
    def __init__(self, processor, max_pixels=200704, min_pixels=100352):
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, rows):
        images, prompts = [], []
        for row in rows:
            with Image.open(row["image"]) as im:
                images.append(im.convert("RGB"))
            # Labels remain outside the processor inputs.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": row["image"]},
                        {"type": "text", "text": row["text"]},
                    ],
                }
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        ids = inputs["input_ids"]
        visual = ids.eq(self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        text = torch.zeros_like(visual)
        # Select caption tokens between the image and the end of the user turn.
        im_end = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for k in range(len(rows)):
            start = int(ids[k].eq(end_id).nonzero()[-1]) + 1
            after = (ids[k, start:].eq(im_end)).nonzero()
            stop = (
                start + int(after[0])
                if len(after)
                else int(inputs["attention_mask"][k].sum())
            )
            text[k, start:stop] = True
            if not text[k].any() or not visual[k].any():
                raise ValueError("empty modality mask")
        return dict(
            inputs=inputs,
            text_mask=text,
            visual_mask=visual,
            labels=torch.tensor([r["label"] for r in rows]),
            ids=[str(r["id"]) for r in rows],
        )


def device_batch(batch, device):
    return {
        k: (
            {n: t.to(device) for n, t in v.items()}
            if k == "inputs"
            else v.to(device) if torch.is_tensor(v) else v
        )
        for k, v in batch.items()
    }


def metrics(y, p):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    return dict(
        samples=len(y),
        accuracy=float(accuracy_score(y, p)),
        macro_f1=float(f1_score(y, p, average="macro")),
        confusion_matrix=confusion_matrix(y, p, labels=[0, 1]).tolist(),
        per_class=classification_report(
            y, p, labels=[0, 1], output_dict=True, zero_division=0
        ),
    )
