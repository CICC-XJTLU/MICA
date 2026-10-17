"""CPU checkpoint restoration tests for optimizer, scheduler, and RNG state."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train import load_trainable, restore_rng, save_resume, trainable_state


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Dropout(0.3),
            torch.nn.GELU(),
            torch.nn.Linear(8, 2),
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9
        )
        self.x, self.y = torch.randn(3, 4), torch.tensor([0, 1, 0])

    def update(self):
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(self.x)
        torch.nn.functional.cross_entropy(logits, self.y).backward()
        self.optimizer.step()
        self.scheduler.step()
        return logits.detach().clone()

    def test_next_update_replays_exactly_on_cpu(self):
        self.update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            identity = {"seed": 42}
            cursor = {"step": 1, "epoch": 0, "next_batch": 8}
            save_resume(
                path,
                self.model,
                self.optimizer,
                self.scheduler,
                identity,
                cursor,
                -1,
                0,
            )
            first = self.update()
            reference = trainable_state(self.model)
            reference_lr = self.scheduler.get_last_lr()
            first_draws = (random.random(), float(np.random.random()))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["identity"], identity)
            self.assertEqual(checkpoint["cursor"], cursor)
            load_trainable(self.model, checkpoint["trainable"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            restore_rng(checkpoint["rng"])
            repeated = self.update()
            replayed = trainable_state(self.model)
            self.assertTrue(torch.equal(first, repeated))
            self.assertTrue(
                all(torch.equal(reference[k], replayed[k]) for k in reference)
            )
            self.assertEqual(reference_lr, self.scheduler.get_last_lr())
            self.assertEqual(first_draws, (random.random(), float(np.random.random())))

    def test_incomplete_parameter_state_is_rejected(self):
        state = trainable_state(self.model)
        state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            load_trainable(self.model, state)

    def test_unexpected_parameter_state_is_rejected(self):
        state = trainable_state(self.model)
        state["extra"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            load_trainable(self.model, state)

    def test_epoch_order_suffix_matches_resume_cursor(self):
        count, batch_size, next_batch = 19344, 2, 8
        original = torch.randperm(
            count, generator=torch.Generator().manual_seed(42)
        ).tolist()
        resumed = torch.randperm(
            count, generator=torch.Generator().manual_seed(42)
        ).tolist()
        self.assertEqual(original[16:], resumed[next_batch * batch_size :])


if __name__ == "__main__":
    unittest.main()
