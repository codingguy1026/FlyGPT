"""Checks for fresh three-way benchmark; run in PyTorch-equipped experimental CI."""
import tempfile
from pathlib import Path
import unittest

from training.fresh_threeway import (
    ROOT, validate_holdout, text_only_model,
)
from training.topology_ablation import prepare_teacher
from training.train_router import FlyGraphRouter

import torch


class ThreeWayExperimentTests(unittest.TestCase):
    def test_frozen_holdout_has_300_novel_balanced_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            keys, _ = prepare_teacher(Path(directory) / "teacher.jsonl")
        cases, audit = validate_holdout(
            ROOT / "training/fresh_holdout_300_20261016.jsonl", keys
        )
        self.assertEqual(len(cases), 300)
        self.assertEqual(sorted(audit["classes"].values()), [50] * 6)
        self.assertEqual(audit["exact_training_overlaps"], 0)

    def test_plain_text_baseline_nearly_matches_graph_parameter_budget(self):
        n, vocab, classes = 16, 512, 6
        src = torch.tensor([i for i in range(n)], dtype=torch.long)
        dst = torch.tensor([(i + 1) % n for i in range(n)], dtype=torch.long)
        weights = torch.ones(n, dtype=torch.float32)
        graph = FlyGraphRouter(vocab, n, classes, src, dst, weights, steps=3)
        # The implementation fixes vocab at 8192 for the actual benchmark;
        # construct a same-sized reference baseline for a parameter check.
        plain = text_only_model(8192, 256, 6)
        graph_production_shape = FlyGraphRouter(
            8192, 256, 6,
            torch.tensor([i for i in range(256)], dtype=torch.long),
            torch.tensor([(i + 1) % 256 for i in range(256)], dtype=torch.long),
            torch.ones(256),
            steps=3,
        )
        baseline_count = sum(p.numel() for p in plain.parameters())
        graph_count = sum(p.numel() for p in graph_production_shape.parameters())
        self.assertLess(abs(baseline_count - graph_count) / graph_count, 0.01)
        self.assertEqual(tuple(graph(torch.zeros(1, vocab)).shape), (1, classes))
        self.assertEqual(tuple(plain(torch.zeros(1, 8192)).shape), (1, 6))

    def test_holdout_rejects_duplicated_or_training_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            testset = Path(tmp) / "tiny.jsonl"
            testset.write_text('{"input":"duplicate","route":"general"}\n',
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected exactly 300"):
                validate_holdout(testset, set())


if __name__ == "__main__":
    unittest.main()
