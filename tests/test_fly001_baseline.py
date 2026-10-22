"""FLY-001 baseline calculations remain isolated from training and app state."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training.fly001_baseline import (
    classification_metrics,
    evaluate,
    latency_summary,
    load_cases,
)


class TestDatasetValidation(unittest.TestCase):
    def write(self, directory: str, rows: list[str]) -> Path:
        path = Path(directory) / "cases.jsonl"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_load_valid_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, [
                '{"input":"hi","route":"general"}',
                '{"input":"2 + 2","route":"math"}',
            ])
            self.assertEqual(len(load_cases(path, ["general", "math"])), 2)

    def test_reject_unknown_labels_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, ['{"input":"hi","route":"research"}'])
            with self.assertRaisesRegex(ValueError, "invalid route"):
                load_cases(path, ["general", "math"])
            path = self.write(folder, [
                '{"input":"Hi","route":"general"}',
                '{"input":" hi ","route":"general"}',
            ])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cases(path, ["general", "math"])

    def test_reject_empty_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, ['{"input":"","route":"general"}'])
            with self.assertRaisesRegex(ValueError, "nonempty"):
                load_cases(path, ["general"])
            path = self.write(folder, ["NOT JSON"])
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_cases(path, ["general"])

    def test_committed_diagnostic_has_100_unique_cases(self):
        dataset = Path(__file__).resolve().parents[1] / "training/fly001_diagnostic.jsonl"
        rows = load_cases(dataset, ["general", "math", "code", "memory", "summarize", "research"])
        self.assertEqual(len(rows), 100)
        self.assertEqual(sum(x["route"] == "general" for x in rows), 20)


class TestBaselineMetrics(unittest.TestCase):
    def test_gate_vs_raw_correctness(self):
        samples = [
            {"expected": "general", "raw": {"route": "general", "accepted": False}},
            {"expected": "math", "raw": {"route": "math", "accepted": True}},
            {"expected": "math", "raw": {"route": "general", "accepted": True}},
        ]
        metrics = classification_metrics(samples, "raw", ["general", "math"])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3, places=6)
        self.assertEqual(metrics["accepted"], 2)
        self.assertEqual(metrics["accepted_correct"], 1)
        self.assertAlmostEqual(metrics["usable_rate"], 1 / 3, places=6)
        self.assertEqual(metrics["confusion"]["math"]["general"], 1)
        self.assertEqual(metrics["by_route"]["general"]["support"], 1)
        self.assertEqual(metrics["by_route"]["math"]["support"], 2)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 2 / 3) / 2, places=6)

    def test_latency_nearest_rank(self):
        lat = latency_summary([1, 2, 3, 4, 100])
        self.assertEqual(lat["p95_ms"], 100)
        self.assertEqual(lat["median_ms"], 3)


class FakeRuntime:
    routes = ["general", "math"]

    def __init__(self):
        self.calls = 0

    def predict(self, text: str, *, top_k: int):
        self.calls += 1
        # Simulate the real misclassification preserved by the guard.
        return {
            "route": "math",
            "confidence": 0.8,
            "margin": 0.65,
            "accepted": True,
            "top_routes": [
                {"route": "math", "confidence": 0.8},
                {"route": "general", "confidence": 0.15},
            ],
        }


class TestEffectiveRoute(unittest.TestCase):
    def test_semantic_guard_does_not_erase_raw_prediction(self):
        runtime = FakeRuntime()
        results = evaluate(
            runtime,
            [{"input": "너 누구야?", "route": "general"}],
            warmup=0,
        )
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(results["raw"]["accuracy"], 0.0)
        self.assertEqual(results["effective"]["accuracy"], 1.0)
        self.assertEqual(results["overrides"]["corrected"], 1)
        self.assertEqual(results["cases"][0]["raw"]["route"], "math")
        self.assertEqual(results["cases"][0]["effective"]["route"], "general")

    def test_warmup_predictions_are_not_counted(self):
        runtime = FakeRuntime()
        results = evaluate(
            runtime,
            [{"input": "2+2는?", "route": "math"}],
            warmup=3,
        )
        self.assertEqual(runtime.calls, 4)
        self.assertEqual(results["raw"]["total"], 1)
        self.assertEqual(results["effective"]["total"], 1)


if __name__ == "__main__":
    unittest.main()
