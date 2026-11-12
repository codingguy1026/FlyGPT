"""Tests for v0.5 all-neuron multi-capability policy. No live download."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from training.papiti_v0_5_policy import (
    CAPABILITIES, FullMaleCNSPolicy, enforce_unseen, load_policy_examples,
    measure, STRENGTH,
)


class CategoryFreePolicyTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([101, 203, 305, 407, 509], dtype=torch.long)
        self.keys = torch.tensor([
            [0.7, 0.1, 0.2], [-0.1, 0.6, 0.4], [0.3, 0.7, -0.2],
            [-0.5, 0.1, 0.8], [0.1, -0.2, 0.9],
        ])
        self.model = FullMaleCNSPolicy(self.ids, self.keys, hidden=16)

    def test_exact_population_weighted_direct_sum(self):
        q = torch.tensor([[0.3, 0.7, 0.5]])
        with torch.no_grad():
            actual = self.model.full_population_state(q)
            keys = F.normalize(self.keys, dim=1)
            normalized_q = F.normalize(q, dim=1)[0]
            gains = torch.ones(len(keys))
            weight = gains * (1.0 + STRENGTH * (keys @ normalized_q))
            expected = (weight[:, None] * keys).sum(0) / weight.sum()
        self.assertTrue(torch.allclose(actual[0], expected, atol=1e-6))

    def test_every_neuron_has_a_gradient(self):
        q = torch.tensor([[0.25, 0.15, 0.8]])
        vector = self.model.full_population_state(q)
        loss = (vector * torch.tensor([[1.0, 0.2, -0.4]])).sum()
        loss.backward()
        grad = self.model.neuron_gain_logits.grad
        self.assertEqual(grad.numel(), len(self.ids))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.all(grad.abs() > 1e-9))

    def test_each_neuron_changes_population_state(self):
        q = torch.tensor([[0.25, 0.15, 0.8]])
        baseline = self.model.full_population_state(q).detach()
        with torch.no_grad():
            self.model.neuron_gain_logits[-1] = 2.0
        changed = self.model.full_population_state(q).detach()
        self.assertFalse(torch.allclose(baseline, changed, atol=1e-7))

    def test_multiple_or_zero_capabilities_without_route_category(self):
        with torch.no_grad():
            self.model.action_head.weight.zero_()
            self.model.action_head.bias.fill_(9.0)
        selected = self.model.decide("지난번 환율로 계산해 줘", [0.5]*len(CAPABILITIES))
        self.assertEqual(selected["selected_capabilities"], list(CAPABILITIES))
        self.assertNotIn("route", selected)
        self.assertFalse(selected["tools_executed"])
        with torch.no_grad():
            self.model.action_head.bias.fill_(-9.0)
        unused = self.model.decide("그냥 이야기해", [0.5]*len(CAPABILITIES))
        self.assertEqual(unused["selected_capabilities"], [])

    def test_multilabel_accuracy_and_unnecessary_calls(self):
        with torch.no_grad():
            self.model.action_head.weight.zero_()
            self.model.action_head.bias.copy_(torch.tensor([9.0, 9.0, -9.0]))
        x = torch.zeros((2, 8192))
        y = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
        metrics = measure(self.model, x, y, [0.5]*len(CAPABILITIES))
        self.assertEqual(metrics["exact_action_set_accuracy"], .5)
        self.assertEqual(metrics["unnecessary_tool_rate_on_no_tool_requests"], 1.0)

    def test_holdout_leakage_rejected(self):
        rows = [{"input": "질문", "tools": []}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            enforce_unseen(rows, [{"input": " 질문 ", "tools": []}])

    def test_unknown_capability_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            path.write_text(json.dumps({"input": "hello", "tools": ["fixed_general"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown capability"):
                load_policy_examples(path)

    def test_incomplete_population_cannot_claim_full_training(self):
        with self.assertRaises(ValueError):
            FullMaleCNSPolicy(self.ids, self.keys[:-1])
        with self.assertRaises(ValueError):
            FullMaleCNSPolicy(torch.tensor([3, 2, 1]), self.keys[:3])


if __name__ == "__main__":
    unittest.main()
