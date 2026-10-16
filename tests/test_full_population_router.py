"""Offline checks of exact whole-population summaries and test-site adapter.

Use synthetic keys to check the mathematics; this does NOT claim the official
MaleCNS 1.1GB data has been downloaded or trained in CI.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from training.full_population_router import FullPopulationRouter, load_all_structural_keys


class WholePopulationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.ids = torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int64)
        self.keys = F.normalize(torch.randn(6, 8), dim=1)
        self.model = FullPopulationRouter(self.ids, self.keys, 3, vocab_size=32, hidden=10)

    def test_all_neurons_have_positive_weight_in_exact_attention(self):
        q = F.normalize(torch.randn(3, 8), dim=1)
        exact_weights = 1. + .85 * (q @ self.model.keys.T)
        self.assertTrue(bool((exact_weights > 0).all()))
        full = (exact_weights @ self.model.keys) / exact_weights.sum(1)[:, None]
        moments = self.model.full_population_state(q)
        torch.testing.assert_close(full, moments, atol=2e-6, rtol=2e-6)

    def test_no_eligible_neuron_is_omitted_from_pooled_state(self):
        q = F.normalize(torch.randn(1, 8), dim=1)
        full = self.model.full_population_state(q)
        after_dropping_last = FullPopulationRouter(
            self.ids[:-1], self.keys[:-1], 3, vocab_size=32, hidden=10
        ).full_population_state(q)
        self.assertFalse(torch.allclose(full, after_dropping_last))

    def test_source_provenance_rejects_partial_sample(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)/"bad.pt"
            torch.save({
                "metadata": {
                    "dataset": "male-cns:v1.0",
                    "full_source_scanned_every_epoch": False,
                    "no_topk_or_node_sampling": False,
                    "training_kind": "offline_self_supervised_directed_edge_reconstruction",
                    "annotated_neurons_eligible": 6,
                },
                "body_ids": self.ids,
                "presynaptic_embeddings": self.keys[:, :4],
                "postsynaptic_embeddings": self.keys[:, 4:],
            }, p)
            with self.assertRaisesRegex(ValueError, "unsampled full MaleCNS"):
                load_all_structural_keys(p, minimum_neurons=1)

    def test_forward_and_top_contributors(self):
        out = self.model(torch.randn(2, 32))
        self.assertEqual(tuple(out.shape), (2, 3))
        visited = self.model.top_contributors(torch.randn(1, 32), limit=6)
        self.assertEqual(len(visited), 6)
        self.assertEqual({int(x["body_id"]) for x in visited}, set(self.ids.tolist()))

    def test_site_adapter_requires_explicit_opt_in(self):
        # Test only the flag, never use a synthetic checkpoint to claim real data.
        from full_population_runtime import FullPopulationRouterRuntime
        with patch.dict("os.environ", {"FLYGPT_ENABLE_FULL_POPULATION_ROUTER": "0"}):
            with self.assertRaisesRegex(ValueError, "explicit opt-in"):
                FullPopulationRouterRuntime("/nonexistent/model.pt")


if __name__ == "__main__":
    unittest.main()
