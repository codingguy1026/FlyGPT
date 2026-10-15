"""Biological-role priors remain optional, reproducible and non-destructive."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

from training.role_prior import ROLE_GAINS, validate_role_assignment


def sample_assignment():
    return {
        "source": "Janelia MaleCNS v1.0 via neuPrint",
        "dataset": "male-cns:v1.0",
        "roles": {
            "1": "sensory", "2": "interneuron",
            "3": "motor", "4": "unclassified",
        },
    }


class RoleValidationTests(unittest.TestCase):
    def setUp(self):
        self.meta = {"kind": "male-cns:v1.0", "root_ids": [1, 2, 3, 4]}

    def test_exact_ids_order_and_coverage(self):
        roles, meta = validate_role_assignment(sample_assignment(), self.meta, 4)
        self.assertEqual(roles, ["sensory", "interneuron", "motor", "unclassified"])
        self.assertEqual(meta["counts"]["sensory"], 1)
        self.assertEqual(meta["annotated_count"], 3)

    def test_reject_mismatch_and_unknown(self):
        for roles in [
            {"1": "sensory", "2": "motor", "3": "motor"},
            {"1": "sensory", "2": "motor", "3": "motor", "4": "unknown-role"},
            {"1": "sensory", "2": "motor", "3": "motor", "4": "unclassified", "5": "sensory"},
        ]:
            with self.subTest(roles=roles):
                bad = sample_assignment()
                bad["roles"] = roles
                with self.assertRaises(ValueError):
                    validate_role_assignment(bad, self.meta, 4)

    def test_reject_bad_provenance_or_different_scaffold(self):
        bad = sample_assignment()
        bad["source"] = "user generated"
        with self.assertRaises(ValueError):
            validate_role_assignment(bad, self.meta, 4)
        with self.assertRaises(ValueError):
            validate_role_assignment(sample_assignment(), {"kind": "synthetic", "root_ids": [1, 2, 3, 4]}, 4)

    def test_all_unclassified_does_not_fake_role_training(self):
        bad = sample_assignment()
        bad["roles"] = {str(i): "unclassified" for i in range(1, 5)}
        with self.assertRaisesRegex(ValueError, "No identified"):
            validate_role_assignment(bad, self.meta, 4)

    def test_all_role_gains_nonzero(self):
        for category, gains in ROLE_GAINS.items():
            with self.subTest(role=category):
                self.assertEqual(len(gains), 3)
                self.assertTrue(all(value > 0 for value in gains))
        self.assertEqual(ROLE_GAINS["unclassified"], (1.0, 1.0, 1.0))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional PyTorch")
class RoleGraphModelTests(unittest.TestCase):
    def test_neutral_role_router_matches_base_with_same_parameters(self):
        import torch
        from training.train_router import FlyGraphRouter, RoleAwareFlyGraphRouter

        torch.manual_seed(4)
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        weight = torch.ones(4)
        base = FlyGraphRouter(32, 4, 2, src, dst, weight, 3)
        role = RoleAwareFlyGraphRouter(32, 4, 2, src, dst, weight, 3, roles=["unclassified"] * 4)
        role.load_state_dict(base.state_dict(), strict=False)
        x = torch.randn(3, 32)
        self.assertTrue(torch.allclose(base(x), role(x), atol=1e-6))

    def test_biology_bias_changes_model_while_preserving_all_edges(self):
        import torch
        from training.train_router import FlyGraphRouter, RoleAwareFlyGraphRouter

        torch.manual_seed(4)
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        weight = torch.ones(4)
        base = FlyGraphRouter(32, 4, 2, src, dst, weight, 2)
        role = RoleAwareFlyGraphRouter(
            32, 4, 2, src, dst, weight, 2,
            roles=["sensory", "interneuron", "motor", "projection"],
        )
        role.load_state_dict(base.state_dict(), strict=False)
        x = torch.randn(3, 32)
        self.assertFalse(torch.allclose(base(x), role(x)))
        self.assertEqual(role.src.tolist(), base.src.tolist())
        self.assertEqual(role.dst.tolist(), base.dst.tolist())

    def test_role_aware_checkpoint_runtime_uses_exact_training_forward(self):
        import torch
        from training.train_router import RoleAwareFlyGraphRouter, vectorize
        from router_runtime import FlyRouterRuntime

        torch.manual_seed(4)
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([1, 2, 3, 0])
        weights = torch.ones(4)
        roles = ["sensory", "interneuron", "motor", "projection"]
        model = RoleAwareFlyGraphRouter(32, 4, 2, src, dst, weights, 2, roles=roles)
        model.eval()
        ckpt = {
            "model_state": model.state_dict(),
            "role_aware": True,
            "role_labels": roles,
            "role_metadata": {"source": "synthetic-test"},
            "routes": ["general", "math"], "vocab_size": 32, "n_nodes": 4,
            "steps": 2, "src": src, "dst": dst, "base_weight": weights,
            "scaffold_meta": {"kind": "male-cns:v1.0", "root_ids": [1, 2, 3, 4]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_role.pt"
            torch.save(ckpt, path)
            runtime = FlyRouterRuntime(path)
            emitted = []
            result = runtime.predict("안녕", on_frame=emitted.append)
            with torch.no_grad():
                exact = runtime.model(vectorize("안녕", 32).unsqueeze(0))
                probs = torch.softmax(exact / runtime.temperature, dim=1)
            self.assertEqual(result["role_aware_experimental"], True)
            self.assertEqual(result["trace"], emitted)
            self.assertAlmostEqual(result["confidence"], probs.max().item(), places=5)


if __name__ == "__main__":
    unittest.main()
