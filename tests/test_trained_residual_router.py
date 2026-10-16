"""Production-runtime compatibility tests for opt-in, genuinely trained MaleCNS model."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU PyTorch")
class TrainedResidualRouterTests(unittest.TestCase):
    def test_experimental_model_class_forward_and_trace_agree(self):
        import torch
        from training.train_router import FlyGraphRouter, ResidualFlyGraphRouter

        torch.manual_seed(7)
        src = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        dst = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        weights = torch.ones(4)
        router = ResidualFlyGraphRouter(32, 4, 2, src, dst, weights, steps=3)
        router.eval()
        data = torch.rand((2, 32))
        with torch.no_grad():
            trace = []
            actual = router.output(router.propagate(
                data, on_stage=lambda h, stage: trace.append(stage)
            ))
            direct = router(data)
            h_text = router.norm(router._encode(data))
            h_graph = FlyGraphRouter.propagate(router, data)
            expected = router.output(
                (1-router.graph_share())*h_text + router.graph_share()*h_graph
            )
        torch.testing.assert_close(direct, actual)
        torch.testing.assert_close(direct, expected)
        self.assertEqual(trace, ["input", "step_1", "step_2", "step_3", "residual_blend"])

    @unittest.skipUnless(os.environ.get("FLYGPT_TEST_RESIDUAL_CKPT"), "optional archived real checkpoint")
    def test_real_research_checkpoint_requires_opt_in_and_is_genuine(self):
        import torch
        from router_runtime import FlyRouterRuntime
        from training.train_router import FlyGraphRouter, vectorize

        path = Path(os.environ["FLYGPT_TEST_RESIDUAL_CKPT"])
        self.assertTrue(path.is_file())
        with patch.dict(os.environ, {"FLYGPT_ALLOW_RESEARCH_ROUTER": "0"}):
            with self.assertRaisesRegex(ValueError, "Research router disabled"):
                FlyRouterRuntime(path)

        with patch.dict(os.environ, {"FLYGPT_ALLOW_RESEARCH_ROUTER": "1"}):
            runtime = FlyRouterRuntime(path)
        self.assertTrue(runtime.residual_research)
        self.assertEqual(runtime.n_nodes, 256)
        self.assertEqual(runtime.src.numel(), 948)
        self.assertEqual(runtime.scaffold_meta["kind"], "male-cns:v1.0")
        self.assertEqual(len(runtime.scaffold_meta["root_ids"]), 256)
        self.assertEqual(runtime.vectorizer_version, "v6")
        self.assertTrue(0 < runtime.model.graph_share().item() < 1)
        self.assertEqual(runtime.model_path, path)

        question = "자바스크립트에서 클릭을 처리하는 코드 만들어줘"
        frames = []
        result = runtime.predict(question, on_frame=frames.append)
        self.assertEqual(frames, result["trace"])
        self.assertEqual(
            [frame["stage"] for frame in frames],
            ["input", "step_1", "step_2", "step_3", "residual_blend"]
        )
        with torch.no_grad():
            x = vectorize(question, runtime.vocab_size, runtime.vectorizer_version).unsqueeze(0)
            state_text = runtime.model.norm(runtime.model._encode(x))
            state_graph = FlyGraphRouter.propagate(runtime.model, x)
            alpha = runtime.model.graph_share()
            expected = runtime.model.output((1-alpha)*state_text + alpha*state_graph)
            scores = torch.softmax(expected / runtime.temperature, dim=1)
        expected_top = float(scores.max().item())
        self.assertAlmostEqual(result["confidence"], expected_top, places=5)
        self.assertTrue(result["residual_research"])
        self.assertIn(result["route"], runtime.routes)

        # Corrupted topology must be refused rather than misrepresented as
        # authentic MaleCNS. Do not alter the original input checkpoint.
        original = torch.load(path, map_location="cpu", weights_only=True)
        original["src"] = original["src"].clone()
        original["src"][0] = (original["src"][0] + 1) % 256
        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / "corrupt.pt"
            torch.save(original, broken)
            with patch.dict(os.environ, {"FLYGPT_ALLOW_RESEARCH_ROUTER": "1"}):
                with self.assertRaisesRegex(ValueError, "does not match the genuine MaleCNS graph"):
                    FlyRouterRuntime(broken)


if __name__ == "__main__":
    unittest.main()
