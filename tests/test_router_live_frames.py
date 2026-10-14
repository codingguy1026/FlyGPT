"""Trace callback must match the actual forward pass (not a demo)."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class RouterFrameTests(unittest.TestCase):
    def test_emitted_frames_match_final_trace_exactly(self):
        import torch
        from router_runtime import FlyRouterRuntime
        from training.train_router import FlyGraphRouter

        torch.manual_seed(7)
        src = torch.tensor([0, 1, 2, 3, 2, 1], dtype=torch.long)
        dst = torch.tensor([1, 2, 3, 0, 0, 3], dtype=torch.long)
        weight = torch.ones(len(src), dtype=torch.float32)
        model = FlyGraphRouter(
            vocab_size=32, n_nodes=4, n_classes=2,
            src=src, dst=dst, base_weight=weight, steps=2,
        )
        checkpoint = {
            "routes": ["general", "math"],
            "vocab_size": 32,
            "n_nodes": 4,
            "steps": 2,
            "src": src,
            "dst": dst,
            "base_weight": weight,
            "model_state": model.state_dict(),
            "scaffold_meta": {"kind": "synthetic-test", "root_ids": [1, 2, 3, 4]},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.pt"
            torch.save(checkpoint, path)
            runtime = FlyRouterRuntime(path)
            frames = []
            live_result = runtime.predict("안녕", top_k=2, on_frame=frames.append)
            normal_result = runtime.predict("안녕", top_k=2)

        self.assertEqual(len(frames), 3)
        self.assertEqual([row["stage"] for row in frames], ["input", "step_1", "step_2"])
        self.assertEqual(frames, live_result["trace"])
        self.assertEqual(live_result["trace"], normal_result["trace"])
        self.assertEqual(live_result["route"], normal_result["route"])


if __name__ == "__main__":
    unittest.main()
