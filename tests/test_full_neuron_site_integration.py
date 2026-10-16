"""Verify a real saved full-neuron model can power the FlyGPT site adapter.

Skipped in ordinary dependency-light tests; experimental GitHub Actions passes
FLYGPT_TEST_FULL_POPULATION_CKPT after verifying archived artifact SHA256.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    os.environ.get("FLYGPT_TEST_FULL_POPULATION_CKPT"),
    "requires real 166700-neuron GitHub Actions checkpoint",
)
class ActualFullNeuronSiteTests(unittest.TestCase):
    def test_real_checkpoint_load_and_route(self):
        from full_population_runtime import FullPopulationRouterRuntime

        path = os.environ["FLYGPT_TEST_FULL_POPULATION_CKPT"]
        with patch.dict(os.environ, {"FLYGPT_ENABLE_FULL_POPULATION_ROUTER": "0"}):
            with self.assertRaisesRegex(ValueError, "explicit opt-in"):
                FullPopulationRouterRuntime(path)

        with patch.dict(os.environ, {"FLYGPT_ENABLE_FULL_POPULATION_ROUTER": "1"}):
            router = FullPopulationRouterRuntime(path)
        self.assertEqual(router.n_nodes, 166700)
        self.assertEqual(router.n_edges, 25582938)
        self.assertEqual(router.scaffold_meta["kind"], "male-cns:v1.0")
        self.assertEqual(
            router.model.keys.shape[0],
            router.n_nodes,
        )
        self.assertEqual(len(router.routes), 6)

        graph = router.graph()
        self.assertTrue(graph["full_population_summary"])
        self.assertEqual(graph["n_nodes"], 166700)
        self.assertEqual(graph["nodes"], [])
        self.assertEqual(graph["edges"], [])

        stream_frames = []
        prediction = router.predict(
            "파이썬으로 정렬 알고리즘 코드를 만들어 줘",
            on_frame=stream_frames.append,
        )
        self.assertIn(prediction["route"], router.routes)
        self.assertTrue(prediction["all_curated_neurons_attended"])
        self.assertTrue(prediction["full_population_summary"])
        self.assertEqual(prediction["trace"], [])
        self.assertEqual(stream_frames, [])

    def test_runtime_full_population_math_matches_exact_all_neurons(self):
        import torch
        from full_population_runtime import FullPopulationRouterRuntime
        from training.train_router import vectorize

        with patch.dict(os.environ, {"FLYGPT_ENABLE_FULL_POPULATION_ROUTER": "1"}):
            router = FullPopulationRouterRuntime(
                os.environ["FLYGPT_TEST_FULL_POPULATION_CKPT"]
            )
        x = vectorize(
            "오늘 대구 날씨 관련 공식 최신 자료를 찾아줘",
            8192, "v6",
        ).unsqueeze(0)
        with torch.no_grad():
            _, q = router.model.query_vector(x)
            exact_weights = 1 + .85*(q @ router.model.keys.T)
            self.assertTrue(bool((exact_weights > 0).all()))
            direct = (exact_weights @ router.model.keys) / exact_weights.sum(dim=1)[:, None]
            cached = router.model.full_population_state(q)
        torch.testing.assert_close(cached, direct, atol=5e-6, rtol=5e-6)


if __name__ == "__main__":
    unittest.main()
