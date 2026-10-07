from __future__ import annotations

import unittest

from brain_runtime import FlyBrainRuntime


class FlyBrainRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = FlyBrainRuntime()

    def test_memory_plan_uses_router_trace(self) -> None:
        state = self.brain.plan(
            route_info={
                "route": "memory",
                "confidence": 0.91,
                "margin": 0.33,
                "accepted": True,
                "trace": [
                    {"stage": "input", "activations": [0.1, 0.2, 0.3]},
                    {"stage": "step_3", "activations": [0.4, 0.9, 0.6]},
                ],
            },
            dispatch={
                "route": "memory",
                "status": "ready",
                "confidence": 0.91,
                "margin": 0.33,
                "tool_context": None,
            },
        )

        self.assertEqual(state["route"], "memory")
        self.assertEqual(state["retrieval"], "memory")
        self.assertEqual(state["certainty"], "high")
        self.assertEqual(state["neural_signature"][0]["node_index"], 1)
        self.assertEqual(state["neural_signature"][0]["activation"], 0.9)

    def test_memory_finalize_fails_closed_without_hits(self) -> None:
        state = self.brain.plan(
            route_info={
                "route": "memory",
                "confidence": 0.8,
                "margin": 0.2,
                "accepted": True,
                "trace": [],
            },
            dispatch={
                "route": "memory",
                "status": "ready",
                "confidence": 0.8,
                "margin": 0.2,
                "tool_context": None,
            },
        )
        final = self.brain.finalize(state, memory_hits=[])

        self.assertFalse(final["evidence"]["available"])
        self.assertEqual(final["evidence"]["count"], 0)
        self.assertTrue(
            any("no relevant prior-session memory" in item for item in final["directives"])
        )

    def test_math_tool_becomes_exact_math_retrieval(self) -> None:
        state = self.brain.plan(
            route_info={
                "route": "math",
                "confidence": 0.88,
                "margin": 0.3,
                "accepted": True,
                "trace": [],
            },
            dispatch={
                "route": "math",
                "status": "ready",
                "confidence": 0.88,
                "margin": 0.3,
                "tool_context": "Exact math tool result: 42",
            },
        )
        final = self.brain.finalize(
            state,
            tool_context="Exact math tool result: 42",
        )

        self.assertEqual(final["retrieval"], "exact_math")
        self.assertTrue(final["evidence"]["available"])
        self.assertEqual(final["evidence"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
