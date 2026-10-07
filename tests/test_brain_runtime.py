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

    def test_simple_greeting_produces_ready_mouth_only_plan(self) -> None:
        state = self.brain.plan(
            route_info={
                "route": "general",
                "confidence": 0.97,
                "margin": 0.5,
                "accepted": True,
                "trace": [],
            },
            dispatch={
                "route": "general",
                "status": "ready",
                "confidence": 0.97,
                "margin": 0.5,
                "tool_context": None,
            },
            message="안녕",
        )

        plan = state["utterance_plan"]
        self.assertEqual(plan["contract"], "mouth_only_v1")
        self.assertTrue(plan["ready"])
        self.assertEqual(plan["speech_act"], "return_greeting")
        self.assertFalse(plan["permissions"]["infer_new_meaning"])
        self.assertFalse(plan["permissions"]["add_new_questions"])

    def test_unplanned_general_question_stays_closed(self) -> None:
        state = self.brain.plan(
            route_info={
                "route": "general",
                "confidence": 0.97,
                "margin": 0.5,
                "accepted": True,
                "trace": [],
            },
            dispatch={
                "route": "general",
                "status": "ready",
                "confidence": 0.97,
                "margin": 0.5,
                "tool_context": None,
            },
            message="왜 하늘은 파래?",
        )

        self.assertFalse(state["utterance_plan"]["ready"])
        self.assertEqual(state["utterance_plan"]["speech_act"], "unresolved")

    def test_exact_math_result_becomes_explicit_content_unit(self) -> None:
        state = self.brain.plan(
            route_info={
                "route": "math",
                "confidence": 0.93,
                "margin": 0.4,
                "accepted": True,
                "trace": [],
            },
            dispatch={
                "route": "math",
                "status": "ready",
                "confidence": 0.93,
                "margin": 0.4,
                "tool_context": "Exact math tool result: x = 5",
            },
            message="3x+7=22 풀어줘",
        )
        final = self.brain.finalize(
            state,
            tool_context="Exact math tool result: x = 5",
        )

        plan = final["utterance_plan"]
        self.assertTrue(plan["ready"])
        self.assertEqual(plan["speech_act"], "state_exact_math_result")
        self.assertEqual(
            plan["content_units"][0]["value"],
            "Exact math tool result: x = 5",
        )


if __name__ == "__main__":
    unittest.main()
