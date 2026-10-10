"""Regression tests for richer brain plans and safe language realization.

No neuPrint credentials, network calls, or local language-model download needed.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from generator_runtime import (
    GenerationResult,
    GeneratorRuntime,
    _contains_internal_plan,
)


def planned(message: str, route: str = "general") -> dict:
    return FlyBrainRuntime().plan(
        route_info={
            "route": route,
            "confidence": 0.98,
            "margin": 0.48,
            "accepted": True,
            "trace": [
                {"stage": "step_3", "activations": [0.2, 0.9, 0.3]},
            ],
        },
        dispatch={"route": route, "status": "ready"},
        message=message,
    )


def fake_generated(text: str) -> GenerationResult:
    return GenerationResult(
        used=True,
        provider="ollama-local",
        model="qwen2.5:0.5b-instruct",
        answer=text,
        latency_ms=25,
    )


class ExtendedBrainPlansTests(unittest.TestCase):
    def test_supported_ko_and_en_intents(self):
        examples = {
            "너 누구야?": "introduce_flygpt",
            "Who are you?": "introduce_flygpt",
            "너 뭐 할 수 있어?": "describe_supported_features",
            "What can you do?": "describe_supported_features",
            "MaleCNS가 뭐야?": "explain_malecns",
            "What is MaleCNS?": "explain_malecns",
            "잘 지내?": "acknowledge_presence",
            "How are you?": "acknowledge_presence",
            "도와줘": "request_specific_task",
            "Help me!": "request_specific_task",
            "반가워": "return_welcome",
            "Nice to meet you": "return_welcome",
        }
        for message, act in examples.items():
            with self.subTest(message=message):
                state = planned(message)
                plan = state["utterance_plan"]
                self.assertTrue(plan["ready"])
                self.assertEqual(plan["speech_act"], act)
                self.assertTrue(plan["content_units"])
                self.assertEqual(plan["semantic_authority"], "fly_brain")
                self.assertFalse(plan["permissions"]["infer_new_meaning"])
                self.assertEqual(state["source"], "connectome_router")
                self.assertEqual(state["neural_signature"][0]["node_index"], 1)

    def test_unsupported_questions_still_fail_closed(self):
        for message in (
            "왜 하늘이 파란가?",
            "이상한 질문을 해결해 줘",
            "뭐야?",
            "Tell me the latest baseball scores",
        ):
            with self.subTest(message=message):
                plan = planned(message)["utterance_plan"]
                self.assertFalse(plan["ready"])
                self.assertEqual(plan["speech_act"], "unresolved")

    def test_non_general_routes_do_not_use_social_plan(self):
        for route in ("memory", "code", "research", "math", "summarize"):
            with self.subTest(route=route):
                state = planned("너 누구야?", route=route)
                self.assertEqual(state["route"], route)
                self.assertFalse(state["utterance_plan"]["ready"])

    def test_verified_saved_knowledge_can_be_reported_with_provenance(self):
        brain = FlyBrainRuntime()
        state = planned("지난번에 기억한 내용을 알려줘")
        final = brain.finalize(state, knowledge_hits=[
            {
                "status": "verified",
                "kind": "fact",
                "statement": "A verified repository fact.",
            },
            {
                "status": "asserted",
                "kind": "personal",
                "statement": "The user likes a certain team.",
            },
            {
                "status": "claim",
                "kind": "fact",
                "statement": "UNVERIFIED: should not be shown.",
            },
            {
                "status": "disputed",
                "kind": "fact",
                "statement": "DISPUTED: should not be shown.",
            },
        ])
        plan = final["utterance_plan"]
        self.assertTrue(plan["ready"])
        self.assertEqual(plan["speech_act"], "report_relevant_stored_knowledge")
        self.assertEqual(
            [item["provenance"] for item in plan["content_units"]],
            ["verified", "user-asserted"],
        )
        self.assertEqual(final["evidence"]["count"], 2)
        self.assertNotIn("UNVERIFIED", str(plan))
        self.assertNotIn("DISPUTED", str(plan))

    def test_unverified_claims_do_not_make_plan_ready(self):
        final = FlyBrainRuntime().finalize(
            planned("왜 하늘이 파란가?"),
            knowledge_hits=[
                {"status": "claim", "kind": "fact", "statement": "Fake fact."},
                {"status": "asserted", "kind": "fact", "statement": "Another fake fact."},
            ],
        )
        self.assertFalse(final["utterance_plan"]["ready"])

    def test_previously_planned_intent_is_not_overwritten_by_knowledge(self):
        final = FlyBrainRuntime().finalize(
            planned("안녕"),
            knowledge_hits=[{
                "status": "verified", "kind": "fact", "statement": "A saved fact."
            }],
        )
        self.assertEqual(final["utterance_plan"]["speech_act"], "return_greeting")


class PlanDisclosureGuardTests(unittest.TestCase):
    def make_runtime(self):
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "1",
            "FLYGPT_GENERATOR_MODEL": "qwen2.5:0.5b-instruct",
        }, clear=True):
            return GeneratorRuntime()

    def test_detects_json_and_partial_serialized_plan(self):
        for text in (
            '{"contract":"mouth_only_v1","speech_act":"return_greeting"}',
            '```json\n{"utterance_plan":{"content_units":[]}}\n```',
            '안녕하세요! {"semantic_authority": "fly_brain"}',
            "speech_act: return_greeting",
            "Here is the Trusted upstream utterance plan:",
            "Final answer: brain_state={'route':'general'}",
        ):
            with self.subTest(text=text):
                self.assertTrue(_contains_internal_plan(text))

        for text in (
            "안녕! 오늘도 반가워.",
            "MaleCNS 데이터는 neuPrint에서 조회해요.",
            '{"status":"ok","result":42}',
            "설명에 따르면 이 답은 42예요.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_contains_internal_plan(text))

    def test_unsafe_greeting_is_replaced_with_fixed_prose(self):
        runtime = self.make_runtime()
        with patch.object(
            runtime, "_generate_once",
            return_value=fake_generated(
                '알겠어! {"contract":"mouth_only_v1","content_units":[]}'
            ),
        ) as gen:
            result = runtime.generate(
                "안녕",
                "general",
                brain_state=planned("안녕"),
            )
        self.assertTrue(result.used)
        self.assertEqual(result.answer, "안녕! 🪰")
        self.assertEqual(result.finish_reason, "safe_surface_fallback")
        gen.assert_called_once()

    def test_internal_metadata_blocked_for_exact_math_result(self):
        runtime = self.make_runtime()
        brain = FlyBrainRuntime()
        state = planned("4+4=", route="math")
        state = brain.finalize(state, tool_context="Exact math tool result: 8")
        self.assertTrue(state["utterance_plan"]["ready"])
        with patch.object(
            runtime, "_generate_once",
            return_value=fake_generated(
                'Here is the plan: {"speech_act":"state_exact_math_result"}'
            ),
        ):
            result = runtime.generate("4+4=", "math", brain_state=state)
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.finish_reason, "blocked_internal_plan")
        self.assertNotIn("speech_act", result.error)

    def test_stored_knowledge_has_no_predefined_fallback(self):
        runtime = self.make_runtime()
        plan = planned("왜 하늘이 파란가?")
        plan = FlyBrainRuntime().finalize(plan, knowledge_hits=[{
            "status": "verified",
            "kind": "fact",
            "statement": "Verified fact.",
        }])
        self.assertEqual(
            plan["utterance_plan"]["speech_act"],
            "report_relevant_stored_knowledge",
        )
        with patch.object(
            runtime, "_generate_once",
            return_value=fake_generated("speech_act: report_relevant_stored_knowledge"),
        ):
            result = runtime.generate(
                "왜 하늘이 파란가?",
                "general",
                brain_state=plan,
            )
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)

    def test_clean_prose_still_passes_unchanged(self):
        runtime = self.make_runtime()
        with patch.object(
            runtime, "_generate_once",
            return_value=fake_generated("안녕! 반가워."),
        ):
            result = runtime.generate(
                "안녕",
                "general",
                brain_state=planned("안녕"),
            )
        self.assertTrue(result.used)
        self.assertEqual(result.answer, "안녕! 반가워.")
        self.assertNotEqual(result.finish_reason, "safe_surface_fallback")


if __name__ == "__main__":
    unittest.main()
