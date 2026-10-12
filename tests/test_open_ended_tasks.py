"""Regression coverage for the previously unreachable open-ended chat paths.

No neuPrint access or model download is needed. The model is stubbed to
verify that accepted reasoning requests *reach* generation, while retrieval
and authenticated memory restrictions remain enforced.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from generator_runtime import GenerationResult, GeneratorRuntime
from research_runtime import ResearchResult, WebSource


LOGIC_PUZZLE = (
    "A, B, C, D, E 5명이 일렬로 서 있습니다. A는 C의 바로 오른쪽에 있고, "
    "B는 양 끝에 서지 않습니다. D는 E와 이웃해 있으며, E는 A와 두 칸 떨어져 "
    "있습니다. C가 왼쪽에서 두 번째라면 정확한 순서는 무엇인가요?"
)


def prepare(text: str, route: str = "general") -> dict:
    brain = FlyBrainRuntime()
    return brain.plan(
        route_info={"route": route, "confidence": 0.8, "margin": 0.3, "accepted": True},
        dispatch={"route": route, "status": "ready"},
        message=text,
    )


def runtime() -> GeneratorRuntime:
    with patch.dict(
        os.environ,
        {"FLYGPT_LOCAL_ONLY": "1", "FLYGPT_GENERATOR_MODEL": "qwen2.5:0.5b-instruct"},
        clear=True,
    ):
        return GeneratorRuntime()


def generated(text: str) -> GenerationResult:
    return GenerationResult(
        used=True,
        provider="ollama-local",
        model="qwen2.5:0.5b-instruct",
        answer=text,
        latency_ms=15,
    )


class OpenEndedTaskTests(unittest.TestCase):
    def test_logic_puzzle_reaches_generator_and_preserves_question(self):
        state = prepare(LOGIC_PUZZLE)
        self.assertFalse(state["utterance_plan"]["ready"])
        state = FlyBrainRuntime().finalize(state)
        self.assertEqual(state["task_plan"]["contract"], "bounded_task_v1")
        self.assertTrue(state["task_plan"]["ready"])

        rt = runtime()
        with patch.object(
            rt,
            "_generate_once",
            return_value=generated("조건을 모두 만족하는 배치는 없습니다."),
        ) as call:
            result = rt.generate(LOGIC_PUZZLE, "general", brain_state=state)

        self.assertTrue(result.used)
        self.assertIn("배치는 없습니다", result.answer)
        call.assert_called_once()
        messages = call.call_args.kwargs["messages"]
        self.assertEqual(messages[-1]["content"], LOGIC_PUZZLE)
        self.assertIn("check all candidates", messages[0]["content"])
        self.assertNotIn("utterance_plan", str(messages))

    def test_code_math_summarize_are_delegated_only_without_exact_plan(self):
        examples = [
            ("code", "이 Python 코드를 고쳐 줘: print((", "print('ok')"),
            ("math", "논리 퍼즐의 조건을 검증해 줘", "조건을 검토하겠습니다."),
            ("summarize", "다음 글을 요약해: 햇빛이 비친다.", "햇빛이 비칩니다."),
        ]
        for route, query, answer in examples:
            with self.subTest(route=route):
                state = FlyBrainRuntime().finalize(prepare(query, route))
                self.assertTrue(state["task_plan"]["ready"])
                rt = runtime()
                with patch.object(rt, "_generate_once", return_value=generated(answer)):
                    result = rt.generate(query, route, brain_state=state)
                self.assertTrue(result.used)
                self.assertEqual(result.answer, answer)

        exact = FlyBrainRuntime().finalize(
            prepare("3x+7=22", "math"), tool_context="Exact math tool result: x = 5"
        )
        self.assertTrue(exact["utterance_plan"]["ready"])
        self.assertNotIn("task_plan", exact)

    def test_search_result_round_trip_enables_evidence_bound_generation(self):
        result = ResearchResult(
            query="recent research",
            provider="brave-llm-context",
            sources=[WebSource(
                url="https://example.org/study",
                title="Study",
                hostname="example.org",
                root_domain="example.org",
                snippets=["Research found a result."],
                age=[],
                authority=0.8,
            )],
            retrieved_at=10.0,
        )
        self.assertTrue(result.used)
        self.assertTrue(result.to_dict()["used"])
        state = FlyBrainRuntime().finalize(
            prepare("Summarize recent research", "research"),
            research_result=result.to_dict(),
            tool_context=result.context(),
        )
        self.assertTrue(state["task_plan"]["ready"])
        self.assertEqual(state["task_plan"]["grounding"], "retrieved_sources")

        rt = runtime()
        with patch.object(rt, "_generate_once", return_value=generated(
            "제공된 자료에는 연구 결과가 나와 있습니다."
        )) as call:
            output = rt.generate(
                "Summarize recent research",
                "research",
                tool_context=result.context(),
                brain_state=state,
            )
        self.assertTrue(output.used)
        self.assertIn("example.org/study", str(call.call_args.kwargs["messages"]))

    def test_no_search_evidence_does_not_allow_open_ended_current_facts(self):
        no_evidence = ResearchResult(
            query="today scores", provider="brave-llm-context",
            sources=[], retrieved_at=10.0,
        )
        self.assertFalse(no_evidence.to_dict()["used"])
        state = FlyBrainRuntime().finalize(
            prepare("today scores", "research"),
            research_result=no_evidence.to_dict(),
        )
        self.assertNotIn("task_plan", state)
        self.assertEqual(state["utterance_plan"]["speech_act"], "report_no_external_evidence")

    def test_unfinalized_or_forged_task_context_is_still_rejected(self):
        state = prepare("Why?", "general")
        rt = runtime()
        with patch.object(rt, "_generate_once") as call:
            result = rt.generate("Why?", "general", brain_state=state)
            self.assertFalse(result.used)
            self.assertIn("no authorized", result.error)
            call.assert_not_called()

        state["task_plan"] = {
            "contract": "bounded_task_v1",
            "ready": True,
            "route": "general",
            "grounding": "retrieved_sources",
        }
        with patch.object(rt, "_generate_once") as call:
            result = rt.generate("Why?", "general", brain_state=state)
            self.assertFalse(result.used)
            call.assert_not_called()

    def test_grounded_fact_plans_cannot_be_bypassed_by_task_mode(self):
        state = prepare("너 누구야?")
        self.assertTrue(state["utterance_plan"]["ready"])
        state["utterance_plan"]["semantic_steps"][0]["fact_id"] = "invented"
        state["task_plan"] = {
            "contract": "bounded_task_v1",
            "ready": True,
            "route": "general",
            "grounding": "user_request",
        }
        rt = runtime()
        with patch.object(rt, "_generate_once") as call:
            result = rt.generate("너 누구야?", "general", brain_state=state)
        self.assertFalse(result.used)
        self.assertEqual(result.error, "grounded semantic fact plan is invalid")
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
