"""Compositional semantic planning and language-generation regression tests.

All LLM output is stubbed. These tests assert the grounding contract and
generation safety, not fluency of a real installed Ollama model.
"""

from __future__ import annotations

from copy import deepcopy
import os
import unittest
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from generator_runtime import (
    GenerationResult,
    GeneratorRuntime,
    _contains_internal_plan,
    _wrong_output_language,
)
from semantic_planner import (
    PROJECT_FACTS,
    compose_fact_plan,
    trusted_fact_fallback,
)


def state_for(message: str, route: str = "general") -> dict:
    return FlyBrainRuntime().plan(
        message=message,
        route_info={
            "route": route,
            "confidence": 0.90,
            "margin": 0.31,
            "accepted": True,
            "trace": [{"stage": "step_3", "activations": [0.2, 0.8, 0.1]}],
        },
        dispatch={"route": route, "status": "ready"},
    )


def model_answer(text: str) -> GenerationResult:
    return GenerationResult(
        used=True, provider="ollama-local", model="qwen2.5:0.5b-instruct",
        answer=text, latency_ms=20,
    )


class CompositionalPlannerTests(unittest.TestCase):
    def test_intro_question_focus_selects_distinct_facts(self):
        generic = state_for("너 누구야?")["utterance_plan"]
        named = state_for("너 이름이 뭐야?")["utterance_plan"]
        brain = state_for("네 뇌가 뭐야?")["utterance_plan"]
        for plan in (generic, named, brain):
            self.assertTrue(plan["ready"])
            self.assertEqual(plan["speech_act"], "introduce_flygpt")
            self.assertTrue(plan["semantic_steps"])
            self.assertEqual(plan["style"]["language"], "ko")
            self.assertEqual(
                [x["fact_id"] for x in plan["content_units"]],
                [x["fact_id"] for x in plan["semantic_steps"]],
            )
        self.assertEqual(
            [x["fact_id"] for x in named["content_units"]],
            ["identity.name", "identity.role"],
        )
        self.assertIn(
            "architecture.language",
            [x["fact_id"] for x in brain["content_units"]],
        )
        self.assertNotEqual(
            [x["fact_id"] for x in generic["content_units"]],
            [x["fact_id"] for x in brain["content_units"]],
        )
        self.assertEqual(
            [x["fact_id"] for x in generic["content_units"]],
            ["identity.name", "identity.role", "architecture.router"],
        )

    def test_addressed_variation_is_understood(self):
        for text in ("파피티야, 너 누구야?", "파피티야 누구야?"):
            with self.subTest(text=text):
                self.assertTrue(state_for(text)["utterance_plan"]["ready"])

    def test_locally_grounded_korean_and_english_facts(self):
        cases = [
            ("너 누구야?", "ko"),
            ("What are you?", "en"),
            ("What is your name?", "en"),
            ("What is your brain?", "en"),
            ("MaleCNS가 뭐야?", "ko"),
            ("What is MaleCNS?", "en"),
            ("너 뭐 할 수 있어?", "ko"),
            ("What can you do?", "en"),
        ]
        for text, language in cases:
            with self.subTest(text=text):
                plan = state_for(text)["utterance_plan"]
                self.assertTrue(plan["ready"])
                self.assertEqual(plan["style"]["language"], language)
                for item in plan["content_units"]:
                    self.assertEqual(item["kind"], "grounded_project_fact")
                    self.assertEqual(
                        item["value"], PROJECT_FACTS[item["fact_id"]][language]
                    )
                self.assertEqual(
                    trusted_fact_fallback(plan),
                    " ".join(unit["value"] for unit in plan["content_units"]),
                )

    def test_disallowed_facts_and_steps_cannot_be_fallback_evidence(self):
        plan = state_for("너 누구야?")["utterance_plan"]
        for edit in ("incorrect_fact", "unknown_fact_id", "wrong_order", "user_provenance"):
            with self.subTest(edit=edit):
                altered = deepcopy(plan)
                if edit == "incorrect_fact":
                    altered["content_units"][0]["value"] = "I am secretly a human."
                elif edit == "unknown_fact_id":
                    altered["content_units"][0]["fact_id"] = "invented.fact"
                elif edit == "wrong_order":
                    altered["semantic_steps"] = list(reversed(altered["semantic_steps"]))
                elif edit == "user_provenance":
                    altered["content_units"][0]["provenance"] = "user_assertion"
                self.assertIsNone(trusted_fact_fallback(altered))

    def test_unplanned_and_non_general_routes_remain_closed(self):
        for question in ("왜 하늘은 파래?", "뭐야?", "무작위 질문"):
            self.assertFalse(state_for(question)["utterance_plan"]["ready"])
        self.assertFalse(state_for("너 누구야?", "math")["utterance_plan"]["ready"])
        self.assertIsNone(compose_fact_plan("test", "unknown_intent", "ko"))


class LocalizedGenerationTests(unittest.TestCase):
    def runtime(self):
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "1",
            "FLYGPT_GENERATOR_MODEL": "qwen2.5:0.5b-instruct",
        }, clear=True):
            return GeneratorRuntime()

    def test_compact_grounded_prompt_contains_facts_not_raw_json(self):
        plan = state_for("너 누구야?")
        runtime = self.runtime()
        messages = runtime._messages(
            "SECRET USER QUESTION", "general",
            [{"role": "assistant", "content": "SECRET HISTORY"}],
            tool_context="SECRET TOOL",
            brain_state=plan,
        )
        joined = "\n".join(item["content"] for item in messages)
        self.assertIn("최종 한국어 답변", joined)
        self.assertIn("파피티", joined)
        self.assertNotIn("mouth_only_v1", joined)
        self.assertNotIn('"speech_act"', joined)
        self.assertNotIn("semantic_steps", joined)
        self.assertNotIn("fact_id", joined)
        self.assertNotIn("SECRET USER QUESTION", joined)
        self.assertNotIn("SECRET HISTORY", joined)
        self.assertNotIn("SECRET TOOL", joined)
        self.assertEqual(len(messages), 2)

    def test_english_prompt_in_english_with_facts(self):
        messages = self.runtime()._messages(
            "Who are you?", "general", None,
            brain_state=state_for("Who are you?"),
        )
        self.assertIn("Final English reply", messages[-1]["content"])
        self.assertIn("The project is named FlyGPT", messages[-1]["content"])

    def test_english_echo_for_korean_question_falls_back_to_atomic_facts(self):
        runtime = self.runtime()
        plan = state_for("너 누구야?")
        with patch.object(runtime, "_generate_once", return_value=model_answer(
            "FlyGPT, also called Papiti, is an experimental AI assistant."
        )) as generate:
            result = runtime.generate("너 누구야?", "general", brain_state=plan)
        self.assertTrue(result.used)
        self.assertEqual(result.finish_reason, "safe_surface_fallback")
        self.assertEqual(
            result.answer,
            trusted_fact_fallback(plan["utterance_plan"]),
        )
        self.assertIn("파피티", result.answer)
        generate.assert_called_once()

    def test_structural_leaks_are_detected_and_replaced(self):
        runtime = self.runtime()
        plan = state_for("너 누구야?")
        for leaked in (
            '{"semantic_steps":[{"fact_id":"identity.name"}]}',
            "말할 목적: 누구인지 설명\n사용할 사실: ...",
            "speech_act: introduce_flygpt",
        ):
            with self.subTest(leaked=leaked), patch.object(
                runtime, "_generate_once", return_value=model_answer(leaked)
            ):
                result = runtime.generate("너 누구야?", "general", brain_state=plan)
                self.assertTrue(result.used)
                self.assertNotIn("fact_id", result.answer)
                self.assertEqual(result.finish_reason, "safe_surface_fallback")

    def test_valid_korean_model_output_passes_unchanged(self):
        runtime = self.runtime()
        with patch.object(
            runtime, "_generate_once",
            return_value=model_answer("나는 FlyGPT 파피티야. MaleCNS를 활용해서 질문 종류를 구분해."),
        ):
            result = runtime.generate(
                "너 누구야?", "general", brain_state=state_for("너 누구야?")
            )
        self.assertTrue(result.used)
        self.assertEqual(result.finish_reason, None)
        self.assertIn("나는 FlyGPT", result.answer)

    def test_tampered_grounding_is_rejected_before_model_call(self):
        runtime = self.runtime()
        state = state_for("너 누구야?")
        state["utterance_plan"]["content_units"][0]["value"] = "new fabricated fact"
        with patch.object(runtime, "_generate_once") as generate:
            result = runtime.generate("너 누구야?", "general", brain_state=state)
        self.assertFalse(result.used)
        self.assertEqual(result.error, "grounded semantic fact plan is invalid")
        generate.assert_not_called()

    def test_unrelated_generations_keep_the_existing_contract(self):
        plan = state_for("안녕")["utterance_plan"]
        self.assertNotIn("semantic_steps", plan)
        self.assertFalse(_wrong_output_language("Hello", plan))
        self.assertFalse(_contains_internal_plan("안녕, 반가워!"))


if __name__ == "__main__":
    unittest.main()
