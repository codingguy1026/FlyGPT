"""No-network regression tests for the experimental Papiti-native voice."""
from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from generator_runtime import GeneratorRuntime


ROUTE = {"route": "general", "confidence": 0.96, "margin": 0.75, "accepted": True}


def plan_for(message: str, route: str = "general", **finalize):
    brain = FlyBrainRuntime()
    route_info = {**ROUTE, "route": route}
    state = brain.plan(route_info=route_info, dispatch={"route": route}, message=message)
    return brain.finalize(state, **finalize)


def native_runtime():
    return GeneratorRuntime()


class NativeVoiceTests(unittest.TestCase):
    def test_greeting_requires_no_http_or_model_credentials(self):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "native",
            "FLYGPT_LOCAL_ONLY": "1",
            "FLYGPT_GENERATOR_URL": "https://untrusted.example/chat",
            "FLYGPT_GENERATOR_MODEL": "stale-model",
            "GEMINI_API_KEY": "stale-key",
        }, clear=True), patch("generator_runtime.urllib.request.urlopen") as send:
            runtime = native_runtime()
            result = runtime.generate("안녕", "general", brain_state=plan_for("안녕"))
        self.assertTrue(runtime.configured)
        self.assertEqual(runtime.provider_name, "papiti-native")
        self.assertEqual(result.provider, "papiti-native")
        self.assertTrue(result.used)
        self.assertIn("안녕", result.answer)
        self.assertIsNone(result.model)
        send.assert_not_called()
        self.assertFalse(runtime.status()["uses_language_model"])

    def test_self_description_uses_only_validated_fact_atoms(self):
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            runtime = native_runtime()
            state = plan_for("너 누구야?")
            result = runtime.generate("너 누구야?", "general", brain_state=state)
            self.assertTrue(result.used)
            self.assertIn("FlyGPT", result.answer)
            self.assertNotIn("utterance_plan", result.answer)
            bad = copy.deepcopy(state)
            bad["utterance_plan"]["content_units"][0]["value"] = "가짜로 바꾼 내용"
            rejected = runtime.generate("너 누구야?", "general", brain_state=bad)
            self.assertFalse(rejected.used)
            self.assertEqual(rejected.error_code, "native_plan_unavailable")

    def test_math_tool_result_comes_from_exact_upstream_plan(self):
        context = "Exact math tool result: 42"
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            runtime = native_runtime()
            state = plan_for("6*7", route="math", tool_context=context)
            result = runtime.generate("6*7", "math", brain_state=state, tool_context=context)
            self.assertEqual(result.answer, "🧮 42")
            forged = runtime.generate("6*7", "math", brain_state=state,
                                      tool_context="Exact math tool result: 99")
            self.assertFalse(forged.used)

    def test_missing_memory_is_reported_without_invention(self):
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            result = native_runtime().generate(
                "내가 전에 뭐라고 했어?", "memory",
                brain_state=plan_for("내가 전에 뭐라고 했어?", route="memory", memory_hits=[]),
            )
        self.assertTrue(result.used)
        self.assertIn("찾지 못했어", result.answer)

    def test_memory_quotes_retrieved_evidence_only(self):
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            state = plan_for("내가 전에 뭐라고 했어?", route="memory",
                             memory_hits=[{"role": "user", "content": "좋아하는 과일은 사과야"}])
            result = native_runtime().generate("내가 전에 뭐라고 했어?", "memory",
                                               brain_state=state)
        self.assertTrue(result.used)
        self.assertIn("사과", result.answer)

    def test_open_ended_question_is_not_fabricated(self):
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            runtime = native_runtime()
            message = "왜 하늘이 파래?"
            result = runtime.generate(message, "general", brain_state=plan_for(message))
        self.assertTrue(result.used)
        self.assertIn("아직 직접 풀 수 없어", result.answer)
        self.assertNotIn("레이리", result.answer)

    def test_missing_contract_is_not_silently_answered(self):
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            result = native_runtime().generate("안녕", "general",
                                               brain_state={"utterance_plan": {}})
        self.assertFalse(result.used)
        self.assertEqual(result.error_code, "native_plan_unavailable")

    def test_unsupported_route_does_not_generate(self):
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            result = native_runtime().generate(
                "안녕", "not-a-route", brain_state=plan_for("안녕"),
            )
        self.assertFalse(result.used)


if __name__ == "__main__":
    unittest.main()
