"""Offline regression tests for Papiti's native-first hybrid router."""
from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

from generator_runtime import GeneratorRuntime, GenerationResult
from brain_runtime import FlyBrainRuntime

ROUTE = {"route": "general", "confidence": 0.96, "margin": 0.75, "accepted": True}


def plan_for(message: str, route: str = "general", **finalize):
    brain = FlyBrainRuntime()
    info = {**ROUTE, "route": route}
    state = brain.plan(route_info=info, dispatch={"route": route}, message=message)
    return brain.finalize(state, **finalize)


class PapitiHybridTests(unittest.TestCase):
    def test_native_greeting_never_calls_model(self):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_URL": "http://127.0.0.1:12345/v1/chat/completions",
            "FLYGPT_HYBRID_MODEL": "experimental-small-model",
        }, clear=True):
            runtime = GeneratorRuntime()
            with patch.object(runtime, "_generate_once") as send:
                result = runtime.generate("안녕", "general",
                                          brain_state=plan_for("안녕"))
                send.assert_not_called()
        self.assertTrue(result.used)
        self.assertIn("안녕", result.answer)
        self.assertEqual(result.provider, "papiti-hybrid-native")
        self.assertIsNone(result.model)
        self.assertTrue(runtime.status()["hybrid_model_configured"])

    def test_unconfigured_hybrid_fails_closed_without_legacy_model_or_network(self):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_GENERATOR_URL": "https://untrusted.example/v1/chat/completions",
            "FLYGPT_GENERATOR_MODEL": "papiti-local-fixture:1b",
            "GEMINI_API_KEY": "stale-key",
        }, clear=True):
            runtime = GeneratorRuntime()
            with patch("generator_runtime.urllib.request.urlopen") as send:
                result = runtime.generate("왜 하늘이 파래?", "general",
                                         brain_state=plan_for("왜 하늘이 파래?"))
                send.assert_not_called()
        self.assertEqual(runtime.url, "")
        self.assertEqual(runtime.model, "")
        self.assertEqual(runtime.provider_name, "papiti-hybrid")
        self.assertFalse(runtime.status()["uses_language_model"])
        self.assertTrue(result.used)
        self.assertIn("아직 직접 풀 수 없어", result.answer)

    def test_open_ended_task_uses_opt_in_local_generator(self):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_URL": "http://127.0.0.1:12345/v1/chat/completions",
            "FLYGPT_HYBRID_MODEL": "papiti-small-experiment",
            "FLYGPT_GENERATOR_RETRY_DELAY": "0",
        }, clear=True):
            runtime = GeneratorRuntime()
            expected = GenerationResult(
                used=True, provider="papiti-hybrid",
                model="papiti-small-experiment",
                answer="하늘은 대기 중 빛의 산란 때문에 파랗게 보여요.",
                finish_reason="stop",
            )
            with patch.object(runtime, "_generate_once", return_value=expected) as send:
                result = runtime.generate("왜 하늘이 파래?", "general",
                                          brain_state=plan_for("왜 하늘이 파래?"))
                send.assert_called_once()
                self.assertEqual(send.call_args.kwargs["model"],
                                 "papiti-small-experiment")
        self.assertEqual(result.answer, expected.answer)
        self.assertEqual(result.model, "papiti-small-experiment")

    def test_invalid_verified_fact_must_not_fall_through_to_model(self):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_URL": "http://127.0.0.1:12345/v1/chat/completions",
            "FLYGPT_HYBRID_MODEL": "papiti-small-experiment",
        }, clear=True):
            runtime = GeneratorRuntime()
            forged = copy.deepcopy(plan_for("너 누구야?"))
            forged["utterance_plan"]["content_units"][0]["value"] = "허위 사실"
            with patch.object(runtime, "_generate_once") as send:
                result = runtime.generate("너 누구야?", "general",
                                          brain_state=forged)
                send.assert_not_called()
        self.assertFalse(result.used)
        self.assertEqual(result.error_code, "native_plan_unavailable")

    def test_hybrid_rejects_hosted_endpoint_even_when_configured(self):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_URL": "https://untrusted.example/v1/chat/completions",
            "FLYGPT_HYBRID_MODEL": "papiti-small-experiment",
        }, clear=True):
            runtime = GeneratorRuntime()
            with patch("generator_runtime.urllib.request.urlopen") as send:
                result = runtime.generate("왜 하늘이 파래?", "general",
                                          brain_state=plan_for("왜 하늘이 파래?"))
                send.assert_not_called()
        self.assertFalse(runtime.local_endpoint_allowed)
        self.assertFalse(runtime.status()["hybrid_model_configured"])
        self.assertIn("아직 직접 풀 수 없어", result.answer)

    def test_verified_math_does_not_use_model(self):
        context = "Exact math tool result: 42"
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_URL": "http://127.0.0.1:12345/v1/chat/completions",
            "FLYGPT_HYBRID_MODEL": "papiti-small-experiment",
        }, clear=True):
            runtime = GeneratorRuntime()
            with patch.object(runtime, "_generate_once") as send:
                result = runtime.generate(
                    "6*7", "math",
                    brain_state=plan_for("6*7", route="math", tool_context=context),
                    tool_context=context,
                )
                send.assert_not_called()
        self.assertEqual(result.answer, "🧮 42")


if __name__ == "__main__":
    unittest.main()
