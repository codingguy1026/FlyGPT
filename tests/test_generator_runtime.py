import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from generator_runtime import (
    GenerationResult,
    GeneratorRuntime,
    _clean_assistant_memory,
    _extract_json_object,
    _memory_messages,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def unavailable_error() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=503,
        msg="Service Unavailable",
        hdrs=None,
        fp=io.BytesIO(b'{"error":{"message":"temporarily unavailable"}}'),
    )


def ready_brain_state() -> dict:
    return {
        "route": "general",
        "utterance_plan": {
            "contract": "mouth_only_v1",
            "semantic_authority": "fly_brain",
            "ready": True,
            "speech_act": "return_greeting",
            "content_units": [
                {
                    "kind": "communicative_act",
                    "value": "Return the user's greeting briefly and warmly.",
                }
            ],
            "style": {
                "language": "ko",
                "length": "short",
                "tone": "natural",
            },
            "permissions": {
                "infer_new_meaning": False,
                "add_new_facts": False,
                "add_new_questions": False,
                "change_objective": False,
            },
        },
    }


class GeneratorRuntimeTests(unittest.TestCase):
    def test_unconfigured_generator_uses_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            result = runtime.generate(
                "안녕",
                "general",
                brain_state=ready_brain_state(),
            )

        self.assertFalse(runtime.configured)
        self.assertFalse(result.used)
        self.assertEqual(result.provider, "fallback")
        self.assertIsNone(result.answer)

    def test_unknown_route_is_never_sent(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "http://127.0.0.1:9/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "test-model",
            },
            clear=True,
        ):
            runtime = GeneratorRuntime()
            result = runtime.generate(
                "알 수 없는 경로",
                "not-a-route",
                brain_state=ready_brain_state(),
            )

        self.assertTrue(runtime.configured)
        self.assertFalse(result.used)
        self.assertIsNone(result.error)

    def test_strict_gate_rejects_missing_semantic_plan(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "fly-model",
            },
            clear=True,
        ), patch("generator_runtime.urllib.request.urlopen") as urlopen:
            runtime = GeneratorRuntime()
            result = runtime.generate(
                "왜 하늘은 파래?",
                "general",
                brain_state={
                    "utterance_plan": {
                        "contract": "mouth_only_v1",
                        "ready": False,
                    }
                },
            )

        self.assertFalse(result.used)
        self.assertEqual(result.error, "mouth-only semantic plan is incomplete")
        urlopen.assert_not_called()

    def test_stale_internal_metadata_is_removed_from_assistant_memory(self):
        content = (
            "안녕하세요! 🪰 FlyGPT v0.5 · confidence_gate 질문을 더 자세히 말해 주세요.\n"
            "Router raw: general · 92.9%\n"
            "Decision: confidence_gate"
        )
        cleaned = _clean_assistant_memory(content)

        self.assertIn("안녕하세요!", cleaned)
        self.assertNotIn("v0.5", cleaned)
        self.assertNotIn("confidence_gate", cleaned)
        self.assertNotIn("Router raw", cleaned)
        self.assertNotIn("Decision:", cleaned)

    def test_status_exposes_mouth_only_role_without_api_key(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "fly-model",
                "FLYGPT_GENERATOR_API_KEY": "secret-value",
            },
            clear=True,
        ):
            runtime = GeneratorRuntime()
            status = runtime.status()

        self.assertTrue(status["configured"])
        self.assertTrue(status["api_key_configured"])
        self.assertEqual(status["role"], "mouth_only_language_realizer")
        self.assertTrue(status["strict_semantic_gate"])
        self.assertNotIn("secret-value", repr(status))

    def test_invalid_numeric_settings_fall_back_safely(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "fly-model",
                "FLYGPT_GENERATOR_TIMEOUT": "not-a-number",
                "FLYGPT_GENERATOR_BUDGET": "not-a-number",
                "FLYGPT_GENERATOR_TEMPERATURE": "99",
                "FLYGPT_GENERATOR_MAX_TOKENS": "1",
            },
            clear=True,
        ):
            runtime = GeneratorRuntime()

        self.assertEqual(runtime.timeout, 45.0)
        self.assertEqual(runtime.budget, 25.0)
        self.assertEqual(runtime.temperature, 2.0)
        self.assertEqual(runtime.max_tokens, 32)

    def test_generation_budget_caps_effective_http_timeout(self):
        observed_timeouts: list[float] = []

        def fake_urlopen(request, timeout):
            observed_timeouts.append(float(timeout))
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "안녕!"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "gemini-3.8-flash",
                "FLYGPT_GENERATOR_TIMEOUT": "60",
                "FLYGPT_GENERATOR_BUDGET": "25",
            },
            clear=True,
        ), patch("generator_runtime.urllib.request.urlopen", side_effect=fake_urlopen):
            runtime = GeneratorRuntime()
            result = runtime.generate(
                "안녕",
                "general",
                brain_state=ready_brain_state(),
                budget_seconds=7.0,
            )

        self.assertTrue(result.used)
        self.assertEqual(len(observed_timeouts), 1)
        self.assertLessEqual(observed_timeouts[0], 7.0)
        self.assertGreater(observed_timeouts[0], 0.0)
        self.assertEqual(runtime.status()["budget_seconds"], 25.0)

    def test_reasoning_effort_is_validated_and_reported(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "gemini-3.8-flash",
                "FLYGPT_GENERATOR_REASONING_EFFORT": "low",
            },
            clear=True,
        ):
            runtime = GeneratorRuntime()
            status = runtime.status()

        self.assertEqual(runtime.reasoning_effort, "low")
        self.assertEqual(status["reasoning_effort"], "low")

        with patch.dict(
            os.environ,
            {"FLYGPT_GENERATOR_REASONING_EFFORT": "turbo"},
            clear=True,
        ):
            invalid = GeneratorRuntime()

        self.assertIsNone(invalid.reasoning_effort)

    def test_transient_503_retries_primary_once(self):
        calls: list[str] = []

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data.decode("utf-8"))
            calls.append(payload["model"])
            if len(calls) == 1:
                raise unavailable_error()
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "안녕!"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "gemini-3.8-flash",
                "FLYGPT_GENERATOR_RETRY_DELAY": "0",
            },
            clear=True,
        ), patch("generator_runtime.urllib.request.urlopen", side_effect=fake_urlopen):
            runtime = GeneratorRuntime()
            result = runtime.generate(
                "안녕",
                "general",
                brain_state=ready_brain_state(),
            )

        self.assertTrue(result.used)
        self.assertEqual(result.model, "gemini-3.8-flash")
        self.assertEqual(calls, ["gemini-3.8-flash", "gemini-3.8-flash"])

    def test_repeated_503_uses_configured_fallback_model(self):
        calls: list[str] = []

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data.decode("utf-8"))
            model = payload["model"]
            calls.append(model)
            if model == "gemini-3.8-flash":
                raise unavailable_error()
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "fallback ok"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "gemini-3.8-flash",
                "FLYGPT_GENERATOR_FALLBACK_MODEL": "gemini-3.7-flash",
                "FLYGPT_GENERATOR_RETRY_DELAY": "0",
            },
            clear=True,
        ), patch("generator_runtime.urllib.request.urlopen", side_effect=fake_urlopen):
            runtime = GeneratorRuntime()
            result = runtime.generate(
                "안녕",
                "general",
                brain_state=ready_brain_state(),
            )

        self.assertTrue(result.used)
        self.assertEqual(result.model, "gemini-3.7-flash")
        self.assertEqual(
            calls,
            ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.7-flash"],
        )

    def test_non_retryable_http_error_is_not_retried(self):
        calls = 0

        def fake_urlopen(request, timeout):
            nonlocal calls
            calls += 1
            raise urllib.error.HTTPError(
                url="https://example.invalid/v1/chat/completions",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"bad request"}}'),
            )

        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "gemini-3.8-flash",
                "FLYGPT_GENERATOR_FALLBACK_MODEL": "gemini-3.7-flash",
                "FLYGPT_GENERATOR_RETRY_DELAY": "0",
            },
            clear=True,
        ), patch("generator_runtime.urllib.request.urlopen", side_effect=fake_urlopen):
            runtime = GeneratorRuntime()
            result = runtime.generate(
                "안녕",
                "general",
                brain_state=ready_brain_state(),
            )

        self.assertFalse(result.used)
        self.assertEqual(result.http_status, 400)
        self.assertEqual(calls, 1)

    def test_mouth_only_messages_do_not_expose_raw_user_or_history(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            messages = runtime._messages(
                "왜 하늘은 파래?",
                "general",
                [
                    {"role": "user", "content": "비밀 대화"},
                    {"role": "assistant", "content": "이전 답변"},
                ],
                tool_context="tool secret",
                knowledge_context="knowledge secret",
                brain_state=ready_brain_state(),
            )

        joined = "\n".join(item["content"] for item in messages)
        self.assertIn("mouth", messages[0]["content"].lower())
        self.assertIn("return_greeting", joined)
        self.assertNotIn("왜 하늘은 파래", joined)
        self.assertNotIn("비밀 대화", joined)
        self.assertNotIn("tool secret", joined)
        self.assertNotIn("knowledge secret", joined)
        self.assertEqual(
            messages[-1]["content"],
            "Render the supplied utterance plan as the final reply. Add no semantic content of your own.",
        )

    def test_prompt_forbids_reasoning_and_new_questions(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            prompt = runtime._system_prompt("general")

        self.assertIn("only its mouth", prompt)
        self.assertIn("Do not solve, infer, retrieve", prompt)
        self.assertIn("Do not add facts, questions", prompt)
        self.assertIn("content_units", prompt)

    def test_short_general_turn_drops_assistant_echo_history_helper(self):
        history = _memory_messages(
            "반가워",
            "general",
            [
                {"role": "user", "content": "안녕"},
                {"role": "assistant", "content": "안녕하세요! 어떻게 도와드릴까요?"},
                {"role": "user", "content": "반가워"},
                {"role": "assistant", "content": "안녕하세요! 어떻게 도와드릴까요?"},
            ],
        )

        self.assertTrue(any(item["role"] == "user" for item in history))
        self.assertFalse(any(item["role"] == "assistant" for item in history))

    def test_extract_json_object_accepts_fenced_json(self):
        payload = _extract_json_object(
            '~~~json\n{"facts":[{"statement":"supported"}]}\n~~~'.replace("~~~", "```")
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["facts"][0]["statement"], "supported")

    def test_supported_fact_extraction_returns_only_json_fact_objects(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "test-model",
            },
            clear=True,
        ):
            runtime = GeneratorRuntime()

        fake_result = GenerationResult(
            used=True,
            provider="test",
            model="test-model",
            answer=(
                '{"facts":['
                '{"statement":"Fact A","evidence":['
                '{"url":"https://a.test","quote":"exact supporting phrase"}],'
                '"confidence":0.9,"volatility":"stable"},'
                '"ignore-me"]}'
            ),
        )

        with patch.object(runtime, "_generate_once", return_value=fake_result) as mocked:
            facts = runtime.extract_supported_facts(
                query="question",
                answer="answer",
                evidence_context="[source 1] evidence",
                allowed_urls=["https://a.test"],
            )

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["statement"], "Fact A")
        messages = mocked.call_args.kwargs["messages"]
        joined = "\n".join(item["content"] for item in messages)
        self.assertIn("https://a.test", joined)
        self.assertIn("verbatim phrase", joined)
        self.assertIn("Prefer zero facts", joined)


if __name__ == "__main__":
    unittest.main()
