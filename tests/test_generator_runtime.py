import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from generator_runtime import GeneratorRuntime, _clean_assistant_memory, _memory_messages


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


class GeneratorRuntimeTests(unittest.TestCase):
    def test_unconfigured_generator_uses_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            result = runtime.generate("왜 하늘은 파래?", "general")

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
            result = runtime.generate("알 수 없는 경로", "not-a-route")

        self.assertTrue(runtime.configured)
        self.assertFalse(result.used)
        self.assertIsNone(result.error)

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

    def test_status_does_not_expose_api_key(self):
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
        self.assertNotIn("secret-value", repr(status))


    def test_invalid_numeric_settings_fall_back_safely(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "fly-model",
                "FLYGPT_GENERATOR_TIMEOUT": "not-a-number",
                "FLYGPT_GENERATOR_TEMPERATURE": "99",
                "FLYGPT_GENERATOR_MAX_TOKENS": "1",
            },
            clear=True,
        ):
            runtime = GeneratorRuntime()

        self.assertEqual(runtime.timeout, 45.0)
        self.assertEqual(runtime.temperature, 2.0)
        self.assertEqual(runtime.max_tokens, 32)

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
            result = runtime.generate("안녕", "general")

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
            result = runtime.generate("안녕", "general")

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
            result = runtime.generate("안녕", "general")

        self.assertFalse(result.used)
        self.assertEqual(result.http_status, 400)
        self.assertEqual(calls, 1)

    def test_memory_context_is_built_without_exposing_extra_messages(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            messages = runtime._messages(
                "계속 설명해줘",
                "general",
                [
                    {"role": "user", "content": "앞에서 파리 뇌 구조를 물어봤어"},
                    {"role": "assistant", "content": "FlyWire 그래프를 사용한다고 답했어"},
                ],
            )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "계속 설명해줘"})
        self.assertIn("앞에서 파리 뇌 구조", messages[1]["content"])

    def test_memory_and_research_routes_have_specialized_prompts(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()

        self.assertIn("MEMORY", runtime._system_prompt("memory"))
        self.assertIn("RESEARCH", runtime._system_prompt("research"))

    def test_general_prompt_discourages_stock_greeting_bot_replies(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            prompt = runtime._system_prompt("general")

        self.assertIn("not a customer-service greeting bot", prompt)
        self.assertIn("Do not default to stock phrases", prompt)
        self.assertIn("emoticon", prompt)

    def test_prompt_enforces_natural_consistent_korean_tone(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            prompt = runtime._system_prompt("general")

        self.assertIn("consistent level of formality", prompt)
        self.assertIn("do not mix casual second-person forms", prompt)
        self.assertIn("Do not introduce yourself", prompt)
        self.assertIn("Do not turn a simple greeting into a self-introduction", prompt)

    def test_short_general_turn_drops_assistant_echo_history(self):
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

    def test_long_general_history_deduplicates_identical_assistant_replies(self):
        history = _memory_messages(
            "아까 이야기한 파리 뇌 구조를 계속 설명해줘",
            "general",
            [
                {"role": "assistant", "content": "같은 답변"},
                {"role": "assistant", "content": "같은 답변"},
                {"role": "user", "content": "앞에서 파리 뇌를 물어봤어"},
            ],
        )

        assistant_items = [item for item in history if item["role"] == "assistant"]
        self.assertEqual(len(assistant_items), 1)

    def test_tool_context_is_injected_before_user_message(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            messages = runtime._messages(
                "3x+7=22 풀어줘",
                "math",
                [],
                "Exact math tool result: x = 5",
            )

        self.assertEqual(messages[-1], {"role": "user", "content": "3x+7=22 풀어줘"})
        self.assertTrue(any("x = 5" in item["content"] for item in messages[:-1]))

    def test_knowledge_context_preserves_provenance_rules(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            messages = runtime._messages(
                "내 프로젝트 백엔드 뭐였지?",
                "memory",
                [],
                None,
                (
                    "[verified] source=repository ref=app.py: backend is FastAPI\n"
                    "[user-asserted] source=user_self: project nickname is 파피티"
                ),
            )

        self.assertEqual(
            messages[-1],
            {"role": "user", "content": "내 프로젝트 백엔드 뭐였지?"},
        )
        system_context = "\n".join(
            item["content"] for item in messages[:-1] if item["role"] == "system"
        )
        self.assertIn("[verified]", system_context)
        self.assertIn("[user-asserted]", system_context)
        self.assertIn("not independently verified", system_context)

if __name__ == "__main__":
    unittest.main()
