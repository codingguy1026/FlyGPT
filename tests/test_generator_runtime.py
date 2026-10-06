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


def http_error(code: int, url: str = "https://example.invalid") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=url,
        code=code,
        msg="error",
        hdrs=None,
        fp=io.BytesIO(b'{"error":{"message":"temporary failure"}}'),
    )


def openai_ok(text: str = "openai ok") -> FakeHTTPResponse:
    return FakeHTTPResponse(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
    )


def gemini_ok(text: str = "gemini ok") -> FakeHTTPResponse:
    return FakeHTTPResponse(
        {
            "choices": [
                {
                    "message": {"content": text},
                    "finish_reason": "stop",
                }
            ]
        }
    )


MULTI_ENV = {
    "OPENAI_API_KEY": "openai-secret",
    "FLYGPT_OPENAI_MODEL": "gpt-6-luna",
    "FLYGPT_OPENAI_URL": "https://api.openai.test/v1/responses",
    "FLYGPT_GENERATOR_URL": "https://gemini.test/v1/chat/completions",
    "FLYGPT_GENERATOR_MODEL": "gemini-3.8-flash",
    "FLYGPT_GENERATOR_PROVIDER": "gemini",
    "FLYGPT_GENERATOR_API_KEY": "gemini-secret",
    "FLYGPT_GENERATOR_FALLBACK_MODEL": "gemini-3.7-flash",
    "FLYGPT_GENERATOR_REASONING_EFFORT": "low",
}


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
            {"OPENAI_API_KEY": "secret-value"},
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

    def test_status_does_not_expose_api_keys(self):
        with patch.dict(os.environ, MULTI_ENV, clear=True):
            runtime = GeneratorRuntime()
            status = runtime.status()

        self.assertTrue(status["configured"])
        self.assertEqual(status["provider"], "multi")
        self.assertTrue(status["api_key_configured"])
        self.assertNotIn("openai-secret", repr(status))
        self.assertNotIn("gemini-secret", repr(status))
        self.assertEqual(status["openai"]["model"], "gpt-6-luna")
        self.assertEqual(status["gemini"]["model"], "gemini-3.8-flash")

    def test_invalid_numeric_settings_fall_back_safely(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret",
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
        with patch.dict(os.environ, MULTI_ENV, clear=True):
            runtime = GeneratorRuntime()
            status = runtime.status()

        self.assertEqual(runtime.reasoning_effort, "low")
        self.assertEqual(status["reasoning_effort"], "low")

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret",
                "FLYGPT_GENERATOR_REASONING_EFFORT": "turbo",
            },
            clear=True,
        ):
            invalid = GeneratorRuntime()

        self.assertIsNone(invalid.reasoning_effort)

    def test_general_prefers_openai(self):
        calls: list[str] = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], "gpt-6-luna")
            return openai_ok("안녕!")

        with patch.dict(os.environ, MULTI_ENV, clear=True), patch(
            "generator_runtime.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            runtime = GeneratorRuntime()
            result = runtime.generate("안녕", "general")

        self.assertTrue(result.used)
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "gpt-6-luna")
        self.assertEqual(calls, ["https://api.openai.test/v1/responses"])

    def test_code_prefers_gemini(self):
        calls: list[str] = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["model"], "gemini-3.8-flash")
            return gemini_ok("코드 답변")

        with patch.dict(os.environ, MULTI_ENV, clear=True), patch(
            "generator_runtime.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            runtime = GeneratorRuntime()
            result = runtime.generate("파이썬 코드 짜줘", "code")

        self.assertTrue(result.used)
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.model, "gemini-3.8-flash")
        self.assertEqual(calls, ["https://gemini.test/v1/chat/completions"])

    def test_gemini_503_immediately_falls_back_to_openai(self):
        calls: list[str] = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if "gemini.test" in request.full_url:
                raise http_error(503, request.full_url)
            return openai_ok("OpenAI fallback")

        with patch.dict(os.environ, MULTI_ENV, clear=True), patch(
            "generator_runtime.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            runtime = GeneratorRuntime()
            result = runtime.generate("코드 고쳐줘", "code")

        self.assertTrue(result.used)
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "gpt-6-luna")
        self.assertEqual(
            calls,
            [
                "https://gemini.test/v1/chat/completions",
                "https://api.openai.test/v1/responses",
            ],
        )

    def test_openai_503_immediately_falls_back_to_gemini(self):
        calls: list[str] = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if "openai.test" in request.full_url:
                raise http_error(503, request.full_url)
            return gemini_ok("Gemini fallback")

        with patch.dict(os.environ, MULTI_ENV, clear=True), patch(
            "generator_runtime.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            runtime = GeneratorRuntime()
            result = runtime.generate("안녕", "general")

        self.assertTrue(result.used)
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.model, "gemini-3.8-flash")
        self.assertEqual(
            calls,
            [
                "https://api.openai.test/v1/responses",
                "https://gemini.test/v1/chat/completions",
            ],
        )

    def test_non_retryable_http_error_is_not_cross_routed(self):
        calls = 0

        def fake_urlopen(request, timeout):
            nonlocal calls
            calls += 1
            raise http_error(400, request.full_url)

        with patch.dict(os.environ, MULTI_ENV, clear=True), patch(
            "generator_runtime.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
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


if __name__ == "__main__":
    unittest.main()
