import os
import unittest
from unittest.mock import patch

from generator_runtime import GeneratorRuntime, _clean_assistant_memory


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
