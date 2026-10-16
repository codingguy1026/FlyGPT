"""Regression tests for OpenAI-compatible completion response formats.

All tests run against synthetic response objects, with no provider account.
"""
import json
import os
import unittest
from unittest.mock import patch

from generator_runtime import (
    GenerationResult, GeneratorRuntime, _extract_chat_completion,
)


class ResponseEnvelopeTests(unittest.TestCase):
    def test_text_only_chat_reply(self):
        text, finish = _extract_chat_completion({
            "choices": [{"message": {"content": " 반가워! "}, "finish_reason": "stop"}]
        })
        self.assertEqual((text, finish), ("반가워!", "stop"))

    def test_typed_text_parts_join_into_fresh_prose(self):
        text, finish = _extract_chat_completion({
            "choices": [{
                "message": {
                    "content": [
                        {"type": "text", "text": "첫 문장이야."},
                        {"type": "reasoning", "text": "내부 판단, 출력 금지"},
                        {"type": "text", "text": "둘째 문장이야."},
                    ]
                },
                "finish_reason": "stop"
            }]
        })
        self.assertEqual(text, "첫 문장이야.\n둘째 문장이야.")
        self.assertNotIn("내부 판단", text)
        self.assertEqual(finish, "stop")

    def test_no_tool_or_hidden_content_is_exposed(self):
        with self.assertRaisesRegex(ValueError, "missing visible assistant text"):
            _extract_chat_completion({"choices": [{
                "message": {"content": None, "reasoning_content": "private"},
                "finish_reason": "stop",
            }]})

    def test_empty_but_truncated_can_be_retried(self):
        self.assertEqual(
            _extract_chat_completion({"choices": [{
                "message": {"content": None}, "finish_reason": "length"
            }]}),
            ("", "length"),
        )

    def test_malformed_responses_are_refused(self):
        for data in ([], {}, {"choices": []}, {"choices": [None]},
                     {"choices": [{"message": {}}]},
                     {"choices": [{"message": {"content": {"text": "no"}}}]}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                _extract_chat_completion(data)


class ProviderRecoveryTests(unittest.TestCase):
    def runtime(self):
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
            "FLYGPT_GENERATOR_MODEL": "fake-provider",
            "FLYGPT_GENERATOR_RETRY_DELAY": "0",
        }, clear=True):
            return GeneratorRuntime()

    @staticmethod
    def state():
        return {"utterance_plan": {
            "contract": "mouth_only_v1",
            "ready": True,
            "speech_act": "return_greeting",
            "content_units": [{"kind": "communicative_act", "value": "Return greeting."}],
            "style": {"language": "ko"},
        }}

    def test_bad_first_http_200_answer_retries_instead_of_canned(self):
        runtime = self.runtime()
        with patch.object(runtime, "_generate_once", side_effect=[
            GenerationResult(used=True, provider="compatible-http", model="fake-provider",
                             answer='{"speech_act":"return_greeting"}', http_status=200),
            GenerationResult(used=True, provider="compatible-http", model="fake-provider",
                             answer="반가워, 오늘 어떤 이야기를 해볼까?", http_status=200),
        ]) as generate:
            result = runtime.generate("안녕", "general", brain_state=self.state())
        self.assertTrue(result.used)
        self.assertEqual(result.answer, "반가워, 오늘 어떤 이야기를 해볼까?")
        self.assertEqual(generate.call_count, 2)

    def test_malformed_then_typed_content_retry(self):
        runtime = self.runtime()
        class FakeResponse:
            def __init__(self, content):
                self.content = content
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps(self.content).encode("utf-8")
        with patch("generator_runtime.urllib.request.urlopen", side_effect=[
            FakeResponse({"not_choices": "nope"}),
            FakeResponse({"choices": [{
                "message": {"content": [{"type": "text", "text": "두 번째 응답은 정상이야."}]},
                "finish_reason": "stop"
            }]}),
        ]) as network:
            result = runtime.generate("안녕", "general", brain_state=self.state())
        self.assertTrue(result.used)
        self.assertEqual(result.answer, "두 번째 응답은 정상이야.")
        self.assertEqual(network.call_count, 2)


if __name__ == "__main__":
    unittest.main()
