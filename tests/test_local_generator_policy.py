"""Strict local-only language generation regression tests (no network needed)."""

import json
import os
import unittest
from unittest.mock import Mock, patch

from generator_runtime import GeneratorRuntime, _NoGeneratorRedirects
from local_generator_policy import enabled, is_local_generator_url


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "안녕!"}, "finish_reason": "stop"}],
        }).encode("utf-8")


def _ready_plan():
    return {
        "utterance_plan": {
            "contract": "mouth_only_v1",
            "ready": True,
            "speech_act": "return_greeting",
            "content_units": [{
                "kind": "communicative_act",
                "value": "Return the user's greeting briefly and warmly.",
            }],
        }
    }


class LocalGenerationPolicyTests(unittest.TestCase):
    def test_flag_parsing(self):
        self.assertTrue(enabled("true"))
        self.assertTrue(enabled(" YES "))
        self.assertFalse(enabled("0"))
        self.assertFalse(enabled(None))

    def test_local_generator_hosts_only(self):
        for url in (
            "http://127.0.0.1:11434/v1/chat/completions",
            "http://127.9.8.7:11434/v1/chat/completions",
            "http://localhost:11434/v1/chat/completions",
            "http://[::1]:11434/v1/chat/completions",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_local_generator_url(url))
        for url in (
            "https://localhost:11434/v1/chat/completions",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "http://localhost.evil.example/v1/chat/completions",
            "http://127.0.0.1.evil.example/v1/chat/completions",
            "http://192.168.1.10:11434/v1/chat/completions",
            "http://10.0.0.1:11434/v1/chat/completions",
            "http://user:pass@127.0.0.1:11434/v1/chat/completions",
            "http://127.0.0.1:99999/v1/chat/completions",
            "http://[::1",
            "not a url",
            "",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_local_generator_url(url))

    def test_local_mode_defaults_and_ignores_hosted_settings(self):
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "1",
            "GEMINI_API_KEY": "very-secret",
            "FLYGPT_GENERATOR_API_KEY": "another-secret",
            "FLYGPT_GENERATOR_FALLBACK_MODEL": "gemini-fallback",
            "FLYGPT_GENERATOR_REASONING_EFFORT": "high",
        }, clear=True):
            runtime = GeneratorRuntime()
            status = runtime.status()
        self.assertTrue(runtime.configured)
        self.assertTrue(status["local_only"])
        self.assertTrue(status["local_endpoint_allowed"])
        self.assertEqual(runtime.provider_name, "ollama-local")
        self.assertEqual(runtime.model, "qwen2.5:0.5b-instruct")
        self.assertEqual(runtime.url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertIsNone(runtime.reasoning_effort)
        self.assertEqual(runtime.api_key, "")
        self.assertEqual(runtime.fallback_model, "")
        self.assertFalse(status["api_key_configured"])
        self.assertNotIn("very-secret", str(status))
        self.assertNotIn("another-secret", str(status))

    def test_hosted_endpoint_blocked_without_network_request(self):
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "yes",
            "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
            "FLYGPT_GENERATOR_MODEL": "gemini-3.8-flash",
        }, clear=True), patch("generator_runtime.urllib.request.urlopen") as urlopen, patch(
            "generator_runtime.urllib.request.build_opener"
        ) as build_opener:
            runtime = GeneratorRuntime()
            self.assertFalse(runtime.configured)
            result = runtime.generate("안녕", "general", brain_state=_ready_plan())
            direct_result = runtime._generate_once(model="test", messages=[])
        self.assertFalse(result.used)
        self.assertIn("blocked", result.error)
        self.assertIn("blocked", direct_result.error)
        urlopen.assert_not_called()
        build_opener.assert_not_called()

    def test_local_mode_uses_proxy_free_redirect_free_opener(self):
        opener = Mock()
        opener.open.return_value = _Response()
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "on",
            "FLYGPT_GENERATOR_MODEL": "qwen2.5:0.5b-instruct",
        }, clear=True), patch(
            "generator_runtime.urllib.request.build_opener", return_value=opener
        ) as build_opener, patch(
            "generator_runtime.urllib.request.urlopen"
        ) as urlopen:
            runtime = GeneratorRuntime()
            result = runtime.generate("안녕", "general", brain_state=_ready_plan())
        self.assertTrue(result.used)
        self.assertEqual(result.answer, "안녕!")
        urlopen.assert_not_called()
        args = build_opener.call_args.args
        self.assertEqual(args[0].proxies, {})
        self.assertIsInstance(args[1], _NoGeneratorRedirects)
        self.assertIsNone(args[1].redirect_request(None, None, 302, "", {}, "https://evil.example"))

    def test_local_mode_preserves_semantic_gate(self):
        with patch.dict(os.environ, {"FLYGPT_LOCAL_ONLY": "1"}, clear=True), patch(
            "generator_runtime.urllib.request.build_opener"
        ) as build_opener:
            runtime = GeneratorRuntime()
            result = runtime.generate("왜?", "general", brain_state={
                "utterance_plan": {"contract": "mouth_only_v1", "ready": False}
            })
        self.assertFalse(result.used)
        self.assertEqual(result.error, "mouth-only semantic plan is incomplete")
        build_opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
