"""Regression tests for FlyGPT's opt-in fast local generation."""
import json
import os
import unittest
from unittest.mock import patch

from generator_runtime import GeneratorRuntime, _ollama_performance


class _Response:
    def __init__(self, data):
        self._payload = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._payload


class FastLocalGenerationTests(unittest.TestCase):
    def test_ollama_metrics_are_converted_from_ns(self):
        metrics = _ollama_performance({
            "total_duration": 2_000_000_000,
            "load_duration": 500_000_000,
            "prompt_eval_duration": 400_000_000,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 40,
            "eval_count": 20,
        })
        self.assertEqual(metrics["total_ms"], 2000.0)
        self.assertEqual(metrics["load_ms"], 500.0)
        self.assertEqual(metrics["prompt_eval_ms"], 400.0)
        self.assertEqual(metrics["eval_ms"], 1000.0)
        self.assertEqual(metrics["tokens_per_second"], 20.0)
        self.assertEqual(metrics["eval_count"], 20)

    def test_social_generation_uses_short_limit_and_no_thinking(self):
        captured = []
        payload = {
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "안녕하세요!"},
            "load_duration": 200_000_000,
            "prompt_eval_duration": 400_000_000,
            "eval_duration": 300_000_000,
            "eval_count": 9,
        }

        def fake_open(req, timeout):
            self.assertGreater(timeout, 0)
            captured.append(json.loads(req.data.decode("utf-8")))
            return _Response(payload)

        brain_state = {
            "utterance_plan": {
                "contract": "mouth_only_v1",
                "ready": True,
                "speech_act": "return_greeting",
                "content_units": [
                    {"kind": "communicative_act", "value": "Return a greeting."}
                ],
                "style": {"language": "ko"},
            }
        }
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "1",
            "FLYGPT_GENERATOR_URL": "http://127.0.0.1:11434/v1/chat/completions",
            "FLYGPT_GENERATOR_MODEL": "papiti-local-fixture:1b",
            "FLYGPT_GENERATOR_MAX_TOKENS": "384",
        }, clear=True), patch("generator_runtime.urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = fake_open
            runtime = GeneratorRuntime()
            result = runtime.generate("안녕", "general", brain_state=brain_state)

        self.assertTrue(result.used)
        self.assertEqual(result.answer, "안녕하세요!")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["model"], "papiti-local-fixture:1b")
        self.assertIs(captured[0]["think"], False)
        self.assertEqual(captured[0]["options"]["num_predict"], 64)
        self.assertEqual(result.ollama_metrics["eval_count"], 9)
        self.assertEqual(result.ollama_metrics["eval_ms"], 300.0)

    def test_metrics_reject_invalid_data(self):
        self.assertIsNone(_ollama_performance(None))
        self.assertIsNone(_ollama_performance({}))
        self.assertEqual(_ollama_performance({"eval_count": True, "eval_duration": -1}), None)


if __name__ == "__main__":
    unittest.main()
