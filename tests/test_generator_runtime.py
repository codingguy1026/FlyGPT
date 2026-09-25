import os
import unittest
from unittest.mock import patch

from generator_runtime import GeneratorRuntime


class GeneratorRuntimeTests(unittest.TestCase):
    def test_unconfigured_generator_uses_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = GeneratorRuntime()
            result = runtime.generate("왜 하늘은 파래?", "general")

        self.assertFalse(runtime.configured)
        self.assertFalse(result.used)
        self.assertEqual(result.provider, "fallback")
        self.assertIsNone(result.answer)

    def test_non_generative_route_is_never_sent(self):
        with patch.dict(
            os.environ,
            {
                "FLYGPT_GENERATOR_URL": "http://127.0.0.1:9/v1/chat/completions",
                "FLYGPT_GENERATOR_MODEL": "test-model",
            },
            clear=True,
        ):
            runtime = GeneratorRuntime()
            result = runtime.generate("17 × 23은?", "math")

        self.assertTrue(runtime.configured)
        self.assertFalse(result.used)
        self.assertIsNone(result.error)

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


if __name__ == "__main__":
    unittest.main()
