import unittest
from unittest.mock import patch
from unittest.mock import Mock
from verified_realizer import realize_verified


class VerifiedRealizerTests(unittest.TestCase):
    def test_absent_model(self):
        text, meta = realize_verified("x = 3", Mock(configured=False, model="Qwen"))
        self.assertEqual(text, "x = 3")
        self.assertFalse(meta["attempted"])

    @patch.dict("os.environ", {"FLYGPT_VERIFIED_REALIZER_MODE": "always"})
    def test_whitespace(self):
        model = Mock(configured=True, local_only=False, model="Qwen")
        model._generate_once.return_value = Mock(used=True, answer="D는 참이다.\nA도 참이다.")
        text, meta = realize_verified("D는 참이다. A도 참이다.", model)
        self.assertTrue(meta["accepted"])
        self.assertIn("\n", text)

    @patch.dict("os.environ", {"FLYGPT_VERIFIED_REALIZER_MODE": "always"})
    def test_hallucination_rejected(self):
        model = Mock(configured=True, local_only=False, model="Qwen")
        model._generate_once.return_value = Mock(used=True, answer="x = 4")
        text, meta = realize_verified("x = 3", model)
        self.assertEqual(text, "x = 3")
        self.assertEqual(meta["reason"], "content_changed")

    @patch.dict("os.environ", {"FLYGPT_VERIFIED_REALIZER_MODE": "auto"})
    def test_auto_skips_slow_model(self):
        model = Mock(configured=True, local_only=False, model="Qwen")
        text, meta = realize_verified("D는 참이다.", model)
        self.assertEqual(text, "D는 참이다.")
        self.assertEqual(meta["reason"], "verified_fast_path")
        model._generate_once.assert_not_called()

    @patch.dict("os.environ", {"FLYGPT_VERIFIED_REALIZER_MODE": "off"})
    def test_off_skips_model(self):
        model = Mock(configured=True, local_only=False, model="Qwen")
        _, meta = realize_verified("D는 참이다.", model)
        self.assertFalse(meta["attempted"])
        model._generate_once.assert_not_called()
