"""CLI and comparison protocol smoke tests (no downloads or training)."""
import sys
import unittest
from unittest.mock import patch

from scripts import train_papiti_foundation as foundation
from scripts import eval_papiti_capacity as capacity


class CapacityTests(unittest.TestCase):
    def test_default_width_preserves_old_resume(self):
        with patch.object(sys, "argv", ["train", "train", "--resume"]), \
             patch.object(foundation, "train") as run:
            foundation.main()
        args = run.call_args.args[0]
        self.assertEqual(args.width, 128)
        self.assertTrue(args.resume)

    def test_width_192_requires_separate_output(self):
        with patch.object(sys, "argv", ["train", "train", "--width", "192"]), \
             patch.object(foundation, "train") as run:
            with self.assertRaises(SystemExit):
                foundation.main()
        run.assert_not_called()

    def test_width_192_with_explicit_output(self):
        with patch.object(sys, "argv", ["train", "train", "--width", "192",
                                       "--output-dir", "artifacts/papiti_v071_capacity",
                                       "--epochs", "16"]), \
             patch.object(foundation, "train") as run:
            foundation.main()
        args = run.call_args.args[0]
        self.assertEqual(args.width, 192)
        self.assertEqual(args.epochs, 16)
        self.assertFalse(args.resume)

    def test_new_prefixes_are_distinct(self):
        for lang in ("ko", "en"):
            self.assertEqual(len(capacity.PROBES[lang]), 5)
            self.assertEqual(len(capacity.NEW_PROBES[lang]), 5)
            self.assertFalse(set(capacity.PROBES[lang]) & set(capacity.NEW_PROBES[lang]))


if __name__ == "__main__":
    unittest.main()
