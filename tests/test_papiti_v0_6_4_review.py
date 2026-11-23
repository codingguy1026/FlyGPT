"""Check v0.6.4 strict, human-reviewed transfer acceptance conditions."""
from __future__ import annotations

import csv
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.review_prose_quality import FIELDS, score


class ReviewTargetsTests(unittest.TestCase):
    def _score(self, rows):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "review.csv"
            with path.open("w", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            output = StringIO()
            with redirect_stdout(output):
                score(path, v064_goals=True)
            return json.loads(output.getvalue())["v064_goals"]

    @staticmethod
    def _rows():
        # 29/36 natural, 33/36 meaningful, 10 copied, 19/26 natural
        # novel outputs, no language leaks or severe factual errors.
        return [
            {
                "index": str(i + 1), "purpose": "transfer", "prompt": "x",
                "expected": "x", "generated": "y",
                "copied_training": str(i < 10),
                "grammar_ok": str(int(i < 29)),
                "meaning_ok": str(int(i < 33)),
                "severe_fact_error": "0", "language_mix": "0", "notes": "",
            }
            for i in range(36)
        ]

    def test_threshold_boundary_passes(self):
        goals = self._score(self._rows())
        self.assertEqual(goals["broken_count"], 7)
        self.assertEqual(goals["novel_natural_count"], 19)
        self.assertEqual(goals["copied_count"], 10)
        self.assertTrue(goals["all_met"])

    def test_eight_broken_fail(self):
        rows = self._rows()
        rows[28]["grammar_ok"] = "0"
        goals = self._score(rows)
        self.assertFalse(goals["checks"]["broken_responses_at_most_7"])
        self.assertFalse(goals["all_met"])

    def test_one_language_leak_fails(self):
        rows = self._rows()
        rows[0]["language_mix"] = "1"
        self.assertFalse(self._score(rows)["all_met"])

    def test_missing_manual_labels_rejected(self):
        rows = self._rows()
        rows[0]["language_mix"] = ""
        with self.assertRaises(ValueError):
            self._score(rows)


if __name__ == "__main__":
    unittest.main()
