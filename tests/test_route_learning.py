from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from route_learning import RouteLearningStore


def base_route(route: str = "general") -> dict:
    return {
        "route": route,
        "confidence": 0.72,
        "margin": 0.28,
        "accepted": True,
        "min_confidence": 0.55,
        "min_margin": 0.10,
        "top_routes": [
            {"route": "general", "confidence": 0.72},
            {"route": "code", "confidence": 0.12},
            {"route": "research", "confidence": 0.08},
            {"route": "memory", "confidence": 0.04},
            {"route": "math", "confidence": 0.02},
            {"route": "summarize", "confidence": 0.02},
        ],
    }


class RouteLearningStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "route_learning.sqlite3"
        self.store = RouteLearningStore(self.path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_account_scoped_feedback_can_shift_a_similar_phrase(self) -> None:
        phrase = "코드 ㄱㄱ 검색창 만들어줘"
        for _ in range(2):
            self.assertTrue(
                self.store.feedback("user-a", phrase, "code")
            )

        result = self.store.personalize(
            "user-a",
            "코드 ㄱㄱ 검색창 짜줘",
            base_route("general"),
        )

        self.assertTrue(result["personalization"]["applied"])
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(result["route"], "code")

        untouched = self.store.personalize(
            "user-b",
            "코드 ㄱㄱ 검색창 짜줘",
            base_route("general"),
        )
        self.assertEqual(untouched["route"], "general")
        self.assertFalse(untouched["personalization"]["applied"])

    def test_auto_learning_requires_comfortably_confident_base_route(self) -> None:
        info = base_route("general")
        info["confidence"] = 0.70
        info["margin"] = 0.12
        self.assertFalse(
            self.store.observe_if_confident("u", "뭐임ㅋㅋ", info)
        )

        info["confidence"] = 0.93
        info["margin"] = 0.50
        self.assertTrue(
            self.store.observe_if_confident("u", "뭐임ㅋㅋ", info)
        )
        self.assertEqual(self.store.stats("u")["auto_examples"], 1)

    def test_raw_prompt_is_not_stored(self) -> None:
        prompt = "내 비밀 원문은 DB에 저장하지 마"
        self.store.feedback("u", prompt, "general")

        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT text_hash, feature_json FROM route_examples WHERE user_id = ?",
                ("u",),
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertNotIn(prompt, str(row["text_hash"]))
        self.assertNotIn(prompt, str(row["feature_json"]))

    def test_clear_only_removes_one_users_learning(self) -> None:
        self.store.feedback("a", "코드 만들어줘", "code")
        self.store.feedback("b", "코드 만들어줘", "code")

        self.assertEqual(self.store.clear("a"), 1)
        self.assertEqual(self.store.stats("a")["examples"], 0)
        self.assertEqual(self.store.stats("b")["examples"], 1)


if __name__ == "__main__":
    unittest.main()
