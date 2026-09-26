import tempfile
import unittest
from pathlib import Path

from memory_store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tempdir.name) / "memory.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_recent_memory_survives_new_store_instance(self):
        self.store.add("session-a", "user", "내 프로젝트 이름은 FlyGPT야")
        reopened = MemoryStore(self.store.path)
        rows = reopened.recent("session-a")
        self.assertEqual(rows[-1]["content"], "내 프로젝트 이름은 FlyGPT야")

    def test_search_finds_related_prior_message(self):
        self.store.add("session-a", "user", "파이썬 리스트 정렬은 sorted를 쓸 거야")
        self.store.add("session-a", "assistant", "알겠어요")
        hits = self.store.search("session-a", "전에 말한 파이썬 정렬 다시 알려줘")
        self.assertTrue(hits)
        self.assertIn("파이썬", hits[0]["content"])

    def test_sessions_are_isolated(self):
        self.store.add("session-a", "user", "비밀 프로젝트 이름은 FlyGPT")
        self.store.add("session-b", "user", "다른 이야기")
        rows = self.store.recent("session-b")
        self.assertEqual(len(rows), 1)
        self.assertNotIn("FlyGPT", rows[0]["content"])


    def test_eight_user_turns_create_long_term_capsule(self):
        for index in range(8):
            self.store.add(
                "session-a",
                "user",
                f"프로젝트 메모 {index}: FlyGPT 기능 테스트",
            )

        capsules = self.store.capsules("session-a")
        self.assertEqual(len(capsules), 1)
        self.assertIn("FlyGPT", capsules[0]["summary"])
        self.assertEqual(self.store.stats("session-a")["capsules"], 1)

    def test_search_can_return_capsule(self):
        for index in range(8):
            text = (
                "내 프로젝트 이름은 FlyGPT야"
                if index == 0
                else f"일반 대화 {index}"
            )
            self.store.add("session-a", "user", text)

        hits = self.store.search(
            "session-a",
            "전에 말한 프로젝트 FlyGPT 기억해?",
            limit=10,
        )
        self.assertTrue(
            any(row.get("source") == "capsule" for row in hits)
        )

    def test_clear_removes_capsules_too(self):
        for index in range(8):
            self.store.add("session-a", "user", f"기억 {index}")

        self.assertEqual(self.store.stats("session-a")["capsules"], 1)
        self.store.clear("session-a")
        self.assertEqual(self.store.stats("session-a")["capsules"], 0)

    def test_clear_removes_only_target_session(self):
        self.store.add("session-a", "user", "A")
        self.store.add("session-b", "user", "B")
        self.store.clear("session-a")
        self.assertEqual(self.store.recent("session-a"), [])
        self.assertEqual(len(self.store.recent("session-b")), 1)


if __name__ == "__main__":
    unittest.main()
