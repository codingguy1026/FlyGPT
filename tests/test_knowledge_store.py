from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from knowledge_store import KnowledgeStore


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "knowledge.sqlite3"
        self.store = KnowledgeStore(self.path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_personal_fact_is_asserted_and_retrievable(self) -> None:
        items = self.store.learn_from_user_text(
            "user-a",
            "내 프로젝트 이름은 fly GPT야",
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "asserted")
        self.assertEqual(items[0]["kind"], "project")
        self.assertEqual(items[0]["source_type"], "user_self")

        context, hits = self.store.context_for_query(
            "user-a",
            "내 프로젝트 이름 뭐였지?",
        )
        self.assertEqual(len(hits), 1)
        self.assertIsNotNone(context)
        self.assertIn("[user-asserted]", context or "")
        self.assertIn("fly GPT", context or "")

    def test_objective_user_claim_is_not_used_as_fact_before_verification(self) -> None:
        items = self.store.learn_from_user_text(
            "user-a",
            "기억해 파리는 날개가 8개야",
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["status"], "claim")
        context, hits = self.store.context_for_query(
            "user-a",
            "파리 날개가 몇 개야?",
        )
        self.assertIsNone(context)
        self.assertEqual(hits, [])

    def test_verified_claim_becomes_trusted_context(self) -> None:
        item = self.store.add(
            "user-a",
            "FlyGPT backend is FastAPI",
            kind="project",
            source_type="user_claim",
        )
        self.assertIsNotNone(item)

        verified = self.store.verify(
            "user-a",
            int(item["id"]),
            source_type="repository",
            source_ref="app.py",
        )
        self.assertIsNotNone(verified)
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["source_type"], "repository")

        context, hits = self.store.context_for_query(
            "user-a",
            "FlyGPT backend 뭐야?",
        )
        self.assertEqual(len(hits), 1)
        self.assertIn("[verified]", context or "")
        self.assertIn("app.py", context or "")

    def test_same_personal_slot_is_superseded_by_correction(self) -> None:
        first = self.store.learn_from_user_text(
            "user-a",
            "내 프로젝트 이름은 FlewGPT야",
        )[0]
        second = self.store.learn_from_user_text(
            "user-a",
            "내 프로젝트 이름은 fly GPT야",
        )[0]

        rows = self.store.list_items("user-a")
        by_id = {int(row["id"]): row for row in rows}

        self.assertEqual(by_id[int(first["id"])]["status"], "superseded")
        self.assertEqual(by_id[int(second["id"])]["status"], "asserted")
        self.assertEqual(second["supersedes_id"], first["id"])

    def test_weaker_conflict_does_not_override_verified_fact(self) -> None:
        original = self.store.add(
            "user-a",
            "내 프로젝트 백엔드는 FastAPI야",
            kind="project",
            source_type="repository",
            source_ref="app.py",
            verified=True,
            subject="project",
            predicate="백엔드",
            value="FastAPI",
            slot_key="project:백엔드",
        )
        self.assertIsNotNone(original)

        conflict = self.store.add(
            "user-a",
            "내 프로젝트 백엔드는 Flask야",
            kind="project",
            source_type="user_self",
            subject="project",
            predicate="백엔드",
            value="Flask",
            slot_key="project:백엔드",
        )
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["status"], "disputed")
        self.assertEqual(conflict["conflicts_with_id"], original["id"])

        context, hits = self.store.context_for_query(
            "user-a",
            "프로젝트 백엔드 뭐였지?",
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["status"], "verified")
        self.assertIn("FastAPI", context or "")
        self.assertNotIn("Flask", context or "")

    def test_expired_verified_fact_is_not_returned(self) -> None:
        self.store.add(
            "user-a",
            "현재 모델 버전은 v1",
            kind="fact",
            source_type="official",
            verified=True,
            expires_at=time.time() - 10,
        )
        context, hits = self.store.context_for_query(
            "user-a",
            "현재 모델 버전 뭐야?",
        )
        self.assertIsNone(context)
        self.assertEqual(hits, [])

    def test_secret_like_text_is_never_auto_learned(self) -> None:
        items = self.store.learn_from_user_text(
            "user-a",
            "기억해 api_key = sk-abcdefghijklmnopqrstuvwxyz123456",
        )
        self.assertEqual(items, [])
        self.assertEqual(self.store.stats("user-a")["items"], 0)

    def test_casual_correction_word_does_not_pollute_knowledge(self) -> None:
        items = self.store.learn_from_user_text("user-a", "아니 ㅋㅋㅋㅋ")
        self.assertEqual(items, [])

    def test_knowledge_is_account_scoped(self) -> None:
        self.store.learn_from_user_text(
            "user-a",
            "내 프로젝트 이름은 fly GPT야",
        )
        context, hits = self.store.context_for_query(
            "user-b",
            "프로젝트 이름 뭐야?",
        )
        self.assertIsNone(context)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
