from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")
_STOPWORDS = {
    "내가", "전에", "말한", "뭐", "뭐였지", "다시", "알려줘", "기억", "기억해",
    "what", "did", "i", "say", "before", "remember", "again", "the", "a", "an",
}
_CAPSULE_BATCH = 8


class MemoryStore:
    """Local SQLite short-term + compressed long-term memory for FlyGPT."""

    def __init__(self, path: str | Path = "data/flygpt_memory.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_capsules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    start_message_id INTEGER NOT NULL,
                    end_message_id INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_capsules_session_id
                ON memory_capsules(session_id, id)
                """
            )

    def add(self, session_id: str, role: str, content: str) -> int:
        clean_session = session_id.strip()[:128]
        clean_role = role.strip()[:32]
        clean_content = content.strip()[:12000]
        if not clean_session or not clean_content:
            return 0

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_session, clean_role, clean_content, time.time()),
            )
            message_id = int(cursor.lastrowid)
            self._prune(conn, clean_session, keep=200)

        if clean_role == "user":
            self.compact_if_needed(clean_session)

        return message_id

    def _prune(self, conn: sqlite3.Connection, session_id: str, keep: int) -> None:
        conn.execute(
            """
            DELETE FROM messages
            WHERE session_id = ?
              AND id NOT IN (
                  SELECT id
                  FROM messages
                  WHERE session_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (session_id, session_id, keep),
        )

    def recent(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 50))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [dict(row) for row in reversed(rows)]

    def _last_capsule_end(self, conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            """
            SELECT MAX(end_message_id) AS end_id
            FROM memory_capsules
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return int(row["end_id"] or 0)

    @staticmethod
    def _compress_user_messages(rows: list[sqlite3.Row]) -> str:
        snippets: list[str] = []
        seen: set[str] = set()

        for row in rows:
            text = re.sub(r"\s+", " ", str(row["content"])).strip()
            if not text:
                continue

            # Keep user statements compact and avoid preserving huge raw blocks.
            if len(text) > 180:
                text = text[:177] + "..."

            normalized = re.sub(r"[^가-힣A-Za-z0-9]+", "", text).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            snippets.append(text)

        return " | ".join(snippets[:_CAPSULE_BATCH])

    def compact_if_needed(self, session_id: str) -> dict[str, Any] | None:
        """Compress each batch of user turns into a durable searchable capsule."""

        with self._connect() as conn:
            last_end = self._last_capsule_end(conn, session_id)
            rows = conn.execute(
                """
                SELECT id, content, created_at
                FROM messages
                WHERE session_id = ?
                  AND role = 'user'
                  AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, last_end, _CAPSULE_BATCH),
            ).fetchall()

            if len(rows) < _CAPSULE_BATCH:
                return None

            summary = self._compress_user_messages(rows)
            if not summary:
                return None

            cursor = conn.execute(
                """
                INSERT INTO memory_capsules(
                    session_id,
                    start_message_id,
                    end_message_id,
                    summary,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    int(rows[0]["id"]),
                    int(rows[-1]["id"]),
                    summary,
                    time.time(),
                ),
            )

            return {
                "id": int(cursor.lastrowid),
                "start_message_id": int(rows[0]["id"]),
                "end_message_id": int(rows[-1]["id"]),
                "summary": summary,
            }

    def capsules(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, start_message_id, end_message_id, summary, created_at
                FROM memory_capsules
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    @staticmethod
    def _score_text(query_tokens: set[str], content: str) -> float:
        content_tokens = {
            token.lower()
            for token in _TOKEN_RE.findall(content)
            if token.lower() not in _STOPWORDS
        }
        content_lower = content.lower()

        if not query_tokens:
            return 0.1

        score = 0.0
        for query_token in query_tokens:
            if query_token in content_tokens:
                score += 1.0
                continue

            if query_token in content_lower or any(
                token.startswith(query_token) or query_token.startswith(token)
                for token in content_tokens
                if len(token) >= 2
            ):
                score += 0.7

        return score

    def search(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 5,
        exclude_content: str | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = {
            token.lower()
            for token in _TOKEN_RE.findall(query)
            if token.lower() not in _STOPWORDS and len(token) >= 2
        }

        scored: list[tuple[float, int, dict[str, Any]]] = []

        for row in self.recent(session_id, limit=50):
            if exclude_content is not None and row["content"].strip() == exclude_content.strip():
                continue

            score = self._score_text(query_tokens, row["content"])
            if row["role"] == "user":
                score += 0.15

            item = dict(row)
            item["source"] = "recent"
            scored.append((score, int(row["id"]), item))

        for capsule in self.capsules(session_id, limit=50):
            score = self._score_text(query_tokens, capsule["summary"])
            item = {
                "id": int(capsule["id"]),
                "role": "memory_capsule",
                "content": capsule["summary"],
                "created_at": capsule["created_at"],
                "source": "capsule",
                "start_message_id": capsule["start_message_id"],
                "end_message_id": capsule["end_message_id"],
            }
            # A capsule represents several turns, so give an exact topical match
            # a small bonus without letting it dominate unrelated recent memory.
            scored.append((score + (0.25 if score > 0 else 0.0), int(capsule["id"]), item))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        if query_tokens:
            matched = [item for item in scored if item[0] > 0.15]
            recall_words = ("아까", "전에", "지난", "마지막", "뭐였", "기억")
            if matched:
                scored = matched
            elif any(word in query for word in recall_words):
                scored = sorted(scored, key=lambda item: item[1], reverse=True)
            else:
                scored = []

        return [row for _, _, row in scored[: max(1, min(int(limit), 10))]]

    def clear(self, session_id: str) -> int:
        with self._connect() as conn:
            message_cursor = conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            capsule_cursor = conn.execute(
                "DELETE FROM memory_capsules WHERE session_id = ?",
                (session_id,),
            )
            return int(message_cursor.rowcount or 0) + int(capsule_cursor.rowcount or 0)

    def stats(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            message_row = conn.execute(
                """
                SELECT COUNT(*) AS messages, MAX(created_at) AS last_message_at
                FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            capsule_row = conn.execute(
                """
                SELECT COUNT(*) AS capsules, MAX(created_at) AS last_capsule_at
                FROM memory_capsules
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return {
            "messages": int(message_row["messages"] or 0),
            "capsules": int(capsule_row["capsules"] or 0),
            "last_message_at": message_row["last_message_at"],
            "last_capsule_at": capsule_row["last_capsule_at"],
            "capsule_batch_size": _CAPSULE_BATCH,
            "path": str(self.path),
        }


def format_recall(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "🧠 아직 이 브라우저 세션에서 떠올릴 만한 이전 대화를 찾지 못했어요."

    lines = ["🧠 이전 대화에서 관련 있는 내용을 찾았어요:"]
    for row in rows:
        if row.get("source") == "capsule" or row["role"] == "memory_capsule":
            speaker = "장기기억"
        else:
            speaker = "너" if row["role"] == "user" else "FlyGPT"

        text = row["content"].replace("\n", " ").strip()
        if len(text) > 240:
            text = text[:237] + "..."
        lines.append(f"• {speaker}: {text}")

    return "\n".join(lines)
