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


class MemoryStore:
    """Tiny local SQLite conversation memory for FlyGPT v0.5."""

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

    def search(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 5,
        exclude_content: str | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self.recent(session_id, limit=50)
        query_tokens = {
            token.lower()
            for token in _TOKEN_RE.findall(query)
            if token.lower() not in _STOPWORDS and len(token) >= 2
        }

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in candidates:
            if exclude_content is not None and row["content"].strip() == exclude_content.strip():
                continue

            content_tokens = {
                token.lower()
                for token in _TOKEN_RE.findall(row["content"])
                if token.lower() not in _STOPWORDS
            }
            content_lower = row["content"].lower()

            overlap = 0.0
            for query_token in query_tokens:
                if query_token in content_tokens:
                    overlap += 1.0
                    continue

                # Korean particles/endings often stay attached to the token.
                if query_token in content_lower or any(
                    token.startswith(query_token) or query_token.startswith(token)
                    for token in content_tokens
                    if len(token) >= 2
                ):
                    overlap += 0.7

            score = overlap

            if not query_tokens:
                score = 0.1

            if row["role"] == "user":
                score += 0.15

            scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)

        if query_tokens:
            matched = [item for item in scored if item[0] > 0.15]
            recall_words = ("아까", "전에", "지난", "마지막", "뭐였", "기억")
            if matched:
                scored = matched
            elif any(word in query for word in recall_words):
                # A vague recall request is better served by the latest messages
                # than by pretending nothing was remembered.
                scored = sorted(scored, key=lambda item: item[1]["id"], reverse=True)
            else:
                scored = []

        return [row for _, row in scored[: max(1, min(int(limit), 10))]]

    def clear(self, session_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            return int(cursor.rowcount or 0)

    def stats(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS messages, MAX(created_at) AS last_message_at
                FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return {
            "messages": int(row["messages"] or 0),
            "last_message_at": row["last_message_at"],
            "path": str(self.path),
        }


def format_recall(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "🧠 아직 이 브라우저 세션에서 떠올릴 만한 이전 대화를 찾지 못했어요."

    lines = ["🧠 이전 대화에서 관련 있는 내용을 찾았어요:"]
    for row in rows:
        speaker = "너" if row["role"] == "user" else "FlyGPT"
        text = row["content"].replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:217] + "..."
        lines.append(f"• {speaker}: {text}")

    return "\n".join(lines)
