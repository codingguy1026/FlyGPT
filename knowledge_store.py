from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9_]+")
_RECALL_HINTS = (
    "기억", "전에", "아까", "내 정보", "나에 대해", "내가 말한",
    "remember", "about me", "what do you know",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:password|passwd|비밀번호)\b\s*[:=]"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|bearer)\b\s*[:=]?\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)

SOURCE_TRUST = {
    "official": 1.00,
    "repository": 0.98,
    "manual_verified": 0.95,
    "web": 0.90,
    "user_self": 0.88,
    "user_claim": 0.55,
    "model_inference": 0.30,
}
TRUSTED_VERIFICATION_SOURCES = {"official", "repository", "manual_verified", "web"}
ACTIVE_STATUSES = {"asserted", "claim", "verified", "disputed"}

_USER_FACT_PREFIXES = (
    "나는 ", "난 ", "내 ", "내가 ", "저는 ", "제 ", "제가 ",
    "우리 프로젝트 ", "내 프로젝트 ", "제 프로젝트 ",
)
_EXPLICIT_REMEMBER = (
    "기억해", "기억해둬", "기억해 줘", "기억해줘",
    "알아둬", "저장해", "메모해", "remember this",
)
_CORRECTION_HINTS = (
    "정정", "아니 ", "아니,", "이제는 ", "이제 ", "바꿨어", "바뀌었",
)

_KOREAN_SLOT_RE = re.compile(
    r"^(?P<prefix>내|제|우리|내 프로젝트|제 프로젝트|우리 프로젝트)\s+"
    r"(?P<key>[^\n.!?]{1,60}?)(?:은|는|이|가)\s+"
    r"(?P<value>.+)$"
)
_PROJECT_SLOT_RE = re.compile(
    r"^(?P<prefix>프로젝트|fly\s*gpt|FlyGPT|파피티)\s+"
    r"(?P<key>[^\n.!?]{1,60}?)(?:은|는|이|가)\s+"
    r"(?P<value>.+)$",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _statement_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).lower().encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 2
    }


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _strip_fact_ending(value: str) -> str:
    clean = value.strip().rstrip(" .!?~ㅋㅋㅎ")
    endings = (
        "입니다", "이에요", "예요", "이야", "야", "임", "이다",
        "이었어", "였어", "이었음", "였음",
    )
    for ending in endings:
        if clean.endswith(ending) and len(clean) > len(ending):
            clean = clean[: -len(ending)].rstrip()
            break
    return clean or value.strip()


def _strip_remember_prefix(text: str) -> str:
    clean = text.strip()
    for marker in _EXPLICIT_REMEMBER:
        lower = clean.lower()
        idx = lower.find(marker.lower())
        if idx == -1:
            continue
        after = clean[idx + len(marker):].lstrip(" :,-.!")
        if after:
            return after
    return clean


def _derive_slot(statement: str) -> tuple[str | None, str | None, str | None, str | None]:
    clean = _normalize_text(statement)
    match = _KOREAN_SLOT_RE.match(clean) or _PROJECT_SLOT_RE.match(clean)
    if not match:
        return None, None, None, None

    prefix = match.group("prefix").strip().lower()
    key = _normalize_text(match.group("key"))
    value = _strip_fact_ending(match.group("value"))

    subject = "project" if "프로젝트" in prefix or prefix in {"flygpt", "fly gpt", "파피티"} else "user"
    predicate = key.lower()
    slot_key = f"{subject}:{predicate}"
    return subject, predicate, value, slot_key


def _is_personal_or_project(statement: str) -> bool:
    clean = _normalize_text(statement)
    lower = clean.lower()
    if any(clean.startswith(prefix) for prefix in _USER_FACT_PREFIXES):
        return True
    return lower.startswith(("프로젝트 ", "flygpt ", "fly gpt ", "파피티 "))


def _eligible_for_auto_learning(message: str) -> bool:
    clean = _normalize_text(message)
    lower = clean.lower()
    return (
        any(clean.startswith(prefix) for prefix in _USER_FACT_PREFIXES)
        or lower.startswith(("프로젝트 ", "flygpt ", "fly gpt ", "파피티 "))
        or any(marker.lower() in lower for marker in _EXPLICIT_REMEMBER)
        or any(marker in clean for marker in _CORRECTION_HINTS)
    )


class KnowledgeStore:
    """Account-scoped long-term knowledge with provenance and verification state."""

    def __init__(self, path: str | Path = "data/flygpt_knowledge.sqlite3") -> None:
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
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    statement_hash TEXT NOT NULL,
                    subject TEXT,
                    predicate TEXT,
                    value TEXT,
                    slot_key TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    verified_at REAL,
                    expires_at REAL,
                    supersedes_id INTEGER,
                    conflicts_with_id INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_user_hash_active
                ON knowledge_items(user_id, statement_hash, source_type)
                WHERE status IN ('asserted', 'claim', 'verified', 'disputed')
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_user_slot
                ON knowledge_items(user_id, slot_key, status, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_user_updated
                ON knowledge_items(user_id, updated_at DESC)
                """
            )

    def _row(self, conn: sqlite3.Connection, item_id: int) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM knowledge_items WHERE id = ?",
            (int(item_id),),
        ).fetchone()
        return dict(row) if row is not None else None

    def _active_slot_row(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        slot_key: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT *
            FROM knowledge_items
            WHERE user_id = ?
              AND slot_key = ?
              AND status IN ('asserted', 'claim', 'verified', 'disputed')
            ORDER BY
              CASE status WHEN 'verified' THEN 3 WHEN 'asserted' THEN 2 ELSE 1 END DESC,
              confidence DESC,
              updated_at DESC
            LIMIT 1
            """,
            (user_id, slot_key),
        ).fetchone()
        return dict(row) if row is not None else None

    def add(
        self,
        user_id: str,
        statement: str,
        *,
        kind: str = "claim",
        source_type: str = "user_claim",
        source_ref: str | None = None,
        confidence: float | None = None,
        verified: bool = False,
        expires_at: float | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        value: str | None = None,
        slot_key: str | None = None,
    ) -> dict[str, Any] | None:
        clean_user = user_id.strip()[:128]
        clean_statement = _normalize_text(statement)[:1600]
        if not clean_user or not clean_statement or _contains_secret(clean_statement):
            return None

        source_type = source_type if source_type in SOURCE_TRUST else "user_claim"
        trust = SOURCE_TRUST[source_type]
        if confidence is None:
            confidence = trust
        confidence = max(0.0, min(float(confidence), trust))

        if verified and source_type not in TRUSTED_VERIFICATION_SOURCES:
            verified = False

        if not subject and not predicate and not value and not slot_key:
            subject, predicate, value, slot_key = _derive_slot(clean_statement)

        clean_kind = kind.strip().lower()
        if clean_kind not in {"personal", "project", "claim", "fact"}:
            clean_kind = "claim"

        if verified:
            status = "verified"
        elif source_type == "user_self" and clean_kind in {"personal", "project"}:
            status = "asserted"
        else:
            status = "claim"

        digest = _statement_hash(clean_statement)
        now = time.time()

        with self._connect() as conn:
            duplicate = conn.execute(
                """
                SELECT *
                FROM knowledge_items
                WHERE user_id = ?
                  AND statement_hash = ?
                  AND source_type = ?
                  AND status IN ('asserted', 'claim', 'verified', 'disputed')
                LIMIT 1
                """,
                (clean_user, digest, source_type),
            ).fetchone()

            if duplicate is not None:
                next_confidence = max(float(duplicate["confidence"]), confidence)
                conn.execute(
                    """
                    UPDATE knowledge_items
                    SET confidence = ?,
                        evidence_count = evidence_count + 1,
                        updated_at = ?,
                        source_ref = COALESCE(?, source_ref),
                        expires_at = COALESCE(?, expires_at)
                    WHERE id = ?
                    """,
                    (
                        next_confidence,
                        now,
                        source_ref,
                        expires_at,
                        int(duplicate["id"]),
                    ),
                )
                return self._row(conn, int(duplicate["id"]))

            supersedes_id = None
            conflicts_with_id = None

            if slot_key:
                existing = self._active_slot_row(conn, clean_user, slot_key)
                if existing is not None:
                    existing_value = _normalize_text(str(existing.get("value") or "")).lower()
                    new_value = _normalize_text(str(value or "")).lower()
                    same_value = bool(existing_value and new_value and existing_value == new_value)

                    if same_value:
                        conn.execute(
                            """
                            UPDATE knowledge_items
                            SET evidence_count = evidence_count + 1,
                                confidence = MAX(confidence, ?),
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (confidence, now, int(existing["id"])),
                        )
                        return self._row(conn, int(existing["id"]))

                    existing_trust = SOURCE_TRUST.get(str(existing["source_type"]), 0.0)
                    can_supersede = (
                        source_type == "user_self" and str(existing["source_type"]) == "user_self"
                    ) or (
                        trust >= existing_trust
                        and status in {"asserted", "verified"}
                        and str(existing["status"]) != "verified"
                    ) or (
                        status == "verified"
                        and trust >= existing_trust
                    )

                    if can_supersede:
                        supersedes_id = int(existing["id"])
                        conn.execute(
                            """
                            UPDATE knowledge_items
                            SET status = 'superseded', updated_at = ?
                            WHERE id = ?
                            """,
                            (now, supersedes_id),
                        )
                    else:
                        conflicts_with_id = int(existing["id"])
                        if status == "claim":
                            status = "disputed"

            cursor = conn.execute(
                """
                INSERT INTO knowledge_items(
                    user_id, statement, statement_hash,
                    subject, predicate, value, slot_key,
                    kind, status, source_type, source_ref,
                    confidence, evidence_count,
                    created_at, updated_at, verified_at, expires_at,
                    supersedes_id, conflicts_with_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_user,
                    clean_statement,
                    digest,
                    subject,
                    predicate,
                    value,
                    slot_key,
                    clean_kind,
                    status,
                    source_type,
                    source_ref,
                    confidence,
                    now,
                    now,
                    now if status == "verified" else None,
                    expires_at,
                    supersedes_id,
                    conflicts_with_id,
                ),
            )
            return self._row(conn, int(cursor.lastrowid))

    def learn_from_user_text(self, user_id: str, message: str) -> list[dict[str, Any]]:
        clean = _normalize_text(message)
        if not clean or len(clean) > 1600 or _contains_secret(clean):
            return []
        if not _eligible_for_auto_learning(clean):
            return []

        statement = _strip_remember_prefix(clean)
        personal = _is_personal_or_project(statement)
        source_type = "user_self" if personal else "user_claim"
        kind = "project" if "프로젝트" in statement or statement.lower().startswith(("flygpt", "fly gpt", "파피티")) else ("personal" if personal else "claim")

        item = self.add(
            user_id,
            statement,
            kind=kind,
            source_type=source_type,
            confidence=0.88 if personal else 0.55,
        )
        return [item] if item is not None else []

    def verify(
        self,
        user_id: str,
        item_id: int,
        *,
        source_type: str,
        source_ref: str | None = None,
        confidence: float | None = None,
        expires_at: float | None = None,
    ) -> dict[str, Any] | None:
        if source_type not in TRUSTED_VERIFICATION_SOURCES:
            raise ValueError("verification requires a trusted source type")

        trust = SOURCE_TRUST[source_type]
        verified_confidence = trust if confidence is None else max(0.0, min(float(confidence), trust))
        now = time.time()

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM knowledge_items
                WHERE id = ? AND user_id = ?
                """,
                (int(item_id), user_id),
            ).fetchone()
            if row is None:
                return None

            conn.execute(
                """
                UPDATE knowledge_items
                SET status = 'verified',
                    source_type = ?,
                    source_ref = ?,
                    confidence = ?,
                    verified_at = ?,
                    updated_at = ?,
                    expires_at = ?
                WHERE id = ?
                """,
                (
                    source_type,
                    source_ref,
                    verified_confidence,
                    now,
                    now,
                    expires_at,
                    int(item_id),
                ),
            )
            return self._row(conn, int(item_id))

    def reject(self, user_id: str, item_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE knowledge_items
                SET status = 'rejected', updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (time.time(), int(item_id), user_id),
            )
            return bool(cursor.rowcount)

    def search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 6,
        include_unverified: bool = False,
    ) -> list[dict[str, Any]]:
        now = time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM knowledge_items
                WHERE user_id = ?
                  AND status IN ('asserted', 'claim', 'verified', 'disputed')
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC
                LIMIT 300
                """,
                (user_id, now),
            ).fetchall()

        query_tokens = _tokens(query)
        broad_recall = any(hint in query.lower() for hint in _RECALL_HINTS)
        scored: list[tuple[float, dict[str, Any]]] = []

        for raw in rows:
            row = dict(raw)
            status = str(row["status"])
            kind = str(row["kind"])

            trusted_context = status == "verified" or (
                status == "asserted" and kind in {"personal", "project"}
            )
            if not include_unverified and not trusted_context:
                continue

            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("statement", "subject", "predicate", "value")
            )
            row_tokens = _tokens(haystack)
            overlap = len(query_tokens & row_tokens)

            if query_tokens and overlap == 0 and not broad_recall:
                continue

            status_score = {
                "verified": 1.4,
                "asserted": 1.0,
                "claim": 0.25,
                "disputed": 0.05,
            }.get(status, 0.0)
            source_score = SOURCE_TRUST.get(str(row["source_type"]), 0.0) * 0.6
            overlap_score = overlap * 1.25
            recency_days = max(0.0, (now - float(row["updated_at"])) / 86400.0)
            recency_score = 0.35 / (1.0 + recency_days / 120.0)

            if not query_tokens and broad_recall:
                overlap_score = 0.2

            score = status_score + source_score + overlap_score + recency_score
            scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        return [row for _, row in scored[: max(1, min(int(limit), 20))]]

    def context_for_query(self, user_id: str, query: str, *, limit: int = 6) -> tuple[str | None, list[dict[str, Any]]]:
        rows = self.search(user_id, query, limit=limit, include_unverified=False)
        if not rows:
            return None, []

        lines = [
            "Account-scoped long-term knowledge relevant to this request.",
            "Use [verified] items as factual evidence.",
            "Use [user-asserted] items only as the user's own profile/project context; "
            "do not present them as independently verified external facts.",
        ]
        for row in rows:
            label = "verified" if row["status"] == "verified" else "user-asserted"
            source = str(row["source_type"])
            source_ref = str(row.get("source_ref") or "").strip()
            source_text = f" source={source}"
            if source_ref:
                source_text += f" ref={source_ref[:240]}"
            lines.append(
                f"- [{label}]{source_text} confidence={float(row['confidence']):.2f}: "
                f"{row['statement']}"
            )

        return "\n".join(lines), rows

    def list_items(self, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM knowledge_items
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self, user_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM knowledge_items WHERE user_id = ?",
                (user_id,),
            )
            return int(cursor.rowcount or 0)

    def stats(self, user_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS items,
                    SUM(CASE WHEN status = 'verified' THEN 1 ELSE 0 END) AS verified,
                    SUM(CASE WHEN status = 'asserted' THEN 1 ELSE 0 END) AS asserted,
                    SUM(CASE WHEN status IN ('claim', 'disputed') THEN 1 ELSE 0 END) AS unverified,
                    SUM(CASE WHEN status = 'superseded' THEN 1 ELSE 0 END) AS superseded,
                    MAX(updated_at) AS last_updated_at
                FROM knowledge_items
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

        return {
            "enabled": True,
            "items": int(row["items"] or 0),
            "verified": int(row["verified"] or 0),
            "asserted": int(row["asserted"] or 0),
            "unverified": int(row["unverified"] or 0),
            "superseded": int(row["superseded"] or 0),
            "last_updated_at": row["last_updated_at"],
            "path": str(self.path),
        }
