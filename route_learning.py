from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9_]+")
_ROUTE_MAX_EXAMPLES = 240
_AUTO_WEIGHT = 0.34
_FEEDBACK_WEIGHT = 2.0
_MAX_ROW_WEIGHT = 4.0
_MIN_SIMILARITY = 0.26
_SHORT_MIN_SIMILARITY = 0.38
_MAX_BLEND = 0.40
_HALF_LIFE_DAYS = 180.0


def _stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    # Keep the value inside SQLite's signed INTEGER range when needed elsewhere.
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _text_digest(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _feature_fingerprint(text: str) -> set[int]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    features: set[int] = set()

    for token in tokens:
        features.add(_stable_hash("w:" + token))

        padded = "^" + token + "$"
        for n in (2, 3):
            if len(padded) >= n:
                for i in range(len(padded) - n + 1):
                    features.add(_stable_hash(f"cg{n}:{padded[i:i+n]}"))

        if len(token) >= 2:
            features.add(_stable_hash("pre2:" + token[:2]))
            features.add(_stable_hash("suf2:" + token[-2:]))
        if len(token) >= 3:
            features.add(_stable_hash("pre3:" + token[:3]))
            features.add(_stable_hash("suf3:" + token[-3:]))

    for left, right in zip(tokens, tokens[1:]):
        features.add(_stable_hash(f"wb:{left}|{right}"))

    for i in range(len(tokens) - 2):
        features.add(_stable_hash(f"ws:{tokens[i]}|{tokens[i+2]}"))

    punctuation = {
        "question": r"[?？]",
        "equation": r"=",
        "mathop": r"[+×✕÷*/%^]",
        "codebrace": r"[{}\[\]]",
        "codepunct": r"[;:]",
    }
    for name, pattern in punctuation.items():
        if re.search(pattern, text):
            features.add(_stable_hash("punct:" + name))

    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        features.add(_stable_hash("shape:url"))
    if re.search(
        r"\b(?:git|npm|pip|python|javascript|typescript|sql|react|fastapi)\b",
        text,
        re.IGNORECASE,
    ):
        features.add(_stable_hash("shape:devterm"))

    return features


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


class RouteLearningStore:
    """Account-scoped online routing adaptation without storing raw prompts.

    The global graph-router checkpoint never changes at request time. Instead,
    FlyGPT stores a compact hashed fingerprint of high-confidence user phrasing
    and uses similar prior examples as a bounded personalization signal.
    """

    def __init__(self, path: str | Path = "data/flygpt_route_learning.sqlite3") -> None:
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
                CREATE TABLE IF NOT EXISTS route_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    route TEXT NOT NULL,
                    feature_json TEXT NOT NULL,
                    weight REAL NOT NULL,
                    confirmations INTEGER NOT NULL DEFAULT 1,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, text_hash, route)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_route_examples_user_updated
                ON route_examples(user_id, updated_at DESC)
                """
            )

    def observe(
        self,
        user_id: str,
        text: str,
        route: str,
        *,
        source: str = "auto",
        weight: float | None = None,
    ) -> bool:
        clean_user = user_id.strip()[:128]
        clean_route = route.strip()[:64]
        features = _feature_fingerprint(text)
        if not clean_user or not clean_route or not features:
            return False

        source = "feedback" if source == "feedback" else "auto"
        increment = (
            _FEEDBACK_WEIGHT if source == "feedback" else _AUTO_WEIGHT
        ) if weight is None else max(0.05, min(float(weight), _MAX_ROW_WEIGHT))

        digest = _text_digest(text)
        feature_json = json.dumps(sorted(features), separators=(",", ":"))
        now = time.time()

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, weight, confirmations, source
                FROM route_examples
                WHERE user_id = ? AND text_hash = ? AND route = ?
                """,
                (clean_user, digest, clean_route),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO route_examples(
                        user_id, text_hash, route, feature_json, weight,
                        confirmations, source, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        clean_user,
                        digest,
                        clean_route,
                        feature_json,
                        min(increment, _MAX_ROW_WEIGHT),
                        source,
                        now,
                        now,
                    ),
                )
            else:
                next_weight = min(float(row["weight"]) + increment, _MAX_ROW_WEIGHT)
                next_source = (
                    "feedback"
                    if source == "feedback" or str(row["source"]) == "feedback"
                    else "auto"
                )
                conn.execute(
                    """
                    UPDATE route_examples
                    SET feature_json = ?,
                        weight = ?,
                        confirmations = ?,
                        source = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        feature_json,
                        next_weight,
                        int(row["confirmations"]) + 1,
                        next_source,
                        now,
                        int(row["id"]),
                    ),
                )

            self._prune(conn, clean_user)

        return True

    def _prune(self, conn: sqlite3.Connection, user_id: str) -> None:
        conn.execute(
            """
            DELETE FROM route_examples
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id
                  FROM route_examples
                  WHERE user_id = ?
                  ORDER BY
                    CASE source WHEN 'feedback' THEN 1 ELSE 0 END DESC,
                    updated_at DESC
                  LIMIT ?
              )
            """,
            (user_id, user_id, _ROUTE_MAX_EXAMPLES),
        )

    def examples(self, user_id: str, *, limit: int = 160) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), _ROUTE_MAX_EXAMPLES))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT route, feature_json, weight, confirmations, source, updated_at
                FROM route_examples
                WHERE user_id = ?
                ORDER BY
                    CASE source WHEN 'feedback' THEN 1 ELSE 0 END DESC,
                    updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def personalize(
        self,
        user_id: str,
        text: str,
        base: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(base)
        ranked = [dict(item) for item in (base.get("top_routes") or [])]
        if not ranked:
            return result

        query_features = _feature_fingerprint(text)
        if not query_features:
            result["personalization"] = {
                "applied": False,
                "examples_considered": 0,
                "neighbors_used": 0,
                "blend": 0.0,
                "nearest_similarity": 0.0,
            }
            return result

        rows = self.examples(user_id)
        if not rows:
            result["personalization"] = {
                "applied": False,
                "examples_considered": 0,
                "neighbors_used": 0,
                "blend": 0.0,
                "nearest_similarity": 0.0,
            }
            return result

        short_query = len(_TOKEN_RE.findall(text)) <= 2
        threshold = _SHORT_MIN_SIMILARITY if short_query else _MIN_SIMILARITY
        now = time.time()
        evidence: dict[str, float] = defaultdict(float)
        used = 0
        nearest = 0.0
        total_evidence = 0.0

        for row in rows:
            try:
                stored_features = {int(value) for value in json.loads(str(row["feature_json"]))}
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            similarity = _jaccard(query_features, stored_features)
            if similarity < threshold:
                continue

            age_days = max(0.0, (now - float(row["updated_at"])) / 86400.0)
            recency = 0.5 ** (age_days / _HALF_LIFE_DAYS)
            row_weight = max(0.0, min(float(row["weight"]), _MAX_ROW_WEIGHT))
            contribution = (similarity ** 2) * row_weight * recency
            if contribution <= 0:
                continue

            route = str(row["route"])
            evidence[route] += contribution
            total_evidence += contribution
            nearest = max(nearest, similarity)
            used += 1

        if not evidence or total_evidence <= 0:
            result["personalization"] = {
                "applied": False,
                "examples_considered": len(rows),
                "neighbors_used": 0,
                "blend": 0.0,
                "nearest_similarity": round(nearest, 4),
            }
            return result

        base_scores = {
            str(item["route"]): max(0.0, float(item["confidence"]))
            for item in ranked
        }
        for route in list(evidence):
            base_scores.setdefault(route, 0.0)

        total_base = sum(base_scores.values()) or 1.0
        normalized_base = {
            route: value / total_base for route, value in base_scores.items()
        }
        personalized = {
            route: value / total_evidence for route, value in evidence.items()
        }

        blend = min(_MAX_BLEND, total_evidence * 0.10)
        adjusted = {
            route: (1.0 - blend) * normalized_base.get(route, 0.0)
            + blend * personalized.get(route, 0.0)
            for route in normalized_base
        }

        total_adjusted = sum(adjusted.values()) or 1.0
        reranked = sorted(
            (
                {
                    "route": route,
                    "confidence": score / total_adjusted,
                }
                for route, score in adjusted.items()
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )

        confidence = float(reranked[0]["confidence"])
        second = float(reranked[1]["confidence"]) if len(reranked) > 1 else 0.0
        margin = max(0.0, confidence - second)
        min_confidence = float(base.get("min_confidence", 0.55))
        min_margin = float(base.get("min_margin", 0.10))

        result.update(
            {
                "base_route": str(base.get("route", reranked[0]["route"])),
                "base_confidence": float(base.get("confidence", confidence)),
                "base_margin": float(base.get("margin", margin)),
                "base_accepted": bool(base.get("accepted", True)),
                "route": reranked[0]["route"],
                "confidence": confidence,
                "margin": margin,
                "accepted": confidence >= min_confidence and margin >= min_margin,
                "top_routes": reranked,
                "personalization": {
                    "applied": blend > 0,
                    "examples_considered": len(rows),
                    "neighbors_used": used,
                    "blend": round(blend, 4),
                    "nearest_similarity": round(nearest, 4),
                },
            }
        )
        return result

    def observe_if_confident(
        self,
        user_id: str,
        text: str,
        route_info: dict[str, Any],
    ) -> bool:
        base_route = str(route_info.get("base_route", route_info.get("route", "")))
        final_route = str(route_info.get("route", ""))
        base_confidence = float(
            route_info.get("base_confidence", route_info.get("confidence", 0.0))
        )
        base_margin = float(route_info.get("base_margin", route_info.get("margin", 0.0)))
        base_accepted = bool(route_info.get("base_accepted", route_info.get("accepted", False)))
        min_confidence = float(route_info.get("min_confidence", 0.55))
        min_margin = float(route_info.get("min_margin", 0.10))

        # Auto-learning is deliberately conservative. It only learns examples
        # where the frozen router and personalized route agree and the frozen
        # router was comfortably above its own gate.
        if not base_accepted or base_route != final_route:
            return False
        if base_confidence < max(0.82, min_confidence + 0.10):
            return False
        if base_margin < max(0.22, min_margin + 0.10):
            return False

        return self.observe(user_id, text, final_route, source="auto")

    def feedback(self, user_id: str, text: str, route: str) -> bool:
        return self.observe(user_id, text, route, source="feedback")

    def clear(self, user_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM route_examples WHERE user_id = ?",
                (user_id,),
            )
            return int(cursor.rowcount or 0)

    def stats(self, user_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            aggregate = conn.execute(
                """
                SELECT
                    COUNT(*) AS examples,
                    SUM(CASE source WHEN 'feedback' THEN 1 ELSE 0 END) AS feedback_examples,
                    SUM(CASE source WHEN 'auto' THEN 1 ELSE 0 END) AS auto_examples,
                    SUM(confirmations) AS confirmations,
                    MAX(updated_at) AS last_learned_at
                FROM route_examples
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            route_rows = conn.execute(
                """
                SELECT route, COUNT(*) AS examples
                FROM route_examples
                WHERE user_id = ?
                GROUP BY route
                ORDER BY examples DESC, route ASC
                """,
                (user_id,),
            ).fetchall()

        return {
            "enabled": True,
            "examples": int(aggregate["examples"] or 0),
            "auto_examples": int(aggregate["auto_examples"] or 0),
            "feedback_examples": int(aggregate["feedback_examples"] or 0),
            "confirmations": int(aggregate["confirmations"] or 0),
            "last_learned_at": aggregate["last_learned_at"],
            "routes": {
                str(row["route"]): int(row["examples"])
                for row in route_rows
            },
            "stores_raw_prompts": False,
            "max_examples": _ROUTE_MAX_EXAMPLES,
            "path": str(self.path),
        }
