from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


PASSWORD_ITERATIONS = 240_000
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30


class AuthStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                ON auth_sessions(user_id);
                """
            )

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "email": str(row["email"]),
            "display_name": str(row["display_name"]),
            "created_at": int(row["created_at"]),
        }

    @staticmethod
    def _derive_password(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PASSWORD_ITERATIONS,
            dklen=32,
        )

    def create_user(
        self,
        email: str,
        password: str,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_email(email)
        if "@" not in normalized or "." not in normalized.rsplit("@", 1)[-1]:
            raise ValueError("올바른 이메일 주소를 입력해 주세요.")
        if len(password) < 8:
            raise ValueError("비밀번호는 8자 이상이어야 합니다.")
        if len(password) > 256:
            raise ValueError("비밀번호가 너무 깁니다.")

        name = (display_name or normalized.split("@", 1)[0]).strip()
        if not name:
            name = "FlyGPT User"
        name = name[:50]

        salt = os.urandom(16)
        password_hash = self._derive_password(password, salt)
        user_id = str(uuid.uuid4())
        created_at = int(time.time())

        try:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO users (
                        id, email, display_name, password_salt, password_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        normalized,
                        name,
                        salt,
                        password_hash,
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("이미 가입된 이메일입니다.") from exc

        return {
            "id": user_id,
            "email": normalized,
            "display_name": name,
            "created_at": created_at,
        }

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        normalized = self.normalize_email(email)
        with self._connect() as db:
            row = db.execute(
                """
                SELECT id, email, display_name, password_salt, password_hash, created_at
                FROM users
                WHERE email = ?
                """,
                (normalized,),
            ).fetchone()

        if row is None:
            return None

        candidate = self._derive_password(password, bytes(row["password_salt"]))
        if not hmac.compare_digest(candidate, bytes(row["password_hash"])):
            return None

        return self.public_user(row)

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        token_hash = self._token_hash(token)
        now = int(time.time())
        expires_at = now + SESSION_TTL_SECONDS

        with self._connect() as db:
            db.execute(
                """
                INSERT INTO auth_sessions (
                    token_hash, user_id, created_at, expires_at
                ) VALUES (?, ?, ?, ?)
                """,
                (token_hash, user_id, now, expires_at),
            )
            db.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ?",
                (now,),
            )

        return token

    def user_for_token(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None

        now = int(time.time())
        token_hash = self._token_hash(token)

        with self._connect() as db:
            row = db.execute(
                """
                SELECT u.id, u.email, u.display_name, u.created_at, s.expires_at
                FROM auth_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()

        if row is None:
            return None
        return self.public_user(row)

    def revoke_session(self, token: str | None) -> None:
        if not token:
            return
        with self._connect() as db:
            db.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (self._token_hash(token),),
            )
