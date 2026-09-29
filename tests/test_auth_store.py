from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from auth_store import AuthStore


class AuthStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = AuthStore(Path(self.tempdir.name) / "auth.sqlite3")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_signup_authenticate_session_and_logout(self):
        user = self.store.create_user(
            "Pilot@Example.com",
            "correct-horse-battery",
            "Pilot",
        )
        self.assertEqual(user["email"], "pilot@example.com")
        self.assertEqual(user["display_name"], "Pilot")

        authenticated = self.store.authenticate(
            "pilot@example.com",
            "correct-horse-battery",
        )
        self.assertIsNotNone(authenticated)
        self.assertEqual(authenticated["id"], user["id"])

        self.assertIsNone(
            self.store.authenticate("pilot@example.com", "wrong-password")
        )

        token = self.store.create_session(user["id"])
        session_user = self.store.user_for_token(token)
        self.assertIsNotNone(session_user)
        self.assertEqual(session_user["id"], user["id"])

        self.store.revoke_session(token)
        self.assertIsNone(self.store.user_for_token(token))

    def test_duplicate_email_is_rejected(self):
        self.store.create_user("pilot@example.com", "12345678", "Pilot")
        with self.assertRaises(ValueError):
            self.store.create_user("PILOT@example.com", "abcdefgh", "Other")

    def test_short_password_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create_user("pilot@example.com", "short", "Pilot")


if __name__ == "__main__":
    unittest.main()
