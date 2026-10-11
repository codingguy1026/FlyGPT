"""No-network tests for Codespaces diagnostics."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.diagnose_codespaces import (
    Probe,
    diagnose,
    forwarded_address,
    is_backend_ok,
)

HEALTH = json.dumps({"status": "ok"})


class CodespacesDoctorTests(unittest.TestCase):
    def test_address_comes_from_codespaces_environment(self):
        self.assertEqual(
            forwarded_address({
                "CODESPACE_NAME": "curly-spork-abc123",
                "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN": "app.github.dev",
            }),
            "https://curly-spork-abc123-3000.app.github.dev/",
        )
        self.assertIsNone(forwarded_address({}))
        self.assertIsNone(forwarded_address({"CODESPACE_NAME": "../../secrets"}))

    def test_health_requires_valid_json_status(self):
        self.assertTrue(is_backend_ok(Probe(200, HEALTH)))
        self.assertFalse(is_backend_ok(Probe(200, "not json")))
        self.assertFalse(is_backend_ok(Probe(404, HEALTH)))

    def test_local_healthy_remote_404_is_not_mistaken_for_broken_app(self):
        def fake_probe(url, *, host=None):
            if url.endswith(":3000/api/health") or url.endswith(":8000/api/health"):
                return Probe(200, HEALTH)
            if url.startswith("https://"):
                return Probe(404, error="HTTP 404")
            return Probe(200, "<title>FlyGPT</title>")

        output = io.StringIO()
        with patch("scripts.diagnose_codespaces.probe", side_effect=fake_probe):
            with patch.dict("os.environ", {
                "CODESPACE_NAME": "my-codespace",
                "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN": "app.github.dev",
                "GITHUB_TOKEN": "SECRET_NOT_FOR_OUTPUT",
            }):
                with redirect_stdout(output):
                    result = diagnose()

        self.assertTrue(result)
        self.assertIn("forwarded link returned 404", output.getvalue())
        self.assertIn("https://my-codespace-3000.app.github.dev/", output.getvalue())
        self.assertNotIn("SECRET_NOT_FOR_OUTPUT", output.getvalue())

    def test_failed_frontend_proxy_is_reported(self):
        def fake_probe(url, *, host=None):
            if url.endswith(":3000/api/health"):
                return Probe(404)
            if url.endswith(":8000/api/health"):
                return Probe(200, HEALTH)
            return Probe(200, "<title>FlyGPT</title>")
        out = io.StringIO()
        with patch("scripts.diagnose_codespaces.probe", side_effect=fake_probe):
            with patch.dict("os.environ", {"CODESPACE_NAME": ""}, clear=True):
                with redirect_stdout(out):
                    result = diagnose(external=False)
        self.assertFalse(result)
        self.assertIn("Frontend API rewrite failed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
