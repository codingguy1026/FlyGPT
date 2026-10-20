from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import app as flygpt_app


class HealthEndpointTests(unittest.TestCase):
    def test_local_cpu_budget_is_distinct_from_hosted_budget(self) -> None:
        with patch.dict(os.environ, {"FLYGPT_LOCAL_ONLY": "1"}, clear=True):
            self.assertEqual(flygpt_app._chat_request_budget_seconds(), 150.0)
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "1", "FLYGPT_CHAT_BUDGET": "200",
        }, clear=True):
            self.assertEqual(flygpt_app._chat_request_budget_seconds(), 200.0)
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "1", "FLYGPT_CHAT_BUDGET": "999",
        }, clear=True):
            self.assertEqual(flygpt_app._chat_request_budget_seconds(), 240.0)
        with patch.dict(os.environ, {"FLYGPT_LOCAL_ONLY": "0"}, clear=True):
            self.assertEqual(flygpt_app._chat_request_budget_seconds(), 22.0)
        with patch.dict(os.environ, {
            "FLYGPT_LOCAL_ONLY": "0", "FLYGPT_CHAT_BUDGET": "999",
        }, clear=True):
            self.assertEqual(flygpt_app._chat_request_budget_seconds(), 28.0)

    def test_health_reports_malecns_without_local_parquet(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NEUPRINT_DATASET": "male-cns:v1.0",
                "NEUPRINT_TOKEN": "test-token",
            },
            clear=False,
        ):
            payload = flygpt_app.health()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["connectome"]["provider"], "neuPrint")
        self.assertEqual(payload["connectome"]["dataset"], "male-cns:v1.0")
        self.assertTrue(payload["connectome"]["configured"])
        self.assertNotIn("data_dir", payload)
        self.assertNotIn("data_available", payload)
        self.assertTrue(payload["knowledge"]["enabled"])
        self.assertTrue(payload["knowledge"]["account_scoped"])
        self.assertTrue(payload["knowledge"]["provenance_aware"])
        self.assertEqual(payload["research"]["provider"], "brave-llm-context")
        self.assertIn("configured", payload["research"])
        self.assertEqual(
            payload["chat_request_budget_seconds"],
            flygpt_app.CHAT_REQUEST_BUDGET_SECONDS,
        )
        self.assertLess(payload["chat_request_budget_seconds"], 30.0)


if __name__ == "__main__":
    unittest.main()
