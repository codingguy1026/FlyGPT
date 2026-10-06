from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import app as flygpt_app


class HealthEndpointTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
