"""Unit tests for repeated local serving probes (no network)."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from scripts.check_stability import check_stability
from scripts.diagnose_codespaces import Probe


class StabilityCheckTests(unittest.TestCase):
    def test_healthy_samples_are_successful(self):
        hits = []

        def prober(url):
            hits.append(url)
            if url.endswith("/api/health"):
                return Probe(200, json.dumps({"status": "ok"}))
            return Probe(200, "<title>FlyGPT</title>")

        out = io.StringIO()
        with redirect_stdout(out):
            healthy = check_stability(
                samples=3, interval_seconds=0, prober=prober, sleeper=lambda _: None
            )

        self.assertTrue(healthy)
        self.assertEqual(len(hits), 9)
        self.assertIn("All local probes succeeded", out.getvalue())

    def test_one_intermediate_proxy_failure_is_detected(self):
        proxy_calls = 0

        def prober(url):
            nonlocal proxy_calls
            if url.endswith(":3000/api/health"):
                proxy_calls += 1
                if proxy_calls == 2:
                    return Probe(502)
            if url.endswith("/api/health"):
                return Probe(200, '{"status": "ok"}')
            return Probe(200, "<title>FlyGPT</title>")

        out = io.StringIO()
        with redirect_stdout(out):
            healthy = check_stability(
                samples=3, interval_seconds=0, prober=prober, sleeper=lambda _: None
            )

        self.assertFalse(healthy)
        self.assertIn("proxy=1/3", out.getvalue())

    def test_invalid_sample_count_is_rejected(self):
        with self.assertRaises(ValueError):
            check_stability(samples=0)


if __name__ == "__main__":
    unittest.main()
