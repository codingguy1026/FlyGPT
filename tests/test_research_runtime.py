from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from research_runtime import (
    BraveResearchRuntime,
    ResearchResult,
    WebSource,
    validated_fact_candidates,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def source(url: str, *, authority: float = 0.75) -> WebSource:
    hostname = url.split("//", 1)[1].split("/", 1)[0]
    labels = hostname.split(".")
    root = ".".join(labels[-2:]) if len(labels) >= 2 else hostname
    return WebSource(
        url=url,
        title=hostname,
        hostname=hostname,
        root_domain=root,
        snippets=["supporting evidence"],
        age=[],
        authority=authority,
    )


class BraveResearchRuntimeTests(unittest.TestCase):
    def test_unconfigured_search_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = BraveResearchRuntime()
            result = runtime.search("latest FlyGPT facts")

        self.assertFalse(runtime.configured)
        self.assertFalse(result.used)
        self.assertIn("BRAVE_SEARCH_API_KEY", result.error or "")

    def test_search_parses_llm_context_sources(self):
        payload = {
            "grounding": {
                "generic": [
                    {
                        "url": "https://example.com/fact",
                        "title": "Example Fact",
                        "snippets": ["Relevant extracted text."],
                    },
                    {
                        "url": "https://www.cdc.gov/topic",
                        "title": "CDC Topic",
                        "snippets": ["Primary source text."],
                    },
                ]
            },
            "sources": {
                "https://example.com/fact": {
                    "hostname": "example.com",
                    "age": ["", "2026-10-01"],
                },
                "https://www.cdc.gov/topic": {
                    "hostname": "www.cdc.gov",
                    "age": ["", "2026-10-02"],
                },
            },
        }

        with patch.dict(
            os.environ,
            {"BRAVE_SEARCH_API_KEY": "test-key"},
            clear=True,
        ), patch(
            "research_runtime.urllib.request.urlopen",
            return_value=FakeHTTPResponse(payload),
        ) as mocked:
            runtime = BraveResearchRuntime()
            result = runtime.search("test query")

        self.assertTrue(result.used)
        self.assertEqual(len(result.sources), 2)
        self.assertEqual(result.sources[0].hostname, "example.com")
        self.assertGreaterEqual(result.sources[1].authority, 0.99)

        request = mocked.call_args.args[0]
        self.assertEqual(request.get_header("X-subscription-token"), "test-key")
        self.assertIn("context_threshold_mode=strict", request.full_url)
        self.assertIn("safesearch=strict", request.full_url)

    def test_two_independent_domains_can_verify_candidate(self):
        research = ResearchResult(
            query="what is x",
            provider="brave-llm-context",
            sources=[
                source("https://alpha.example/fact"),
                source("https://beta.test/fact"),
            ],
            retrieved_at=1000.0,
        )
        facts = [
            {
                "statement": "X has the value Y.",
                "subject": "X",
                "predicate": "value",
                "value": "Y",
                "slot_key": "X:value",
                "support_urls": [
                    "https://alpha.example/fact",
                    "https://beta.test/fact",
                ],
                "confidence": 0.96,
                "volatility": "stable",
            }
        ]

        accepted = validated_fact_candidates(facts, research, query="what is x")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(accepted[0]["support_domains"]), 2)
        self.assertGreater(accepted[0]["expires_at"], 1000.0)

    def test_same_root_domain_does_not_count_as_two_sources(self):
        research = ResearchResult(
            query="fact",
            provider="brave-llm-context",
            sources=[
                WebSource(
                    url="https://a.example.com/fact",
                    title="A",
                    hostname="a.example.com",
                    root_domain="example.com",
                    snippets=["evidence"],
                    age=[],
                    authority=0.75,
                ),
                WebSource(
                    url="https://b.example.com/fact",
                    title="B",
                    hostname="b.example.com",
                    root_domain="example.com",
                    snippets=["evidence"],
                    age=[],
                    authority=0.75,
                ),
            ],
            retrieved_at=1000.0,
        )
        facts = [
            {
                "statement": "A supported statement.",
                "support_urls": [
                    "https://a.example.com/fact",
                    "https://b.example.com/fact",
                ],
                "confidence": 0.99,
            }
        ]

        self.assertEqual(
            validated_fact_candidates(facts, research, query="fact"),
            [],
        )

    def test_single_primary_authority_can_pass(self):
        research = ResearchResult(
            query="public statistic",
            provider="brave-llm-context",
            sources=[source("https://cdc.gov/fact", authority=1.0)],
            retrieved_at=1000.0,
        )
        facts = [
            {
                "statement": "A primary-source statistic is reported.",
                "support_urls": ["https://cdc.gov/fact"],
                "confidence": 0.95,
                "volatility": "medium",
            }
        ]

        accepted = validated_fact_candidates(
            facts,
            research,
            query="current public statistic",
        )
        self.assertEqual(len(accepted), 1)
        self.assertLess(accepted[0]["expires_at"], 1000.0 + 31 * 86400)


if __name__ == "__main__":
    unittest.main()
