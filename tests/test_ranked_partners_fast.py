"""Server-side ranked partner queries, directional weights, and bounded cache."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from connectome import MaleCNSConnectome


class RankedPartnerClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def fetch_custom(self, query: str) -> pd.DataFrame:
        self.queries.append(query)
        if "MATCH (selected:Neuron {bodyId: 10009})-[edge:ConnectsTo]-(partner:Neuron)" in query:
            # The server returns the *globally* strongest incoming/outgoing
            # edges in a single sorted, limited result set.
            return pd.DataFrame([
                {
                    "partner_root_id": 10836, "syn_count": 918,
                    "consensus_nt": "acetylcholine", "direction": "inputs",
                    "roi_info": '{"a":{"post":400},"b":{"post":518}}',
                },
                {
                    "partner_root_id": 537032, "syn_count": 288,
                    "consensus_nt": "GABA", "direction": "outputs",
                    "roi_info": '{"e":{"post":288}}',
                },
                {
                    "partner_root_id": 10532, "syn_count": 144,
                    "consensus_nt": "ACH", "direction": "inputs",
                    "roi_info": {"c": {"post": 144}, "d": {"post": 0}},
                },
                {
                    "partner_root_id": 537032, "syn_count": 126,
                    "consensus_nt": "GABA", "direction": "inputs",
                    "roi_info": None,
                },
            ])
        if "<-[edge:ConnectsTo]-" in query:
            return pd.DataFrame([
                {
                    "partner_root_id": 10836, "syn_count": 918,
                    "consensus_nt": "acetylcholine",
                    "roi_info": '{"a":{"post":400},"b":{"post":518}}',
                },
                {
                    "partner_root_id": 10532, "syn_count": 144,
                    "consensus_nt": "ACH",
                    "roi_info": {"c": {"post": 144}, "d": {"post": 0}},
                },
                {
                    "partner_root_id": 537032, "syn_count": 126,
                    "consensus_nt": "GABA", "roi_info": None,
                },
            ])
        if "-[edge:ConnectsTo]->" in query:
            return pd.DataFrame([
                {
                    "partner_root_id": 537032, "syn_count": 288,
                    "consensus_nt": "GABA",
                    "roi_info": '{"e":{"post":288}}',
                },
                {
                    "partner_root_id": 11684, "syn_count": 83,
                    "consensus_nt": "GABA", "roi_info": "{}",
                },
            ])
        raise AssertionError("Query did not anchor directed connection matching")


class RankedPartnerFastPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = RankedPartnerClient()
        self.connectome = MaleCNSConnectome(client=self.client)

    def test_bidirectional_query_uses_server_sort_limit_and_real_weights(self) -> None:
        with patch("connectome.fetch_adjacencies") as slow:
            result = self.connectome.ranked_partners(10009, limit=4)
        slow.assert_not_called()

        self.assertEqual(len(self.client.queries), 1)
        query = self.client.queries[0]
        self.assertIn("selected:Neuron {bodyId: 10009}", query)
        self.assertIn("-[edge:ConnectsTo]-(partner:Neuron)", query)
        self.assertIn("startNode(edge) AS presynaptic", query)
        self.assertIn("presynaptic.consensusNt AS consensus_nt", query)
        self.assertIn("CASE WHEN presynaptic = selected", query)
        self.assertIn("edge.weight AS syn_count", query)
        self.assertIn(
            "ORDER BY syn_count DESC, partner_root_id ASC, direction ASC",
            query,
        )
        self.assertIn("LIMIT 4", query)
        self.assertNotIn("fetch_adjacencies", query)
        self.assertEqual(
            [(row["partner_root_id"], row["direction"], row["syn_count"])
             for row in result],
            [
                (10836, "inputs", 918),
                (537032, "outputs", 288),
                (10532, "inputs", 144),
                (537032, "inputs", 126),
            ],
        )
        self.assertEqual(result[0]["dominant_nt"], "ACH")
        self.assertEqual(result[0]["roi_count"], 2)
        self.assertEqual(result[2]["roi_count"], 1)
        self.assertIsNone(result[3]["roi_count"])

    def test_bidirectional_empty_result_returns_empty_list(self) -> None:
        with patch.object(
            self.client, "fetch_custom", return_value=pd.DataFrame()
        ) as query:
            result = self.connectome.ranked_partners(10009, limit=10)
        self.assertEqual(result, [])
        query.assert_called_once()

    def test_single_query_preserves_two_opposite_edges_of_same_partner(self) -> None:
        result = self.connectome.ranked_partners(10009, limit=4)
        same_partner = [row for row in result if row["partner_root_id"] == 537032]
        self.assertEqual(
            [(row["direction"], row["syn_count"]) for row in same_partner],
            [("outputs", 288), ("inputs", 126)],
        )

    def test_invalid_combined_direction_fails_instead_of_silently_guessing(self) -> None:
        with patch.object(
            self.client,
            "fetch_custom",
            return_value=pd.DataFrame([{
                "partner_root_id": 20000, "syn_count": 5,
                "consensus_nt": "ACH", "roi_info": None, "direction": "unknown",
            }]),
        ) as query:
            with self.assertRaisesRegex(ValueError, "invalid connection direction"):
                self.connectome.ranked_partners(10009, limit=5)
        query.assert_called_once()
        self.assertNotIn((10009, "both", 1, 5), self.connectome._ranked_partners_cache)

    def test_direction_output_queries_once_and_presynaptic_nt_is_selected(self) -> None:
        result = self.connectome.ranked_partners(
            10009, direction="outputs", min_synapses=20, limit=2
        )
        self.assertEqual(len(self.client.queries), 1)
        self.assertIn("selected.consensusNt AS consensus_nt", self.client.queries[0])
        self.assertIn("WHERE edge.weight >= 20", self.client.queries[0])
        self.assertEqual(result[0]["direction"], "outputs")
        self.assertEqual(result[0]["syn_count"], 288)

    def test_direction_input_queries_once_and_presynaptic_nt_is_partner(self) -> None:
        result = self.connectome.ranked_partners(
            10009, direction="inputs", limit=2
        )
        self.assertEqual(len(self.client.queries), 1)
        self.assertIn("partner.consensusNt AS consensus_nt", self.client.queries[0])
        self.assertEqual(result[0]["partner_root_id"], 10836)

    def test_repeat_within_ttl_avoids_all_network_calls_and_returns_copy(self) -> None:
        first = self.connectome.ranked_partners(10009, limit=3)
        self.assertEqual(len(self.client.queries), 1)
        first[0]["syn_count"] = -999
        second = self.connectome.ranked_partners(10009, limit=3)
        self.assertEqual(len(self.client.queries), 1)
        self.assertEqual(second[0]["syn_count"], 918)

    def test_cache_expiration_forces_new_query(self) -> None:
        self.connectome.ranked_partners(10009, direction="inputs")
        self.assertEqual(len(self.client.queries), 1)
        key = (10009, "inputs", 1, 10)
        timestamp, rows = self.connectome._ranked_partners_cache[key]
        self.connectome._ranked_partners_cache[key] = (
            timestamp - self.connectome._ranked_partners_ttl - 1, rows,
        )
        self.connectome.ranked_partners(10009, direction="inputs")
        self.assertEqual(len(self.client.queries), 2)

    def test_disabled_cache_does_not_reuse_stale_results(self) -> None:
        with patch.dict(os.environ, {"FLYGPT_PARTNER_CACHE_TTL": "0"}):
            conn = MaleCNSConnectome(client=self.client)
        conn.ranked_partners(10009, direction="inputs", limit=1)
        conn.ranked_partners(10009, direction="inputs", limit=1)
        self.assertEqual(len(self.client.queries), 2)

    def test_invalid_inputs_do_not_query_server(self) -> None:
        for kwargs in (
            {"direction": "sideways"},
            {"min_synapses": 0},
            {"limit": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.connectome.ranked_partners(10009, **kwargs)
        self.assertEqual(len(self.client.queries), 0)

    def test_full_neuron_detail_path_is_not_replaced_by_fast_ranking(self) -> None:
        # ROI-specific/full-partner pages still use the existing detailed
        # fetch_adjacencies API, with their established ROI semantics.
        with patch("connectome.fetch_adjacencies", return_value=(None, pd.DataFrame())) as adj:
            result = self.connectome.neuron(10009, direction="inputs", limit=5)
        self.assertEqual(result, {"inputs": []})
        adj.assert_called_once()
        self.assertEqual(len(self.client.queries), 0)


if __name__ == "__main__":
    unittest.main()
