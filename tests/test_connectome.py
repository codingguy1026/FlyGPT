from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from connectome import MaleCNSConnectome


class FakeClient:
    def __init__(self) -> None:
        self.top_neuron_queries = 0

    def fetch_custom(self, query: str):
        if "count(n) AS neurons" in query:
            return pd.DataFrame([{"neurons": 3}])
        if "AS body_id" in query and "total_synapses" in query:
            self.top_neuron_queries += 1
            return pd.DataFrame(
                [
                    {
                        "body_id": 2,
                        "type": "DNp01",
                        "instance": "DNp01_R",
                        "consensus_nt": "GABA",
                        "total_synapses": 21,
                        "outgoing_synapses": 9,
                        "incoming_synapses": 12,
                    },
                    {
                        "body_id": 1,
                        "type": "MBON",
                        "instance": "MBON_L",
                        "consensus_nt": "acetylcholine",
                        "total_synapses": 13,
                        "outgoing_synapses": 8,
                        "incoming_synapses": 5,
                    },
                ]
            )
        if "pre_pt_root_id" in query and "consensus_nt" in query:
            return pd.DataFrame(
                [
                    {
                        "pre_pt_root_id": 1,
                        "post_pt_root_id": 2,
                        "syn_count": 8,
                        "consensus_nt": "acetylcholine",
                    },
                    {
                        "pre_pt_root_id": 3,
                        "post_pt_root_id": 1,
                        "syn_count": 5,
                        "consensus_nt": "GABA",
                    },
                ]
            )
        if "RETURN pre.bodyId AS pre, post.bodyId AS post" in query:
            return pd.DataFrame(
                [
                    {"pre": 1, "post": 2, "weight": 8},
                    {"pre": 3, "post": 1, "weight": 5},
                ]
            )
        raise AssertionError(f"Unexpected query: {query}")


def sample_adjacencies(*, sources=None, targets=None, **kwargs):
    neurons = pd.DataFrame(
        [
            {"bodyId": 1, "consensusNt": "acetylcholine"},
            {"bodyId": 2, "consensusNt": "GABA"},
            {"bodyId": 3, "consensusNt": "glutamate"},
        ]
    )
    if sources == [1] and targets == [2]:
        edges = pd.DataFrame(
            [
                {"bodyId_pre": 1, "bodyId_post": 2, "roi": "AL(R)", "weight": 3},
                {"bodyId_pre": 1, "bodyId_post": 2, "roi": "LH(R)", "weight": 2},
            ]
        )
    elif sources == [1]:
        edges = pd.DataFrame(
            [
                {"bodyId_pre": 1, "bodyId_post": 2, "roi": "AL(R)", "weight": 3},
                {"bodyId_pre": 1, "bodyId_post": 2, "roi": "LH(R)", "weight": 2},
            ]
        )
    else:
        edges = pd.DataFrame(
            [
                {"bodyId_pre": 3, "bodyId_post": 1, "roi": "AL(R)", "weight": 5},
            ]
        )
    return neurons, edges


class MaleCNSConnectomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.connectome = MaleCNSConnectome(client=self.client)

    @patch(
        "connectome.fetch_meta",
        return_value={
            "dataset": "male-cns:v1.0",
            "totalPreCount": 100,
            "totalPostCount": 200,
            "primaryRois": ["AL(R)", "LH(R)"],
            "lastDatabaseEdit": "2026-06-08",
        },
    )
    def test_stats(self, _mock_meta) -> None:
        stats = self.connectome.stats()
        self.assertEqual(stats["dataset"], "male-cns:v1.0")
        self.assertEqual(stats["neurons"], 3)
        self.assertEqual(stats["presynaptic_sites"], 100)
        self.assertEqual(stats["postsynaptic_sites"], 200)
        self.assertEqual(stats["neuropils"], 2)

    @patch("connectome.fetch_adjacencies", side_effect=sample_adjacencies)
    def test_output_partner_aggregation(self, _mock_adj) -> None:
        outputs = self.connectome.neuron(1, direction="outputs")["outputs"]
        self.assertEqual(outputs[0]["partner_root_id"], 2)
        self.assertEqual(outputs[0]["syn_count"], 5)
        self.assertEqual(outputs[0]["neuropil_count"], 2)
        self.assertEqual(outputs[0]["dominant_nt"], "ACH")

    @patch("connectome.fetch_adjacencies", side_effect=sample_adjacencies)
    def test_pair(self, _mock_adj) -> None:
        pair = self.connectome.pair(1, 2)
        self.assertEqual(pair["syn_count"], 5)
        self.assertEqual(len(pair["by_neuropil"]), 2)
        self.assertEqual(pair["by_neuropil"][0]["dominant_nt"], "ACH")

    def test_top_connections(self) -> None:
        rows = self.connectome.top_connections(limit=2)
        self.assertEqual(rows[0]["pre_pt_root_id"], 1)
        self.assertEqual(rows[0]["post_pt_root_id"], 2)
        self.assertEqual(rows[0]["syn_count"], 8)
        self.assertEqual(rows[0]["dominant_nt"], "ACH")

    @patch(
        "connectome.fetch_meta",
        return_value={"lastDatabaseEdit": "edit-1"},
    )
    def test_top_neurons(self, _mock_meta) -> None:
        rows = self.connectome.top_neurons(limit=2)
        self.assertEqual(rows[0]["body_id"], 2)
        self.assertEqual(rows[0]["total_synapses"], 21)
        self.assertEqual(rows[0]["incoming_synapses"], 12)
        self.assertEqual(rows[0]["outgoing_synapses"], 9)
        self.assertEqual(rows[0]["dominant_nt"], "GABA")
        self.assertEqual(rows[0]["type"], "DNp01")
        self.assertEqual(rows[0]["metric"], "pre_plus_post_synaptic_sites")
        self.assertEqual(self.client.top_neuron_queries, 1)

    @patch(
        "connectome.fetch_meta",
        return_value={"lastDatabaseEdit": "edit-1"},
    )
    def test_top_neurons_reuses_cache_within_ttl(self, mock_meta) -> None:
        first = self.connectome.top_neurons(limit=2)
        second = self.connectome.top_neurons(limit=2)

        self.assertEqual(first, second)
        self.assertEqual(self.client.top_neuron_queries, 1)
        self.assertEqual(mock_meta.call_count, 1)

    @patch(
        "connectome.fetch_meta",
        return_value={"lastDatabaseEdit": "edit-1"},
    )
    def test_top_neurons_keeps_cache_after_ttl_when_database_is_unchanged(
        self,
        mock_meta,
    ) -> None:
        self.connectome.top_neurons(limit=2)
        self.connectome._top_neurons_cache_checked_at -= (
            self.connectome._top_neurons_cache_ttl + 1
        )
        self.connectome.top_neurons(limit=2)

        self.assertEqual(self.client.top_neuron_queries, 1)
        self.assertEqual(mock_meta.call_count, 2)

    @patch(
        "connectome.fetch_meta",
        side_effect=[
            {"lastDatabaseEdit": "edit-1"},
            {"lastDatabaseEdit": "edit-2"},
        ],
    )
    def test_top_neurons_refreshes_cache_when_database_changes(self, mock_meta) -> None:
        self.connectome.top_neurons(limit=2)
        self.connectome._top_neurons_cache_checked_at -= (
            self.connectome._top_neurons_cache_ttl + 1
        )
        self.connectome.top_neurons(limit=2)

        self.assertEqual(self.client.top_neuron_queries, 2)
        self.assertEqual(mock_meta.call_count, 2)

    def test_scaffold_edges(self) -> None:
        self.assertEqual(
            self.connectome.scaffold_edges(candidate_edges=2),
            [(1, 2, 8), (3, 1, 5)],
        )


if __name__ == "__main__":
    unittest.main()
