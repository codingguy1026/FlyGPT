from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from connectome import MaleCNSConnectome


class FakeClient:
    def fetch_custom(self, query: str):
        if "count(n) AS neurons" in query:
            return pd.DataFrame([{"neurons": 3}])
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
        self.connectome = MaleCNSConnectome(client=FakeClient())

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

    def test_scaffold_edges(self) -> None:
        self.assertEqual(
            self.connectome.scaffold_edges(candidate_edges=2),
            [(1, 2, 8), (3, 1, 5)],
        )


if __name__ == "__main__":
    unittest.main()
