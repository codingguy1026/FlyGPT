"""Regression tests: a neuron's strongest neighbors are not the global leaderboard."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from starlette.requests import Request

import app as flygpt_app


USER_QUESTION = (
    "MaleCNS에서 뉴런 10009(CT1)와 가장 강하게 연결된 뉴런 10개를 찾아줘. "
    "각각의 뉴런 ID, 연결 방향(입력/출력), 시냅스 가중치를 보여줘. "
    "추측하지 말고 실제 데이터를 조회해."
)


class NeuronPartnerRankingTests(unittest.TestCase):
    def test_named_neuron_ranking_is_not_global_top_neurons(self):
        self.assertTrue(flygpt_app.is_neuron_partner_ranking_query(USER_QUESTION))
        self.assertFalse(flygpt_app.is_strongest_neuron_query(USER_QUESTION))
        self.assertFalse(flygpt_app.is_top_connection_query(USER_QUESTION))
        self.assertEqual(flygpt_app.extract_root_ids(USER_QUESTION), [10009])
        self.assertEqual(flygpt_app.extract_limit(USER_QUESTION), 10)

    def test_korean_number_suffix_parsing_excludes_unrelated_years(self):
        self.assertEqual(flygpt_app.extract_root_ids("10009번 뉴런"), [10009])
        self.assertEqual(flygpt_app.extract_root_ids("10009에 연결된 뉴런"), [10009])
        self.assertEqual(flygpt_app.extract_root_ids("10009와 이웃한 뉴런"), [10009])
        self.assertEqual(flygpt_app.extract_root_ids("10009(CT1)"), [10009])
        self.assertEqual(flygpt_app.extract_root_ids("2026년 통계"), [])

    def test_global_leaderboard_remains_supported(self):
        question = "네 뉴런중 연결이 가장 강한 뉴런은 뭐야?"
        self.assertTrue(flygpt_app.is_strongest_neuron_query(question))
        self.assertFalse(flygpt_app.is_neuron_partner_ranking_query(question))

    def test_single_neuron_input_only_ranking(self):
        question = "뉴런 10009에 가장 강하게 연결된 입력 뉴런 상위 5개"
        self.assertTrue(flygpt_app.is_neuron_partner_ranking_query(question))
        self.assertEqual(flygpt_app.extract_limit(question), 5)

    def test_no_specific_neuron_does_not_hijack_global_connection_query(self):
        question = "가장 강한 연결 순위 TOP 10 알려줘"
        self.assertFalse(flygpt_app.is_neuron_partner_ranking_query(question))
        self.assertTrue(flygpt_app.is_top_connection_query(question))

    def test_partner_ranking_calls_real_neuron_query_instead_of_global_top(self):
        conn = MagicMock()
        conn.neuron.return_value = {
            "inputs": [
                {
                    "partner_root_id": 22333,
                    "syn_count": 120,
                    "neuropil_count": 2,
                    "dominant_nt": "ACH",
                },
                {
                    "partner_root_id": 33444,
                    "syn_count": 10,
                    "neuropil_count": 1,
                    "dominant_nt": "GABA",
                },
            ],
            "outputs": [
                {
                    "partner_root_id": 44555,
                    "syn_count": 90,
                    "neuropil_count": 3,
                    "dominant_nt": "GABA",
                },
                {
                    "partner_root_id": 55666,
                    "syn_count": 45,
                    "neuropil_count": 1,
                    "dominant_nt": "GABA",
                },
            ],
        }
        with (
            patch.object(flygpt_app, "_require_user", return_value={"id": "test-user"}),
            patch.object(flygpt_app._knowledge, "context_for_query", return_value=(None, [])),
            patch.object(flygpt_app._knowledge, "learn_from_user_text", return_value=[]),
            patch.object(flygpt_app, "get_connectome", return_value=conn),
            patch.object(flygpt_app, "predict_route") as router,
            patch.object(flygpt_app._generator, "generate") as generator,
        ):
            payload = flygpt_app.chat_endpoint(
                flygpt_app.ChatRequest(message=USER_QUESTION),
                Request({"type": "http"}),
            )

        self.assertEqual(payload["type"], "neuron_partner_ranking")
        self.assertEqual(payload["data"]["neuron_id"], 10009)
        self.assertEqual(payload["data"]["metric"], "directed_connects_to_synapse_weight")
        self.assertEqual(
            [(entry["partner_root_id"], entry["direction"], entry["syn_count"])
             for entry in payload["data"]["partners"]],
            [
                (22333, "입력", 120),
                (44555, "출력", 90),
                (55666, "출력", 45),
                (33444, "입력", 10),
            ],
        )
        self.assertIn("22333 → 10009", payload["answer"])
        self.assertIn("10009 → 44555", payload["answer"])
        self.assertNotIn("시냅스 수 상위 뉴런", payload["answer"])
        conn.neuron.assert_called_once_with(10009, direction="both", limit=10)
        conn.top_neurons.assert_not_called()
        conn.top_connections.assert_not_called()
        router.assert_not_called()
        generator.assert_not_called()
        self.assertIn("query_ms", payload["timings"])
        self.assertIn("total_ms", payload["timings"])

    def test_only_requested_input_direction_is_queried(self):
        conn = MagicMock()
        conn.neuron.return_value = {
            "inputs": [{
                "partner_root_id": 22333, "syn_count": 12,
                "neuropil_count": 1, "dominant_nt": "GABA",
            }],
        }
        question = "10009번 뉴런과 가장 강하게 연결된 입력 뉴런 상위 5개"
        with (
            patch.object(flygpt_app, "_require_user", return_value={"id": "test-user"}),
            patch.object(flygpt_app._knowledge, "context_for_query", return_value=(None, [])),
            patch.object(flygpt_app._knowledge, "learn_from_user_text", return_value=[]),
            patch.object(flygpt_app, "get_connectome", return_value=conn),
        ):
            payload = flygpt_app.chat_endpoint(
                flygpt_app.ChatRequest(message=question),
                Request({"type": "http"}),
            )

        self.assertEqual(payload["type"], "neuron_partner_ranking")
        self.assertEqual(payload["data"]["direction"], "inputs")
        conn.neuron.assert_called_once_with(10009, direction="inputs", limit=5)


if __name__ == "__main__":
    unittest.main()
