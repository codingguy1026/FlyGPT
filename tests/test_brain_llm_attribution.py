"""Unit tests for honest attribution accounting (no Ollama or GitHub tokens)."""
from __future__ import annotations
import unittest

from training.brain_llm_attribution import run_arm_brain_only, summarize


class AttributionAuditTests(unittest.TestCase):
    def test_unresolved_brain_does_not_invent_an_answer(self):
        o = run_arm_brain_only(
            {"utterance_plan": {"ready": False}},
            {"status": "ready", "answer": ""},
        )
        self.assertIsNone(o["answer"])
        self.assertFalse(o["llm_called"])
        self.assertFalse(o["solved_open_ended_question"])
        self.assertEqual(o["answer_origin"], "unresolved_no_language_model")

    def test_exact_tool_answer_is_not_credited_to_fly_reasoning(self):
        o = run_arm_brain_only(
            {"utterance_plan": {
                "ready": True, "speech_act": "state_exact_math_result",
                "content_units": [{"kind": "exact_result",
                                   "value": "Exact math tool result: 12"}],
            }},
            {"status": "ready", "answer": ""},
        )
        self.assertEqual(o["answer_origin"], "exact_tool")
        self.assertIn("12", o["answer"])
        self.assertFalse(o["solved_open_ended_question"])

    def test_gate_failure_keeps_language_model_closed(self):
        o = run_arm_brain_only(
            {"utterance_plan": {"ready": True, "content_units": []}},
            {"status": "uncertain", "answer": "need clarification"},
        )
        self.assertEqual(o["answer_origin"], "router_gate")
        self.assertFalse(o["llm_called"])

    def test_summary_separates_math_markers_from_generation_quality(self):
        o = summarize([{
            "route_match": True, "router_accepted": True,
            "bounded_task_delegated_to_llm": True,
            "same_original_user_question_passed_to_combined_llm": True,
            "brain_only": {
                "answer_origin": "unresolved_no_language_model",
                "solved_open_ended_question": False,
            },
            "llm_only": {"answer": "some real output"},
            "brain_plus_llm": {"answer": "different real output"},
            "dispatch_status": "ready",
            "expected_math_marker": None,
            "math_marker_observed": None,
        }])
        self.assertEqual(o["delegated_bounded_task"], 1)
        self.assertEqual(o["brain_only_open_ended_solved"], 0)
        self.assertEqual(o["llm_only_successful_response"], 1)
        self.assertEqual(o["combined_successful_response"], 1)
        self.assertEqual(o["math_marker_checks"], 0)


if __name__ == "__main__":
    unittest.main()
