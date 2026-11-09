"""Regression audit: MaleCNS route decisions must influence answer planning.

This checks the deterministic brain boundary, not the fluency of Ollama.
Run: python -m unittest tests.test_malecns_answer_influence -v
"""
from __future__ import annotations

import unittest

from brain_runtime import FlyBrainRuntime


def brain_state(message: str, route: str) -> dict:
    return FlyBrainRuntime().plan(
        message=message,
        route_info={
            "route": route,
            "confidence": 0.95,
            "margin": 0.40,
            "accepted": True,
            "trace": [{"stage": "step_3", "activations": [0.1, 0.9]}],
        },
        dispatch={"route": route, "status": "ready"},
    )


class MaleCNSAnswerInfluenceAudit(unittest.TestCase):
    def test_route_changes_response_objective(self):
        message = "파피티야, 너 누구야?"
        general = brain_state(message, "general")
        math = brain_state(message, "math")
        self.assertNotEqual(general["objective"], math["objective"])
        self.assertNotEqual(general["response_mode"], math["response_mode"])

    def test_only_appropriate_route_can_use_grounded_identity_plan(self):
        message = "파피티야, 너 누구야?"
        general = brain_state(message, "general")
        math = brain_state(message, "math")
        self.assertTrue(general["utterance_plan"]["ready"])
        self.assertFalse(math["utterance_plan"]["ready"])

    def test_different_intents_produce_distinct_content_units(self):
        name = brain_state("너 이름이 뭐야?", "general")["utterance_plan"]
        brain = brain_state("네 뇌가 뭐야?", "general")["utterance_plan"]
        self.assertTrue(name["ready"])
        self.assertTrue(brain["ready"])
        self.assertNotEqual(
            [item["fact_id"] for item in name["content_units"]],
            [item["fact_id"] for item in brain["content_units"]],
        )


if __name__ == "__main__":
    unittest.main()
