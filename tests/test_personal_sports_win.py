"""Regression tests for conversational self-reported sports wins."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from dispatcher import dispatch
from generator_runtime import GeneratorRuntime
from semantic_route_guard import apply_semantic_route_override


def predicted_research():
    return {
        "route": "research",
        "confidence": 0.75,
        "margin": 0.51,
        "accepted": True,
        "feature_signal": 1.0,
        "top_routes": [
            {"route": "research", "confidence": 0.75},
            {"route": "general", "confidence": 0.18},
            {"route": "math", "confidence": 0.07},
        ],
        "trace": [
            {"stage": "input", "activations": [0.1, 0.3]},
            {"stage": "step_3", "activations": [1.0, 0.2]},
        ],
    }


def conversational_state(text, route_info=None):
    fixed = apply_semantic_route_override(
        text, route_info if route_info is not None else predicted_research(),
    )
    dispatched = dispatch(text, fixed)
    brain = FlyBrainRuntime()
    state = brain.plan(route_info=fixed, dispatch=dispatched.to_dict(), message=text)
    return fixed, brain.finalize(state)


class PersonalSportsWinTests(unittest.TestCase):
    def test_research_misroute_is_corrected_without_faking_model_confidence(self):
        raw = predicted_research()
        fixed, state = conversational_state("오늘 학교에서 축구 이김", raw)
        self.assertEqual(raw["route"], "research")
        self.assertEqual(fixed["route"], "general")
        self.assertEqual(fixed["confidence"], 0.18)
        self.assertEqual(fixed["margin"], 0.0)
        self.assertEqual(fixed["trace"], raw["trace"])
        self.assertEqual(fixed["semantic_override"]["model_route"], "research")
        self.assertEqual(fixed["semantic_override"]["speech_act"],
                         "acknowledge_personal_sports_win")
        self.assertEqual(state["retrieval"], "none")
        self.assertTrue(state["utterance_plan"]["ready"])

    def test_personal_sports_examples_work_without_a_language_model(self):
        for message in (
            "오늘 학교에서 축구 이김",
            "오늘 학교에서 축구 이겼어!",
            "우리 반 축구 이겼어",
            "나 오늘 축구 이겼어",
            "오늘 야구 이겼어ㅋㅋㅋ",
            "학교에서 반대항전 승리함",
        ):
            with self.subTest(message=message), patch.dict(
                os.environ, {"FLYGPT_GENERATOR_MODE": "hybrid"}, clear=True,
            ):
                fixed, state = conversational_state(message)
                runtime = GeneratorRuntime()
                with patch("generator_runtime.urllib.request.urlopen") as network:
                    output = runtime.generate(message, fixed["route"], brain_state=state)
                    network.assert_not_called()
                self.assertEqual(output.answer, "축하해! 몇 대 몇으로 이겼어?")
                self.assertEqual(output.provider, "papiti-hybrid-native")
                self.assertFalse(state["evidence"]["available"])

    def test_native_mode_handles_personal_win(self):
        text = "오늘 학교에서 축구 이김"
        fixed, state = conversational_state(text)
        with patch.dict(os.environ, {"FLYGPT_GENERATOR_MODE": "native"}, clear=True):
            output = GeneratorRuntime().generate(text, fixed["route"],
                                                 brain_state=state)
        self.assertEqual(output.answer, "축하해! 몇 대 몇으로 이겼어?")
        self.assertEqual(output.provider, "papiti-native")

    def test_explicit_searches_questions_and_external_claims_not_overridden(self):
        raw = predicted_research()
        for text in (
            "오늘 학교에서 축구 이겼어?",
            "오늘 한화가 축구 이김",
            "오늘 축구 경기 결과 알려줘",
            "오늘 축구 승리 원인 분석해줘",
            "오늘 학교에서 축구 이겼어 그리고 경기 기록 검색해줘",
            "오늘 축구에서 누가 이겼어?",
            "왜 하늘은 파란색이야?",
            "오늘 학교에서 축구 졌어",
            "17 * 23",
        ):
            with self.subTest(message=text):
                fixed = apply_semantic_route_override(text, raw)
                self.assertIs(fixed, raw)
                plan = FlyBrainRuntime()._initial_utterance_plan(text, "general")
                self.assertNotEqual(plan["speech_act"],
                                    "acknowledge_personal_sports_win")

    def test_no_intent_overrides_a_ready_general_model_prediction(self):
        raw = predicted_research()
        raw.update(route="general", confidence=0.8, accepted=True)
        fixed = apply_semantic_route_override("오늘 학교에서 축구 이김", raw)
        self.assertIs(fixed, raw)
        self.assertNotIn("semantic_override", fixed)


if __name__ == "__main__":
    unittest.main()
