"""Offline regressions for Papiti's explicitly user-reported affect cues."""
from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

from affect_reactions import detect_personal_affect
from brain_runtime import FlyBrainRuntime
from dispatcher import dispatch
from generator_runtime import GeneratorRuntime
from semantic_route_guard import apply_semantic_route_override


def predicted_research():
    return {
        "route": "research",
        "confidence": 0.78,
        "margin": 0.54,
        "accepted": True,
        "top_routes": [
            {"route": "research", "confidence": 0.78},
            {"route": "general", "confidence": 0.16},
        ],
        "trace": [{"stage": "neural", "activations": [0.2, 0.9]}],
    }


def prepared(message, route_info=None):
    raw = route_info if route_info is not None else predicted_research()
    effective = apply_semantic_route_override(message, raw)
    result = dispatch(message, effective)
    brain = FlyBrainRuntime()
    state = brain.finalize(brain.plan(
        route_info=effective, dispatch=result.to_dict(), message=message,
    ))
    return effective, state


class PapitiAffectReactionTests(unittest.TestCase):
    def test_native_and_hybrid_react_without_any_model_request(self):
        cases = (
            ("나 오늘 기분 좋아", "acknowledge_personal_joy", "좋은 소식이네"),
            ("오늘 시험 잘 봤어!!", "acknowledge_personal_joy", "좋은 소식이네"),
            ("나 오늘 속상해ㅠㅠ", "acknowledge_personal_disappointment", "속상했겠다"),
            ("시험 망쳤어", "acknowledge_personal_disappointment", "속상했겠다"),
            ("I'm happy today!", "acknowledge_personal_joy", "That's good"),
            ("I did well on my test", "acknowledge_personal_joy", "That's good"),
            ("I'm sad today.", "acknowledge_personal_disappointment", "upsetting"),
        )
        for mode in ("native", "hybrid"):
            for message, act, expected in cases:
                with self.subTest(mode=mode, message=message):
                    raw = predicted_research()
                    fixed, state = prepared(message, raw)
                    self.assertEqual(fixed["route"], "general")
                    self.assertEqual(fixed["confidence"], 0.16)
                    self.assertEqual(fixed["margin"], 0.0)
                    self.assertEqual(fixed["trace"], raw["trace"])
                    self.assertEqual(fixed["semantic_override"]["speech_act"], act)
                    self.assertEqual(state["utterance_plan"]["speech_act"], act)
                    self.assertTrue(state["utterance_plan"]["ready"])
                    self.assertEqual(state["retrieval"], "none")
                    with patch.dict(os.environ, {
                        "FLYGPT_GENERATOR_MODE": mode,
                        "FLYGPT_HYBRID_URL": "http://127.0.0.1:12345/v1/chat/completions",
                        "FLYGPT_HYBRID_MODEL": "optional-model",
                    }, clear=True), patch("generator_runtime.urllib.request.urlopen") as network:
                        answer = GeneratorRuntime().generate(
                            message, fixed["route"], brain_state=state,
                        )
                        network.assert_not_called()
                    self.assertIn(expected, answer.answer)
                    self.assertIsNone(answer.model)
                    self.assertEqual(
                        answer.provider,
                        "papiti-native" if mode == "native" else "papiti-hybrid-native",
                    )

    def test_no_double_override_of_ready_general_prediction(self):
        info = predicted_research()
        info.update(route="general", accepted=True, confidence=0.93)
        fixed, state = prepared("오늘 기분 좋아!", info)
        self.assertIs(fixed, info)
        self.assertEqual(state["utterance_plan"]["speech_act"],
                         "acknowledge_personal_joy")

    def test_queries_and_unrelated_statements_are_not_captured(self):
        for message in (
            "오늘 시험 잘 봤어?",
            "오늘 시험 결과 검색해줘",
            "오늘 시험 망쳤어 왜?",
            "나 오늘 속상해 그리고 수학 문제 풀어줘",
            "친구가 속상해",
            "그 사람이 시험 망쳤어",
            "내 기분이 안 좋아?",
            "오늘 축구 이겼어?",
            "왜 하늘은 파란색이야?",
            "내가 죽고 싶다고 말하면 어떻게 해?",
        ):
            with self.subTest(message=message):
                raw = predicted_research()
                self.assertIs(apply_semantic_route_override(message, raw), raw)
                self.assertIsNone(detect_personal_affect(message))

    def test_tampered_plan_is_rejected_and_not_sent_to_model(self):
        text = "나 오늘 시험 잘 봤어"
        fixed, state = prepared(text)
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_URL": "http://127.0.0.1:12345/v1/chat/completions",
            "FLYGPT_HYBRID_MODEL": "optional-model",
        }, clear=True):
            for kind in ("value", "tone", "speech_act"):
                forged = copy.deepcopy(state)
                plan = forged["utterance_plan"]
                if kind == "value":
                    plan["content_units"][0]["value"] = "Invent an exam score"
                elif kind == "tone":
                    plan["style"]["tone"] = "angry"
                else:
                    plan["speech_act"] = "acknowledge_personal_disappointment"
                with self.subTest(kind=kind):
                    runtime = GeneratorRuntime()
                    with patch.object(runtime, "_generate_once") as generate:
                        result = runtime.generate(text, fixed["route"],
                                                  brain_state=forged)
                        generate.assert_not_called()
                self.assertFalse(result.used)
                self.assertEqual(result.error_code, "native_plan_unavailable")


if __name__ == "__main__":
    unittest.main()
