"""Regress the misrouting of '너 누구야?' into MATH.

The MaleCNS model and neuPrint calls are intentionally not mocked away from
production; this test uses a fake raw prediction to isolate the routing bridge.
"""

import unittest
from unittest.mock import Mock, patch

import app as flygpt_app
from brain_runtime import FlyBrainRuntime
from dispatcher import dispatch
from semantic_route_guard import apply_semantic_route_override


def model_prediction(route="math", accepted=True):
    return {
        "model": "fly_router_malecns_v0_4.pt",
        "route": route,
        "confidence": 0.625,
        "margin": 0.41,
        "accepted": accepted,
        "feature_signal": 1.0,
        "top_routes": [
            {"route": route, "confidence": 0.625},
            {"route": "general", "confidence": 0.24},
            {"route": "research", "confidence": 0.135},
        ],
        "trace": [
            {"stage": "input", "activations": [0.1, 0.2, 0.3]},
            {"stage": "step_3", "activations": [0.3, 0.9, 0.4]},
        ],
    }


class SemanticOverrideTests(unittest.TestCase):
    def test_identity_misclassified_as_math_is_dispatched_to_general(self):
        raw = model_prediction()
        fixed = apply_semantic_route_override("너 누구야?", raw)
        self.assertEqual(raw["route"], "math")  # immutable source inference
        self.assertEqual(raw["confidence"], 0.625)
        self.assertEqual(fixed["route"], "general")
        self.assertEqual(fixed["confidence"], 0.24)
        self.assertEqual(fixed["margin"], 0.0)
        self.assertTrue(fixed["accepted"])
        self.assertEqual(fixed["trace"], raw["trace"])
        self.assertEqual(fixed["top_routes"], raw["top_routes"])
        self.assertEqual(fixed["semantic_override"]["model_route"], "math")
        self.assertEqual(fixed["semantic_override"]["model_confidence"], 0.625)
        self.assertEqual(fixed["semantic_override"]["speech_act"], "introduce_flygpt")

        result = dispatch("너 누구야?", fixed)
        self.assertEqual(result.route, "general")
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.handler, "generator")

        state = FlyBrainRuntime().plan(
            route_info=fixed,
            dispatch=result.to_dict(),
            message="너 누구야?",
        )
        self.assertTrue(state["utterance_plan"]["ready"])
        self.assertEqual(state["utterance_plan"]["speech_act"], "introduce_flygpt")
        self.assertEqual(state["neural_signature"][0]["node_index"], 1)

    def test_predict_route_still_invokes_malecns_model_first(self):
        mock_router = Mock()
        mock_router.routes = ["math", "general", "research"]
        mock_router.predict.return_value = model_prediction()
        with patch.object(flygpt_app, "get_router", return_value=mock_router):
            output = flygpt_app.predict_route("너 누구야?")

        mock_router.predict.assert_called_once_with("너 누구야?", top_k=3)
        self.assertEqual(output["route"], "general")
        self.assertEqual(output["semantic_override"]["model_route"], "math")
        self.assertIn("runtime_timings", output)

    def test_only_known_fly_brain_intents_are_overridden(self):
        raw = model_prediction()
        examples = {
            "너 누구야?": "introduce_flygpt",
            "너 뭐 할 수 있어?": "describe_supported_features",
            "MaleCNS가 뭐야?": "explain_malecns",
            "잘 지내?": "acknowledge_presence",
            "도와줘": "request_specific_task",
            "반가워": "return_welcome",
        }
        for message, speech_act in examples.items():
            with self.subTest(message=message):
                fixed = apply_semantic_route_override(message, raw)
                self.assertEqual(fixed["route"], "general")
                self.assertEqual(fixed["semantic_override"]["speech_act"], speech_act)

    def test_actual_math_and_unknown_queries_follow_model(self):
        raw = model_prediction()
        for message in (
            "3x+7=22에서 x는?",
            "17 * 23",
            "왜 하늘은 파래?",
            "뭐야?",
            "너 누구야? 그리고 6*7도 계산해줘",
            "인공지능의 새로운 아키텍처를 작성해 줘",
        ):
            with self.subTest(message=message):
                output = apply_semantic_route_override(message, raw)
                self.assertIs(output, raw)
                self.assertEqual(output["route"], "math")

    def test_existing_accepted_general_is_not_modified(self):
        raw = model_prediction(route="general")
        fixed = apply_semantic_route_override("너 누구야?", raw)
        self.assertIs(fixed, raw)
        self.assertNotIn("semantic_override", fixed)

    def test_rejected_general_is_rescued_without_fabricated_model_confidence(self):
        raw = model_prediction(route="general", accepted=False)
        raw["confidence"] = 0.42
        raw["top_routes"][0]["confidence"] = 0.42
        fixed = apply_semantic_route_override("너 누구야?", raw)
        self.assertTrue(fixed["accepted"])
        self.assertEqual(fixed["confidence"], 0.42)
        self.assertEqual(fixed["semantic_override"]["model_accepted"], False)
        self.assertNotIn("semantic_override", raw)


if __name__ == "__main__":
    unittest.main()
