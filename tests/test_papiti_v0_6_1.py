"""Papiti 0.6.1 tests: brain-to-mouth separation, no answer lookup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from brain_runtime import FlyBrainRuntime
from generator_runtime import GeneratorRuntime
from hybrid_speech import check_candidate, generate_speech
from papiti_lm.v0_6_1_data import create_corpus, packet_text


def plan_for(message: str, route: str = "general", tool_context: str | None = None):
    brain = FlyBrainRuntime()
    state = brain.plan(
        route_info={"route": route, "accepted": True, "confidence": 0.98, "margin": 0.8},
        dispatch={"route": route}, message=message,
    )
    return brain.finalize(state, tool_context=tool_context)


class SemanticContractTests(unittest.TestCase):
    def test_social_packet_is_meaning_from_existing_brain(self):
        plan = plan_for("안녕")["utterance_plan"]
        data = json.loads(packet_text(plan))
        self.assertEqual(data["act"], "return_greeting")
        self.assertIn("Return the user's greeting", data["content"][0]["value"])
        with self.assertRaises(ValueError):
            packet_text({"contract": "mouth_only_v1", "ready": False})

    def test_training_test_split_never_shares_semantic_packet(self):
        first = create_corpus(seed=42)
        second = create_corpus(seed=42)
        self.assertEqual(first, second)
        self.assertGreater(len(first.train), 100)
        self.assertGreater(len(first.test), 10)
        train = {x.user for x in first.train}
        valid = {x.user for x in first.valid}
        test = {x.user for x in first.test}
        self.assertFalse(train & valid)
        self.assertFalse(train & test)
        self.assertFalse(valid & test)

    def test_math_candidate_never_replaced_by_calculator(self):
        plan = plan_for("6*7", "math", "Exact math tool result: 42")["utterance_plan"]
        self.assertIsNone(check_candidate(plan, "42"))
        self.assertEqual(check_candidate(plan, "13"), "exact_result_not_preserved")

    def test_meaning_and_repetition_guard(self):
        plan = plan_for("안녕")["utterance_plan"]
        self.assertIsNone(check_candidate(plan, "안녕! 반가워."))
        self.assertEqual(
            check_candidate(plan, "안녕! 도움이 됐다니 됐다니 줘."),
            "repetitive_speech",
        )
        fact = plan_for("너 누구야?")["utterance_plan"]
        self.assertEqual(
            check_candidate(fact, "아무튼 안녕."),
            "semantic_content_not_preserved",
        )

    def test_bridge_uses_only_local_python_no_http(self):
        plan = plan_for("안녕")["utterance_plan"]
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "trained.pt"
            checkpoint.write_bytes(b"fixture")  # subprocess is mocked, not loaded
            with patch.dict(os.environ, {
                "FLYGPT_HYBRID_SPEECH_CHECKPOINT": str(checkpoint),
                "FLYGPT_HYBRID_SPEECH_PYTHON": "python-fixture",
            }, clear=False), patch(
                "hybrid_speech.subprocess.run",
                return_value=Mock(returncode=0, stdout="안녕!\n", stderr=""),
            ) as proc:
                answer = generate_speech(plan)
        self.assertEqual(answer, "안녕!")
        command = proc.call_args.args[0]
        self.assertEqual(command[:3], ["python-fixture", "-m", "papiti_lm.v0_6_1"])
        self.assertEqual(json.loads(proc.call_args.kwargs["input"])["speech_act"],
                         "return_greeting")


class HybridIntegrationTests(unittest.TestCase):
    def _runtime(self):
        return patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_SPEAKER": "v0.6.1",
        }, clear=True)

    def test_ready_social_uses_genuine_neural_speaker_when_available(self):
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech", return_value="안녕! 반가워.") as speaker, \
                 patch.object(runtime, "_generate_once") as http:
                result = runtime.generate("안녕", "general", brain_state=plan_for("안녕"))
        self.assertEqual(result.provider, "papiti-hybrid-v0.6.1")
        self.assertEqual(result.model, "papiti-language-v0.6.1")
        self.assertEqual(result.finish_reason, "neural_semantic_realization")
        speaker.assert_called_once()
        http.assert_not_called()

    def test_bad_neural_output_falls_back_with_honest_label(self):
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech", return_value="도움이 됐다니 됐다니 줘."):
                result = runtime.generate("안녕", "general", brain_state=plan_for("안녕"))
        self.assertEqual(result.provider, "papiti-hybrid-native")
        self.assertEqual(result.finish_reason, "native_fallback_v0.6.1")
        self.assertIn("repetitive_speech", result.error)
        self.assertIn("안녕", result.answer)

    def test_unresolved_request_never_promoted_to_neural_thought(self):
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech") as speaker:
                result = runtime.generate("왜 하늘은 파래?", "general",
                                          brain_state=plan_for("왜 하늘은 파래?"))
        speaker.assert_not_called()
        self.assertEqual(result.provider, "papiti-hybrid-native")

    def test_verified_math_fails_closed_if_neural_changes_result(self):
        context = "Exact math tool result: 42"
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech", return_value="17"):
                result = runtime.generate(
                    "6*7", "math",
                    brain_state=plan_for("6*7", "math", context),
                    tool_context=context,
                )
        self.assertEqual(result.provider, "papiti-hybrid-native")
        self.assertEqual(result.answer, "🧮 42")
        self.assertIn("exact_result_not_preserved", result.error)

    def test_missing_checkpoint_does_not_claim_neural_generation(self):
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=False), \
                 patch("generator_runtime.generate_speech") as speaker:
                result = runtime.generate("안녕", "general", brain_state=plan_for("안녕"))
        self.assertEqual(result.provider, "papiti-hybrid-native")
        speaker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
