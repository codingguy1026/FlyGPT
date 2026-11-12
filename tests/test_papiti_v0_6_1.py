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
from hybrid_speech import check_candidate, generate_speech, supports_v061_plan
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

    def test_historical_story_without_sources_never_accepts_self_intro(self):
        brain = FlyBrainRuntime()
        message = "흥미로운 역사 이야기를 들려줘"
        state = brain.plan(
            route_info={"route": "general", "accepted": True,
                        "confidence": 0.63, "margin": 0.2},
            dispatch={"route": "general"}, message=message,
        )
        state["retrieval"] = "research"
        state = brain.finalize(
            state,
            research_result={"used": False, "sources": []},
        )
        plan = state["utterance_plan"]
        self.assertEqual(plan["speech_act"], "report_no_external_evidence")
        self.assertFalse(supports_v061_plan(plan))
        wrong = (
            "MaleCNS 이 파피티는 한국어 애칭이다. "
            "파피티는 실험 단계의 AI 대화 도우미다. "
            "MaleCNS 기반 작업 라우터가 사용자 질문의 처리 유형을 분류한다."
        )
        self.assertEqual(check_candidate(plan, wrong), "missing_status_semantics")
        self.assertIsNone(check_candidate(plan, "확인된 자료가 없어서 지금은 답을 검증할 수 없어."))
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_SPEAKER": "v0.6.1",
        }, clear=True):
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech") as speaker:
                result = runtime.generate(message, "general", brain_state=state)
        speaker.assert_not_called()
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error_code, "hybrid_speaker_unsupported")

    def test_untrained_sourced_search_cannot_be_sent_to_tiny_speaker(self):
        brain = FlyBrainRuntime()
        state = brain.plan(
            route_info={"route": "general", "accepted": True, "confidence": 0.63},
            dispatch={"route": "general"}, message="흥미로운 역사 이야기를 들려줘",
        )
        state["retrieval"] = "research"
        state = brain.finalize(state, research_result={
            "used": True,
            "sources": [{
                "url": "https://example.org/history",
                "title": "Historical source",
                "snippets": ["역사 기록에서 확인할 수 있는 사건입니다."],
            }],
        })
        plan = state["utterance_plan"]
        self.assertEqual(plan["speech_act"], "report_retrieved_web_evidence")
        self.assertFalse(supports_v061_plan(plan))
        self.assertEqual(
            check_candidate(plan, "파피티는 AI 도우미야."),
            "semantic_content_not_preserved",
        )
        with patch.dict(os.environ, {
            "FLYGPT_GENERATOR_MODE": "hybrid",
            "FLYGPT_HYBRID_SPEAKER": "v0.6.1",
        }, clear=True):
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech") as speaker:
                result = runtime.generate(
                    "흥미로운 역사 이야기를 들려줘", "general", brain_state=state,
                )
        speaker.assert_not_called()
        self.assertEqual(result.error_code, "hybrid_speaker_unsupported")

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

    def test_bad_neural_output_never_falls_back_to_native(self):
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech", return_value="도움이 됐다니 됐다니 줘."):
                result = runtime.generate("안녕", "general", brain_state=plan_for("안녕"))
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.provider, "papiti-hybrid")
        self.assertEqual(result.error_code, "hybrid_neural_rejected")
        self.assertIn("repetitive_speech", result.error)

    def test_unresolved_request_never_promoted_to_neural_thought(self):
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech") as speaker:
                result = runtime.generate("왜 하늘은 파래?", "general",
                                          brain_state=plan_for("왜 하늘은 파래?"))
        speaker.assert_not_called()
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error_code, "hybrid_language_model_required")

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
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error_code, "hybrid_neural_rejected")
        self.assertIn("exact_result_not_preserved", result.error)

    def test_missing_checkpoint_does_not_claim_neural_generation(self):
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=False), \
                 patch("generator_runtime.generate_speech") as speaker:
                result = runtime.generate("안녕", "general", brain_state=plan_for("안녕"))
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error_code, "hybrid_speaker_missing")
        speaker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
