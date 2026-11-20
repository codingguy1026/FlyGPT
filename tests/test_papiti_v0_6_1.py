"""Papiti 0.6.1 tests: brain-to-mouth separation, no answer lookup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

from brain_runtime import FlyBrainRuntime
from generator_runtime import GeneratorRuntime
from hybrid_speech import check_candidate, generate_speech, reply_key, supports_v061_plan
from hybrid_speech import _request_speech, stop_speech_worker
from papiti_lm.v0_6_1 import _fresh_wording, _load_speaker, _speaker_model
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

    def test_mirror_greeting_preserves_original_form(self):
        plan = plan_for("  안녕 !?  ")["utterance_plan"]
        self.assertEqual(plan["greeting"], "안녕 !?")
        self.assertEqual(packet_text(plan, mirror_greetings=True), "Mirror greeting: 안녕 !?")
        self.assertNotIn("안녕 !?", packet_text(plan))

    def test_greeting_copy_corpus_holds_out_entire_punctuation_patterns(self):
        corpus = create_corpus(diverse_social=True, mirror_greetings=True)
        for rows in (corpus.train, corpus.valid, corpus.test):
            greetings = [row for row in rows if row.group == "mirror-greeting"]
            self.assertTrue(greetings)
            for row in greetings:
                self.assertEqual(row.user, "Mirror greeting: " + row.assistant)
        held_out = packet_text(plan_for("안녕!?")["utterance_plan"], mirror_greetings=True)
        self.assertIn(held_out, {row.user for row in corpus.test})
        self.assertNotIn(held_out, {row.user for row in corpus.train})

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

    def test_social_packet_keeps_multiple_targets_but_math_stays_exact(self):
        corpus = create_corpus(diverse_social=True)
        greeting = packet_text(plan_for("안녕")["utterance_plan"])
        targets = {row.assistant for row in corpus.train if row.user == greeting}
        self.assertGreaterEqual(len(targets), 3)
        for row in corpus.train:
            if row.group == "verified-number":
                packet = json.loads(row.user)
                expected = packet["content"][0]["value"].removeprefix("Exact math tool result: ")
                self.assertEqual(row.assistant, expected)

    def test_varied_fact_training_repeats_the_meaning_in_different_wording(self):
        plan = plan_for("너 누구야?")["utterance_plan"]
        corpus = create_corpus(diverse_social=True, phrasing_cues=True)
        answers = {row.assistant for row in corpus.train if row.user == packet_text(plan)}
        self.assertGreaterEqual(len({reply_key(answer) for answer in answers}), 3)
        self.assertTrue(all(check_candidate(plan, answer) is None for answer in answers))
        self.assertFalse(any(row.group == "mirror-greeting" for row in corpus.train))

    def test_new_wording_skips_cosmetic_changes_and_rejected_candidates(self):
        plan = plan_for("안녕")["utterance_plan"]
        decode = Mock(side_effect=["안녕 !", "반가워~", '{"speech_act":"return_greeting"}', "만나서 반가워."])
        result = _fresh_wording(plan, decode, (reply_key("안녕"), reply_key("반가워!")))
        self.assertEqual(result, "만나서 반가워.")
        self.assertGreater(decode.call_args_list[0].kwargs["temperature"], 0)

    def test_no_valid_new_wording_never_replays_an_old_answer(self):
        plan = plan_for("안녕")["utterance_plan"]
        decode = Mock(return_value="안녕!")
        self.assertEqual(_fresh_wording(plan, decode, (reply_key("안녕"),)), "")
        self.assertLessEqual(decode.call_count, 6)

    def test_social_companion_speaks_greetings_while_math_uses_base(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "trained.pt"
            companion = base.with_stem("trained_social")
            for path in (base, companion):
                path.touch()
            with (
                patch.dict(os.environ, {"FLYGPT_HYBRID_SPEECH_CHECKPOINT": str(base)}),
                patch("hybrid_speech._request_speech", return_value="reply") as worker,
            ):
                generate_speech(plan_for("안녕")["utterance_plan"])
                self.assertEqual(worker.call_args.args[0]["checkpoint"], str(companion.resolve()))
                generate_speech(plan_for("6*7", "math", "Exact math tool result: 42")["utterance_plan"])
                self.assertEqual(worker.call_args.args[0]["checkpoint"], str(base.resolve()))

    def test_math_candidate_never_replaced_by_calculator(self):
        plan = plan_for("6*7", "math", "Exact math tool result: 42")["utterance_plan"]
        self.assertIsNone(check_candidate(plan, "42"))
        self.assertEqual(check_candidate(plan, "13"), "exact_result_not_preserved")

    def test_everyday_dialogue_selects_the_dialogue_checkpoint(self):
        from hybrid_speech import checkpoint_present

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "trained.pt"
            dialogue = base.with_stem("trained_dialogue")
            base.touch()
            plan = plan_for("좀 피곤해요.")["utterance_plan"]
            with patch.dict(os.environ, {"FLYGPT_HYBRID_SPEECH_CHECKPOINT": str(base)}):
                self.assertFalse(checkpoint_present(plan))
                dialogue.touch()
                self.assertTrue(checkpoint_present(plan))
                with patch("hybrid_speech._request_speech", return_value="피곤했겠다.") as worker:
                    self.assertEqual(generate_speech(plan), "피곤했겠다.")
                self.assertEqual(worker.call_args.args[0]["checkpoint"], str(dialogue.resolve()))

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

    def test_bridge_sends_plan_and_digests_without_reply_history(self):
        plan = plan_for("안녕")["utterance_plan"]
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "trained.pt"
            checkpoint.write_bytes(b"fixture")  # subprocess is mocked, not loaded
            with patch.dict(os.environ, {
                "FLYGPT_HYBRID_SPEECH_CHECKPOINT": str(checkpoint),
            }, clear=False), patch(
                "hybrid_speech._request_speech", return_value="안녕!",
            ) as worker:
                answer = generate_speech(plan, avoid_reply_keys=(reply_key("이전 대화의 답변"),))
        self.assertEqual(answer, "안녕!")
        payload = worker.call_args.args[0]
        self.assertEqual(payload["plan"]["speech_act"], "return_greeting")
        self.assertNotIn("이전 대화의 답변", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(payload["avoid_reply_keys"], (reply_key("이전 대화의 답변"),))


class SpeechWorkerLifecycleTests(unittest.TestCase):
    def test_reuses_worker_and_discards_timed_out_reply(self):
        command = [sys.executable, "-c", """
import json, os, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request.get('stall'):
        time.sleep(5)
    print(json.dumps({'answer': f"{os.getpid()}:{request['value']}"}), flush=True)
"""]
        self.addCleanup(stop_speech_worker)
        with patch("hybrid_speech._worker_command", return_value=command) as start:
            first = _request_speech({"value": "first"}, time.monotonic() + 5)
            second = _request_speech({"value": "second"}, time.monotonic() + 5)
            self.assertEqual(first.split(":")[0], second.split(":")[0])
            self.assertTrue(second.endswith(":second"))
            start.assert_called_once()
            with self.assertRaises(TimeoutError):
                _request_speech({"value": "stale", "stall": True}, time.monotonic() + 0.05)
            recovered = _request_speech({"value": "recovered"}, time.monotonic() + 5)
            self.assertTrue(recovered.endswith(":recovered"))
            self.assertEqual(start.call_count, 2)

    def test_reloads_checkpoint_after_atomic_replacement(self):
        self.addCleanup(_load_speaker.cache_clear)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "speaker.pt"
            path.write_bytes(b"first")
            with patch("papiti_lm.model.load_checkpoint", side_effect=Path.read_bytes) as load:
                self.assertEqual(_speaker_model(str(path)), b"first")
                self.assertEqual(_speaker_model(str(path)), b"first")
                load.assert_called_once_with(path.resolve())
                replacement = path.with_suffix(".tmp")
                replacement.write_bytes(b"second")
                replacement.replace(path)
                self.assertEqual(_speaker_model(str(path)), b"second")
                self.assertEqual(load.call_count, 2)


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

    def test_repeated_question_rejects_neural_duplicate(self):
        history = [
            {"role": "user", "content": "안녕!"},
            {"role": "assistant", "content": "안녕! 반가워."},
            {"role": "user", "content": "고마워"},
            {"role": "assistant", "content": "천만에."},
        ]
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech", return_value="안녕, 반가워!") as speaker, \
                 patch("generator_runtime.realize_native") as native:
                result = runtime.generate("안녕", "general", brain_state=plan_for("안녕"),
                                          memory_context=history)
        self.assertEqual(speaker.call_args.kwargs["avoid_reply_keys"], (reply_key("안녕! 반가워."),))
        self.assertEqual(result.error_code, "hybrid_new_wording_unavailable")
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        native.assert_not_called()

    def test_repeated_exact_result_can_still_be_identical(self):
        context = "Exact math tool result: 42"
        with self._runtime():
            runtime = GeneratorRuntime()
            with patch("generator_runtime.speech_checkpoint_present", return_value=True), \
                 patch("generator_runtime.generate_speech", return_value="42") as speaker:
                result = runtime.generate(
                    "6*7", "math", brain_state=plan_for("6*7", "math", context),
                    tool_context=context, memory_context=[
                        {"role": "user", "content": "6*7"},
                        {"role": "assistant", "content": "42"},
                    ],
                )
        self.assertEqual(result.answer, "42")
        self.assertEqual(speaker.call_args.kwargs["avoid_reply_keys"], ())

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
