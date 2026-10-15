"""Confirm the tiny language model cannot pass a question echo as a solution."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from generator_runtime import GeneratorRuntime, GenerationResult


QUESTION = (
    "가상의 세계에서 모라는 숫자에 3을 더하는 연산이고 "
    "테라는 숫자에 2를 곱하는 연산이야. "
    "4에 모라를 적용한 다음 테라를 적용하면?"
)
PARROT = "네, 4에 모라를 적용한 다음 테라를 적용하면 됩니다."


def make_result(answer: str) -> GenerationResult:
    return GenerationResult(
        used=True, provider="ollama", model="gemma3:270m",
        answer=answer, latency_ms=3,
    )


def make_runtime() -> GeneratorRuntime:
    with patch.dict(os.environ, {
        "FLYGPT_LOCAL_ONLY": "1",
        "FLYGPT_GENERATOR_MODEL": "gemma3:270m",
        "FLYGPT_GENERATOR_URL": "http://127.0.0.1:11434/v1/chat/completions",
    }, clear=True):
        return GeneratorRuntime()


def make_state() -> dict:
    brain = FlyBrainRuntime()
    raw = brain.plan(
        route_info={
            "route": "general", "confidence": 0.95,
            "margin": 0.6, "accepted": True,
        },
        dispatch={"route": "general", "status": "ready"},
        message=QUESTION,
    )
    return brain.finalize(raw)


class ParrotGeneratorTests(unittest.TestCase):
    def test_echo_is_not_accepted_when_retry_echoes_again(self):
        runtime = make_runtime()
        with patch.object(runtime, "_generate_once", side_effect=[
            make_result(PARROT), make_result(PARROT),
        ]) as invoke:
            result = runtime.generate(QUESTION, "general", brain_state=make_state())
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error_code, "non_answer_echo")
        self.assertEqual(invoke.call_count, 2)
        self.assertIn("previous completion", str(invoke.call_args_list[1]).lower())

    def test_retry_can_supply_a_real_answer(self):
        runtime = make_runtime()
        with patch.object(runtime, "_generate_once", side_effect=[
            make_result(PARROT), make_result("4+3=7, 이어서 7×2=14이므로 정답은 14야."),
        ]) as invoke:
            result = runtime.generate(QUESTION, "general", brain_state=make_state())
        self.assertTrue(result.used)
        self.assertIn("14", result.answer)
        self.assertEqual(invoke.call_count, 2)


class RepeatedPromptTests(unittest.TestCase):
    def test_repeated_general_prompt_is_reworded_once(self):
        runtime = make_runtime()
        history = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "정답은 14입니다."},
        ]
        with patch.object(runtime, "_generate_once", side_effect=[
            make_result("정답은 14입니다."),
            make_result("모라부터 적용하면 4+3=7, 테라를 적용하면 7×2=14야."),
        ]) as call:
            result = runtime.generate(
                QUESTION, "general", brain_state=make_state(),
                memory_context=history,
            )
        self.assertEqual(call.call_count, 2)
        self.assertTrue(result.used)
        self.assertIn("14", result.answer)
        self.assertIn("temperature_override", call.call_args_list[1].kwargs)
        self.assertIn("identical", str(call.call_args_list[1]).lower())

    def test_same_prompt_with_changed_punctuation_does_not_mutate_facts(self):
        from generator_runtime import _same_surface_answer
        self.assertTrue(_same_surface_answer("정답은 14입니다.", "정답은 14입니다!"))
        self.assertFalse(_same_surface_answer("정답은 14", "정답은 11"))

    def test_new_prompt_never_fires_diversity_retry(self):
        runtime = make_runtime()
        history = [
            {"role": "user", "content": "전혀 다른 질문"},
            {"role": "assistant", "content": "정답은 14입니다."},
        ]
        with patch.object(
            runtime, "_generate_once", return_value=make_result("정답은 14입니다.")
        ) as call:
            result = runtime.generate(
                QUESTION, "general", brain_state=make_state(),
                memory_context=history,
            )
        self.assertTrue(result.used)
        call.assert_called_once()

    def test_grounded_identity_reply_is_not_diversity_retried(self):
        runtime = make_runtime()
        message = "너 누구야?"
        brain = FlyBrainRuntime()
        state = brain.finalize(brain.plan(
            route_info={"route": "general", "confidence": 0.99,
                        "margin": 0.8, "accepted": True},
            dispatch={"route": "general", "status": "ready"},
            message=message,
        ))
        answer = "나는 FlyGPT 파피티야. 초파리 커넥톰을 이용해 질문 종류를 구분해."
        history = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        with patch.object(
            runtime, "_generate_once", return_value=make_result(answer)
        ) as call:
            result = runtime.generate(
                message, "general", brain_state=state,
                memory_context=history,
            )
        self.assertTrue(result.used)
        call.assert_called_once()

    def test_repeat_retry_must_not_replace_good_answer_with_error(self):
        runtime = make_runtime()
        history = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "정답은 14입니다."},
        ]
        failed = GenerationResult(
            used=False, provider="ollama", model="gemma3:270m",
            answer=None, error="temporarily unavailable",
        )
        with patch.object(runtime, "_generate_once", side_effect=[
            make_result("정답은 14입니다."), failed,
        ]):
            result = runtime.generate(
                QUESTION, "general", brain_state=make_state(),
                memory_context=history,
            )
        self.assertTrue(result.used)
        self.assertEqual(result.answer, "정답은 14입니다.")


if __name__ == "__main__":
    unittest.main()
