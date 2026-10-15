"""Regression: novel user rules must be executed, not restated."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from starlette.requests import Request
import app as flygpt_app
from novel_rule_solver import solve_defined_operations
from generator_runtime import _is_parroting_answer, GenerationResult


QUESTION = (
    "가상의 세계에서 '모라'는 숫자에 3을 더하는 연산이고, "
    "'테라'는 숫자에 2를 곱하는 연산이야. "
    "4에 모라를 적용한 다음 테라를 적용하면?"
)


class NovelRuleSolverTests(unittest.TestCase):
    def test_screenshot_regression_returns_14(self):
        result = solve_defined_operations(QUESTION)
        self.assertIsNotNone(result)
        self.assertEqual(result.result, "14")
        self.assertEqual([step["name"] for step in result.steps], ["모라", "테라"])
        self.assertIn("4 + 3 = 7", result.answer)
        self.assertIn("7 × 2 = 14", result.answer)

    def test_follow_up_uses_prior_user_rules_not_assistant_inventions(self):
        prior = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": "대답을 되풀이하면 안 돼."},
        ]
        result = solve_defined_operations(
            "이번에는 4에 테라를 먼저 적용하고 모라를 나중에 적용하면?",
            prior,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.result, "11")
        self.assertEqual([step["name"] for step in result.steps], ["테라", "모라"])

    def test_unknown_or_conflicting_rules_never_get_guessed(self):
        self.assertIsNone(solve_defined_operations("4에 모라를 적용하면?"))
        self.assertIsNone(solve_defined_operations(
            QUESTION.replace("테라를 적용하면", "아무거나를 적용하면")
        ))
        self.assertIsNone(solve_defined_operations(
            QUESTION.replace("4에 모라를", "4에 모라를").replace(
                "4에 모라를 적용한", "'모라'는 숫자에 7을 더하는 연산. 4에 모라를 적용한"
            )
        ))

    def test_zero_division_is_explicitly_not_an_answer(self):
        result = solve_defined_operations(
            "'제로'는 숫자에 0을 나누는 연산이야. 7에 제로를 적용하면?"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.result, "undefined")
        self.assertIn("0으로 나누", result.answer)

    def test_ordinary_chat_and_maths_not_hijacked(self):
        for prompt in ("안녕ㅋㅋㅋ", "너 누구야?", "11*11", "모라의 뜻을 설명해줘"):
            with self.subTest(prompt=prompt):
                self.assertIsNone(solve_defined_operations(prompt))

    def test_reject_only_user_answer_echo_not_new_numbers(self):
        bad = "네, 4에 모라를 적용한 다음 테라를 적용하면 됩니다."
        self.assertTrue(_is_parroting_answer(QUESTION, bad))
        self.assertFalse(_is_parroting_answer(QUESTION, "4 + 3 = 7, 7 × 2 = 14. 정답은 14야."))
        self.assertFalse(_is_parroting_answer("안녕", "안녕! 🪰"))
        self.assertFalse(_is_parroting_answer("2 더하기 2", "4"))

    def test_verified_solver_avoids_router_and_generator(self):
        request = Request({"type": "http"})
        with (
            patch.object(flygpt_app, "_require_user", return_value={"id": "test-user"}),
            patch.object(flygpt_app._knowledge, "context_for_query", return_value=(None, [])),
            patch.object(flygpt_app._knowledge, "learn_from_user_text", return_value=[]),
            patch.object(flygpt_app, "predict_route") as router,
            patch.object(flygpt_app._generator, "generate") as generator,
        ):
            result = flygpt_app.chat_endpoint(
                flygpt_app.ChatRequest(message=QUESTION), request,
            )
        self.assertEqual(result["type"], "logic_verified")
        self.assertEqual(result["data"]["solver"], "user_defined_arithmetic_v1")
        self.assertEqual(result["data"]["result"], "14")
        router.assert_not_called()
        generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
