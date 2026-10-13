"""Check proved linear seating answers, including contradictory constraints."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from starlette.requests import Request

import app as flygpt_app
from seating_solver import solve_linear_seating


PUZZLE = (
    "A, B, C, D, E 5명이 일렬로 서 있습니다. "
    "A는 C의 바로 오른쪽에 있고, B는 양 끝에 서지 않습니다. "
    "D는 E와 이웃해 있으며, E는 A와 두 칸 떨어져 있습니다. "
    "만약 C가 왼쪽에서 두 번째에 있다면, "
    "5명의 정확한 배치 순서는 무엇인가요? 단계별로 추론 과정을 설명해 주세요."
)


class SeatingSolverTests(unittest.TestCase):
    def test_exact_reported_false_answer_is_detected_as_contradictory(self) -> None:
        solved = solve_linear_seating(PUZZLE)
        self.assertIsNotNone(solved)
        self.assertEqual(solved.checked_count, 120)
        self.assertEqual(solved.rule_count, 5)
        self.assertEqual(solved.possible_count, 0)
        self.assertIn("가능한 배치는 없습니다", solved.answer)
        self.assertIn("C는 왼쪽에서 2번째", solved.answer)
        self.assertIn("A는 C의 바로 오른쪽", solved.answer)
        self.assertIn("B는 양 끝이 아님", solved.answer)
        self.assertIn("D와 E는 이웃", solved.answer)
        self.assertIn("남은 배치 0가지", solved.answer)

    def test_three_person_unique_solution(self) -> None:
        text = (
            "A, B, C 3명이 일렬로 서 있습니다. "
            "B가 왼쪽에서 첫 번째에 있습니다. "
            "A는 B의 바로 오른쪽에 있고 "
            "C가 왼쪽에서 세 번째에 있습니다. 정확한 순서는?"
        )
        solved = solve_linear_seating(text)
        self.assertIsNotNone(solved)
        self.assertEqual(solved.checked_count, 6)
        self.assertEqual(solved.possible_count, 1)
        self.assertIn("B → A → C", solved.answer)

    def test_insufficient_information_does_not_fabricate_unique_solution(self) -> None:
        text = (
            "A, B, C, D, E 5명이 일렬로 서 있습니다. "
            "A는 C의 바로 오른쪽에 있습니다. "
            "B는 양 끝에 서지 않습니다. D는 E와 이웃해 있습니다. 순서는?"
        )
        solved = solve_linear_seating(text)
        self.assertIsNotNone(solved)
        self.assertGreater(solved.possible_count, 1)
        self.assertIn("하나로 확정할 수 없습니다", solved.answer)

    def test_unknown_constraints_disable_verified_badge(self) -> None:
        text = PUZZLE + " D는 B보다 왼쪽에 서 있습니다."
        self.assertIsNone(solve_linear_seating(text))

    def test_unknown_puzzle_format_does_not_claim_verified(self) -> None:
        self.assertIsNone(solve_linear_seating("일렬로 서는 법을 알려줘."))
        self.assertIsNone(solve_linear_seating("A, B, C가 일렬로 서 있습니다."))

    def test_partial_recognition_never_proves_answer(self) -> None:
        text = (
            "A, B, C, D, E 5명이 일렬로 서 있습니다. "
            "A는 C의 바로 오른쪽에 있습니다. "
            "B는 양 끝에 서지 않습니다. "
            "D는 E와 이웃해 있으며 D는 B의 왼쪽으로 두 칸 이동합니다."
        )
        self.assertIsNone(solve_linear_seating(text))

    def test_api_uses_verified_result_without_loading_model(self) -> None:
        request = Request({"type": "http"})
        with (
            patch.object(flygpt_app, "_require_user", return_value={"id": "test-user"}),
            patch.object(flygpt_app._knowledge, "context_for_query", return_value=(None, [])),
            patch.object(flygpt_app._knowledge, "learn_from_user_text", return_value=[]),
            patch.object(flygpt_app, "predict_route") as router,
            patch.object(flygpt_app._generator, "generate") as generator,
        ):
            result = flygpt_app.chat_endpoint(
                flygpt_app.ChatRequest(message=PUZZLE), request
            )

        self.assertEqual(result["type"], "logic_verified")
        self.assertIn("가능한 배치는 없습니다", result["answer"])
        self.assertEqual(result["data"]["possible_count"], 0)
        router.assert_not_called()
        generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
