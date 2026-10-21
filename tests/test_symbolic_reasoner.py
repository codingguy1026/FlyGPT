"""Symbolic proof must enumerate admissible cases, not recall a memorized answer."""
import unittest
from symbolic_reasoner import solve_truth_teller_puzzle

PUZZLE = (
    '어느 섬에 세 사람이 살고 있다. A는 항상 진실만 말하고, '
    'B는 항상 거짓말만 한다. C는 진실과 거짓을 자유롭게 말한다. '
    '세 사람 중 한 명이 이렇게 말했다. '
    '"나는 A가 아니다. 그리고 B도 아니다." '
    '이 말을 한 사람이 누구인지 반드시 알아낼 수 있을까? '
    '가능하다면 누구인지, 불가능하다면 이유를 설명해 줘.'
)


class SymbolicReasonerTests(unittest.TestCase):
    def test_enumerates_cases_with_no_false_unique_claim(self):
        proof = solve_truth_teller_puzzle(PUZZLE)
        self.assertIsNotNone(proof)
        self.assertEqual(proof.possible_speakers, ("B", "C"))
        self.assertEqual(proof.truth_values, {"A": False, "B": False, "C": True})
        self.assertEqual(proof.speaker_cases, {"A": False, "B": True, "C": True})
        self.assertIn("한 명으로 확정할 수 없어", proof.explanation)

    def test_refuses_changed_or_unrelated_puzzle(self):
        variants = (
            PUZZLE.replace("항상 거짓말만 한다", "항상 진실만 말한다"),
            PUZZLE.replace("그리고 B도 아니다", "그리고 C도 아니다"),
            PUZZLE.replace('한 명이 이렇게 말했다', '다른 사람이 이렇게 말했다'),
            '어제 네가 말한 A와 B에 대한 기억을 찾아줘',
            '거짓말쟁이와 진실쟁이 게임을 설명해 줘',
        )
        for case in variants:
            with self.subTest(case=case):
                self.assertIsNone(solve_truth_teller_puzzle(case))


if __name__ == "__main__":
    unittest.main()
