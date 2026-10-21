"""Regression and mutation tests for symbolic speaker reasoning."""
import unittest
from symbolic_reasoner import solve_truth_teller_puzzle

ORIGINAL = (
    '어느 섬에 세 사람이 살고 있다. A는 항상 진실만 말하고, '
    'B는 항상 거짓말만 한다. C는 진실과 거짓을 자유롭게 말한다. '
    '세 사람 중 한 명이 이렇게 말했다. '
    '"나는 A가 아니다. 그리고 B도 아니다." '
    '이 말을 한 사람이 누구인지 반드시 알아낼 수 있을까? '
    '가능하다면 누구인지, 불가능하다면 이유를 설명해 줘.'
)
VARIANT = (
    "A는 항상 진실을 말하고 B는 항상 거짓말을 하며 C는 자유롭게 말한다. "
    "누군가 '나는 B가 아니다'라고 말했다. "
    "발언자를 확정할 수 있을까? 각 경우를 검토해 줘."
)


class SymbolicReasonerTests(unittest.TestCase):
    def test_original_puzzle(self):
        proof = solve_truth_teller_puzzle(ORIGINAL)
        self.assertIsNotNone(proof)
        self.assertEqual(proof.possible_speakers, ("B", "C"))
        self.assertEqual(proof.truth_values, {"A": False, "B": False, "C": True})

    def test_unseen_utterance_can_have_three_possible_speakers(self):
        proof = solve_truth_teller_puzzle(VARIANT)
        self.assertIsNotNone(proof)
        self.assertEqual(proof.possible_speakers, ("A", "B", "C"))
        self.assertEqual(proof.truth_values, {"A": True, "B": False, "C": True})

    def test_mutations_change_proof_not_just_wording(self):
        p = solve_truth_teller_puzzle(ORIGINAL.replace("그리고 B도 아니다", "그리고 C도 아니다"))
        self.assertIsNotNone(p)
        self.assertEqual(p.possible_speakers, ("C",))
        self.assertEqual(p.truth_values, {"A": False, "B": True, "C": False})
        p = solve_truth_teller_puzzle(VARIANT.replace("나는 B가 아니다", "나는 B가 맞아"))
        self.assertIsNotNone(p)
        self.assertEqual(p.possible_speakers, ("C",))

    def test_fail_closed(self):
        variants = (
            ORIGINAL.replace("항상 거짓말만 한다", "항상 진실만 말한다"),
            ORIGINAL.replace('한 명이 이렇게 말했다', '다른 사람이 이렇게 말했다'),
            "어제 네가 말한 A와 B에 대한 기억을 찾아줘",
            "거짓말쟁이와 진실쟁이 게임을 설명해 줘",
            VARIANT.replace("'나는 B가 아니다'", "'만약 내가 B가 아니라면'"),
        )
        for case in variants:
            with self.subTest(case=case):
                self.assertIsNone(solve_truth_teller_puzzle(case))


if __name__ == "__main__":
    unittest.main()
