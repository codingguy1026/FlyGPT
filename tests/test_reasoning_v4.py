import unittest
from reasoning_v4 import reason_v4


class ReasoningV4Tests(unittest.TestCase):
    def test_quadratic_two_roots(self):
        r=reason_v4("다음 2차방정식의 모든 실근을 구해. x² - 5x + 6 = 0")
        self.assertIsNotNone(r)
        self.assertEqual(r.evidence["roots"],["2","3"])
        self.assertEqual(r.evidence["discriminant"],1)

    def test_double_root(self):
        r=reason_v4("2차방정식 x² - 4x + 4 = 0의 실근을 구해")
        self.assertEqual(r.evidence["roots"],["2"])

    def test_no_real_root(self):
        r=reason_v4("2차방정식 x² + 4 = 0의 실근을 구해")
        self.assertEqual(r.evidence["roots"],[])
        self.assertLess(r.evidence["discriminant"],0)

    def test_irrational_roots(self):
        r=reason_v4("2차방정식 x² - 2 = 0의 모든 실근")
        self.assertEqual(len(r.evidence["roots"]),2)

    def test_non_quadratic_not_accepted(self):
        for question in ["x + 4 = 0", "2차방정식 x + 4 = 0의 근", "2차방정식 x³ - 4 = 0의 근"]:
            with self.subTest(question=question):
                self.assertIsNone(reason_v4(question))

    def test_syllogism_does_not_invent_reverse_implication(self):
        q="모든 루미는 파랑이다. 일부 파랑은 동그라미이다. 모든 루미가 동그라미라고 결론 내릴 수 있을까?"
        r=reason_v4(q)
        self.assertIsNotNone(r)
        self.assertFalse(r.evidence["entails"])
        self.assertIsNotNone(r.evidence["counterexample"])

    def test_supported_syllogism_entails(self):
        q="모든 루미는 파랑이다. 모든 파랑은 동그라미이다. 모든 루미가 동그라미라고 결론 내릴 수 있을까?"
        r=reason_v4(q)
        self.assertIsNotNone(r)
        self.assertTrue(r.evidence["entails"])


if __name__=="__main__":
    unittest.main()
