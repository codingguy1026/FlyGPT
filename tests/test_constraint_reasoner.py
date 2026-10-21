import unittest
from constraint_reasoner import solve_integer_constraints as solve


class ConstraintReasonerTests(unittest.TestCase):
    def test_unique_solution_with_offset(self):
        r = solve("A와 B는 0부터 10까지의 정수다. A는 B보다 2 크다. B는 3이다. A는 얼마?")
        self.assertIsNotNone(r)
        self.assertEqual(len(r.solutions),1)
        self.assertEqual(r.solutions[0],{"A":5,"B":3})
        self.assertEqual(r.checked_count,121)

    def test_order_matters_no_premature_uniqueness(self):
        r = solve("A와 B는 0부터 4까지의 정수다. A는 B보다 크다. A는 얼마?")
        self.assertIsNotNone(r)
        self.assertEqual({v["A"] for v in r.solutions},{1,2,3,4})

    def test_contradiction(self):
        r=solve("A와 B는 0부터 5까지의 정수다. A는 B보다 2 크다. A는 B보다 작다. A는 얼마?")
        self.assertIsNotNone(r)
        self.assertEqual(r.solutions,())

    def test_reject_unknown_or_unbounded(self):
        for text in (
            "A는 B보다 2 크다. B는 3이다. A는 얼마?",
            "A와 B는 0부터 10까지의 정수다. A는 B보다 2 크다. 사실 A는 짝수다. B는 3이다. A는 얼마?",
            "A와 B는 0부터 500까지의 정수다. A는 B보다 크다. A는 얼마?",
            "A와 B는 0부터 10까지의 정수다. A는 B보다 클 수도 있다. A는 얼마?",
        ):
            with self.subTest(text=text):
                self.assertIsNone(solve(text))


if __name__=="__main__":
    unittest.main()
