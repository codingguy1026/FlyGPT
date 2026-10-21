import unittest
from chain_reasoner import solve_chain


class ChainReasonerTests(unittest.TestCase):
    def test_three_steps(self):
        r = solve_chain("A이면 B이다. B이면 C이다. C이면 D이다. A는 참이다. D는 참인가?")
        self.assertIsNotNone(r)
        self.assertTrue(r.evidence["entailed"])
        self.assertEqual([p["fact"] for p in r.evidence["proof_steps"]],["A","B","C","D"])

    def test_out_of_order_rules(self):
        r=solve_chain("C이면 D이다. B이면 C이다. A이면 B이다. A는 참이다. D는 참인가?")
        self.assertTrue(r.evidence["entailed"])
        self.assertIn("D",r.evidence["derived_facts"])

    def test_no_reverse_implication(self):
        r=solve_chain("A이면 B이다. B이면 C이다. C는 참이다. A는 참인가?")
        self.assertFalse(r.evidence["entailed"])
        self.assertNotIn("A",r.evidence["derived_facts"])

    def test_missing_fact_not_false(self):
        r=solve_chain("A이면 B이다. B이면 C이다. A는 참이다. D는 참인가?")
        self.assertFalse(r.evidence["entailed"])
        self.assertIn("참이 아니라는 뜻은 아니야",r.answer)

    def test_cycle_halts(self):
        r=solve_chain("A이면 B이다. B이면 A이다. A는 참이다. B는 참인가?")
        self.assertTrue(r.evidence["entailed"])

    def test_and_missing_premise_is_unknown(self):
        q = "A와 B가 모두 참이면 C가 참이다. A는 참이다. B는 참인지 모른다. C가 참이라고 확정할 수 있을까? 이유를 설명해 줘."
        result = solve_chain(q)
        self.assertIsNotNone(result)
        self.assertFalse(result.evidence["entailed"])
        self.assertEqual(result.evidence["unknown_facts"], ["B"])

    def test_and_both_premises_true(self):
        result = solve_chain("A와 B가 모두 참이면 C가 참이다. A는 참이다. B는 참이다. C는 참인가?")
        self.assertTrue(result.evidence["entailed"])

    def test_neural_ablation_changes_rule_order_not_truth(self):
        q = "A와 B가 모두 참이면 C가 참이다. A이면 B이다. A는 참이다. C는 참인가?"
        baseline = solve_chain(q)
        neural = solve_chain(q, neural_route={
            "accepted": True, "model": "synthetic",
            "top_routes": [{"route": "math", "confidence": .8}, {"route": "general", "confidence": .2}],
        })
        self.assertEqual(neural.evidence["entailed"], baseline.evidence["entailed"])
        self.assertTrue(neural.evidence["entailed"])
        self.assertNotEqual(neural.evidence["neural_selection"]["rule_order"],
                            baseline.evidence["neural_selection"]["rule_order"])

    def test_rejected_signal_falls_back(self):
        q = "A와 B가 모두 참이면 C가 참이다. A이면 B이다. A는 참이다. C는 참인가?"
        result = solve_chain(q, neural_route={"accepted": False,
            "top_routes": [{"route": "math", "confidence": .99}]})
        self.assertEqual(result.evidence["neural_selection"]["mode"], "baseline")

    def test_fail_closed(self):
        for q in [
            "A이면 B이다. B이면 C이다. A는 거짓이다. C는 참인가?",
            "A이면 B이다. B일 수도 있다. A는 참이다. B는 참인가?",
            "A이면 B이다. B이면 C이다. A는 참이다. C는 참인가? 추가 조건은 D다",
            "A이면 B이다. B이면 C이다. A는 참이다. 왜?",
        ]:
            with self.subTest(q=q):
                self.assertIsNone(solve_chain(q))


if __name__=="__main__":
    unittest.main()
