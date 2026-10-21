import unittest
from chain_reasoner import solve_chain
from scripts.benchmark_neural_rule_scheduler import cases, measure, summarize


class RuleSchedulerBenchmarkTests(unittest.TestCase):
    def test_100_deterministic_cases_without_real_checkpoint(self):
        examples = cases()
        self.assertEqual(len(examples), 100)
        for query, expected in examples:
            solved = solve_chain(query)
            self.assertIsNotNone(solved)
            self.assertEqual(solved.evidence["entailed"], expected)

    def test_summary_and_ablation_with_stubbed_signals(self):
        class FakeRouter:
            routes = ["math", "general"]
            def predict(self, _query, top_k=None):
                return {"accepted": True, "model": "FAKE_ONLY",
                        "top_routes": [{"route": "math", "confidence": .8},
                                       {"route": "general", "confidence": .2}]}
        report = summarize(measure(FakeRouter(), cases(100)))
        self.assertEqual(report["baseline_accuracy"], 1.0)
        self.assertEqual(report["neural_accuracy"], 1.0)
        self.assertEqual(report["same_conclusions"], 100)
        self.assertEqual(report["rule_order_changed"], 100)
