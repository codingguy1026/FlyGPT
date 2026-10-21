import unittest
from scripts.train_malecns_rule_head import make_examples
from chain_reasoner import solve_chain


class MaleCNSRuleHeadDatasetTests(unittest.TestCase):
    def test_balanced_and_reproducible(self):
        dataset = make_examples(100, 92371)
        self.assertEqual(dataset, make_examples(100, 92371))
        self.assertEqual(sum(label for _, label in dataset), 50)

    def test_all_questions_provable(self):
        for text, label in make_examples(100, 92371):
            with self.subTest(text=text):
                result = solve_chain(text)
                self.assertIsNotNone(result)
                self.assertTrue(result.evidence["entailed"])
