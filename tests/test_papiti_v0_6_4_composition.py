"""Guardrails for v0.6.4 clause-factorized language training."""
from __future__ import annotations

import unittest

from papiti_lm.compositional_curriculum import clause_lessons
from papiti_lm.corpus import DigitSubwordTokenizer, EOS
from papiti_lm.inquiry_curriculum import add_inquiry_training
from papiti_lm.language_curriculum import mixed_style_transfer_probes
from papiti_lm.v0_6_1_prose import create_prose_corpus


class FactorizedLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = create_prose_corpus(repeats=2)
        cls.corpus = add_inquiry_training(cls.base)
        cls.tokenizer = DigitSubwordTokenizer.from_examples(
            cls.corpus.train, max_merges=256
        )
        cls.lessons = clause_lessons(cls.corpus, cls.tokenizer, 448)

    def test_frozen_splits_untouched(self):
        self.assertEqual(94, len(self.corpus.test))
        self.assertEqual(self.base.test, self.corpus.test)
        self.assertEqual(self.base.valid, self.corpus.valid)

    def test_both_clauses_have_independent_targets(self):
        self.assertGreater(len(self.lessons), 0)
        self.assertEqual(len(self.lessons) % 2, 0)
        parts = {"opening": 0, "followup": 0}
        for inputs, labels, prompt, part in self.lessons:
            parts[part] += 1
            self.assertEqual(len(inputs), len(labels))
            supervised = [label for label in labels if label != -100]
            self.assertTrue(supervised)
            if part == "opening":
                self.assertTrue(prompt.endswith("Follow-up style: ?"))
                self.assertNotIn(EOS, supervised)
                self.assertEqual(labels[-1], -100)
            else:
                self.assertIn("\nOpening style: ?\nFollow-up style: ", prompt)
                self.assertEqual(supervised[-1], EOS)
                self.assertEqual(labels[0], -100)
        self.assertEqual(parts["opening"], parts["followup"])

    def test_transfer_pairs_never_appear_in_training(self):
        probes = mixed_style_transfer_probes(self.corpus)
        prior = {row.user for row in (*self.corpus.train, *self.corpus.valid, *self.corpus.test)}
        aux = {prompt for _, _, prompt, _ in self.lessons}
        self.assertEqual(len(probes), 36)
        for row in probes:
            self.assertNotIn(row.user, prior)
            self.assertNotIn(row.user, aux)
            self.assertNotIn("?", row.user.rsplit("\nOpening style: ", 1)[1])

    def test_original_train_rows_are_never_mutated(self):
        self.assertTrue(all(
            "Opening style: ?" not in row.user and "Follow-up style: ?" not in row.user
            for row in self.corpus.train
        ))


if __name__ == "__main__":
    unittest.main()
