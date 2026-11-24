"""Guardrails for Papiti v0.6.6 train-only fluency recovery."""
from __future__ import annotations

import unittest

from papiti_lm.corpus import EOS
from papiti_lm.fluency_curriculum import (
    SEALED_TRANSFER_PAIRS, fluency_epoch_rows, fluency_sequence,
    sealed_transfer_probes,
)
from papiti_lm.ib_curriculum import (
    ACTION_PAIRS, RESERVED_TRANSFER_PAIRS,
    action_transfer_examples, unique_training_rows,
)
from papiti_lm.inquiry_curriculum import add_inquiry_training
from papiti_lm.v0_6_1_prose import create_prose_corpus


class FluencyRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = create_prose_corpus(seed=42, repeats=2)
        cls.corpus = add_inquiry_training(cls.base)
        cls.original = unique_training_rows(cls.corpus)
        cls.action = action_transfer_examples(cls.corpus)
        # Same tokenizer as v0.6.5, built exclusively from the train split.
        from papiti_lm.corpus import DigitSubwordTokenizer
        cls.tokenizer = DigitSubwordTokenizer.from_examples(
            cls.corpus.train, max_merges=256
        )

    def test_weights_keep_all_supervised_targets_and_eos(self):
        row = self.action[0]
        ids, targets, weights = fluency_sequence(
            row, self.tokenizer, 448, followup_weight=1.6, eos_weight=1.4,
        )
        original_ids, original_targets = self.tokenizer.sequence(row, 448)
        self.assertEqual(ids, original_ids)
        self.assertEqual(targets, original_targets)
        self.assertEqual(EOS, targets[-1])
        self.assertAlmostEqual(1.4, weights[-1])
        self.assertIn(1.6, weights)
        self.assertIn(1.0, weights)
        for label, weight in zip(targets, weights):
            self.assertEqual(label == -100, weight == 0)

    def test_unweighted_control_is_original_full_sentence_ce(self):
        for row in (self.original[0], self.action[0]):
            _, targets, weights = fluency_sequence(
                row, self.tokenizer, 448,
                followup_weight=1.0, eos_weight=1.0,
            )
            self.assertEqual(
                weights, [0.0 if x == -100 else 1.0 for x in targets]
            )

    def test_same_examples_in_control_and_fluency_arm(self):
        for epoch in (1, 3, 12):
            candidate = fluency_epoch_rows(
                self.original, self.action, self.tokenizer, 448, epoch,
                followup_weight=1.6, eos_weight=1.4,
            )
            control = fluency_epoch_rows(
                self.original, self.action, self.tokenizer, 448, epoch,
                followup_weight=1.0, eos_weight=1.0,
            )
            self.assertEqual(len(self.original), len(candidate))
            self.assertEqual(
                [(ids, labels) for ids, labels, _ in candidate],
                [(ids, labels) for ids, labels, _ in control],
            )

    def test_new_sealed_36_are_not_training_or_old_diagnostics(self):
        self.assertEqual(
            set(SEALED_TRANSFER_PAIRS).intersection(ACTION_PAIRS), set()
        )
        self.assertEqual(
            set(SEALED_TRANSFER_PAIRS).intersection(RESERVED_TRANSFER_PAIRS), set()
        )
        probes = sealed_transfer_probes(self.corpus)
        self.assertEqual(36, len(probes))
        all_prompts = {
            row.user for row in (
                *self.corpus.train, *self.action, *self.corpus.valid, *self.corpus.test
            )
        }
        for row in probes:
            self.assertNotIn(row.user, all_prompts)
            start, follow = row.user.rsplit("\nOpening style: ", 1)[1].split(
                "\nFollow-up style: ", 1
            )
            self.assertIn((int(start), int(follow)), SEALED_TRANSFER_PAIRS)

    def test_legacy_validation_and_94_test_are_untouched(self):
        self.assertEqual(94, len(self.corpus.test))
        self.assertEqual(self.base.valid, self.corpus.valid)
        self.assertEqual(self.base.test, self.corpus.test)


if __name__ == "__main__":
    unittest.main()
