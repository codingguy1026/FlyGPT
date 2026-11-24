"""Integrity tests for the from-scratch v0.6.7 interleaved curriculum."""
from __future__ import annotations

import unittest

from papiti_lm.compositional_curriculum import clause_training_sequences
from papiti_lm.corpus import DigitSubwordTokenizer, EOS
from papiti_lm.ib_curriculum import action_transfer_examples, unique_training_rows
from papiti_lm.inquiry_curriculum import add_inquiry_training
from papiti_lm.interleaved_curriculum import (
    interleaved_epoch_rows, paired_clause_lessons, sealed_v067_probes,
)
from papiti_lm.v0_6_1_prose import create_prose_corpus


class InterleavedCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = create_prose_corpus(seed=42, repeats=2)
        cls.corpus = add_inquiry_training(cls.base)
        cls.unique = unique_training_rows(cls.corpus)
        cls.actions = action_transfer_examples(cls.corpus)
        cls.tokenizer = DigitSubwordTokenizer.from_examples(
            cls.corpus.train, max_merges=256,
        )
        cls.paired = paired_clause_lessons(cls.corpus, cls.tokenizer, 448)

    def test_factorized_labels_reproduce_existing_valid_curriculum(self):
        existing = clause_training_sequences(self.corpus, self.tokenizer, 448)
        self.assertEqual(existing, [partial for _, partial in self.paired])
        self.assertGreater(len(existing), 0)
        self.assertEqual(0, len(existing) % 2)

    def test_no_label_loss_and_masked_variants(self):
        for full, partial in self.paired:
            source, labels = partial
            self.assertEqual(len(source), len(labels))
            self.assertEqual(EOS, full[1][-1])
            self.assertTrue(any(y == -100 for y in labels))
            self.assertTrue(any(y != -100 for y in labels))
            self.assertEqual(len(full[0]), len(full[1]))
            self.assertEqual(full[1].count(EOS), 1)

    def test_both_training_arms_have_equal_batches_and_full_rows(self):
        original = [self.tokenizer.sequence(r, 448) for r in self.unique]
        action = [self.tokenizer.sequence(r, 448) for r in self.actions]
        for epoch in (7, 12, 13, 30):
            treatment = interleaved_epoch_rows(
                original, action, self.paired, epoch, control=False,
            )
            control = interleaved_epoch_rows(
                original, action, self.paired, epoch, control=True,
            )
            self.assertEqual(len(treatment), len(control))
            self.assertEqual(treatment[:len(original)], control[:len(original)])
            self.assertEqual(
                len(treatment) - len(original),
                max(1, len(original) // 5),
            )
            self.assertNotEqual(treatment[len(original):], control[len(original):])

    def test_blind_exam_excludes_every_current_training_and_old_probe_pair(self):
        self.assertEqual(94, len(self.corpus.test))
        self.assertEqual(self.base.valid, self.corpus.valid)
        self.assertEqual(self.base.test, self.corpus.test)
        probes = sealed_v067_probes(self.corpus)
        self.assertEqual(36, len(probes))
        train_prompts = {r.user for r in (*self.corpus.train, *self.actions)}
        self.assertTrue(all(p.user not in train_prompts for p in probes))
        self.assertEqual(36, len({p.user for p in probes}))
        reserved = {
            (4, 0), (0, 5), (4, 1), (1, 5),
            (5, 0), (0, 4),  # already inspected in v0.6.6
            (2, 4), (3, 5), (5, 2), (5, 3),  # v0.6.5 training
        }
        for row in probes:
            cues = row.user.rsplit("\nOpening style: ", 1)[1]
            pair = tuple(map(int, cues.split("\nFollow-up style: ", 1)))
            self.assertIn(pair, {(4, 2), (2, 5)})
            self.assertNotIn(pair, reserved)


if __name__ == "__main__":
    unittest.main()
