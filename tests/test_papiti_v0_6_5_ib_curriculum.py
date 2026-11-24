"""Guardrails for the experimental IB-inspired Papiti v0.6.5 curriculum."""
from __future__ import annotations

import unittest

from hybrid_speech import reply_key
from papiti_lm.ib_curriculum import (
    ACTION_PAIRS, RESERVED_TRANSFER_PAIRS, action_transfer_examples,
    balanced_epoch_rows, stage_for_epoch, unique_training_rows,
)
from papiti_lm.inquiry_curriculum import add_inquiry_training
from papiti_lm.language_curriculum import mixed_style_transfer_probes
from papiti_lm.v0_6_1_prose import create_prose_corpus


class IbLanguageCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = create_prose_corpus(seed=42, repeats=2)
        cls.corpus = add_inquiry_training(cls.base)
        cls.actions = action_transfer_examples(cls.corpus)

    def test_old_validation_and_test_are_unchanged(self):
        self.assertEqual(94, len(self.corpus.test))
        self.assertEqual(self.corpus.valid, self.base.valid)
        self.assertEqual(self.corpus.test, self.base.test)

    def test_training_recombination_never_sees_frozen_answers(self):
        existing_prompts = {
            row.user for row in (
                *self.corpus.train, *self.corpus.valid, *self.corpus.test
            )
        }
        heldout_answers = {
            reply_key(row.assistant)
            for row in (*self.corpus.valid, *self.corpus.test)
        }
        action_prompts = {row.user for row in self.actions}
        self.assertEqual(len(self.actions), len(action_prompts))
        self.assertTrue(self.actions)
        for row in self.actions:
            self.assertNotIn(row.user, existing_prompts)
            self.assertNotIn(reply_key(row.assistant), heldout_answers)
            self.assertEqual(row.assistant.count("\n"), 1)
            start, end = row.user.rsplit("\nOpening style: ", 1)[1].split(
                "\nFollow-up style: ", 1
            )
            pair = (int(start), int(end))
            self.assertIn(pair, ACTION_PAIRS)
            self.assertNotIn(pair, RESERVED_TRANSFER_PAIRS)

    def test_old_and_new_36_probe_cues_remain_untrained(self):
        action_prompts = {row.user for row in self.actions}
        old_probes = mixed_style_transfer_probes(self.corpus)
        self.assertEqual(36, len(old_probes))
        for row in old_probes:
            self.assertNotIn(row.user, action_prompts)
        for row in self.actions:
            suffix = row.user.rsplit("\nOpening style: ", 1)[1]
            pair = tuple(map(int, suffix.split("\nFollow-up style: ")))
            self.assertNotIn(pair, {(4, 0), (0, 5), (4, 1), (1, 5)})

    def test_duplicate_reduction_and_fixed_epoch_budget(self):
        original = unique_training_rows(self.corpus)
        self.assertLess(len(original), len(self.corpus.train))
        for epoch in (7, 12, 13, 22):
            rows = balanced_epoch_rows(original, self.actions, epoch)
            self.assertEqual(len(original), len(rows))
            focused = balanced_epoch_rows(
                original, self.actions, epoch, focus=self.actions[:6]
            )
            self.assertEqual(len(original), len(focused))

    def test_three_stages_and_true_control(self):
        self.assertEqual("ib_inquiry", stage_for_epoch(6, 36))
        self.assertEqual("ib_action", stage_for_epoch(7, 36))
        self.assertEqual("ib_action", stage_for_epoch(12, 36))
        self.assertEqual("ib_reflection", stage_for_epoch(13, 36))
        original = list(unique_training_rows(self.corpus))
        self.assertEqual(original, balanced_epoch_rows(original, [], 22))


if __name__ == "__main__":
    unittest.main()
