"""Check that v0.6.3 lessons and transfer probes cannot contaminate old tests."""
from __future__ import annotations

import unittest

from papiti_lm.corpus import DigitSubwordTokenizer
from papiti_lm.inquiry_curriculum import add_inquiry_training
from papiti_lm.language_curriculum import (
    foundation_epoch_count, foundation_sequences, mixed_style_transfer_probes,
)
from papiti_lm.v0_6_1_prose import create_prose_corpus


class LanguageCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = create_prose_corpus(repeats=2)
        cls.corpus = add_inquiry_training(cls.base)

    def test_preserves_frozen_holdouts(self):
        self.assertEqual(len(self.base.test), 94)
        self.assertEqual(self.corpus.test, self.base.test)
        self.assertEqual(self.corpus.valid, self.base.valid)

    def test_foundation_only_trains_clean_opening_targets(self):
        tokenizer = DigitSubwordTokenizer.from_examples(
            self.corpus.train, max_merges=256
        )
        lessons = foundation_sequences(self.corpus, tokenizer, 448)
        self.assertGreater(len(lessons), 0)
        for inputs, targets in lessons:
            self.assertEqual(len(inputs), len(targets))
            self.assertTrue(any(target != -100 for target in targets))
            # The clean opening target is supervised, but EOS and the
            # following line are never taught as a one-line reply.
            last_supervised = max(i for i, token in enumerate(targets)
                                  if token != -100)
            self.assertTrue(all(token == -100 for token in targets[last_supervised + 1:]))
            self.assertNotEqual(targets[last_supervised], 4)  # EOS
        self.assertEqual(foundation_epoch_count(30), 6)
        self.assertEqual(foundation_epoch_count(1), 1)

    def test_untrained_mixed_style_combinations_are_disjoint(self):
        probes = mixed_style_transfer_probes(self.corpus)
        self.assertEqual(len(probes), 36)
        prompts = {row.user for row in (
            *self.corpus.train, *self.corpus.valid, *self.corpus.test
        )}
        self.assertEqual(len({row.user for row in probes}), len(probes))
        for row in probes:
            self.assertNotIn(row.user, prompts)
            self.assertEqual(row.group, "inquiry-mixed-transfer")
            self.assertEqual(row.assistant.count("\n"), 1)
            self.assertTrue(
                row.user.endswith("Opening style: 4\nFollow-up style: 0")
                or row.user.endswith("Opening style: 0\nFollow-up style: 5")
            )


if __name__ == "__main__":
    unittest.main()
