"""Leakage, provenance, evaluation and strict inference regression tests."""
from __future__ import annotations

import copy
import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from papiti_lm.corpus import DigitSubwordTokenizer, Example
from papiti_lm.v0_6_corpus import (
    Record, audit, connected_groups, epoch_examples, normalize_prompt, prepare, training_pools,
)
from papiti_lm.v0_6_eval import contrast_pairs, surface_metrics


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curriculum = prepare(math_train=80, math_holdout=12)

    def test_source_coverage_and_reproducibility(self):
        first = self.curriculum
        self.assertEqual(first, prepare(math_train=80, math_holdout=12))
        self.assertEqual(len(first.manifest["knowledge"]), 50)
        self.assertTrue(any(record.level == 3 for record in first.select("train", "knowledge")))
        for record in first.records:
            source = first.manifest["sources"][record.source]
            self.assertTrue(source["url"].startswith("https://"))
            self.assertTrue(source["provenance"])
        for split in ("train", "valid", "test"):
            self.assertTrue(first.select(split, "language"))
            self.assertTrue(first.select(split, "knowledge"))

    def test_group_and_normalized_prompt_disjointness(self):
        splits = {split: self.curriculum.select(split) for split in ("train", "valid", "test")}
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            self.assertFalse({record.family for record in splits[left]}
                             & {record.family for record in splits[right]})
            self.assertFalse({normalize_prompt(record.example.user) for record in splits[left]}
                             & {normalize_prompt(record.example.user) for record in splits[right]})
        self.assertNotIn("addition:1:1", {record.family for record in splits["train"]})
        self.assertIn("addition:1:1", {record.family for record in splits["test"]})

    def test_audit_rejects_perturbed_prompt_leakage(self):
        rows = list(self.curriculum.records)
        original = next(record for record in rows if record.split == "train")
        duplicate = replace(original, family="new-test-family", split="test",
                            example=replace(original.example, user=original.example.user + " !"))
        with self.assertRaisesRegex(ValueError, "prompt leakage"):
            audit(rows + [duplicate])
        with self.assertRaisesRegex(ValueError, "family leakage"):
            audit(rows + [replace(original, split="test")])

    def test_transitive_duplicates_keep_source_groups_together(self):
        rows = [Example("a", "hello!", "hi"), Example("b", "HELLO", "hey"),
                Example("b", "bye?", "bye"), Example("c", "bye", "see you")]
        merged = connected_groups(rows)
        self.assertEqual(len({row.group for row in merged}), 1)

    def test_test_labels_cannot_change_training_or_tokenizer(self):
        first = self.curriculum
        second = copy.deepcopy(first)
        second.records = [replace(record, example=replace(record.example, assistant="\u2603"))
                          if record.split == "test" else record for record in second.records]
        self.assertEqual(training_pools(first), training_pools(second))
        self.assertEqual(epoch_examples(first, 6, 42), epoch_examples(second, 6, 42))
        pools = training_pools(first)
        tokenizer = DigitSubwordTokenizer.from_examples(
            [example for pool in pools.values() for example in pool], max_merges=8)
        self.assertNotIn("\u2603", tokenizer.characters)

    def test_stages_balance_domains_without_holding_out_augmentation(self):
        early = epoch_examples(self.curriculum, 1, 42)
        late = epoch_examples(self.curriculum, 6, 42)
        advanced = {record.family for record in self.curriculum.select("train", "knowledge")
                    if record.level == 3}
        self.assertFalse({example.group for example in early} & advanced)
        self.assertTrue({example.group for example in late} & advanced)
        self.assertEqual(late, epoch_examples(self.curriculum, 6, 42))
        self.assertEqual(len(early), len(late))
        heldout = {normalize_prompt(record.example.user) for record in self.curriculum.records
                   if record.split != "train"}
        self.assertFalse({normalize_prompt(example.user) for example in late} & heldout)
        for positive, negative in contrast_pairs(self.curriculum, "train"):
            self.assertEqual(positive.user, negative.user)
            self.assertNotEqual(positive.assistant, negative.assistant)

    def test_surface_metrics_are_diagnostics_not_grammar_claims(self):
        self.assertTrue(surface_metrics("")["empty"])
        self.assertTrue(surface_metrics("\ufffd")["malformed_unicode"])
        self.assertGreater(surface_metrics("하나 둘 셋 하나 둘 셋")["repeated_trigram_fraction"], 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch required")
class NeuralTests(unittest.TestCase):
    def test_generic_cli_cannot_enable_calculator_for_v06_checkpoint(self):
        from papiti_lm.cli import _chat
        with patch("papiti_lm.model.load_checkpoint", return_value=(object(), object(), {"version": "v0.6"})), \
                patch("papiti_lm.arithmetic.try_arithmetic") as calculator:
            with self.assertRaisesRegex(ValueError, "model-only"):
                _chat(SimpleNamespace(checkpoint="unused", math_tool=True))
        calculator.assert_not_called()

    def test_chat_preserves_neural_output_despite_math_environment(self):
        from papiti_lm import v0_6
        args = SimpleNamespace(checkpoint="unused", message="1+1은?", max_new_tokens=12,
                               temperature=0, top_k=12, seed=42)
        output = io.StringIO()
        with patch.dict(os.environ, {"PAPITI_LM_ARITHMETIC": "1"}), \
                patch("papiti_lm.model.load_checkpoint", return_value=(object(), object(), {"version": "v0.6"})), \
                patch("papiti_lm.model.generate", return_value="wrong-model-answer") as generated, \
                patch("papiti_lm.arithmetic.try_arithmetic", side_effect=AssertionError("calculator forbidden")), \
                patch("papiti_lm.v0_6_corpus.prepare", side_effect=AssertionError("curriculum forbidden")), \
                redirect_stdout(output):
            v0_6._chat(args)
        self.assertEqual(output.getvalue().strip(), "wrong-model-answer")
        generated.assert_called_once()
        self.assertEqual(generated.call_args.args[2], "1+1은?")

    def test_v06_server_disables_legacy_calculator_even_with_environment(self):
        from papiti_lm import MODEL_NAME
        from papiti_lm.server import ChatRequest, Message, chat
        request = ChatRequest(model=MODEL_NAME, messages=[Message(role="user", content="1+1은?")])
        with patch.dict(os.environ, {"PAPITI_LM_ARITHMETIC": "1"}), \
                patch("papiti_lm.server._loaded", return_value=(SimpleNamespace(papiti_version="v0.6"),
                                                                 SimpleNamespace(vocab_size=60))), \
                patch("papiti_lm.model.generate", return_value="neural prediction") as generated, \
                patch("papiti_lm.arithmetic.try_arithmetic", side_effect=AssertionError("calculator forbidden")):
            response = chat(request)
        self.assertEqual(response["choices"][0]["message"]["content"], "neural prediction")
        generated.assert_called_once()

    def test_checkpoint_fingerprint_and_model_backpropagation(self):
        import torch
        from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint
        from papiti_lm.v0_6_eval import conditional_nll, evaluate_checkpoint
        curriculum = prepare(math_train=20, math_holdout=2)
        examples = [record.example for record in curriculum.select("train")]
        tokenizer = DigitSubwordTokenizer.from_examples(examples, max_merges=8)
        model = PapitiTransformer(ModelConfig(tokenizer.vocab_size, context=384,
                                              d_model=32, heads=4, layers=1, dropout=0))
        pairs = contrast_pairs(curriculum, "train")
        scores = conditional_nll(model, tokenizer, [pairs[0][0], pairs[0][1]])
        loss = torch.relu(0.5 + scores[0] - scores[1])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "tiny.pt"
            save_checkpoint(checkpoint, model, tokenizer, {"version": "v0.6", "source_sha256": "wrong"})
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                evaluate_checkpoint(str(checkpoint), curriculum)


if __name__ == "__main__":
    unittest.main()
