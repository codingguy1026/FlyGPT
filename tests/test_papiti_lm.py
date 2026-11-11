"""Offline tests for the scratch language model; no weights are downloaded."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from papiti_lm.corpus import (
    ASSISTANT, BOS, EOS, UNK, USER,
    CharTokenizer, Example, load_examples, split_by_group,
)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            Example(group=f"group{i // 4}", user=f"hello {i}",
                    assistant=f"reply {i}") for i in range(16)
        ]

    def test_group_disjoint_validation_and_reproducible_split(self):
        train, valid = split_by_group(self.examples, seed=7)
        self.assertFalse({x.group for x in train} & {x.group for x in valid})
        self.assertEqual((train, valid), split_by_group(self.examples, seed=7))
        self.assertEqual(len(train) + len(valid), 16)

    def test_answer_only_teacher_forcing_and_eos(self):
        tokenizer = CharTokenizer.from_examples(self.examples)
        record = self.examples[0]
        inputs, labels = tokenizer.sequence(record, context=64)
        self.assertEqual(inputs[0:2], [BOS, USER])
        self.assertEqual(inputs[-1], tokenizer.encode(record.assistant)[-1])
        prefix = tokenizer.prompt(record.user)
        self.assertEqual(prefix[-1], ASSISTANT)
        self.assertTrue(all(x == -100 for x in labels[:len(prefix)-1]))
        self.assertEqual(labels[len(prefix)-1], tokenizer.encode(record.assistant)[0])
        self.assertEqual(labels[-1], EOS)

    def test_unknown_character_is_not_leaked_from_holdout(self):
        tokenizer = CharTokenizer.from_examples([Example("a", "a", "b")])
        self.assertEqual(tokenizer.encode("☃"), [UNK])

    def test_reject_malformed_and_duplicate_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                load_examples(path)
            repeated = {"group": "a", "user": "hello", "assistant": "hi"}
            path.write_text("\n".join(json.dumps(repeated) for _ in range(12)),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_examples(path)

    def test_reject_truncation_instead_of_silently_dropping_answer(self):
        tokenizer = CharTokenizer.from_examples(self.examples)
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            tokenizer.sequence(self.examples[0], context=4)

    def test_bundled_demo_is_loadable(self):
        path = Path(__file__).resolve().parents[1] / "training/papiti_demo_dialogues.jsonl"
        examples = load_examples(path)
        self.assertEqual(len(examples), 48)
        train, valid = split_by_group(examples)
        tokenizer = CharTokenizer.from_examples(train)
        for item in train + valid:
            tokenizer.sequence(item, context=192)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch not installed in web CI")
class TransformerSmokeTests(unittest.TestCase):
    def test_causality_gradient_generation_and_safe_reload(self):
        import torch
        from papiti_lm.model import (
            ModelConfig, PapitiTransformer, generate, load_checkpoint, save_checkpoint
        )
        torch.manual_seed(123)
        records = [Example("demo", "안녕", "반가워!"),
                   Example("demo", "고마워", "천만에!")]
        tokenizer = CharTokenizer.from_examples(records)
        model = PapitiTransformer(ModelConfig(
            vocab_size=tokenizer.vocab_size, context=48, d_model=32,
            heads=4, layers=1, dropout=0.0
        ))
        model.eval()
        inputs, labels = tokenizer.sequence(records[0], 48)
        padded = torch.tensor([inputs])
        alternative = padded.clone()
        alternative[0, -1] = tokenizer.encode("!")[0]
        with torch.inference_mode():
            initial = model(padded)
            changed = model(alternative)
        self.assertTrue(torch.allclose(
            initial[:, :-1, :], changed[:, :-1, :], atol=1e-5
        ), "future tokens must not affect earlier logits")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        optimizer.zero_grad()
        loss = model.loss(padded, torch.tensor([labels]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(
            float(model.token_embedding.weight.grad.abs().sum()), 0.0
        )
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scratch.pt"
            save_checkpoint(path, model, tokenizer, {"initialization": "random_from_scratch"})
            loaded, tok, metadata = load_checkpoint(path)
            self.assertEqual(metadata["initialization"], "random_from_scratch")
            self.assertEqual(tok.characters, tokenizer.characters)
            for old, new in zip(model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(old, new))
            answer = generate(loaded, tok, "안녕", max_new_tokens=8, seed=7)
            self.assertIsInstance(answer, str)
            self.assertLessEqual(len(answer), 8)


if __name__ == "__main__":
    unittest.main()
