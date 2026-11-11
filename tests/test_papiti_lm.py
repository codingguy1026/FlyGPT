"""Offline tests for the scratch language model; no weights are downloaded."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from papiti_lm.corpus import (
    ASSISTANT, BOS, EOS, UNK, USER,
    CharTokenizer, JamoTokenizer, SubwordTokenizer, DigitSubwordTokenizer, Example, load_examples, split_by_group,
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

    def test_extended_v0_2_data_is_deduplicated_and_holdout_is_grouped(self):
        path = Path(__file__).resolve().parents[1] / "training/papiti_dialogues_v0_2.jsonl"
        examples = load_examples(path)
        self.assertEqual(len(examples), 225)
        self.assertEqual(len({item.group for item in examples}), 22)
        train, valid = split_by_group(examples, seed=42)
        self.assertFalse({item.group for item in train} & {item.group for item in valid})
        tokenizer = CharTokenizer.from_examples(train)
        for item in train + valid:
            tokenizer.sequence(item, context=192)

    def test_v0_3_expanded_corpus_and_jamo_roundtrip(self):
        path = Path(__file__).resolve().parents[1] / "training/papiti_dialogues_v0_3.jsonl"
        examples = load_examples(path)
        self.assertEqual(len(examples), 721)
        self.assertGreaterEqual(len({row.group for row in examples}), 50)
        train, valid = split_by_group(examples, seed=42)
        self.assertFalse({row.group for row in train} & {row.group for row in valid})
        tokenizer = JamoTokenizer.from_examples(train)
        for row in train + valid:
            tokenizer.sequence(row, context=192)
        # A composed syllable is split into constituent jamo.
        self.assertEqual(tokenizer.decode(tokenizer.encode("한글 테스트!")), "한글 테스트!")
        self.assertLess(tokenizer.vocab_size, 200)

    def test_bpe_is_train_only_and_roundtrip_for_korean(self):
        examples = [Example("a", "안녕하세요", "반가워요!"),
                    Example("b", "안녕하세요!", "좋은 하루 보내요."),
                    Example("c", "안녕", "반가워!")]
        tok = SubwordTokenizer.from_examples(examples, max_merges=24)
        for example in examples:
            for value in (example.user, example.assistant):
                self.assertEqual(tok.decode(tok.encode(value)), value)
        self.assertEqual(tok.encode("☃"), [UNK])
        self.assertTrue(tok.merges)
        tok2 = SubwordTokenizer(tok.characters, tok.merges)
        self.assertEqual(tok2.encode("안녕하세요!"), tok.encode("안녕하세요!"))

    def test_bundled_demo_is_loadable(self):
        path = Path(__file__).resolve().parents[1] / "training/papiti_demo_dialogues.jsonl"
        examples = load_examples(path)
        self.assertEqual(len(examples), 48)
        train, valid = split_by_group(examples)
        tokenizer = CharTokenizer.from_examples(train)
        for item in train + valid:
            tokenizer.sequence(item, context=192)


class V05CurriculumTests(unittest.TestCase):
    def test_arithmetic_splits_hold_out_unordered_pairs(self):
        from papiti_lm.v0_5_corpus import PROBE_PAIRS, arithmetic_examples
        first = arithmetic_examples(seed=42, train_count=200, valid_count=60)
        second = arithmetic_examples(seed=42, train_count=200, valid_count=60)
        self.assertEqual(first, second)
        self.assertFalse(first.train_pairs & first.valid_pairs)
        self.assertEqual(len(first.train), 200)
        self.assertEqual(len(first.valid), 60 + len(PROBE_PAIRS))
        self.assertTrue(all(x.assistant.isdecimal() for x in first.train))
        for a, b in PROBE_PAIRS:
            self.assertNotIn((min(a, b), max(a, b)), first.train_pairs)

    def test_augment_does_not_modify_original_groups_or_answers(self):
        from papiti_lm.v0_5_corpus import augment_language
        original = [Example("only-train", "안녕", "반가워"),
                    Example("only-train", "고마워!", "천만에")]
        augmented = augment_language(original)
        self.assertEqual([x.group for x in augmented], ["only-train"] * 2)
        self.assertEqual([x.assistant for x in augmented], [x.assistant for x in original])
        self.assertEqual([x.user for x in augmented], ["안녕!", "고마워"])

    def test_digit_bpe_never_merges_adjacent_digits(self):
        from papiti_lm.v0_5_corpus import arithmetic_examples
        mathset = arithmetic_examples(seed=42, train_count=100, valid_count=40)
        sentences = [Example("korean", "안녕", "반가워")]
        tok = DigitSubwordTokenizer.from_examples(mathset.train + sentences, max_merges=32)
        sample = "23+47은?"
        self.assertEqual(tok.decode(tok.encode(sample)), sample)
        self.assertEqual(tok.encode("23"), tok.encode("2") + tok.encode("3"))
        self.assertEqual(tok.encode("99"), tok.encode("9") + tok.encode("9"))
        self.assertTrue(all(not any(c.isdigit() for c in p[0] + p[1])
                            for p in tok.merges))
        self.assertFalse(mathset.train_pairs & mathset.valid_pairs)


class CalculatorTests(unittest.TestCase):
    def test_arithmetic_is_explicit_and_never_eval(self):
        from papiti_lm.arithmetic import try_arithmetic
        self.assertEqual(try_arithmetic("1+1은?"), "2야.")
        self.assertEqual(try_arithmetic("1 + 1 =?"), "2야.")
        self.assertEqual(try_arithmetic("(3+4)*2는?"), "14야.")
        self.assertEqual(try_arithmetic("0.1+0.2은?"), "0.3야.")
        self.assertEqual(try_arithmetic("7/2는 얼마야?"), "3.5야.")
        self.assertEqual(try_arithmetic("1/0은?"), "0으로 나눌 수 없어.")
        self.assertEqual(try_arithmetic("2**10은?"), "이 계산식은 아직 지원하지 않아.")
        self.assertIsNone(try_arithmetic("안녕"))
        self.assertIsNone(try_arithmetic("__import__('os').system('echo hello')"))
        self.assertIsNone(try_arithmetic("1+1은?\nprint(7)"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch not installed in web CI")
class TransformerSmokeTests(unittest.TestCase):
    def test_hybrid_openai_adapter_accepts_default_512_budget(self):
        from unittest.mock import patch
        from papiti_lm import MODEL_NAME
        from papiti_lm.server import ChatRequest, Message, chat

        request = ChatRequest(
            model=MODEL_NAME,
            messages=[Message(role="system", content="a plan"),
                      Message(role="user", content="안녕")],
            max_tokens=512,
        )
        from types import SimpleNamespace
        dummy_tokenizer = SimpleNamespace(vocab_size=50)
        with patch("papiti_lm.server._loaded", return_value=(object(), dummy_tokenizer)), \
             patch("papiti_lm.model.generate", return_value="반가워") as mock_generate:
            result = chat(request)
        self.assertEqual(result["choices"][0]["message"]["content"], "반가워")
        self.assertEqual(mock_generate.call_args.kwargs["max_new_tokens"], 128)

    def test_v0_5_checkpoint_and_model_only_math_chat(self):
        import io
        import torch
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import patch
        from papiti_lm.model import ModelConfig, PapitiTransformer, load_checkpoint, save_checkpoint
        from papiti_lm import v0_5

        rows = [Example("t", "1+1은?", "2"),
                Example("t", "안녕", "반가워!")]
        tok = DigitSubwordTokenizer.from_examples(rows, max_merges=8)
        torch.manual_seed(15)
        model = PapitiTransformer(ModelConfig(
            vocab_size=tok.vocab_size, context=64,
            d_model=32, heads=4, layers=1, dropout=0.0
        ))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_only.pt"
            save_checkpoint(path, model, tok, {"version": "v0.5", "epoch": 1})
            _, loaded_tok, meta = load_checkpoint(path)
        self.assertIsInstance(loaded_tok, DigitSubwordTokenizer)
        self.assertEqual(meta["version"], "v0.5")
        self.assertEqual(loaded_tok.encode("23"), tok.encode("23"))
        args = SimpleNamespace(
            checkpoint="unused", message="1+1은?", max_new_tokens=8,
            temperature=0.0, top_k=12, repetition_penalty=1.0, seed=7
        )
        output = io.StringIO()
        with patch.object(v0_5, "_torch"), \
             patch("papiti_lm.model.load_checkpoint",
                   return_value=(model, tok, {"version": "v0.5", "epoch": 1})), \
             patch("papiti_lm.model.generate", return_value="actual-model-output") as called, \
             redirect_stdout(output):
            v0_5._chat(args)
        self.assertIn("actual-model-output", output.getvalue())
        self.assertEqual(called.call_args.args[2], "1+1은?")

    def test_bpe_checkpoint_reload_and_seeded_generation(self):
        import torch
        from papiti_lm.model import (
            ModelConfig, PapitiTransformer, generate, load_checkpoint, save_checkpoint
        )
        rows = [Example("demo", "안녕하세요", "반가워요!"),
                Example("demo", "고마워요", "천만에요!")]
        tok = SubwordTokenizer.from_examples(rows, max_merges=18)
        torch.manual_seed(3)
        model = PapitiTransformer(ModelConfig(
            vocab_size=tok.vocab_size, context=64,
            d_model=32, heads=4, layers=1, dropout=0
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bpe.pt"
            save_checkpoint(path, model, tok, {"epoch": 2})
            loaded, second, metadata = load_checkpoint(path)
            self.assertIsInstance(second, SubwordTokenizer)
            self.assertEqual(tok.merges, second.merges)
            self.assertEqual(tok.encode("안녕하세요"), second.encode("안녕하세요"))
            self.assertEqual(metadata["epoch"], 2)
            self.assertEqual(
                generate(loaded, second, "안녕", max_new_tokens=10, seed=42),
                generate(loaded, second, "안녕", max_new_tokens=10, seed=42),
            )

    def test_opt_in_calculator_bypasses_model_generation(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from papiti_lm import MODEL_NAME
        from papiti_lm.server import ChatRequest, Message, chat
        request = ChatRequest(
            model=MODEL_NAME,
            messages=[Message(role="user", content="1+1은?")],
        )
        with patch.dict(os.environ, {"PAPITI_LM_ARITHMETIC": "1"}), \
             patch("papiti_lm.server._loaded",
                   return_value=(object(), SimpleNamespace(vocab_size=100))), \
             patch("papiti_lm.model.generate") as model_generate:
            answer = chat(request)
        self.assertEqual(answer["choices"][0]["message"]["content"], "2야.")
        model_generate.assert_not_called()

    def test_jamo_checkpoint_reload_and_repetition_penalty(self):
        import torch
        from papiti_lm.model import (
            ModelConfig, PapitiTransformer, generate, load_checkpoint, save_checkpoint
        )
        rows = [Example("demo", "안녕하세요", "반가워요!"),
                Example("demo", "고마워요", "천만에요!")]
        tokenizer = JamoTokenizer.from_examples(rows)
        torch.manual_seed(7)
        model = PapitiTransformer(ModelConfig(
            vocab_size=tokenizer.vocab_size, context=96,
            d_model=32, heads=4, layers=1, dropout=0
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jamo.pt"
            save_checkpoint(path, model, tokenizer, {"epoch": 1})
            loaded, tok, _ = load_checkpoint(path)
            self.assertIsInstance(tok, JamoTokenizer)
            self.assertEqual(tok.decode(tok.encode("안녕하세요")), "안녕하세요")
            text1 = generate(loaded, tok, "안녕", max_new_tokens=10,
                             repetition_penalty=1.12, seed=42)
            text2 = generate(loaded, tok, "안녕", max_new_tokens=10,
                             repetition_penalty=1.12, seed=42)
            self.assertEqual(text1, text2)
            with self.assertRaisesRegex(ValueError, "repetition_penalty"):
                generate(loaded, tok, "안녕", repetition_penalty=0.5)

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
        alternative[0, -1] = tokenizer.encode("안")[0]
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
