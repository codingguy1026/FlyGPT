"""Papiti v0.5 model-only language and arithmetic training experiment.

No inference calculator, canned strings, external model or lookup table.
All responses, including mathematics, are autoregressively generated using
a randomly initialized Transformer trained on supervised examples.

Usage: python -m papiti_lm.v0_5 {train|eval|chat}
"""
from __future__ import annotations

import argparse
import random

from papiti_lm.cli import _collate, _dataset, _evaluate, _torch


def _prepare(data: str, seed: int, fraction: float, math_train: int, math_valid: int):
    from papiti_lm.corpus import DigitSubwordTokenizer
    from papiti_lm.v0_5_corpus import arithmetic_examples, augment_language

    text_train, text_valid, digest = _dataset(data, seed, fraction)
    arithmetic = arithmetic_examples(seed, math_train, math_valid)
    variants = augment_language(text_train)
    # Keep the original Korean task prominent in the mixed curriculum:
    # three copies of original training groups + one punctuation variant
    # and all synthetic training arithmetic examples.
    language_training = text_train * 3 + variants
    full_training = language_training + arithmetic.train
    tokenizer = DigitSubwordTokenizer.from_examples(full_training, max_merges=256)
    return text_train, text_valid, arithmetic, full_training, tokenizer, digest


def _train(args):
    torch = _torch()
    from torch.utils.data import DataLoader
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint

    if not 1 <= args.epochs <= 100 or not 1 <= args.batch_size <= 128:
        raise ValueError("invalid epochs/batch size")
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("invalid learning rate")
    if not 0 <= args.patience <= 50:
        raise ValueError("invalid patience")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    text_train, text_valid, arithmetic, full_training, tok, digest = _prepare(
        args.data, args.seed, args.val_fraction, args.math_train, args.math_valid
    )
    config = ModelConfig(
        vocab_size=tok.vocab_size, context=args.context,
        d_model=args.d_model, heads=args.heads, layers=args.layers,
        dropout=args.dropout,
    )
    model = PapitiTransformer(config)
    language_valid_seq = [tok.sequence(x, config.context) for x in text_valid]
    math_valid_seq = [tok.sequence(x, config.context) for x in arithmetic.valid]
    language_chars = sum(len(x.assistant) + 1 for x in text_valid)
    math_chars = sum(len(x.assistant) + 1 for x in arithmetic.valid)
    training = [tok.sequence(x, config.context) for x in full_training]
    loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    best = float("inf")
    stalled = 0
    print(
        f"v0.5 RANDOM INIT / MODEL ONLY | parameters={sum(p.numel() for p in model.parameters()):,} "
        f"| train={len(full_training)} (language={len(full_training)-len(arithmetic.train)}, "
        f"math={len(arithmetic.train)}) | held-out language={len(text_valid)} "
        f"math={len(arithmetic.valid)} | BPE vocab={tok.vocab_size}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        tokens, total = 0, 0.0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(targets.ne(-100).sum())
            total += float(loss.detach()) * count
            tokens += count
        language = _evaluate(model, language_valid_seq, args.batch_size, language_chars)
        math_metrics = _evaluate(model, math_valid_seq, args.batch_size, math_chars)
        score = 0.6 * language["char_nll"] + 0.4 * math_metrics["char_nll"]
        print(
            f"epoch={epoch} train_nll={total/tokens:.4f} "
            f"language_char_nll={language['char_nll']:.4f} "
            f"math_char_nll={math_metrics['char_nll']:.4f} "
            f"balanced_score={score:.4f}", flush=True
        )
        if score < best:
            best = score
            stalled = 0
            save_checkpoint(args.checkpoint, model, tok, {
                "version": "v0.5",
                "training_kind": "model_only_synthetic_arithmetic_supervision",
                "initialization": "random_from_scratch",
                "source_sha256": digest,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
                "math_train_count": args.math_train,
                "math_valid_count": args.math_valid,
                "language_train_examples": len(text_train),
                "language_heldout_examples": len(text_valid),
                "math_heldout_examples": len(arithmetic.valid),
                "epoch": epoch,
                "language_char_nll": language["char_nll"],
                "math_char_nll": math_metrics["char_nll"],
                "balanced_score": round(score, 5),
            })
        else:
            stalled += 1
        if args.patience and stalled >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
            break
    print(f"model-only v0.5 checkpoint: {args.checkpoint}", flush=True)
    print("No arithmetic engine or answer lookup runs at inference.", flush=True)


def _eval(args):
    _torch()
    from papiti_lm.v0_5_corpus import arithmetic_examples
    from papiti_lm.model import generate, load_checkpoint

    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.5":
        raise ValueError("v0.5 checkpoint required")
    train, valid, digest = _dataset(
        args.data, int(metadata["seed"]), float(metadata["val_fraction"])
    )
    if digest != metadata["source_sha256"]:
        raise ValueError("training data changed since checkpoint creation")
    arithmetic = arithmetic_examples(
        seed=int(metadata["seed"]),
        train_count=int(metadata["math_train_count"]),
        valid_count=int(metadata["math_valid_count"]),
    )
    language_seq = [tokenizer.sequence(x, model.config.context) for x in valid]
    math_seq = [tokenizer.sequence(x, model.config.context) for x in arithmetic.valid]
    language = _evaluate(
        model, language_seq, args.batch_size,
        sum(len(x.assistant) + 1 for x in valid),
    )
    mathematics = _evaluate(
        model, math_seq, args.batch_size,
        sum(len(x.assistant) + 1 for x in arithmetic.valid),
    )
    print(f"held-out language: {language}")
    print(f"held-out mathematics: {mathematics}")
    probes = arithmetic.valid[-8:] + arithmetic.valid[:args.extra_probes]
    hits = 0
    for row in probes:
        # Direct neural output: NO arithmetic parser, hardcoded answer, or tool.
        answer = generate(model, tokenizer, row.user, temperature=0,
                          max_new_tokens=8)
        correct = answer.strip() == row.assistant
        hits += correct
        print(
            f"UNSEEN-MATH {row.user!r}: predicted={answer!r} "
            f"expected={row.assistant!r} correct={correct}"
        )
    print(f"HELDOUT EXACT MATCH: {hits}/{len(probes)} ({hits/len(probes):.1%})")
    for prompt in ("안녕", "고마워", "방금 야구에서 역전승을 했어!"):
        answer = generate(model, tokenizer, prompt, temperature=0,
                          max_new_tokens=40)
        print(f"UNSEEN-DIALOGUE {prompt!r}: {answer!r}")
    print("These are genuine model generations; synthetic targets appear only in training/evaluation.")


def _chat(args):
    _torch()
    from papiti_lm.model import generate, load_checkpoint

    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.5":
        raise ValueError("v0.5 checkpoint required")
    print(
        f"Papiti v0.5 | model-only | trained epoch={metadata['epoch']} "
        "| NO runtime math tool, no canned responses"
    )

    def respond(message: str) -> str:
        return generate(
            model, tokenizer, message,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty, seed=args.seed,
        )

    if args.message is not None:
        print(respond(args.message))
        return
    print("type /quit to leave")
    while True:
        try:
            message = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip() == "/quit":
            break
        if message.strip():
            print("papiti> " + respond(message))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Papiti v0.5 pure learned model; no inference answer hardcoding"
    )
    actions = parser.add_subparsers(dest="command", required=True)
    train = actions.add_parser("train", help="train language + math from random weights")
    train.add_argument("--data", default="training/papiti_dialogues_v0_3.jsonl")
    train.add_argument("--checkpoint", default="artifacts/papiti_language_v0_5.pt")
    train.add_argument("--epochs", type=int, default=22)
    train.add_argument("--patience", type=int, default=7)
    train.add_argument("--batch-size", type=int, default=24)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--context", type=int, default=192)
    train.add_argument("--d-model", type=int, default=160)
    train.add_argument("--heads", type=int, default=5)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--math-train", type=int, default=3000)
    train.add_argument("--math-valid", type=int, default=160)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.set_defaults(func=_train)

    evaluation = actions.add_parser("eval", help="heldout quality + math exact match")
    evaluation.add_argument("--data", default="training/papiti_dialogues_v0_3.jsonl")
    evaluation.add_argument("--checkpoint", default="artifacts/papiti_language_v0_5.pt")
    evaluation.add_argument("--batch-size", type=int, default=24)
    evaluation.add_argument("--extra-probes", type=int, default=24)
    evaluation.set_defaults(func=_eval)

    chat = actions.add_parser("chat", help="only model-generated output, no tools")
    chat.add_argument("--checkpoint", default="artifacts/papiti_language_v0_5.pt")
    chat.add_argument("--message", default=None)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-k", type=int, default=12)
    chat.add_argument("--repetition-penalty", type=float, default=1.0)
    chat.add_argument("--max-new-tokens", type=int, default=48)
    chat.add_argument("--seed", type=int, default=None)
    chat.set_defaults(func=_chat)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
