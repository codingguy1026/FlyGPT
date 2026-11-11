"""Papiti v0.6.1: brain-conditioned neural speaker for Hybrid.

Only the upstream Python/MaleCNS brain can select message content. The
Transformer must turn a completed semantic packet into Korean/English text.
No runtime intent classifier, arithmetic operation or canned answer exists.
"""
from __future__ import annotations

import argparse
import json
import random
import sys


def _train(args):
    from papiti_lm.cli import _collate, _evaluate, _torch
    from papiti_lm.corpus import DigitSubwordTokenizer
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint
    from papiti_lm.v0_6_1_data import create_corpus
    from torch.utils.data import DataLoader

    torch = _torch()
    if not 1 <= args.epochs <= 60 or not 1 <= args.batch_size <= 96:
        raise ValueError("invalid training budget")
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("invalid learning rate")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    corpus = create_corpus(args.seed)
    tokenizer = DigitSubwordTokenizer.from_examples(corpus.train, max_merges=256)
    config = ModelConfig(
        tokenizer.vocab_size, context=args.context, d_model=args.d_model,
        heads=args.heads, layers=args.layers, dropout=0.1,
    )
    model = PapitiTransformer(config)
    training = [tokenizer.sequence(row, config.context) for row in corpus.train]
    validation = [tokenizer.sequence(row, config.context) for row in corpus.valid]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    print(
        f"v0.6.1 model-only speech | random init | "
        f"parameters={sum(p.numel() for p in model.parameters()):,} "
        f"| train={len(corpus.train)} | validation={len(corpus.valid)} "
        f"| untouched test={len(corpus.test)} | vocab={tokenizer.vocab_size}",
        flush=True,
    )
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                            collate_fn=_collate,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = _evaluate(
            model, validation, args.batch_size,
            sum(len(row.assistant) + 1 for row in corpus.valid),
        )
        print(json.dumps({"epoch": epoch, "validation": score}, ensure_ascii=False), flush=True)
        if score["char_nll"] < best:
            best = score["char_nll"]
            stale = 0
            save_checkpoint(args.checkpoint, model, tokenizer, {
                "version": "v0.6.1",
                "role": "brain_conditioned_language_realizer_only",
                "initialization": "random_from_scratch",
                "seed": args.seed, "epoch": epoch,
                "val_char_nll": score["char_nll"],
                "train_count": len(corpus.train),
                "valid_count": len(corpus.valid),
            })
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early_stop at epoch={epoch}", flush=True)
            break
    print(f"Selected checkpoint {args.checkpoint}; test split untouched", flush=True)


def _eval(args):
    from papiti_lm.cli import _evaluate, _torch
    from papiti_lm.model import generate, load_checkpoint
    from papiti_lm.v0_6_1_data import create_corpus

    _torch()
    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.6.1":
        raise ValueError("v0.6.1 checkpoint required")
    corpus = create_corpus(int(metadata["seed"]))
    metric = _evaluate(
        model, [tokenizer.sequence(row, model.config.context) for row in corpus.test],
        args.batch_size, sum(len(row.assistant) + 1 for row in corpus.test),
    )
    results = []
    for row in corpus.test:
        answer = generate(model, tokenizer, row.user, max_new_tokens=64,
                          temperature=0, top_k=12)
        results.append({
            "purpose": row.group,
            "expected": row.assistant,
            "generated": answer,
            "exact_match": answer.strip() == row.assistant,
        })
    report = {
        "version": "v0.6.1", "test_metrics": metric,
        "unseen_exact_match_count": sum(row["exact_match"] for row in results),
        "test_count": len(results),
        "test_generations": results,
        "release_ready": False,  # Human semantic review is required.
        "limitations": [
            "A tiny, from-scratch speaker may not preserve arbitrary upstream meanings.",
            "Exact match on unseen values is not evidence of human-level language competence.",
            "Failed or ungrounded generations must not silently replace verified native answers.",
        ],
    }
    with open(args.report, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _infer(args):
    """Only checkpoint-driven token prediction; does not load training examples."""
    from papiti_lm.cli import _torch
    from papiti_lm.model import generate, load_checkpoint
    from papiti_lm.v0_6_1_data import packet_text

    _torch()
    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.6.1":
        raise ValueError("v0.6.1 semantic speaker checkpoint required")
    plan = json.loads(sys.stdin.read())
    packet = packet_text(plan)
    answer = generate(model, tokenizer, packet, temperature=0, top_k=12,
                      max_new_tokens=args.max_new_tokens)
    # Never emit a manufactured fallback or a training-set answer.
    print(answer, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Papiti v0.6.1 trained brain-to-speech Transformer")
    actions = parser.add_subparsers(dest="command", required=True)
    train = actions.add_parser("train", help="train from scratch on semantic packets")
    evaluation = actions.add_parser("eval", help="score packets withheld from training")
    infer = actions.add_parser("infer", help="read complete brain plan JSON on stdin")
    for cmd in (train, evaluation, infer):
        cmd.add_argument("--checkpoint", default="artifacts/papiti_language_v0_6_1.pt")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--context", type=int, default=768)
    train.add_argument("--d-model", type=int, default=96)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.set_defaults(func=_train)
    evaluation.add_argument("--report", default="reports/papiti_v0_6_1.json")
    evaluation.add_argument("--batch-size", type=int, default=16)
    evaluation.set_defaults(func=_eval)
    infer.add_argument("--max-new-tokens", type=int, default=96)
    infer.set_defaults(func=_infer)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
