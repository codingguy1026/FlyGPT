"""Papiti v0.6 scratch training and strictly neural inference.

python -m papiti_lm.v0_6 {train,eval,chat,serve}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from papiti_lm.cli import _torch


def _train(args):
    torch = _torch()
    from torch.utils.data import DataLoader
    from papiti_lm.cli import _collate, _evaluate
    from papiti_lm.corpus import DigitSubwordTokenizer
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint
    from papiti_lm.v0_6_corpus import epoch_examples, prepare, training_pools
    from papiti_lm.v0_6_eval import conditional_nll, contrast_pairs

    if not 5 <= args.epochs <= 100 or not 1 <= args.batch_size <= 128:
        raise ValueError("epochs must be 5..100; batch size must be 1..128")
    if not 0 < args.learning_rate <= 0.01 or not 0 <= args.contrastive_weight <= 1:
        raise ValueError("invalid learning rate or contrastive weight")
    if args.baseline_checkpoint and Path(args.baseline_checkpoint).resolve() == Path(args.checkpoint).resolve():
        raise ValueError("baseline and selected checkpoint must have different paths")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    curriculum = prepare(args.data, args.legacy, args.seed, args.math_train, args.math_holdout)
    pools = training_pools(curriculum)
    tokenizer = DigitSubwordTokenizer.from_examples(
        [example for pool in pools.values() for example in pool], max_merges=256)
    config = ModelConfig(tokenizer.vocab_size, args.context, args.d_model,
                         args.heads, args.layers, args.dropout)
    model = PapitiTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    validation = {domain: [record.example for record in curriculum.select("valid", domain)]
                  for domain in pools}
    for record in curriculum.records:
        tokenizer.sequence(record.example, config.context)
    negatives = contrast_pairs(curriculum, "train")
    positive_examples = [positive for positive, _negative in negatives]
    negative_examples = [negative for _positive, negative in negatives]
    best = float("inf")
    history = []
    inventory = {split: {domain: len(curriculum.select(split, domain)) for domain in pools}
                 for split in ("train", "valid", "test")}
    print(f"v0.6 random init | parameters={sum(parameter.numel() for parameter in model.parameters()):,} "
          f"| vocab={tokenizer.vocab_size} | split counts={inventory}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        examples = epoch_examples(curriculum, epoch, args.seed)
        loader = DataLoader([tokenizer.sequence(example, config.context) for example in examples],
                            batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
        total, count, ranking_total = 0.0, 0, 0.0
        for step, (inputs, targets) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            cross_entropy = model.loss(inputs, targets)
            loss = cross_entropy
            if args.contrastive_weight and step % 4 == 0:
                scores = conditional_nll(model, tokenizer, positive_examples + negative_examples)
                positive, negative = scores.chunk(2)
                ranking = torch.relu(0.5 + positive - negative).mean()
                ranking_total += float(ranking.detach())
                loss = loss + args.contrastive_weight * ranking
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int(targets.ne(-100).sum())
            total += float(cross_entropy.detach()) * tokens
            count += tokens
        metrics = {}
        for domain, rows in validation.items():
            metrics[domain] = _evaluate(model, [tokenizer.sequence(example, config.context) for example in rows],
                                        args.batch_size, sum(len(example.assistant) + 1 for example in rows))
        score = (0.6 * metrics["language"]["char_nll"]
                 + 0.2 * metrics["arithmetic"]["char_nll"]
                 + 0.2 * metrics["knowledge"]["char_nll"])
        summary = {"epoch": epoch, "train_token_nll": total / count,
                   "contrastive_hinge_sum": ranking_total, "validation": metrics,
                   "selection_score": score, "sampled_examples": len(examples)}
        history.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        metadata = {
            "version": "v0.6", "initialization": "random_from_scratch", "runtime": "transformer_only",
            "source_sha256": curriculum.digest, "seed": args.seed, "epoch": epoch,
            "math_train": args.math_train, "math_holdout": args.math_holdout,
            "inventory": inventory, "selection": "validation only; compare epochs >=5 after full curriculum",
            "selection_score": score, "validation": metrics, "training_history": list(history),
            "contrastive_weight": args.contrastive_weight,
        }
        if epoch == 1 and args.baseline_checkpoint:
            save_checkpoint(args.baseline_checkpoint, model, tokenizer, metadata)
        if epoch >= 5 and score < best:
            best = score
            save_checkpoint(args.checkpoint, model, tokenizer, metadata)
    print(f"Selected checkpoint: {args.checkpoint}; final tests have not been scored.", flush=True)


def _eval(args):
    torch = _torch()
    from papiti_lm.model import load_checkpoint
    from papiti_lm.v0_6_corpus import prepare
    from papiti_lm.v0_6_eval import evaluate_checkpoint

    if not 1 <= args.batch_size <= 128:
        raise ValueError("batch size must be 1..128")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    _model, _tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.6":
        raise ValueError("v0.6 checkpoint required")
    curriculum = prepare(args.data, args.legacy, metadata["seed"],
                         metadata["math_train"], metadata["math_holdout"])
    report = evaluate_checkpoint(args.checkpoint, curriculum, args.batch_size)
    if args.compare_checkpoint:
        comparison = evaluate_checkpoint(args.compare_checkpoint, curriculum, args.batch_size)
        if comparison["metadata"]["seed"] != metadata["seed"] or comparison["metadata"]["inventory"] != metadata["inventory"]:
            raise ValueError("comparison must use the same seed and splits")
        report["one_epoch_control"] = comparison
        report["delta_char_nll"] = {
            domain: report["metrics"][domain]["char_nll"] - comparison["metrics"][domain]["char_nll"]
            for domain in report["metrics"]}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _chat(args):
    _torch()
    from papiti_lm.model import generate, load_checkpoint

    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.6":
        raise ValueError("v0.6 checkpoint required")

    def respond(message):
        return generate(model, tokenizer, message, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, seed=args.seed)

    if args.message is not None:
        print(respond(args.message))
        return
    while True:
        try:
            message = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip() == "/quit":
            break
        if message.strip():
            print("papiti> " + respond(message))


def _serve(args):
    import os
    import uvicorn
    from papiti_lm.model import load_checkpoint

    _model, _tokenizer, metadata = load_checkpoint(args.checkpoint)
    if metadata.get("version") != "v0.6":
        raise ValueError("v0.6 checkpoint required")
    os.environ["PAPITI_LM_CHECKPOINT"] = str(Path(args.checkpoint).resolve())
    from papiti_lm.server import _loaded, app
    _loaded.cache_clear()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Papiti v0.6: source-backed training, pure Transformer answers")
    actions = parser.add_subparsers(dest="command", required=True)
    train = actions.add_parser("train")
    evaluation = actions.add_parser("eval")
    chat = actions.add_parser("chat")
    serve = actions.add_parser("serve")
    for action in (train, evaluation, chat, serve):
        action.add_argument("--checkpoint", default="artifacts/papiti_language_v0_6.pt")
    for action in (train, evaluation):
        action.add_argument("--data", default="training/papiti_v0_6_curriculum.json")
        action.add_argument("--legacy", default="training/papiti_dialogues_v0_3.jsonl")
        action.add_argument("--batch-size", type=int, default=24)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--baseline-checkpoint", default=None)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--context", type=int, default=384)
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--contrastive-weight", type=float, default=0.15)
    train.add_argument("--math-train", type=int, default=1200)
    train.add_argument("--math-holdout", type=int, default=80)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=_train)
    evaluation.add_argument("--report", default="reports/papiti_v0_6.json")
    evaluation.add_argument("--compare-checkpoint", default=None)
    evaluation.set_defaults(func=_eval)
    chat.add_argument("--message", default=None)
    chat.add_argument("--temperature", type=float, default=0)
    chat.add_argument("--top-k", type=int, default=12)
    chat.add_argument("--max-new-tokens", type=int, default=96)
    chat.add_argument("--seed", type=int, default=None)
    chat.set_defaults(func=_chat)
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=_serve)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
