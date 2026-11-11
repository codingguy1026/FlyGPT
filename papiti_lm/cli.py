"""Explicit training, held-out evaluation, and local inference commands.

Running the FlyGPT app never trains or downloads this model automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Install CPU PyTorch in your training venv first."
        ) from exc
    return torch


def _dataset(path: str, seed: int, fraction: float):
    from papiti_lm.corpus import load_examples, split_by_group
    examples = load_examples(path)
    train, valid = split_by_group(examples, seed, fraction)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return train, valid, digest


def _collate(batch):
    import torch
    from papiti_lm.corpus import PAD

    length = max(len(x) for x, _ in batch)
    inputs = torch.full((len(batch), length), PAD, dtype=torch.long)
    targets = torch.full((len(batch), length), -100, dtype=torch.long)
    for row, (source, target) in enumerate(batch):
        inputs[row, :len(source)] = torch.tensor(source, dtype=torch.long)
        targets[row, :len(target)] = torch.tensor(target, dtype=torch.long)
    return inputs, targets


def _evaluate(model, sequences, batch_size: int) -> dict[str, float]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for inputs, targets in DataLoader(
            sequences, batch_size=batch_size, collate_fn=_collate
        ):
            number = int(targets.ne(-100).sum())
            if not number:
                continue
            loss = model.loss(inputs, targets)
            total_loss += float(loss) * number
            total_tokens += number
    if not total_tokens:
        raise ValueError("no validation response tokens")
    nll = total_loss / total_tokens
    return {"nll": round(nll, 5), "perplexity": round(math.exp(min(20, nll)), 3)}


def _train(args):
    torch = _torch()
    from torch.utils.data import DataLoader
    from papiti_lm.corpus import CharTokenizer, JamoTokenizer
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint

    if not 1 <= args.epochs <= 200 or not 1 <= args.batch_size <= 256:
        raise ValueError("epochs and batch size outside supported range")
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("learning rate outside supported range")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train, valid, digest = _dataset(args.data, args.seed, args.val_fraction)
    tokenizer = (JamoTokenizer if args.tokenizer == "jamo" else CharTokenizer).from_examples(train)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context=args.context,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
    )
    model = PapitiTransformer(config)  # Randomly initialized, never downloaded.
    training = [tokenizer.sequence(x, config.context) for x in train]
    validation = [tokenizer.sequence(x, config.context) for x in valid]
    loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    best_nll = float("inf")
    epochs_no_improvement = 0
    print(f"random init | parameters={sum(p.numel() for p in model.parameters()):,} "
          f"| train={len(train)} | held-out={len(valid)} | vocab={tokenizer.vocab_size}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int(targets.ne(-100).sum())
            total += float(loss.detach()) * tokens
            count += tokens
        metrics = _evaluate(model, validation, args.batch_size)
        print(f"epoch={epoch} train_nll={total/count:.4f} "
              f"heldout_nll={metrics['nll']:.4f} "
              f"heldout_ppl={metrics['perplexity']:.3f}")
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            epochs_no_improvement = 0
            save_checkpoint(args.checkpoint, model, tokenizer, {
                "seed": args.seed,
                "source_sha256": digest,
                "train_examples": len(train),
                "heldout_examples": len(valid),
                "val_fraction": args.val_fraction,
                "epoch": epoch,
                "heldout_nll": metrics["nll"],
                "heldout_perplexity": metrics["perplexity"],
                "initialization": "random_from_scratch",
                "tokenizer_type": args.tokenizer,
            })
        else:
            epochs_no_improvement += 1
        if args.patience > 0 and epochs_no_improvement >= args.patience:
            print(f"early_stop epoch={epoch} after {args.patience} non-improving epochs")
            break
    print(f"best checkpoint saved locally: {args.checkpoint}")
    print("Small training corpora are experimental and do not establish fluency.")


def _eval(args):
    _torch()
    from papiti_lm.model import load_checkpoint
    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    _train_set, valid, digest = _dataset(
        args.data, int(metadata["seed"]), float(metadata["val_fraction"])
    )
    if digest != metadata["source_sha256"]:
        raise ValueError("dataset hash changed; evaluation split is no longer comparable")
    validation = [tokenizer.sequence(x, model.config.context) for x in valid]
    print(f"held-out validation (groups unseen in training): "
          f"{_evaluate(model, validation, args.batch_size)}")


def _chat(args):
    _torch()
    from papiti_lm.model import generate, load_checkpoint
    model, tokenizer, metadata = load_checkpoint(args.checkpoint)
    print(f"Papiti scratch Transformer | trained epoch {metadata.get('epoch', '?')}")
    if args.message is not None:
        print(generate(model, tokenizer, args.message,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, top_k=args.top_k,
                       repetition_penalty=args.repetition_penalty,
                       seed=args.seed))
        return
    print("type /quit to leave")
    while True:
        try:
            message = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip() == "/quit":
            break
        if not message.strip():
            continue
        print("papiti> " + generate(
            model, tokenizer, message, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty, seed=args.seed
        ))


def _serve(args):
    _torch()
    from papiti_lm.model import load_checkpoint
    # Fail at launch, rather than pretending a checkpoint is available.
    load_checkpoint(args.checkpoint)
    import os
    import uvicorn
    os.environ["PAPITI_LM_CHECKPOINT"] = args.checkpoint
    uvicorn.run("papiti_lm.server:app", host="127.0.0.1", port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and run Papiti's own tiny Transformer (research-only)"
    )
    actions = parser.add_subparsers(dest="command", required=True)
    train = actions.add_parser("train", help="explicit training from random initialization")
    train.add_argument("--data", default="training/papiti_demo_dialogues.jsonl")
    train.add_argument("--checkpoint", default="artifacts/papiti_language_v0_1.pt")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--tokenizer", choices=("char", "jamo"), default="char")
    train.add_argument("--patience", type=int, default=0, help="early stop after N unimproved validation epochs; 0 disables")
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--context", type=int, default=192)
    train.add_argument("--d-model", type=int, default=96)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.set_defaults(func=_train)

    evaluation = actions.add_parser("eval", help="reproduce grouped hold-out metrics")
    evaluation.add_argument("--data", default="training/papiti_demo_dialogues.jsonl")
    evaluation.add_argument("--checkpoint", default="artifacts/papiti_language_v0_1.pt")
    evaluation.add_argument("--batch-size", type=int, default=8)
    evaluation.set_defaults(func=_eval)

    chat = actions.add_parser("chat", help="run a single-turn interactive chat")
    chat.add_argument("--checkpoint", default="artifacts/papiti_language_v0_1.pt")
    chat.add_argument("--message", default=None)
    chat.add_argument("--max-new-tokens", type=int, default=96)
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--top-k", type=int, default=24)
    chat.add_argument("--repetition-penalty", type=float, default=1.0)
    chat.add_argument("--seed", type=int, default=None)
    chat.set_defaults(func=_chat)

    serve = actions.add_parser("serve", help="loopback OpenAI-compatible inference")
    serve.add_argument("--checkpoint", default="artifacts/papiti_language_v0_1.pt")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=_serve)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
