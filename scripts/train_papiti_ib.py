"""Train Papiti v0.6.5 with an IB-MYP-inspired learning cycle.

Research analogy only: inquiry (independent clause concepts), action (new
training-only style recombinations), reflection (loss-guided practice review).
Not an official IB programme and not reflective language reasoning.

Run with .venv-train/bin/python -m scripts.train_papiti_ib
Use --control for the matched-size, no-action/no-reflection comparator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def _reflection_focus(model, actions, torch, epoch: int, *, sample=18, top=8):
    """Identify harder TRAINING-ONLY action items without reading test labels."""
    if not actions:
        return [], None
    indices = list(range(len(actions)))
    random.Random(6500 + epoch).shuffle(indices)
    model.eval()
    ranked = []
    with torch.inference_mode():
        for index in indices[:min(sample, len(actions))]:
            inputs, targets = actions[index]
            x = torch.tensor([inputs], dtype=torch.long)
            y = torch.tensor([targets], dtype=torch.long)
            loss = float(model.loss(x, y))
            if math.isfinite(loss):
                ranked.append((loss, index))
    ranked.sort(reverse=True)
    selected = [actions[index] for _, index in ranked[:top]]
    mean_loss = sum(value for value, _ in ranked) / len(ranked) if ranked else None
    return selected, mean_loss


def train(args):
    from torch.utils.data import DataLoader

    from papiti_lm.cli import _collate, _evaluate, _torch
    from papiti_lm.compositional_curriculum import clause_training_sequences
    from papiti_lm.corpus import DigitSubwordTokenizer
    from papiti_lm.ib_curriculum import (
        action_transfer_examples, balanced_epoch_rows, stage_for_epoch,
        unique_training_rows,
    )
    from papiti_lm.inquiry_curriculum import add_inquiry_training
    from papiti_lm.language_curriculum import foundation_epoch_count
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint
    from papiti_lm.v0_6_1_data import SpeechCorpus
    from papiti_lm.v0_6_1_prose import create_prose_corpus, prose_fingerprints

    if not 8 <= args.epochs <= 60:
        raise ValueError("IB cycle requires between 8 and 60 epochs")
    if not 0 < args.learning_rate <= 0.01 or args.patience < 1:
        raise ValueError("invalid optimization settings")

    torch = _torch()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = add_inquiry_training(create_prose_corpus(args.seed, repeats=2))
    original_rows = unique_training_rows(corpus)
    action_rows = action_transfer_examples(corpus) if not args.control else []
    # Train-side fingerprints include new action lessons. No frozen heldout
    # response, prompt or target is passed to the optimizer.
    augmented = SpeechCorpus(corpus.train + action_rows, corpus.valid, corpus.test)
    fingerprints = prose_fingerprints(augmented)

    # Fix the tokenizer to the SAME original training-side corpus in both
    # control and IB runs. Do not learn vocabulary from validation/test.
    tokenizer = DigitSubwordTokenizer.from_examples(corpus.train, max_merges=256)
    model = PapitiTransformer(ModelConfig(
        tokenizer.vocab_size, context=448, d_model=128, heads=4,
        layers=3, dropout=0.1,
    ))

    original = [tokenizer.sequence(row, model.config.context) for row in original_rows]
    action = [tokenizer.sequence(row, model.config.context) for row in action_rows]
    inquiry = clause_training_sequences(corpus, tokenizer, model.config.context)
    validation = [tokenizer.sequence(row, model.config.context) for row in corpus.valid]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.01)
    start_reflection = foundation_epoch_count(args.epochs) + min(6, max(2, args.epochs // 6))
    print(json.dumps({
        "experiment": "papiti_v0_6_5_ib_control" if args.control else "papiti_v0_6_5_ib",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "original_unique_train": len(original),
        "action_training_pairs": len(action),
        "frozen_validation": len(corpus.valid),
        "frozen_test": len(corpus.test),
        "inquiry_examples": len(inquiry),
        "action_practice_per_epoch": min(len(action), max(1, len(original) // 5)),
        "reflection_starts_after_epoch": start_reflection,
    }, ensure_ascii=False), flush=True)

    best = float("inf")
    stale = 0
    focus = []
    for epoch in range(1, args.epochs + 1):
        stage = stage_for_epoch(epoch, args.epochs)
        reflection_loss = None
        if stage == "ib_inquiry":
            examples = inquiry
        else:
            if stage == "ib_reflection" and action and (epoch - start_reflection - 1) % 3 == 0:
                focus, reflection_loss = _reflection_focus(
                    model, action, torch, epoch
                )
            examples = balanced_epoch_rows(
                original, action, epoch,
                focus=focus if stage == "ib_reflection" else None,
            )

        loader = DataLoader(
            examples, batch_size=24, shuffle=True, collate_fn=_collate,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        sum_nll, count = 0.0, 0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            nonmasked = int((targets != -100).sum().item())
            sum_nll += float(loss.detach()) * nonmasked
            count += nonmasked
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        score = _evaluate(
            model, validation, 24,
            sum(len(row.assistant) + 1 for row in corpus.valid),
        )
        record = {
            "epoch": epoch,
            "curriculum_stage": stage,
            "train_nll": round(sum_nll / max(1, count), 5),
            "validation": score,
        }
        if reflection_loss is not None:
            record["train_side_reflection_nll"] = round(reflection_loss, 5)
            record["hard_training_practice_count"] = len(focus)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if score["char_nll"] < best:
            best = score["char_nll"]
            stale = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.checkpoint, model, tokenizer, {
                "version": "v0.6.1",
                "role": "brain_conditioned_language_realizer_only",
                "initialization": "random_from_scratch",
                "shared_prose": True,
                "prose_repeats": 2,
                "inquiry_curriculum": True,
                "composition_curriculum": True,
                "ib_curriculum": not args.control,
                "ib_control": args.control,
                "prose_fingerprints": fingerprints,
                "seed": args.seed,
                "epoch": epoch,
                "val_char_nll": score["char_nll"],
                "train_count": len(augmented.train),
                "valid_count": len(corpus.valid),
                "action_training_pairs": len(action_rows),
                "frozen_test_count": len(corpus.test),
            })
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early_stop at epoch={epoch}", flush=True)
            break
    print(f"Selected checkpoint {args.checkpoint}; test split untouched", flush=True)


def main():
    parser = argparse.ArgumentParser(description="IB-inspired Papiti training")
    parser.add_argument("--control", action="store_true",
                        help="same model and epoch budget; no action/reflect transfer examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_5_ib.pt"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
