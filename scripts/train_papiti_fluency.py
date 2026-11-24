"""Papiti v0.6.6 training-only fluency recovery from a local v0.6.5 checkpoint.

The candidate and control start from the SAME v0.6.5 weights, train on the
SAME unique original/action rows and SAME optimizer budget. The sole ablation
is stronger follow-up/EOS cross-entropy weights, not scripted decoding.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def weighted_collate(batch):
    import torch
    from papiti_lm.corpus import PAD

    length = max(len(inputs) for inputs, _, _ in batch)
    inputs = torch.full((len(batch), length), PAD, dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    weights = torch.zeros((len(batch), length), dtype=torch.float32)
    for index, (source, target, loss_weights) in enumerate(batch):
        if len(source) != len(target) or len(target) != len(loss_weights):
            raise AssertionError("incorrectly aligned weighted training row")
        inputs[index, :len(source)] = torch.tensor(source, dtype=torch.long)
        labels[index, :len(target)] = torch.tensor(target, dtype=torch.long)
        weights[index, :len(loss_weights)] = torch.tensor(loss_weights, dtype=torch.float32)
    return inputs, labels, weights


def weighted_next_token_loss(model, inputs, labels, weights):
    import torch.nn.functional as F

    logits = model(inputs)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    weight_sum = weights.sum()
    if not bool(weight_sum > 0):
        raise AssertionError("empty supervised batch")
    return (token_loss * weights).sum() / weight_sum


def train(args):
    from torch.utils.data import DataLoader

    from papiti_lm.cli import _evaluate, _torch
    from papiti_lm.fluency_curriculum import fluency_epoch_rows
    from papiti_lm.ib_curriculum import action_transfer_examples, unique_training_rows
    from papiti_lm.inquiry_curriculum import add_inquiry_training
    from papiti_lm.model import load_checkpoint, save_checkpoint
    from papiti_lm.v0_6_1_data import SpeechCorpus
    from papiti_lm.v0_6_1_prose import create_prose_corpus, prose_fingerprints

    if not 1 <= args.epochs <= 30 or not 1 <= args.patience <= 30:
        raise ValueError("invalid fine-tuning epochs or patience")
    if not 0 < args.learning_rate <= 0.001:
        raise ValueError("fine-tuning learning rate must be in (0, 0.001]")
    if args.base_checkpoint.resolve() == args.checkpoint.resolve():
        raise ValueError("never overwrite the original v0.6.5 checkpoint")
    if args.control and (args.followup_weight != 1.6 or args.eos_weight != 1.4):
        raise ValueError("control uses default options with both effective weights equal to 1")

    torch = _torch()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, tokenizer, parent = load_checkpoint(args.base_checkpoint)
    if not (parent.get("ib_curriculum") and parent.get("composition_curriculum")
            and parent.get("inquiry_curriculum") and parent.get("shared_prose")):
        raise ValueError("v0.6.6 requires the v0.6.5 IB action/reflection checkpoint")
    if int(parent.get("seed", -1)) != args.seed:
        raise ValueError("seed must match the original checkpoint corpus split")

    corpus = add_inquiry_training(create_prose_corpus(args.seed, repeats=2))
    original = unique_training_rows(corpus)
    action = action_transfer_examples(corpus)
    fingerprints = prose_fingerprints(
        SpeechCorpus(corpus.train + action, corpus.valid, corpus.test)
    )
    if parent.get("prose_fingerprints") != fingerprints:
        raise ValueError("training corpus/fingerprints differ from source checkpoint")
    valid = [tokenizer.sequence(row, model.config.context) for row in corpus.valid]
    char_count = sum(len(row.assistant) + 1 for row in corpus.valid)
    initial = _evaluate(model, valid, 24, char_count)

    followup_weight = 1.0 if args.control else args.followup_weight
    eos_weight = 1.0 if args.control else args.eos_weight
    experiment = "papiti_v0_6_6_fluency_control" if args.control else "papiti_v0_6_6_fluency"
    def checkpoint_metadata(epoch, score):
        return {
            **parent,
            "version": "v0.6.6",
            "initialization": "fine_tuned_from_v0_6_5_ib",
            "v066_fluency": not args.control,
            "v066_control": args.control,
            "parent_checkpoint_epoch": parent.get("epoch"),
            "parent_checkpoint_path": str(args.base_checkpoint),
            "fine_tune_epoch": epoch,
            "epoch": epoch,
            "val_char_nll": score["char_nll"],
            "fluency_followup_weight": followup_weight,
            "fluency_eos_weight": eos_weight,
            "action_training_pairs": len(action),
        }

    print(json.dumps({
        "experiment": experiment,
        "parent_epoch": parent.get("epoch"),
        "original_unique_train": len(original),
        "action_training_pairs": len(action),
        "original_validation": len(corpus.valid),
        "frozen_test_count": len(corpus.test),
        "followup_weight": followup_weight,
        "eos_weight": eos_weight,
        "baseline_validation": initial,
    }, ensure_ascii=False), flush=True)

    # A valid output always exists even when the pre-trained baseline wins.
    # A selected epoch 0 means fine-tuning did not improve validation NLL.
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.checkpoint, model, tokenizer, checkpoint_metadata(0, initial))
    best, best_epoch, stale = initial["char_nll"], 0, 0
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    for epoch in range(1, args.epochs + 1):
        rows = fluency_epoch_rows(
            original, action, tokenizer, model.config.context, epoch,
            followup_weight=followup_weight, eos_weight=eos_weight,
        )
        loader = DataLoader(
            rows, batch_size=24, shuffle=True, collate_fn=weighted_collate,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        total, weight_count = 0.0, 0.0
        for inputs, labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_next_token_loss(model, inputs, labels, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite fine-tuning loss")
            mass = float(weights.sum())
            total += float(loss.detach()) * mass
            weight_count += mass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        score = _evaluate(model, valid, 24, char_count)
        if not math.isfinite(score["char_nll"]):
            raise RuntimeError("non-finite validation loss")
        print(json.dumps({
            "fine_tune_epoch": epoch,
            "weighted_train_nll": round(total / weight_count, 5),
            "validation": score,
        }, ensure_ascii=False), flush=True)

        if score["char_nll"] < best:
            best, best_epoch, stale = score["char_nll"], epoch, 0
            save_checkpoint(args.checkpoint, model, tokenizer, checkpoint_metadata(epoch, score))
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop at fine_tune_epoch={epoch}", flush=True)
                break
    print(json.dumps({
        "selected_checkpoint": str(args.checkpoint),
        "selected_fine_tune_epoch": best_epoch,
        "best_val_char_nll": best,
        "baseline_val_char_nll": initial["char_nll"],
        "frozen_test_untouched": True,
    }, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="v0.6.6 fluency recovery / matched control")
    parser.add_argument("--base-checkpoint", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_5_ib.pt"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_6_fluency.pt"))
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--followup-weight", type=float, default=1.6)
    parser.add_argument("--eos-weight", type=float, default=1.4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
