"""From-scratch Papiti v0.6.7: continuous factorized-clause rehearsal.

No direct optimization on frozen validation, legacy 94 tests or previously
viewed/new sealed transfer probes. Both arms share initial weights, corpus,
model/tokenizer, batch budget, and validation-based checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random


def train(args):
    from torch.utils.data import DataLoader

    from papiti_lm.cli import _collate, _evaluate, _torch
    from papiti_lm.compositional_curriculum import clause_training_sequences
    from papiti_lm.corpus import DigitSubwordTokenizer
    from papiti_lm.ib_curriculum import (
        action_transfer_examples, unique_training_rows,
    )
    from papiti_lm.inquiry_curriculum import add_inquiry_training
    from papiti_lm.interleaved_curriculum import (
        interleaved_epoch_rows, paired_clause_lessons,
    )
    from papiti_lm.language_curriculum import foundation_epoch_count
    from papiti_lm.model import ModelConfig, PapitiTransformer, save_checkpoint
    from papiti_lm.v0_6_1_data import SpeechCorpus
    from papiti_lm.v0_6_1_prose import create_prose_corpus, prose_fingerprints

    if not 8 <= args.epochs <= 60 or args.patience < 1:
        raise ValueError("invalid training length/patience")
    if not 0 < args.learning_rate <= 0.01:
        raise ValueError("invalid learning rate")

    torch = _torch()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = add_inquiry_training(create_prose_corpus(args.seed, repeats=2))
    unique = unique_training_rows(corpus)
    actions = action_transfer_examples(corpus)
    training_fingerprints = prose_fingerprints(
        SpeechCorpus(corpus.train + actions, corpus.valid, corpus.test)
    )
    tokenizer = DigitSubwordTokenizer.from_examples(corpus.train, max_merges=256)
    model = PapitiTransformer(ModelConfig(
        tokenizer.vocab_size, context=448, d_model=128, heads=4,
        layers=3, dropout=0.1,
    ))

    original_rows = [tokenizer.sequence(r, model.config.context) for r in unique]
    action_rows = [tokenizer.sequence(r, model.config.context) for r in actions]
    inquiries = clause_training_sequences(corpus, tokenizer, model.config.context)
    paired = paired_clause_lessons(corpus, tokenizer, model.config.context)
    validation = [tokenizer.sequence(r, model.config.context) for r in corpus.valid]
    val_char_count = sum(len(r.assistant) + 1 for r in corpus.valid)
    first_full_epoch = foundation_epoch_count(args.epochs) + 1
    experiment = "papiti_v0_6_7_full_control" if args.control else "papiti_v0_6_7_interleaved"
    print(json.dumps({
        "experiment": experiment,
        "initialized": "random_from_scratch",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "unique_original_train": len(unique),
        "training_action_pairs": len(actions),
        "interleaved_train_only_pairs": len(paired),
        "full_epoch_extra_examples": max(1, len(original_rows) // 5),
        "frozen_validation": len(corpus.valid),
        "frozen_test": len(corpus.test),
        "foundation_epochs": first_full_epoch - 1,
        "balanced_original_and_action": True,
        "sealed_transfer_probes_used_in_training": False,
    }, ensure_ascii=False), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01,
    )
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        stage = ("factorized_foundation" if epoch < first_full_epoch
                 else "full_replay_control" if args.control
                 else "interleaved_clause_rehearsal")
        if epoch < first_full_epoch:
            rows = inquiries
        else:
            # In the full-response stage both arms see the same examples,
            # plus the same number of additional TRAIN-ONLY items.
            # The only treatment change is which parts are supervised
            # in those supplementary lesson items.
            rows = interleaved_epoch_rows(
                original_rows, action_rows, paired, epoch,
                control=args.control,
            )

        loader = DataLoader(
            rows, batch_size=24, shuffle=True, collate_fn=_collate,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        loss_sum, tokens = 0.0, 0
        for x, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            count = int(labels.ne(-100).sum().item())
            loss_sum += float(loss.detach()) * count
            tokens += count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        score = _evaluate(model, validation, 24, val_char_count)
        record = {
            "epoch": epoch,
            "stage": stage,
            "train_nll": round(loss_sum / max(1, tokens), 5),
            "validation": score,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if not math.isfinite(score["char_nll"]):
            raise RuntimeError("non-finite validation loss")
        if score["char_nll"] < best:
            best, stale = score["char_nll"], 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.checkpoint, model, tokenizer, {
                "version": "v0.6.7",
                "role": "brain_conditioned_language_realizer_only",
                "initialization": "random_from_scratch",
                "shared_prose": True,
                "prose_repeats": 2,
                "inquiry_curriculum": True,
                "composition_curriculum": True,
                "interleaved_clauses": not args.control,
                "interleaved_control": args.control,
                "prose_fingerprints": training_fingerprints,
                "seed": args.seed,
                "epoch": epoch,
                "val_char_nll": score["char_nll"],
                "train_count": len(corpus.train) + len(actions),
                "valid_count": len(corpus.valid),
                "frozen_test_count": len(corpus.test),
                "action_training_pairs": len(actions),
                "sealed_v067_used_in_training": False,
            })
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop at epoch={epoch}", flush=True)
                break
    print(json.dumps({
        "experiment": experiment,
        "selected_checkpoint": str(args.checkpoint),
        "best_val_char_nll": best,
        "frozen_test_untouched": True,
    }, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Papiti v0.6.7 interleaved clause training")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_7_interleaved.pt"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
