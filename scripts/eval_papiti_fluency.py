"""One-time sealed v0.6.6 compositional fluency exam.

Only call after both candidate and control have finished training. The two
predeclared 36-probe cue pairs (5,0) and (0,4) are never used in optimization.
Once reports are examined, these probes become development diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(args):
    from hybrid_speech import reply_key
    from papiti_lm.cli import _evaluate
    from papiti_lm.fluency_curriculum import sealed_transfer_probes
    from papiti_lm.inquiry_curriculum import add_inquiry_training
    from papiti_lm.model import generate, load_checkpoint
    from papiti_lm.v0_6_1_prose import create_prose_corpus

    corpus = add_inquiry_training(create_prose_corpus(seed=42, repeats=2))
    probes = sealed_transfer_probes(corpus)
    reports = []
    for name, checkpoint in (
        ("v065_baseline", args.baseline),
        ("v066_control", args.control),
        ("v066_fluency", args.candidate),
    ):
        if checkpoint is None:
            continue
        model, tokenizer, metadata = load_checkpoint(checkpoint)
        if name == "v065_baseline" and not metadata.get("ib_curriculum"):
            raise ValueError("the baseline is not a v0.6.5 IB checkpoint")
        if name == "v066_control" and not metadata.get("v066_control"):
            raise ValueError("the control is not a v0.6.6 control checkpoint")
        if name == "v066_fluency" and not metadata.get("v066_fluency"):
            raise ValueError("the candidate is not a v0.6.6 fluency checkpoint")
        if metadata.get("seed") != 42 or metadata.get("prose_repeats") != 2:
            raise ValueError("the checkpoints must use the frozen seed-42 corpus")
        sequences = [tokenizer.sequence(row, model.config.context) for row in probes]
        score = _evaluate(
            model, sequences, 24, sum(len(row.assistant) + 1 for row in probes),
        )
        training_replies = set(metadata["prose_fingerprints"]["replies"])
        records = []
        for row in probes:
            generated = generate(
                model, tokenizer, row.user,
                max_new_tokens=120, temperature=0, top_k=12,
            )
            records.append({
                "prompt": row.user,
                "expected": row.assistant,
                "generated": generated,
                "exact_match": generated.strip() == row.assistant,
                "copied_training_reply": reply_key(generated) in training_replies,
                "complete_two_clause_shape": (
                    generated.count("\n") == 1
                    and all(part.strip() for part in generated.split("\n"))
                ),
            })
        reports.append({
            "model": name,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": metadata["epoch"],
            "test_metrics": score,
            "exact_match_count": sum(r["exact_match"] for r in records),
            "copied_training_count": sum(r["copied_training_reply"] for r in records),
            "complete_two_clause_shape_count": sum(
                r["complete_two_clause_shape"] for r in records
            ),
            "test_generations": records,
        })

    report = {
        "experiment": "papiti_v0_6_6_predeclared_sealed_fluency_transfer",
        "probe_pairs": [[5, 0], [0, 4]],
        "test_count": len(probes),
        "models": reports,
        "human_review_required": True,
        "limitations": [
            "This is the first evaluation of newly predeclared style pairs.",
            "The halves are already present in training; this tests new joins, not unrestricted chat.",
            "Shape/low NLL/low copying do NOT demonstrate grammatical fluency.",
            "Once inspected, do not use these 36 prompts or answers as optimization targets.",
            "Checkpoint selection uses only the original validation set.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report": str(args.report),
        "test_count": len(probes),
        "summary": [
            {key: item[key] for key in (
                "model", "checkpoint_epoch", "test_metrics", "exact_match_count",
                "copied_training_count", "complete_two_clause_shape_count",
            )}
            for item in reports
        ],
        "human_review_required": True,
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="One-time sealed Papiti fluency exam")
    parser.add_argument("--baseline", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_5_ib.pt"))
    parser.add_argument("--candidate", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_6_fluency.pt"))
    parser.add_argument("--control", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_6_fluency_control.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/papiti_v066_sealed_fluency_eval.json"))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
