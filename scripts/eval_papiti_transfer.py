"""Diagnostic transfer exam for the staged Papiti v0.6.3 language curriculum.

The 36 examples combine *previously taught parts* in untrained pairs. This
probe is separate from the existing 94-pair test and must not be used as
training data. Scores do not certify natural language competence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_transfer(checkpoint: Path, report_path: Path, batch_size: int = 24) -> dict:
    from hybrid_speech import reply_key
    from papiti_lm.cli import _evaluate, _torch
    from papiti_lm.model import generate, load_checkpoint
    from papiti_lm.inquiry_curriculum import add_inquiry_training
    from papiti_lm.language_curriculum import mixed_style_transfer_probes
    from papiti_lm.v0_6_1_prose import create_prose_corpus

    _torch()
    model, tokenizer, metadata = load_checkpoint(checkpoint)
    if not (metadata.get("shared_prose") and metadata.get("inquiry_curriculum")
            and metadata.get("language_curriculum")):
        raise ValueError("the transfer exam requires a v0.6.3 language-curriculum checkpoint")
    corpus = add_inquiry_training(create_prose_corpus(
        int(metadata["seed"]), repeats=int(metadata["prose_repeats"])
    ))
    probes = mixed_style_transfer_probes(corpus)
    sequences = [tokenizer.sequence(row, model.config.context) for row in probes]
    score = _evaluate(
        model, sequences, batch_size,
        sum(len(row.assistant) + 1 for row in probes),
    )
    trained = set(metadata["prose_fingerprints"]["replies"])
    records = []
    for row in probes:
        answer = generate(
            model, tokenizer, row.user,
            max_new_tokens=120, temperature=0, top_k=12,
        )
        records.append({
            "purpose": row.group,
            "prompt": row.user,
            "expected": row.assistant,
            "generated": answer,
            "exact_match": answer.strip() == row.assistant,
            "matches_training_reply": reply_key(answer) in trained,
        })
    report = {
        "experiment": "papiti_v0_6_3_mixed_style_transfer",
        "checkpoint_epoch": metadata["epoch"],
        "test_count": len(records),
        "test_metrics": score,
        "test_generations": records,
        "release_ready": False,
        "limitations": [
            "Pairs are new, but all constituent utterance parts were seen in supervised training.",
            "The 36 examples test recombination, not arbitrary user requests or human-level grammar.",
            "All novel outputs require human grammar/meaning review before improvement claims.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "experiment": report["experiment"],
        "checkpoint_epoch": report["checkpoint_epoch"],
        "test_count": report["test_count"],
        "test_metrics": score,
        "exact_match_count": sum(row["exact_match"] for row in records),
        "copied_training_count": sum(row["matches_training_reply"] for row in records),
        "report": str(report_path),
        "human_review_needed": True,
    }, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent mixed-style transfer exam")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_3_curriculum.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/papiti_v0_6_3_transfer_eval.json"))
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    evaluate_transfer(args.checkpoint, args.report, args.batch_size)


if __name__ == "__main__":
    main()
