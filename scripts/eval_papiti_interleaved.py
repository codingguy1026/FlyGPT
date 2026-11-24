"""One-time, previously unseen 36-pair v0.6.7 transfer exam.

Compare the existing v0.6.5 checkpoint, a scratch-trained full-replay
control, and the scratch-trained interleaved-clause candidate. Do NOT run
until both v0.6.7 arms finish; once inspected probes are no longer blind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(args):
    from hybrid_speech import reply_key
    from papiti_lm.cli import _evaluate
    from papiti_lm.inquiry_curriculum import add_inquiry_training
    from papiti_lm.interleaved_curriculum import sealed_v067_probes
    from papiti_lm.model import generate, load_checkpoint
    from papiti_lm.v0_6_1_prose import create_prose_corpus

    corpus = add_inquiry_training(create_prose_corpus(seed=42, repeats=2))
    probes = sealed_v067_probes(corpus)
    results = []
    checkpoints = (
        ("v065_reference", args.reference, "ib_curriculum"),
        ("v067_full_control", args.control, "interleaved_control"),
        ("v067_interleaved", args.candidate, "interleaved_clauses"),
    )
    for name, path, property_name in checkpoints:
        model, tokenizer, metadata = load_checkpoint(path)
        if not metadata.get(property_name):
            raise ValueError(f"{path} is not a compatible {name} checkpoint")
        if metadata.get("seed") != 42 or metadata.get("prose_repeats") != 2:
            raise ValueError("the evaluation requires the frozen seed-42 corpus")
        sequences = [tokenizer.sequence(x, model.config.context) for x in probes]
        score = _evaluate(
            model, sequences, 24, sum(len(x.assistant) + 1 for x in probes),
        )
        training_replies = set(metadata["prose_fingerprints"]["replies"])
        generations = []
        for probe in probes:
            generated = generate(
                model, tokenizer, probe.user,
                max_new_tokens=120, temperature=0, top_k=12,
            )
            generations.append({
                "prompt": probe.user,
                "expected": probe.assistant,
                "generated": generated,
                "exact_match": generated.strip() == probe.assistant,
                "copied_training_reply": reply_key(generated) in training_replies,
                "complete_two_clause_shape": (
                    generated.count("\n") == 1
                    and all(text.strip() for text in generated.split("\n"))
                ),
            })
        results.append({
            "model": name,
            "checkpoint_epoch": metadata.get("epoch"),
            "checkpoint": str(path),
            "test_metrics": score,
            "exact_match_count": sum(x["exact_match"] for x in generations),
            "copied_training_count": sum(x["copied_training_reply"] for x in generations),
            "complete_two_clause_shape_count": sum(
                x["complete_two_clause_shape"] for x in generations
            ),
            "test_generations": generations,
        })

    report = {
        "experiment": "papiti_v067_interleaved_blind_transfer",
        "probe_style_pairs": [[4, 2], [2, 5]],
        "test_count": len(probes),
        "models": results,
        "human_review_required": True,
        "limitations": [
            "36 unseen cue pairs, but opening/follow-up fragments are in training.",
            "Never train/tune on these diagnostics after opening their outputs.",
            "Copying, two-clause shape and Char NLL alone cannot prove fluency.",
            "Human review must score grammar, meaning, style, and novelty.",
            "The original 94 tests remain separate from this probe.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(args.report),
        "test_count": len(probes),
        "summary": [
            {k: item[k] for k in (
                "model", "checkpoint_epoch", "test_metrics",
                "exact_match_count", "copied_training_count",
                "complete_two_clause_shape_count",
            )}
            for item in results
        ],
        "human_review_required": True,
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="one-time v0.6.7 blind transfer test")
    parser.add_argument("--reference", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_5_ib.pt"))
    parser.add_argument("--control", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_7_full_control.pt"))
    parser.add_argument("--candidate", type=Path,
                        default=Path("artifacts/papiti_language_v0_6_7_interleaved.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/papiti_v067_blind_transfer_eval.json"))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
