"""Evaluate a saved Papiti prose checkpoint without training or changing weights.

Usage:
  .venv-train/bin/python -m papiti_lm.prose_audit \
    --checkpoint artifacts/papiti_language_v0_6_1_prose_best_backup.pt
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from papiti_lm.model import generate, load_checkpoint
from papiti_lm.v0_6_1_prose import prose_packet


def audit(checkpoint: str, *, samples: int = 5, temperatures: tuple[float, ...] = (0.0, 0.3, 0.65),
          output: str = "artifacts/papiti_prose_audit.json") -> dict:
    if samples < 1 or samples > 100:
        raise ValueError("samples must be between 1 and 100")
    model, tokenizer, metadata = load_checkpoint(checkpoint)
    if metadata.get("version") != "v0.6.1" or metadata.get("shared_prose") is not True:
        raise ValueError("expected a v0.6.1 shared-prose checkpoint")
    fingerprints = metadata.get("prose_fingerprints") or {}
    packets = sorted((fingerprints.get("parts") or {}).keys())
    if not packets:
        raise ValueError("checkpoint has no prose packet fingerprints")
    training_replies = set(fingerprints.get("replies") or [])
    from hybrid_speech import reply_key

    records = []
    for packet_index, packet in enumerate(packets):
        prompt = prose_packet(packet, 0, 0)
        for temperature in temperatures:
            for sample_index in range(samples):
                answer = generate(
                    model, tokenizer, prompt, max_new_tokens=96,
                    temperature=temperature, top_k=8,
                    seed=42 + packet_index * 10000 + round(temperature * 1000) * 100 + sample_index,
                )
                first, separator, second = answer.partition("\n")
                first_keys, second_keys = (set(group) for group in fingerprints["parts"][packet])
                records.append({
                    "packet": packet,
                    "temperature": temperature,
                    "sample": sample_index + 1,
                    "answer": answer,
                    "has_two_parts": bool(separator and first.strip() and second.strip()),
                    "training_reply_match": reply_key(answer.replace("\n", " ")) in training_replies
                        or reply_key(answer) in training_replies,
                    "first_part_match": reply_key(first) in first_keys,
                    "second_part_match": reply_key(second) in second_keys if separator else False,
                    "contains_latin": bool(re.search(r"[A-Za-z]", answer)),
                })
    summaries = []
    for temperature in temperatures:
        group = [row for row in records if row["temperature"] == temperature]
        counts = Counter((row["packet"], row["answer"]) for row in group)
        summaries.append({
            "temperature": temperature,
            "total": len(group),
            "two_part_rate": round(sum(row["has_two_parts"] for row in group) / len(group), 4),
            "training_reply_match_rate": round(sum(row["training_reply_match"] for row in group) / len(group), 4),
            "both_parts_match_rate": round(sum(row["first_part_match"] and row["second_part_match"] for row in group) / len(group), 4),
            "latin_character_rate": round(sum(row["contains_latin"] for row in group) / len(group), 4),
            "duplicate_within_packet_rate": round(sum(count - 1 for count in counts.values()) / len(group), 4),
        })
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "model_version": metadata["version"],
        "packet_count": len(packets),
        "samples_per_packet_per_temperature": samples,
        "note": "Heuristic diagnostics only; not a human fluency score or full pipeline acceptance test.",
        "summary": summaries,
        "records": records,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/papiti_language_v0_6_1_prose_best_backup.pt")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.3, 0.65])
    parser.add_argument("--output", default="artifacts/papiti_prose_audit.json")
    args = parser.parse_args()
    result = audit(args.checkpoint, samples=args.samples,
                   temperatures=tuple(args.temperatures), output=args.output)
    print(f"Evaluated {result['packet_count']} plans; report: {args.output}")
    for row in result["summary"]:
        print(f"T={row['temperature']:.2f}: two-part={row['two_part_rate']:.0%}, "
              f"training-match={row['training_reply_match_rate']:.0%}, "
              f"duplicate={row['duplicate_within_packet_rate']:.0%}, "
              f"Latin={row['latin_character_rate']:.0%}")


if __name__ == "__main__":
    main()
