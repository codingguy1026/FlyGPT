"""Compare two frozen Papiti prose evaluation reports without training-data leakage.

This comparison is diagnostic only. Novelty and surface-form heuristics do not
establish semantic correctness; human review is still required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    generations = report["test_generations"]
    n = len(generations)
    if not n:
        raise ValueError(f"Empty evaluation report: {path}")
    copied = sum(row.get("matches_training_reply") is True for row in generations)
    novel = sum(row.get("matches_training_reply") is False for row in generations)
    malformed = 0
    mixed_language = 0
    for row in generations:
        result = row.get("generated", "")
        expected = row.get("expected", "")
        if (not result.strip() or len(result) > 500
                or re.search(r"\b(\w+)(?:\s+\1){2,}\b", result, flags=re.I)):
            malformed += 1
        # A Korean target with Latin chatter, or an English target with Hangul.
        # Domain identifiers like FlyGPT/MaleCNS in verified fact plans are
        # intentionally not flagged for Korean targets.
        if re.search(r"[가-힣]", expected) and re.search(r"[가-힣]", result):
            if re.search(r"\b(?:What|How|You|Can|The|Quick)\b", result):
                mixed_language += 1
        elif not re.search(r"[가-힣]", expected) and re.search(r"[가-힣]", result):
            mixed_language += 1
    return {
        "report": str(path),
        "test_count": n,
        "char_nll": report["test_metrics"]["char_nll"],
        "training_reply_matches": copied,
        "nonmatching_training_reply": novel,
        "surface_malformed_warning": malformed,
        "unexpected_language_warning": mixed_language,
        "note": "Heuristic warnings are not a grammar or meaning score.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    before, after = summarize(args.baseline), summarize(args.candidate)
    if before["test_count"] != after["test_count"]:
        raise ValueError("Different evaluation set sizes. Do not compare unpaired reports.")
    print(json.dumps({"baseline": before, "candidate": after}, indent=2, ensure_ascii=False))
    print("Review all novel responses manually before claiming an improvement.")


if __name__ == "__main__":
    main()
