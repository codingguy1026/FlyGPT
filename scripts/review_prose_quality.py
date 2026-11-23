"""Export a human-review worksheet and score reviewed Papiti generations.

The existing 94 combination tests are a regression suite, not an independent
estimate of free-form chat ability. This script does not manufacture labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "index", "purpose", "prompt", "expected", "generated", "copied_training",
    "grammar_ok", "meaning_ok", "severe_fact_error", "notes",
)


def export(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, generation in enumerate(report["test_generations"], start=1):
            writer.writerow({
                "index": index,
                "purpose": generation.get("purpose", ""),
                "prompt": generation.get("prompt", ""),
                "expected": generation["expected"],
                "generated": generation["generated"],
                "copied_training": str(generation.get("matches_training_reply", "")),
                "grammar_ok": "",
                "meaning_ok": "",
                "severe_fact_error": "",
                "notes": "",
            })
    print(f"Review {len(report['test_generations'])} outputs in {output}.")
    print("Fill grammar_ok, meaning_ok, severe_fact_error with 1 or 0.")


def score(review_path: Path) -> None:
    with review_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Review CSV is empty")
    metrics = ("grammar_ok", "meaning_ok", "severe_fact_error")
    for row in rows:
        for metric in metrics:
            if row[metric] not in ("0", "1"):
                raise ValueError(
                    f"Unreviewed or invalid {metric} at row {row['index']}; "
                    "complete the human review before scoring"
                )
    count = len(rows)
    grammar = sum(row["grammar_ok"] == "1" for row in rows) / count
    meaning = sum(row["meaning_ok"] == "1" for row in rows) / count
    fact_errors = sum(row["severe_fact_error"] == "1" for row in rows)
    copied = sum(row["copied_training"] == "True" for row in rows)
    result = {
        "reviewed": count,
        "grammar_ok_rate": round(grammar, 4),
        "meaning_ok_rate": round(meaning, 4),
        "severe_fact_errors": fact_errors,
        "training_copy_rate": round(copied / count, 4),
        "provisional_targets_met": (
            grammar >= 0.90 and meaning >= 0.95
            and fact_errors == 0 and copied / count <= 0.40
        ),
        "caveat": "Manually scored combination-test results only; unseen conversational intents still require separate testing.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    export_cmd = actions.add_parser("export")
    export_cmd.add_argument("report", type=Path)
    export_cmd.add_argument("output", type=Path)
    score_cmd = actions.add_parser("score")
    score_cmd.add_argument("review_csv", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        export(args.report, args.output)
    else:
        score(args.review_csv)


if __name__ == "__main__":
    main()
