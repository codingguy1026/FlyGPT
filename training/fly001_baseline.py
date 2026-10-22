"""FLY-001: reproducible, read-only baseline for the frozen FlyGPT router.

This measures raw network predictions and the production semantic route guard.
It does NOT call app.py, mutate any checkpoints, personalize a user, train,
access neuPrint, or evaluate downstream LLM answer quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_runtime import FlyRouterRuntime
from semantic_route_guard import apply_semantic_route_override


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path: Path, routes: list[str]) -> list[dict[str, str]]:
    """Validate the benchmark strictly; never silently drop invalid examples."""
    known = set(routes)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            message, route = item.get("input"), item.get("route")
            if not isinstance(message, str) or not message.strip():
                raise ValueError(f"{path}:{line_number}: input must be nonempty")
            if not isinstance(route, str) or route not in known:
                raise ValueError(
                    f"{path}:{line_number}: invalid route {route!r}; valid: {sorted(known)}"
                )
            normalized = " ".join(message.casefold().split())
            if normalized in seen:
                raise ValueError(f"{path}:{line_number}: duplicate input {message!r}")
            seen.add(normalized)
            rows.append({"input": message, "route": route})
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def classification_metrics(
    cases: list[dict[str, Any]], stage: str, labels: list[str],
) -> dict[str, Any]:
    """Accuracy ignores gates; usable_rate additionally requires acceptance."""
    correct = accepted = accepted_correct = 0
    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }
    expected_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()

    for case in cases:
        expected = case["expected"]
        value = case[stage]
        predicted = value["route"]
        passed = bool(value["accepted"])
        is_correct = expected == predicted
        expected_counts[expected] += 1
        predicted_counts[predicted] += 1
        confusion[expected][predicted] += 1
        correct += int(is_correct)
        accepted += int(passed)
        accepted_correct += int(is_correct and passed)

    by_route: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = confusion[label][label]
        n_expected = expected_counts[label]
        n_predicted = predicted_counts[label]
        precision = tp / n_predicted if n_predicted else 0.0
        recall = tp / n_expected if n_expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_route[label] = {
            "support": n_expected,
            "predicted": n_predicted,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": _rate(correct, len(cases)),
        "macro_f1": round(statistics.mean(r["f1"] for r in by_route.values()), 6),
        "accepted": accepted,
        "coverage": _rate(accepted, len(cases)),
        "accepted_correct": accepted_correct,
        "accepted_precision": _rate(accepted_correct, accepted),
        "usable_rate": _rate(accepted_correct, len(cases)),
        "rejected": len(cases) - accepted,
        "by_route": by_route,
        "confusion": confusion,
    }


def latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    # Nearest-rank p95, stable and meaningful for a small diagnostic corpus.
    p95_idx = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[p95_idx], 3),
        "max_ms": round(ordered[-1], 3),
    }


def evaluate(
    runtime: Any, examples: list[dict[str, str]], warmup: int = 2,
) -> dict[str, Any]:
    """Do not change frozen model weights, temperature, or decision gates."""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    from brain_runtime import FlyBrainRuntime
    brain = FlyBrainRuntime()

    for _ in range(warmup):
        runtime.predict(examples[0]["input"], top_k=len(runtime.routes))

    records: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    guard_ms: list[float] = []

    for index, example in enumerate(examples, 1):
        start = time.perf_counter()
        raw = runtime.predict(example["input"], top_k=len(runtime.routes))
        model_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        effective = apply_semantic_route_override(
            example["input"], raw, brain=brain
        )
        override_ms = (time.perf_counter() - start) * 1000
        for stage, item in (("raw", raw), ("effective", effective)):
            if item.get("route") not in runtime.routes:
                raise ValueError(f"case {index}: {stage} returned invalid route")
        source = effective.get("semantic_override") or {}
        records.append({
            "id": index,
            "input": example["input"],
            "expected": example["route"],
            "raw": {
                "route": raw["route"],
                "accepted": bool(raw["accepted"]),
                "confidence": round(float(raw["confidence"]), 6),
                "margin": round(float(raw["margin"]), 6),
            },
            "effective": {
                "route": effective["route"],
                "accepted": bool(effective["accepted"]),
                "confidence": round(float(effective["confidence"]), 6),
                "margin": round(float(effective["margin"]), 6),
                "override": bool(source),
                "override_source": source.get("source"),
                "speech_act": source.get("speech_act"),
            },
            "latency_ms": {
                "inference": round(model_ms, 3),
                "semantic_guard": round(override_ms, 3),
            },
        })
        inference_ms.append(model_ms)
        guard_ms.append(override_ms)

    return {
        "raw": classification_metrics(records, "raw", runtime.routes),
        "effective": classification_metrics(records, "effective", runtime.routes),
        "overrides": {
            "total": sum(int(row["effective"]["override"]) for row in records),
            "corrected": sum(
                int(row["effective"]["override"] and
                    row["effective"]["route"] == row["expected"] and
                    row["raw"]["route"] != row["expected"])
                for row in records
            ),
        },
        "latency": {
            "inference": latency_summary(inference_ms),
            "semantic_guard": latency_summary(guard_ms),
        },
        "cases": records,
    }


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_report(
    model: Path, dataset: Path, runtime: Any,
    evaluation: dict[str, Any], warmup: int,
) -> dict[str, Any]:
    return {
        "schema_version": "fly001-baseline-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "offline_router_and_semantic_guard_no_personalization",
        "limitations": [
            "Not an end-to-end chat, LLM, neuPrint, or reasoning benchmark.",
            "Diagnostic cases are public and not guaranteed independent from training paraphrases.",
            "Latency excludes model load and reflects only the current CPU and process.",
        ],
        "provenance": {
            "git_commit": git_head(),
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "python": platform.python_version(),
            "torch": str(runtime._torch.__version__),
        },
        "model": {
            "name": runtime.model_path.name,
            "routes": list(runtime.routes),
            "n_nodes": runtime.n_nodes,
            "n_edges": len(runtime.src),
            "steps": runtime.steps,
            "vectorizer_version": runtime.vectorizer_version,
            "scaffold_kind": runtime.scaffold_meta.get("kind", "unknown"),
            "temperature": runtime.temperature,
            "min_confidence": runtime.min_confidence,
            "min_margin": runtime.min_margin,
            "checkpoint_best_val_accuracy": runtime.best_val_accuracy,
        },
        "benchmark": {
            "dataset_size": evaluation["raw"]["total"],
            "label_counts": dict(sorted(Counter(
                case["expected"] for case in evaluation["cases"]
            ).items())),
            "warmup_iterations": warmup,
            "raw": evaluation["raw"],
            "effective": evaluation["effective"],
            "semantic_overrides": evaluation["overrides"],
            "latency_ms": evaluation["latency"],
        },
        "cases": evaluation["cases"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FLY-001 frozen-router baseline")
    parser.add_argument("--model", type=Path, required=True,
                        help="Exact checkpoint to measure (prefer the running app's FLYGPT_MODEL_PATH)")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "training/fly001_diagnostic.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/fly001_baseline.json")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}")

    runtime = FlyRouterRuntime(args.model)
    cases = load_cases(args.dataset, runtime.routes)
    measured = evaluate(runtime, cases, warmup=args.warmup)
    report = build_report(args.model, args.dataset, runtime, measured, args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for stage in ("raw", "effective"):
        metrics = measured[stage]
        print(
            f"{stage}: accuracy={metrics['accuracy']:.1%} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"coverage={metrics['coverage']:.1%} "
            f"usable={metrics['usable_rate']:.1%}"
        )
    print(f"semantic_overrides={measured['overrides']['total']}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
