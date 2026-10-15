"""Fair held-out comparison of unchanged v0.5 and role-aware MaleCNS routers.

No checkpoint is promoted and no speed/accuracy improvement is assumed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_runtime import FlyRouterRuntime


DEFAULT_REGRESSIONS = (
    Path("training/regression_v0_5_realworld.jsonl"),
    Path("training/regression_v0_3_9_paraphrase.jsonl"),
    Path("training/regression_v0_3_8_hard_negatives.jsonl"),
)


def load_cases(dataset: Path) -> list[tuple[str, str]]:
    rows = []
    for row_text in dataset.read_text(encoding="utf-8").splitlines():
        if row_text.strip():
            case = json.loads(row_text)
            rows.append((str(case["input"]), str(case["route"])))
    if not rows:
        raise ValueError(f"No cases found: {dataset}")
    return rows


def evaluate(runtime: FlyRouterRuntime, cases: list[tuple[str, str]]) -> dict:
    from collections import Counter

    class_total: dict[str, int] = Counter()
    class_correct: dict[str, int] = Counter()
    passed = 0
    passed_correct = 0
    times_ms = []
    for prompt, expected in cases:
        started = time.perf_counter()
        result = runtime.predict(prompt)
        times_ms.append(1000 * (time.perf_counter() - started))
        class_total[expected] += 1
        is_correct = result["route"] == expected
        class_correct[expected] += int(is_correct)
        is_accepted = bool(result["accepted"])
        passed += int(is_accepted)
        passed_correct += int(is_accepted and is_correct)
    count = len(cases)
    return {
        "cases": count,
        "accuracy": round(sum(class_correct.values()) / count, 6),
        "balanced_accuracy": round(
            sum(class_correct[c] / n for c, n in class_total.items()) / len(class_total), 6,
        ),
        "gate_coverage": round(passed / count, 6),
        "gate_precision": round(passed_correct / passed, 6) if passed else None,
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(sorted(times_ms)[min(len(times_ms) - 1, int(0.95 * len(times_ms)))], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", action="append", type=Path, dest="datasets")
    parser.add_argument("--out", type=Path, default=Path("artifacts/role_aware_comparison.json"))
    args = parser.parse_args()

    baseline = FlyRouterRuntime(args.baseline)
    candidate = FlyRouterRuntime(args.candidate)
    if baseline.role_aware or not candidate.role_aware:
        raise ValueError("Expected unchanged baseline and role-aware candidate checkpoints")
    if (
        baseline.n_nodes != candidate.n_nodes
        or baseline.steps != candidate.steps
        or baseline.routes != candidate.routes
        or baseline.vocab_size != candidate.vocab_size
        or baseline.vectorizer_version != candidate.vectorizer_version
        or baseline.scaffold_meta.get("root_ids") != candidate.scaffold_meta.get("root_ids")
        or not baseline._torch.equal(baseline.src, candidate.src)
        or not baseline._torch.equal(baseline.dst, candidate.dst)
        or not baseline._torch.equal(baseline.base_weight, candidate.base_weight)
    ):
        raise ValueError("Cannot compare checkpoints with different topology, route labels, or text features")

    datasets = args.datasets or list(DEFAULT_REGRESSIONS)
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "scaffold_nodes": baseline.n_nodes,
        "role_metadata": candidate.role_metadata,
        "datasets": {},
        "not_deployed": True,
        "note": "Accuracies and CPU timings must be interpreted with dataset size and hardware.",
    }
    for dataset in datasets:
        cases = load_cases(dataset)
        # Warm up both runtimes before timings, without including cold-start load.
        for model in (baseline, candidate):
            for prompt, _ in cases[:min(5, len(cases))]:
                model.predict(prompt)
        before = evaluate(baseline, cases)
        after = evaluate(candidate, cases)
        report["datasets"][str(dataset)] = {
            "baseline": before,
            "candidate": after,
            "delta_balanced_accuracy": round(
                after["balanced_accuracy"] - before["balanced_accuracy"], 6,
            ),
            "median_latency_ratio": (
                round(after["median_ms"] / before["median_ms"], 4)
                if before["median_ms"] > 0 else None
            ),
        }
        print(f"{dataset} balanced={before['balanced_accuracy']:.1%} -> "
              f"{after['balanced_accuracy']:.1%}, median latency "
              f"{before['median_ms']:.1f} -> {after['median_ms']:.1f} ms")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved=" + str(args.out) + " (requires human review before deployment)")


if __name__ == "__main__":
    main()
