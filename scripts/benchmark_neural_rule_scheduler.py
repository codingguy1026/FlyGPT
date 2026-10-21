"""Reproducible 100-case MaleCNS rule-order ablation.

Run from repository root: python scripts/benchmark_neural_rule_scheduler.py
Measures solver-only time separately from router inference time.
No synthetic router scores are used: loads the actual configured checkpoint.
"""
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from router_runtime import FlyRouterRuntime


def load_router():
    override = os.environ.get("FLYGPT_MODEL_PATH")
    candidates = ([Path(override)] if override else [
        ROOT / "artifacts" / f"fly_router_malecns_v0_{v}.pt"
        for v in (5, 4)
    ] + [
        ROOT / "artifacts" / f"fly_router_v0_3_{v}.pt"
        for v in range(9, 0, -1)
    ] + [ROOT / "artifacts" / "fly_router_v0_3.pt"])
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("No router checkpoint found. Set FLYGPT_MODEL_PATH to its path.")
    return FlyRouterRuntime(path)

from chain_reasoner import solve_chain


def cases(n=100, seed=20261021):
    rng = random.Random(seed)
    result = []
    # 50 provable, 50 not derivable; reproducible varied rule order.
    for index in range(n):
        affirmative = index % 2 == 0
        rules = ["A이면 B이다", "B이면 C이다", "C와 D가 모두 참이면 E가 참이다"]
        if affirmative:
            rules.append("B이면 D이다")
        else:
            rules.append("F이면 D이다")
        rng.shuffle(rules)
        query = ". ".join(rules + ["A는 참이다", "E는 참인가?"])
        result.append((query, affirmative))
    return result


def measure(router, dataset):
    rows = []
    for query, expected in dataset:
        t0 = time.perf_counter_ns()
        route = router.predict(query, top_k=len(router.routes))
        route_ms = (time.perf_counter_ns()-t0) / 1e6
        t1 = time.perf_counter_ns()
        baseline = solve_chain(query)
        baseline_ms = (time.perf_counter_ns()-t1) / 1e6
        t2 = time.perf_counter_ns()
        neural = solve_chain(query, neural_route=route)
        neural_ms = (time.perf_counter_ns()-t2) / 1e6
        assert baseline is not None and neural is not None, query
        rows.append({
            "expected": expected,
            "baseline": baseline.evidence["entailed"],
            "neural": neural.evidence["entailed"],
            "baseline_checks": baseline.evidence["rule_checks"],
            "pruned": baseline.evidence["pruned_rules"],
            "neural_checks": neural.evidence["rule_checks"],
            "rule_order_changed": baseline.evidence["neural_selection"]["rule_order"] != neural.evidence["neural_selection"]["rule_order"],
            "router_accepted": bool(route.get("accepted")),
            "route_ms": route_ms,
            "baseline_solver_ms": baseline_ms,
            "neural_solver_ms": neural_ms,
        })
    return rows


def summarize(rows):
    def mean(key):
        return round(statistics.mean(row[key] for row in rows), 4)
    n = len(rows)
    return {
        "cases": n,
        "baseline_accuracy": sum(r["baseline"] == r["expected"] for r in rows) / n,
        "neural_accuracy": sum(r["neural"] == r["expected"] for r in rows) / n,
        "same_conclusions": sum(r["baseline"] == r["neural"] for r in rows),
        "router_accepted": sum(r["router_accepted"] for r in rows),
        "rule_order_changed": sum(r["rule_order_changed"] for r in rows),
        "average_baseline_rule_checks": mean("baseline_checks"),
        "average_goal_pruned_rules": mean("pruned"),
        "average_neural_rule_checks": mean("neural_checks"),
        "average_router_ms": mean("route_ms"),
        "average_baseline_solver_ms": mean("baseline_solver_ms"),
        "average_neural_solver_ms": mean("neural_solver_ms"),
        "limitations": [
            "This is a route-score-based ordering heuristic, not a trained rule-selection model.",
            "Goal-driven pruning is deterministic and applies to both arms, so it is not a MaleCNS gain.",
            "Rule order alone does not guarantee fewer rule checks; model has no learned rule-selection head.",
            "Measured inference time includes Python overhead; results depend on machine and checkpoint.",
        ],
    }


def main():
    router = load_router()
    dataset = cases()
    report = summarize(measure(router, dataset))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["baseline_accuracy"] == report["neural_accuracy"] == 1.0
    assert report["same_conclusions"] == report["cases"]


if __name__ == "__main__":
    main()
