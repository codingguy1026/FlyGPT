from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from router_runtime import FlyRouterRuntime


def load_cases(path: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("input"), str) or not isinstance(row.get("route"), str):
                raise ValueError(f"Invalid row at line {line_no}")
            cases.append({"input": row["input"], "route": row["route"]})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FlyGPT router on held-out regression cases")
    parser.add_argument("--model", type=Path, default=Path("artifacts/fly_router_v0_1.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("training/regression_v0_2.jsonl"))
    args = parser.parse_args()

    runtime = FlyRouterRuntime(args.model)
    cases = load_cases(args.dataset)

    correct = 0
    per_route_total: Counter[str] = Counter()
    per_route_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    print(
        f"model={args.model} scaffold={runtime.scaffold_meta.get('kind', 'unknown')} "
        f"nodes={runtime.n_nodes} cases={len(cases)}"
    )

    for i, case in enumerate(cases, 1):
        result = runtime.predict(case["input"])
        expected = case["route"]
        predicted = result["route"]
        confidence = result["confidence"]

        per_route_total[expected] += 1
        confusion[expected][predicted] += 1
        ok = predicted == expected
        if ok:
            correct += 1
            per_route_correct[expected] += 1

        marker = "PASS" if ok else "FAIL"
        print(
            f"[{marker}] {i:02d} expected={expected:<9} predicted={predicted:<9} "
            f"confidence={confidence:6.1%} | {case['input']}"
        )

    accuracy = correct / len(cases) if cases else 0.0
    print(f"\noverall={correct}/{len(cases)} accuracy={accuracy:.3f}")

    print("\nper-route:")
    for route in sorted(per_route_total):
        route_correct = per_route_correct[route]
        route_total = per_route_total[route]
        print(f"  {route:<9} {route_correct}/{route_total} = {route_correct / route_total:.3f}")

    failures = [
        (expected, predicted, count)
        for expected, row in confusion.items()
        for predicted, count in row.items()
        if expected != predicted and count
    ]
    if failures:
        print("\nconfusions:")
        for expected, predicted, count in sorted(failures):
            print(f"  {expected} -> {predicted}: {count}")


if __name__ == "__main__":
    main()
