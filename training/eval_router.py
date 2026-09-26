from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_runtime import FlyRouterRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FlyGPT router checkpoint on JSONL examples")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    runtime = FlyRouterRuntime(args.model)
    total = 0
    correct = 0

    with args.dataset.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row["input"])
            expected = str(row["route"])
            pred = runtime.predict(text)
            actual = str(pred["route"])
            confidence = float(pred["confidence"])
            total += 1
            correct += int(actual == expected)
            mark = "OK" if actual == expected else "MISS"
            print(f"{mark:4} expected={expected:10} predicted={actual:10} conf={confidence:.1%} | {text}")

    accuracy = correct / total if total else 0.0
    print(f"\naccuracy={accuracy:.1%} ({correct}/{total})")


if __name__ == "__main__":
    main()
