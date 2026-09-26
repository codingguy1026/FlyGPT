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
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-margin", type=float, default=0.10)
    args = parser.parse_args()

    runtime = FlyRouterRuntime(args.model)
    total = 0
    correct = 0
    gate_pass = 0
    usable = 0

    with args.dataset.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            text_value = str(row["input"])
            expected = str(row["route"])
            pred = runtime.predict(text_value, top_k=3)
            actual = str(pred["route"])
            confidence = float(pred["confidence"])
            ranked = pred.get("top_routes") or []
            second = float(ranked[1]["confidence"]) if len(ranked) > 1 else 0.0
            margin = max(0.0, confidence - second)
            passes = confidence >= args.min_confidence and margin >= args.min_margin
            is_correct = actual == expected

            total += 1
            correct += int(is_correct)
            gate_pass += int(passes)
            usable += int(is_correct and passes)

            mark = "OK" if is_correct else "MISS"
            gate = "PASS" if passes else "HOLD"
            top = ", ".join(
                f"{item['route']}={float(item['confidence']):.1%}"
                for item in ranked
            )
            print(
                f"{mark:4} gate={gate:4} expected={expected:10} "
                f"predicted={actual:10} conf={confidence:.1%} margin={margin:.1%} "
                f"| {text_value} | {top}"
            )

    accuracy = correct / total if total else 0.0
    gate_rate = gate_pass / total if total else 0.0
    usable_rate = usable / total if total else 0.0

    print(f"\nraw_accuracy={accuracy:.1%} ({correct}/{total})")
    print(f"gate_pass_rate={gate_rate:.1%} ({gate_pass}/{total})")
    print(f"usable_accuracy={usable_rate:.1%} ({usable}/{total})")
    print(f"vectorizer={runtime.vectorizer_version} scaffold={runtime.scaffold_meta.get('kind', 'unknown')}")


if __name__ == "__main__":
    main()
