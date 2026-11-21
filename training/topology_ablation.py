"""Train matched MaleCNS and degree-preserving rewired-control task routers.

This study measures ROUTING, not end-to-end conversation ability. The existing
regression suites are exploratory and not a fully blind external benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_FILES = [
    "teacher_seed.jsonl", "curriculum_v0_3_4.jsonl",
    "curriculum_v0_3_5_short.jsonl", "curriculum_v0_3_6_context.jsonl",
    "curriculum_v0_3_7_yugeok.jsonl", "curriculum_v0_3_8_hard_negatives.jsonl",
    "curriculum_v0_3_9.jsonl", "curriculum_v0_5_realworld.jsonl",
]
EVAL_FILES = [
    "regression_v0_5_realworld.jsonl", "regression_v0_3_9_paraphrase.jsonl",
    "regression_v0_3_8_hard_negatives.jsonl",
]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def key_for(prompt):
    return " ".join(prompt.casefold().split())


def prepare_teacher(out):
    seen = {}
    original_count = 0
    for name in TRAIN_FILES:
        for row in load_jsonl(ROOT / "training" / name):
            original_count += 1
            key = key_for(row["input"])
            if key in seen and seen[key]["route"] != row["route"]:
                raise ValueError("Conflicting training labels for " + key)
            seen.setdefault(key, {"input": row["input"], "route": row["route"]})
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                           for row in seen.values()), encoding="utf-8")
    return set(seen), original_count - len(seen)


def filter_heldout(rows, teacher_keys):
    """Remove exact normalized training overlaps and repeated evaluation prompts."""
    seen, filtered = set(), []
    for row in rows:
        key = key_for(row["input"])
        if key not in teacher_keys and key not in seen:
            seen.add(key)
            filtered.append({"input": row["input"], "route": row["route"]})
    return filtered, len(rows) - len(filtered)


def degrees(rows):
    return (Counter(int(x["src"]) for x in rows),
            Counter(int(x["dst"]) for x in rows))


def rewire_scaffold(source, seed, swaps_per_edge=5):
    """Double-edge swaps keep individual in/out degree counts and weight multiset."""
    edges = [dict(x) for x in source["edges"]]
    original = {(int(e["src"]), int(e["dst"])) for e in edges}
    n_nodes = int(source["n_nodes"])
    if len(original) != len(edges) or any(
        u == v or not (0 <= u < n_nodes and 0 <= v < n_nodes)
        for u, v in original
    ):
        raise ValueError("Original scaffold must be a simple, loop-free digraph")
    base_degree, weights = degrees(edges), sorted(float(x["weight"]) for x in edges)
    pairs = set(original)
    rng = random.Random(seed)
    target = len(edges) * swaps_per_edge
    swaps = 0
    for _ in range(target * 50):
        if swaps >= target:
            break
        i, j = rng.sample(range(len(edges)), 2)
        a, b = int(edges[i]["src"]), int(edges[i]["dst"])
        c, d = int(edges[j]["src"]), int(edges[j]["dst"])
        if a == c or b == d or a == d or c == b:
            continue
        if (a, d) in pairs or (c, b) in pairs:
            continue
        pairs.difference_update(((a, b), (c, d)))
        pairs.update(((a, d), (c, b)))
        edges[i]["dst"], edges[j]["dst"] = d, b
        swaps += 1
    changed = len(pairs.symmetric_difference(original)) // 2
    if swaps != target or changed < len(edges) // 2:
        raise RuntimeError("Insufficient degree-preserving rewires")
    if degrees(edges) != base_degree or sorted(float(e["weight"]) for e in edges) != weights:
        raise AssertionError("Control changed degree counts or edge weights")
    control = dict(source)
    control.update({
        "edges": edges,
        "dataset": "degree-preserving-rewired-control",
        "source": "Randomized, non-anatomical control based on MaleCNS scaffold",
        "ablation": {"seed": seed, "swaps": swaps, "changed_pairs": changed},
    })
    return control


def measure(model, rows):
    total = Counter()
    hits = Counter()
    accepted = accepted_correct = 0
    for row in rows:
        prediction = model.predict(row["input"], top_k=2)
        correct = prediction["route"] == row["route"]
        total[row["route"]] += 1
        hits[row["route"]] += int(correct)
        accepted += int(prediction["accepted"])
        accepted_correct += int(prediction["accepted"] and correct)
    n = len(rows)
    return {
        "n": n,
        "accuracy": round(sum(hits.values()) / n, 5),
        "macro_recall": round(statistics.mean(hits[k] / count for k, count in total.items()), 5),
        "gate_coverage": round(accepted / n, 5),
        "gate_precision": round(accepted_correct / accepted, 5) if accepted else None,
        "counts_by_class": dict(total),
    }


def train(teacher, graph, out, seed, epochs, patience):
    cmd = [
        sys.executable, str(ROOT / "training/train_router.py"),
        "--dataset", str(teacher), "--scaffold", str(graph), "--out", str(out),
        "--seed", str(seed), "--epochs", str(epochs), "--patience", str(patience),
        "--vocab-size", "8192", "--vectorizer-version", "v6",
        "--lr", "0.0015", "--weight-decay", "0.0004", "--val-ratio", "0.22",
        "--label-smoothing", "0.04", "--margin", "0.30", "--margin-weight", "0.10",
        "--feature-dropout", "0.08", "--selection-metric", "balanced_accuracy",
        "--gate-target-precision", "0.97", "--gate-min-coverage", "0.55",
    ]
    outcome = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if outcome.returncode:
        raise RuntimeError("Training failed:\n" + outcome.stdout[-4500:]
                           + "\n" + outcome.stderr[-4500:])
    lines = [s for s in outcome.stdout.splitlines() if s.startswith(("saved=", "early_stop"))]
    return " | ".join(lines[-2:])


def run(args):
    from router_runtime import FlyRouterRuntime
    graph = json.loads(args.scaffold.read_text(encoding="utf-8"))
    if graph.get("dataset") != "male-cns:v1.0":
        raise ValueError("Expected genuine frozen MaleCNS v1.0 scaffold")
    with tempfile.TemporaryDirectory(prefix="flygpt-ablation-") as tmp:
        base = Path(tmp)
        teacher = base / "teacher.jsonl"
        keys, duplicates = prepare_teacher(teacher)
        eval_sets, overlap = {}, {}
        for file in EVAL_FILES:
            filtered, removed = filter_heldout(
                load_jsonl(ROOT / "training" / file), keys
            )
            overlap[file] = {"usable": len(filtered), "exact_overlap_removed": removed}
            print(f"OVERLAP {file}: usable={len(filtered)} removed={removed}", flush=True)
            if len(filtered) >= 3:
                eval_sets[file] = filtered
        if not eval_sets:
            raise ValueError("No usable evaluation examples after overlap removal")
        results = []
        for seed in args.seeds:
            control = rewire_scaffold(graph, seed + 10000)
            control_path = base / f"control_{seed}.json"
            control_path.write_text(json.dumps(control), encoding="utf-8")
            by_condition = {}
            for name, topology in (("MaleCNS", args.scaffold), ("rewired", control_path)):
                print(f"TRAIN seed={seed} condition={name}", flush=True)
                ckpt = base / f"{name}_{seed}.pt"
                training_summary = train(teacher, topology, ckpt, seed, args.epochs, args.patience)
                model = FlyRouterRuntime(ckpt)
                scores = {k: measure(model, v) for k, v in eval_sets.items()}
                by_condition[name] = {
                    "training": training_summary,
                    "validation_macro_recall": model.best_val_balanced_accuracy,
                    "scores": scores,
                }
                print(training_summary, flush=True)
                for label, data in scores.items():
                    print(f"EVAL {label}: n={data['n']} accuracy={data['accuracy']:.3f} macro_recall={data['macro_recall']:.3f}", flush=True)
            results.append({"seed": seed, "changed_pairs": control["ablation"]["changed_pairs"],
                            "conditions": by_condition})
        diffs = {}
        for label in eval_sets:
            delta = [
                case["conditions"]["MaleCNS"]["scores"][label]["macro_recall"] -
                case["conditions"]["rewired"]["scores"][label]["macro_recall"]
                for case in results
            ]
            diffs[label] = {"per_seed": [round(v, 5) for v in delta],
                            "mean_delta": round(statistics.mean(delta), 5)}
        return {
            "experiment": "MaleCNS vs degree-preserving rewired graph",
            "pilot": len(args.seeds) < 3,
            "dataset_sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
            "scaffold_sha256": hashlib.sha256(args.scaffold.read_bytes()).hexdigest(),
            "node_count": graph["n_nodes"], "edge_count": len(graph["edges"]),
            "duplicate_training_prompts_removed": duplicates,
            "seeds": args.seeds, "epochs": args.epochs, "patience": args.patience,
            "overlap": overlap, "results": results, "delta_macro_recall": diffs,
            "limitations": [
                "Existing project regression data, not independent blind testing",
                "Exact duplicates removed, but paraphrase leakage possible",
                "Degree sequence and weight multiset conserved; other motifs may differ",
                "Route classification only: no claim of chatbot reasoning",
                "Single-seed pilots are not statistical evidence of superiority",
            ],
            "production_checkpoint_unchanged": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold", type=Path, default=ROOT / "training/malecns_scaffold.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("/tmp/flygpt_topology_ablation.json"))
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be distinct")
    if args.epochs < 2 or args.patience < 1:
        parser.error("epochs must be >= 2 and patience >= 1")
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REPORT=" + str(args.out))
    print("DELTA=" + json.dumps(report["delta_macro_recall"], ensure_ascii=False))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        lines = ["## FlyGPT connectome topology ablation",
                 "", "Exploratory pilot; NOT evidence of biological reasoning.",
                 "", "| Existing regression suite | Delta macro recall (MaleCNS minus rewired) |",
                 "|---|---:|"]
        lines.extend(f"| {name} | {diff['mean_delta']:+.1%} |"
                     for name, diff in report["delta_macro_recall"].items())
        lines.extend(["", "Exact-overlap audit:", json.dumps(report["overlap"], indent=2)])
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as output:
            output.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
