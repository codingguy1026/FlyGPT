"""Research-only natural supervised-learning curves for FlyGPT.

- No memorized evaluation answers, handcrafted route shortcuts or LLM answer labels.
- Group similar training examples *before* train/validation/test splitting.
- Nested training subsets, one fixed validation/test set per random seed.
- Selection based ONLY on validation. The outer held-out test is scored once.
- Do not read or tune against previously inspected 300-question diagnostics.
- Model topology stays the authentic subset of MaleCNS in the biological arm.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from training.topology_ablation import prepare_teacher, load_jsonl, rewire_scaffold
from training.fresh_threeway import text_only_model
from training.residual_connectome_study import ResidualFlyGraphRouter
from training.train_router import (
    Example, batchify, load_scaffold,
    _accuracy_from_logits, _balanced_accuracy_from_logits,
)


def shingles(text: str) -> set[str]:
    # Input-only heuristic grouping, deliberately independent of the labels.
    import re
    normalized = re.sub(r"\s+", "", text.strip().casefold())
    if len(normalized) <= 3:
        return {normalized}
    return {normalized[i:i+3] for i in range(len(normalized)-2)}


def group_near_duplicates(examples: list[Example], threshold: float = 0.86):
    """Cluster very similar questions across all routes before any split.

    Semantic near-duplicates below this threshold can still leak. This is
    mitigated, not a guarantee of independent linguistic families.
    """
    n = len(examples)
    parent = list(range(n))
    grams = [shingles(x.text) for x in examples]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n):
        a = grams[i]
        for j in range(i+1, n):
            b = grams[j]
            if min(len(a), len(b)) == 0:
                continue
            # Reject impossibly different sets before computing intersections.
            if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            intersection = len(a & b)
            overlap = intersection / (len(a)+len(b)-intersection)
            if overlap >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


def grouped_three_way_split(examples: list[Example], seed: int):
    """Deterministic group-stratified ~70/15/15 split.

    Groups are assigned according to majority route; a minority cross-route
    cluster stays intact even when distribution cannot be perfectly balanced.
    """
    groups = group_near_duplicates(examples)
    by_route = defaultdict(list)
    for group in groups:
        major = sorted(Counter(examples[i].route for i in group).items(),
                       key=lambda item: (-item[1], item[0]))[0][0]
        by_route[major].append(group)

    rng = random.Random(seed)
    train_ids, val_ids, test_ids = [], [], []
    for route in sorted(by_route):
        clusters = by_route[route][:]
        rng.shuffle(clusters)
        if len(clusters) < 4:
            raise ValueError(f"Not enough independent {route} clusters for splitting")
        # Strict separate validation & test allocations by distinct clusters.
        n_test = max(1, round(len(clusters)*0.15))
        n_val = max(1, round(len(clusters)*0.15))
        test_ids.extend(i for group in clusters[:n_test] for i in group)
        val_ids.extend(i for group in clusters[n_test:n_test+n_val] for i in group)
        train_ids.extend(i for group in clusters[n_test+n_val:] for i in group)

    a, b, c = set(train_ids), set(val_ids), set(test_ids)
    if a & b or a & c or b & c or a | b | c != set(range(len(examples))):
        raise AssertionError("Group split overlap or omission")
    for items in (train_ids, val_ids, test_ids):
        if set(x.route for x in (examples[i] for i in items)) != {
            x.route for x in examples
        }:
            raise ValueError("Split lacks at least one class")
    return (
        [examples[i] for i in train_ids],
        [examples[i] for i in val_ids],
        [examples[i] for i in test_ids],
        {"n_clusters": len(groups),
         "largest_cluster": max(len(g) for g in groups),
         "train": len(train_ids), "validation": len(val_ids), "test": len(test_ids),
         "test_class_counts": dict(Counter(examples[i].route for i in test_ids))},
    )


def ordered_nested_training(train: list[Example], seed: int):
    """Shuffle within route exactly once so subsets are nested, not cherry-picked."""
    by_class = defaultdict(list)
    for row in train:
        by_class[row.route].append(row)
    rng = random.Random(seed + 100000)
    for group in by_class.values():
        rng.shuffle(group)
    return by_class


def fraction_training(by_class, fraction: float):
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1]")
    subset = []
    for label in sorted(by_class):
        group = by_class[label]
        count = max(2, round(fraction * len(group)))
        subset.extend(group[:min(count, len(group))])
    return subset


def metrics(logits: torch.Tensor, targets: torch.Tensor, routes: list[str]) -> dict:
    pred = logits.argmax(dim=1)
    confusion = Counter()
    true_counts = Counter()
    correct_counts = Counter()
    for exp, got in zip(targets.tolist(), pred.tolist()):
        expected, actual = routes[exp], routes[got]
        confusion[(expected, actual)] += 1
        true_counts[expected] += 1
        correct_counts[expected] += int(exp == got)
    return {
        "n": int(len(targets)),
        "accuracy": round(_accuracy_from_logits(logits, targets), 6),
        "macro_recall": round(_balanced_accuracy_from_logits(
            logits, targets, len(routes)), 6),
        "class_recall": {
            key: round(correct_counts[key] / true_counts[key], 5)
            for key in sorted(true_counts)
        },
        "confusion": [
            {"expected": exp, "predicted": got, "count": count}
            for (exp, got), count in sorted(confusion.items())
        ],
    }


def train_and_score(train, val, test, src, dst, weights,
                    condition: str, seed: int, epochs: int, patience: int):
    """All models have the same text representation and training hyperparams.

    Checkpoint selection uses validation only; *test* is evaluated once
    after training ends and cannot affect model choice.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    classes = sorted({x.route for x in train + val + test})
    label_ids = {name: i for i, name in enumerate(classes)}
    vocab, nodes = 8192, 256
    train_x, train_y = batchify(train, label_ids, vocab, "v6")
    val_x, val_y = batchify(val, label_ids, vocab, "v6")
    test_x, test_y = batchify(test, label_ids, vocab, "v6")

    if condition == "text_only":
        model = text_only_model(vocab, nodes, len(classes))
    elif condition in ("male_cns", "rewired"):
        model = ResidualFlyGraphRouter(vocab, nodes, len(classes), src, dst,
                                       weights, steps=3, initial_graph_share=0.20)
    else:
        raise ValueError("Unexpected condition: " + condition)

    class_counts = torch.bincount(train_y, minlength=len(classes)).float()
    class_weight = class_counts.sum() / (len(classes) * class_counts.clamp_min(1))
    loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0004)
    best, best_acc, best_epoch, stale, best_state = -1.0, -1.0, 0, 0, None
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_noisy = F.dropout(train_x, p=0.08, training=True)
        logits = model(train_noisy)
        target_scores = logits.gather(1, train_y[:, None]).squeeze(1)
        not_true = F.one_hot(train_y, len(classes)).bool()
        runner_up = logits.masked_fill(not_true, float("-inf")).max(1).values
        loss = loss_fn(logits, train_y) + 0.10 * F.relu(
            0.30 - target_scores + runner_up).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            held_logits = model(val_x)
            bal = _balanced_accuracy_from_logits(held_logits, val_y, len(classes))
            val_acc = _accuracy_from_logits(held_logits, val_y)
        if bal > best + 1e-9 or (abs(bal-best) < 1e-9 and val_acc > best_acc):
            best, best_acc, best_epoch, stale = bal, val_acc, epoch, 0
            best_state = {name: t.detach().cpu().clone()
                          for name, t in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint learned")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # Evaluation is performed ONCE per trained model, after all training.
        seen_logits = model(train_x)
        validation_logits = model(val_x)
        test_logits = model(test_x)
    return {
        "train_size": len(train),
        "selected_epoch": best_epoch,
        "train": metrics(seen_logits, train_y, classes),
        "validation": metrics(validation_logits, val_y, classes),
        "test": metrics(test_logits, test_y, classes),
        "learned_graph_share": (
            round(float(model.graph_share().item()), 6)
            if condition != "text_only" else None
        ),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }


def run(args):
    torch.set_num_threads(2)
    source = json.loads(args.scaffold.read_text(encoding="utf-8"))
    if source.get("dataset") != "male-cns:v1.0":
        raise ValueError("Expected authentic male-cns:v1.0 scaffold")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")
    fractions = args.fractions
    if not fractions or sorted(set(fractions)) != fractions or fractions[-1] != 1.0:
        raise ValueError("Fractions must be unique, increasing, ending at 1.0")

    with tempfile.TemporaryDirectory(prefix="flygpt-learning-") as work:
        teacher = Path(work) / "training.jsonl"
        _, dropped = prepare_teacher(teacher)
        corpus_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
        data = [Example(item["input"], item["route"]) for item in load_jsonl(teacher)]
        results = []
        original_nodes, src, dst, weights, meta = load_scaffold(args.scaffold, 256, 42)
        if original_nodes != 256:
            raise ValueError("Pre-registered scaffold requires 256 nodes")
        for seed in args.seeds:
            train, val, test, audit = grouped_three_way_split(data, seed)
            pool = ordered_nested_training(train, seed)
            control = rewire_scaffold(source, seed+10000)
            control_path = Path(work) / f"rewired_{seed}.json"
            control_path.write_text(json.dumps(control), encoding="utf-8")
            _, shuffled_src, shuffled_dst, shuffled_weights, _ = load_scaffold(
                control_path, 256, seed)
            row = {"seed": seed, "split": audit, "fractions": []}
            for frac in fractions:
                subset = fraction_training(pool, frac)
                scores = {}
                for condition in ("male_cns", "rewired", "text_only"):
                    edge_data = ((src, dst, weights)
                                 if condition != "rewired"
                                 else (shuffled_src, shuffled_dst, shuffled_weights))
                    result = train_and_score(
                        subset, val, test, *edge_data, condition, seed,
                        args.epochs, args.patience)
                    scores[condition] = result
                    print(
                        f"RESULT seed={seed} fraction={frac:.2f} "
                        f"model={condition} train={len(subset)} "
                        f"train_acc={result['train']['accuracy']:.3f} "
                        f"val_acc={result['validation']['accuracy']:.3f} "
                        f"test_acc={result['test']['accuracy']:.3f} "
                        f"test_macro={result['test']['macro_recall']:.3f}",
                        flush=True)
                row["fractions"].append({"fraction": frac, "models": scores})
            results.append(row)

    summary = []
    for idx, frac in enumerate(fractions):
        row = {"fraction": frac, "models": {}}
        for name in ("male_cns", "rewired", "text_only"):
            for kind in ("train", "validation", "test"):
                metric_values = [
                    entry["fractions"][idx]["models"][name][kind]["macro_recall"]
                    for entry in results
                ]
                row["models"].setdefault(name, {})[kind + "_macro_mean"] = round(
                    statistics.mean(metric_values), 6)
                row["models"][name][kind + "_macro_per_seed"] = metric_values
        summary.append(row)
    return {
        "study": "Group-aware training-only learning curves",
        "hypothesis": "Does additional supervised data improve true holdout routing performance?",
        "training_corpus_sha256": corpus_sha,
        "scaffold_sha256": hashlib.sha256(args.scaffold.read_bytes()).hexdigest(),
        "scaffold": {"node_count": original_nodes, "edge_count": len(source["edges"]),
                     "dataset": source["dataset"]},
        "corpus": {"examples": len(data), "duplicate_training_inputs_removed": dropped,
                   "no_previously_viewed_300_prompts_used": True},
        "seeds": args.seeds, "training_fractions": fractions,
        "max_epochs": args.epochs, "patience": args.patience,
        "per_seed": results, "summary": summary,
        "scope": "task routing labels only; not natural-language response correctness",
        "caveats": [
            "Supervised learning from labels, not biological synaptic plasticity",
            "Test is unseen during training but drawn from the same teacher corpus",
            "Character-trigram grouping mitigates only obvious near duplicates",
            "Early stopping uses validation only, never the held-out test",
            "Three random seeds not enough for strong significance claims",
            "Fractions are nested per route but not balanced by paraphrase family",
            "Test data varies by seed; model comparisons are paired within seed",
            "Research-only: no checkpoint persisted or promoted to production",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold", type=Path,
                        default=ROOT / "training/malecns_scaffold.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.25, 0.50, 1.0])
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--out", type=Path,
                        default=Path("/tmp/flygpt-learning-curves.json"))
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    for frac in result["summary"]:
        print("SUMMARY fraction=" + str(frac["fraction"]) +
              " test_macro_recall=" +
              json.dumps({k: v["test_macro_mean"]
                          for k, v in frac["models"].items()}), flush=True)
    print("REPORT=" + str(args.out), flush=True)
    import os
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## FlyGPT group-aware natural learning curves\n\n")
            f.write("| Fraction of train split | MaleCNS test macro | Rewired test macro | Text-only test macro |\n")
            f.write("|---:|---:|---:|---:|\n")
            for row in result["summary"]:
                f.write(f"| {row['fraction']:.0%} | "
                        f"{row['models']['male_cns']['test_macro_mean']:.1%} | "
                        f"{row['models']['rewired']['test_macro_mean']:.1%} | "
                        f"{row['models']['text_only']['test_macro_mean']:.1%} |\n")
            f.write("\nTests are group-split from existing teacher corpus; not independent external QA.\n")


if __name__ == "__main__":
    main()
