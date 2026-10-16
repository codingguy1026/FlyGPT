"""Second FlyGPT topology study: 300 held-out prompts, three architectures.

TRAINING/EVALUATION SEPARATION:
  - six-route teacher inputs from frozen existing training files
  - three matched random seeds, same per-seed validation split
  - never use held-out evaluation labels for fitting or checkpoint selection
  - compare MaleCNS FlyGraphRouter, degree-preserving rewired FlyGraphRouter,
    and conventional non-graph text-only encoder (similar parameter budget)

Results are about *route classification only*, not chatbot factual accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

from training.topology_ablation import (
    prepare_teacher, key_for, load_jsonl, rewire_scaffold, measure,
    train as train_graph,
)
from training.train_router import (
    FlyGraphRouter, load_examples, stratified_split, batchify, vectorize,
    _balanced_accuracy_from_logits, _accuracy_from_logits,
    _fit_temperature, _recommend_gate,
)


def validate_holdout(dataset: Path, train_keys: set[str]) -> tuple[list[dict], dict]:
    cases = load_jsonl(dataset)
    counts = Counter(str(x["route"]) for x in cases)
    if len(cases) != 300 or len(counts) != 6 or any(n != 50 for n in counts.values()):
        raise ValueError(f"Expected exactly 300 rows and 50 per class: {counts}")
    keys = [key_for(str(x["input"])) for x in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Held-out questions contain duplicate normalized inputs")
    overlap = set(keys) & train_keys
    if overlap:
        raise ValueError("Train/holdout exact overlap! First examples: " + repr(sorted(overlap)[:5]))
    return cases, {"questions": len(cases), "classes": dict(sorted(counts.items())),
                   "exact_training_overlaps": 0,
                   "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}


def text_only_model(vocab_size: int, hidden: int, classes: int):
    """Parameter-matched conventional text encoder with NO graph message passing.

    Same encoder width and nonlinearities as the FlyGraphRouter. All graph edges,
    weights, and three propagation steps are absent by construction.
    """
    import torch
    from torch import nn

    class TextOnlyRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(vocab_size, hidden)
            self.output = nn.Linear(hidden, classes)
            self.norm = nn.LayerNorm(hidden)

        def forward(self, x):
            h = self.norm(torch.tanh(self.encoder(x)))
            return self.output(h)

    return TextOnlyRouter()


def train_plain(teacher_file: Path, seed: int, epochs: int, patience: int,
                n_nodes: int, graph_param_count: int):
    import torch
    import torch.nn.functional as F
    from torch import nn
    torch.set_num_threads(2)
    random.seed(seed)
    torch.manual_seed(seed)
    examples = load_examples(teacher_file)
    routes = sorted({example.route for example in examples})
    label_to_idx = {route: idx for idx, route in enumerate(routes)}
    train_rows, val_rows = stratified_split(examples, 0.22, seed)
    train_x, train_y = batchify(train_rows, label_to_idx, 8192, "v6")
    val_x, val_y = batchify(val_rows, label_to_idx, 8192, "v6")
    model = text_only_model(8192, n_nodes, len(routes))
    plain_param_count = sum(param.numel() for param in model.parameters())
    relative_difference = abs(plain_param_count - graph_param_count) / graph_param_count
    if relative_difference > 0.01:
        raise AssertionError(f"Parameter budget differs by {relative_difference:.2%}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.0004)
    class_counts = torch.bincount(train_y, minlength=len(routes)).float()
    class_weights = class_counts.sum() / (len(routes) * class_counts.clamp_min(1.0))
    cross_entropy = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)
    best_state = None
    best_score, best_acc, best_epoch, stale = -1.0, -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x = F.dropout(train_x, p=0.08, training=True)
        logits = model(x)
        ce_loss = cross_entropy(logits, train_y)
        mask = F.one_hot(train_y, num_classes=len(routes)).bool()
        target = logits.gather(1, train_y.unsqueeze(1)).squeeze(1)
        runner_up = logits.masked_fill(mask, float("-inf")).max(dim=1).values
        loss = ce_loss + 0.10 * F.relu(0.30 - target + runner_up).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_acc = _accuracy_from_logits(val_logits, val_y)
            val_score = _balanced_accuracy_from_logits(val_logits, val_y, len(routes))
        if val_score > best_score + 1e-9 or (
            abs(val_score - best_score) <= 1e-9 and val_acc > best_acc
        ):
            best_score, best_acc, best_epoch, stale = val_score, val_acc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise AssertionError("Plain model did not train")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
    temperature, calibration = _fit_temperature(val_logits, val_y)
    gate = _recommend_gate(torch.softmax(val_logits / temperature, dim=1), val_y,
                           target_precision=0.97, min_coverage=0.55)

    class PlainRuntime:
        def predict(self, prompt, top_k=2):
            with torch.no_grad():
                x = vectorize(prompt, 8192, "v6").unsqueeze(0)
                scores = torch.softmax(model(x) / temperature, dim=1)[0]
                values, indices = torch.topk(scores, k=2)
                confidence, runner_up = values.tolist()
                label = routes[indices.tolist()[0]]
                accepted = (
                    confidence >= gate["min_confidence"]
                    and confidence - runner_up >= gate["min_margin"]
                    and float(torch.linalg.vector_norm(x).item()) > 1e-8
                )
                return {"route": label, "accepted": bool(accepted)}

    return PlainRuntime(), {
        "best_epoch": best_epoch,
        "validation_macro_recall": round(best_score, 5),
        "validation_accuracy": round(best_acc, 5),
        "temperature": round(float(temperature), 5),
        "params": plain_param_count, "graph_params": graph_param_count,
        "param_difference_percent": round(relative_difference * 100, 4),
        "calibration": calibration, "gate": gate,
    }


def run(args):
    from router_runtime import FlyRouterRuntime
    import torch
    torch.set_num_threads(2)
    scaffold = json.loads(args.scaffold.read_text(encoding="utf-8"))
    if scaffold["dataset"] != "male-cns:v1.0":
        raise ValueError("Only a frozen anatomical MaleCNS source is supported")
    if args.seeds != sorted(set(args.seeds)):
        raise ValueError("Unique sorted seeds are required to make run order reproducible")
    with tempfile.TemporaryDirectory(prefix="flygpt-threeway-") as name:
        temp = Path(name)
        teacher_path = temp / "teacher.jsonl"
        teacher_keys, train_duplicates = prepare_teacher(teacher_path)
        cases, holdout_audit = validate_holdout(args.holdout, teacher_keys)

        n_nodes = int(scaffold["n_nodes"])
        results = []
        for seed in args.seeds:
            print(f"SEED {seed}", flush=True)
            control_scaffold = rewire_scaffold(scaffold, seed + 10000)
            control_path = temp / f"rewired_{seed}.json"
            control_path.write_text(json.dumps(control_scaffold), encoding="utf-8")
            conditions = {}
            graph_param_count = None
            for condition, path in (
                ("male_cns", args.scaffold), ("rewired", control_path)
            ):
                checkpoint = temp / f"{condition}_{seed}.pt"
                summary = train_graph(teacher_path, path, checkpoint, seed,
                                      args.epochs, args.patience)
                runtime = FlyRouterRuntime(checkpoint)
                count = sum(int(p.numel()) for p in runtime.model.parameters())
                if graph_param_count is None:
                    graph_param_count = count
                elif count != graph_param_count:
                    raise AssertionError("Paired graph parameter counts differ")
                score = measure(runtime, cases)
                conditions[condition] = {
                    "metrics": score, "parameters": count,
                    "training_summary": summary,
                    "validation_macro_recall": runtime.best_val_balanced_accuracy,
                }
                print(f"RESULT seed={seed} {condition}: accuracy={score['accuracy']:.5f}"
                      f" macro_recall={score['macro_recall']:.5f} n={score['n']}",
                      flush=True)
            plain, details = train_plain(teacher_path, seed, args.epochs,
                                         args.patience, n_nodes, graph_param_count)
            plain_score = measure(plain, cases)
            conditions["text_only"] = {"metrics": plain_score, "training": details}
            print(f"RESULT seed={seed} text_only: accuracy={plain_score['accuracy']:.5f}"
                  f" macro_recall={plain_score['macro_recall']:.5f} n={plain_score['n']}",
                  flush=True)
            results.append({"seed": seed, "conditions": conditions,
                            "rewired_changed_pairs": control_scaffold["ablation"]["changed_pairs"]})

    summary = {}
    for condition in ("male_cns", "rewired", "text_only"):
        summary[condition] = {}
        for metric in ("accuracy", "macro_recall", "gate_coverage", "gate_precision"):
            values = [row["conditions"][condition]["metrics"][metric] for row in results]
            defined = [v for v in values if v is not None]
            summary[condition][metric] = {
                "per_seed": values,
                "mean": round(statistics.mean(defined), 5) if defined else None,
            }
    diffs = {}
    for condition in ("rewired", "text_only"):
        diffs[condition] = {}
        for metric in ("accuracy", "macro_recall"):
            delta = [
                row["conditions"]["male_cns"]["metrics"][metric] -
                row["conditions"][condition]["metrics"][metric]
                for row in results
            ]
            diffs[condition][metric] = {
                "per_seed": [round(v, 5) for v in delta],
                "mean": round(statistics.mean(delta), 5),
            }
    return {
        "study": "300 held-out questions: MaleCNS, rewired, text-only",
        "scaffold": {"path": str(args.scaffold),
                     "sha256": hashlib.sha256(args.scaffold.read_bytes()).hexdigest(),
                     "nodes": scaffold["n_nodes"],
                     "edges": len(scaffold["edges"])},
        "holdout": {"path": str(args.holdout), **holdout_audit},
        "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest()
            if teacher_path.exists() else None,
        "training_exact_duplicates_removed": train_duplicates,
        "seeds": args.seeds, "epochs_max": args.epochs,
        "patience": args.patience, "results": results,
        "summary": summary, "male_cns_minus_others": diffs,
        "production_checkpoint_unchanged": True,
        "cautions": [
            "This is a freshly drafted, developer-authored holdout, not external blind verification.",
            "No normalized exact train/eval duplicates; semantic paraphrases still possible.",
            "Text-only control is near-parameter-matched but differs in computation depth.",
            "Task routing accuracy does not measure chatbot reasoning or correctness.",
            "Three seeds are exploratory; no significance claim.",
            "No hyperparameters were selected using this 300-question test set.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scaffold", type=Path,
                    default=ROOT / "training/malecns_scaffold.json")
    ap.add_argument("--holdout", type=Path,
                    default=ROOT / "training/fresh_holdout_300_20261016.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--patience", type=int, default=18)
    ap.add_argument("--out", type=Path, default=Path("/tmp/flygpt-threeway-300.json"))
    args = ap.parse_args()
    if not args.seeds or args.epochs < 2 or args.patience < 1:
        ap.error("Non-empty seeds, epochs>=2 and patience>=1 required")
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    for condition, scores in report["summary"].items():
        print(f"SUMMARY {condition}: accuracy={scores['accuracy']['mean']:.5f}"
              f" macro_recall={scores['macro_recall']['mean']:.5f}", flush=True)
    print("DELTAS=" + json.dumps(report["male_cns_minus_others"]), flush=True)
    print("RESULT_FILE=" + str(args.out))
    import os
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## 🪰 Fresh 300-question three-way study\n\n")
            f.write(f"Balanced 300-question frozen holdout SHA256: {report['holdout']['sha256']}\n\n")
            f.write("| Model | Mean route accuracy | Mean macro recall |\n|---|---:|---:|\n")
            for name, vals in report["summary"].items():
                f.write(f"| {name} | {vals['accuracy']['mean']:.1%} | "
                        f"{vals['macro_recall']['mean']:.1%} |\n")
            f.write("\n**Developer-authored exploratory holdout; not proof of biological superiority.**\n")


if __name__ == "__main__":
    main()
