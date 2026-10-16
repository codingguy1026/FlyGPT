"""Experiment 3: can a learnable skip path recover graph-router generalization?

Research-only model. Do not change FlyRouterRuntime, mainline checkpoints, or
production. Compare one prespecified residual architecture on the established
300-question diagnostic benchmark. The 300 questions were seen in a PREVIOUS
experiment, so results here are no longer independently blind.

There is one prespecified residual candidate, with initial graph gate=0.20.
Model selection/early stopping uses only training-derived validation examples.
The benchmark is never used in training or hyperparameter selection.
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

import torch
from torch import nn
from torch.nn import functional as F

from training.topology_ablation import prepare_teacher, rewire_scaffold, measure
from training.fresh_threeway import validate_holdout, train_plain
from training.train_router import (
    FlyGraphRouter, load_examples, stratified_split, batchify, load_scaffold,
    vectorize, _accuracy_from_logits, _balanced_accuracy_from_logits,
    _fit_temperature, _recommend_gate,
)


class ResidualFlyGraphRouter(FlyGraphRouter):
    """Blend unpropagated text encoding with biological graph propagation.

    A learned scalar alpha=σ(graph_logit) controls the graph contribution:
    alpha=0 produces exactly the same function class as the text-only
    baseline (same encoder, tanh, layer norm and output);
    alpha=1 uses the existing original FlyGraphRouter.
    No graphs or biological neuron annotations are fabricated.
    """

    def __init__(self, vocab_size, n_nodes, n_classes, src, dst, base_weight,
                 steps=3, initial_graph_share=0.20):
        super().__init__(vocab_size, n_nodes, n_classes,
                         src, dst, base_weight, steps=steps)
        if not 0 < initial_graph_share < 1:
            raise ValueError("initial_graph_share must be between 0 and 1")
        import math
        self.graph_logit = nn.Parameter(
            torch.tensor(math.log(initial_graph_share / (1-initial_graph_share)),
                         dtype=torch.float32)
        )

    def graph_share(self):
        return torch.sigmoid(self.graph_logit)

    def forward(self, x):
        h_text = self.norm(self._encode(x))
        h_graph = self.propagate(x)
        share = self.graph_share()
        h = (1 - share) * h_text + share * h_graph
        return self.output(h)


class ModelRuntime:
    def __init__(self, model, routes, temperature, gate):
        self.model, self.routes = model, routes
        self.temperature, self.gate = temperature, gate

    def predict(self, text, top_k=2):
        with torch.no_grad():
            x = vectorize(text, 8192, "v6").unsqueeze(0)
            probabilities = torch.softmax(self.model(x) / self.temperature,
                                          dim=1)[0]
            values, indices = torch.topk(probabilities, k=2)
            confidence, second = [float(z) for z in values.tolist()]
            accepted = (
                confidence >= self.gate["min_confidence"] and
                confidence-second >= self.gate["min_margin"] and
                bool(torch.linalg.vector_norm(x).item() > 1e-8)
            )
        return {"route": self.routes[int(indices[0])], "accepted": accepted}


def train_residual(teacher, topology, seed, epochs, patience):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    examples = load_examples(teacher)
    routes = sorted({item.route for item in examples})
    labels = {route: i for i, route in enumerate(routes)}
    train_rows, validation_rows = stratified_split(examples, 0.22, seed)
    tx, ty = batchify(train_rows, labels, 8192, "v6")
    vx, vy = batchify(validation_rows, labels, 8192, "v6")
    n_nodes, src, dst, weights, _ = load_scaffold(topology, 256, seed)
    model = ResidualFlyGraphRouter(8192, n_nodes, len(routes),
                                   src, dst, weights, steps=3,
                                   initial_graph_share=0.20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015,
                                  weight_decay=0.0004)
    counts = torch.bincount(ty, minlength=len(routes)).float()
    class_weights = counts.sum() / (len(routes) * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)
    best_score, best_acc, best_epoch, stale = -1.0, -1.0, 0, 0
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x = F.dropout(tx, p=0.08, training=True)
        logits = model(x)
        loss = criterion(logits, ty)
        target = logits.gather(1, ty.unsqueeze(1)).squeeze(1)
        mask = F.one_hot(ty, num_classes=len(routes)).bool()
        strongest_other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
        loss += 0.10 * F.relu(0.30-target+strongest_other).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(vx)
            score = _balanced_accuracy_from_logits(predictions, vy, len(routes))
            acc = _accuracy_from_logits(predictions, vy)
        if score > best_score + 1e-9 or (
            abs(score-best_score) <= 1e-9 and acc > best_acc
        ):
            best_score, best_acc, best_epoch, stale = score, acc, epoch, 0
            best_state = {key: val.detach().cpu().clone()
                          for key, val in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No best checkpoint found")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid_logits = model(vx)
    temperature, _ = _fit_temperature(valid_logits, vy)
    gate = _recommend_gate(
        torch.softmax(valid_logits / temperature, dim=1), vy,
        target_precision=0.97, min_coverage=0.55,
    )
    share = float(model.graph_share().detach().cpu().item())
    return ModelRuntime(model, routes, temperature, gate), {
        "best_epoch": best_epoch,
        "validation_macro_recall": round(best_score, 5),
        "validation_accuracy": round(best_acc, 5),
        "graph_share": round(share, 6),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }


def run(args):
    torch.set_num_threads(2)
    orig_scaffold = json.loads(args.scaffold.read_text(encoding="utf-8"))
    if orig_scaffold.get("dataset") != "male-cns:v1.0":
        raise ValueError("Authentic MaleCNS scaffold required")
    if sorted(set(args.seeds)) != args.seeds:
        raise ValueError("Seeds must be sorted and unique")
    with tempfile.TemporaryDirectory(prefix="flygpt-residual-") as d:
        teacher = Path(d) / "teacher.jsonl"
        train_keys, _ = prepare_teacher(teacher)
        holdout, audit = validate_holdout(args.dataset, train_keys)
        rows = []
        for seed in args.seeds:
            conditions = {}
            ctrl = rewire_scaffold(orig_scaffold, seed+10000)
            ctrl_path = Path(d) / f"rewired_{seed}.json"
            ctrl_path.write_text(json.dumps(ctrl), encoding="utf-8")
            for name, path in (
                ("residual_malecns", args.scaffold),
                ("residual_rewired", ctrl_path),
            ):
                runtime, training = train_residual(
                    teacher, path, seed, args.epochs, args.patience
                )
                score = measure(runtime, holdout)
                # Within-model inference-only ablation: same trained model
                # with its graph contribution set to ~0, not a fair
                # comparison to a separately trained conventional baseline.
                trained_share = float(runtime.model.graph_share().item())
                with torch.no_grad():
                    runtime.model.graph_logit.fill_(-40.0)
                skip_score = measure(runtime, holdout)
                with torch.no_grad():
                    runtime.model.graph_logit.fill_(
                        torch.logit(torch.tensor(trained_share)).item()
                    )
                conditions[name] = {
                    "score": score, "skip_only_inference": skip_score,
                    "training": training,
                }
                print(
                    f"RESULT seed={seed} {name}: accuracy={score['accuracy']:.4f} "
                    f"macro_recall={score['macro_recall']:.4f} "
                    f"alpha={trained_share:.4f} skip_only={skip_score['accuracy']:.4f}",
                    flush=True,
                )
            plain_runtime, plain_training = train_plain(
                teacher, seed, args.epochs, args.patience,
                orig_scaffold["n_nodes"], conditions["residual_malecns"]["training"]["trainable_parameters"]
            )
            plain_score = measure(plain_runtime, holdout)
            conditions["text_only"] = {"score": plain_score, "training": plain_training}
            print(f"RESULT seed={seed} text_only: accuracy={plain_score['accuracy']:.4f} "
                  f"macro_recall={plain_score['macro_recall']:.4f}", flush=True)
            rows.append({"seed": seed, "conditions": conditions})
    names = ("residual_malecns", "residual_rewired", "text_only")
    summary = {}
    for name in names:
        scores = [row["conditions"][name]["score"]["accuracy"] for row in rows]
        summary[name] = {"per_seed_accuracy": scores,
                         "mean_accuracy": round(statistics.mean(scores), 5)}
    return {
        "title": "Learnable graph/text residual compared with controls",
        "version": "predefined gate=0.2 single architecture",
        "seeds": args.seeds,
        "holdout": audit,
        "results": rows, "summary": summary,
        "limitations": [
            "Previously observed 300-question developer-authored benchmark; diagnostic only",
            "No hyperparameters tuned on 300 test prompts within this run",
            "3 seeds exploratory; no claims of significance",
            "Skip-only ablation is inference-only: not equal to trained text baseline",
            "Positive gate alpha is not proof that graph topology improves generalization",
            "Routing label classification is not full chatbot reasoning",
        ],
        "production_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold", type=Path,
                        default=ROOT / "training/malecns_scaffold.json")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "training/fresh_holdout_300_20261016.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--out", type=Path,
                        default=Path("/tmp/flygpt-residual-study.json"))
    args = parser.parse_args()
    if not args.seeds or args.epochs < 2 or args.patience < 1:
        parser.error("Non-empty seeds, epochs>=2, and patience>=1 required")
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    for name, info in report["summary"].items():
        print(f"SUMMARY {name}: mean_accuracy={info['mean_accuracy']:.4f}",
              flush=True)
    print("REPORT=" + str(args.out), flush=True)


if __name__ == "__main__":
    main()
