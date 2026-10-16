"""Whole-population MaleCNS structural-attention task router (EXPERIMENTAL).

The official full MaleCNS Feather weights file is consumed and structurally
pretrained by training/train_full_connectome.py. This module then learns
task-routing weights over ALL curated neuronal structural representations,
using a memory-efficient exact linear-attention population summary.

IMPORTANT: This does NOT simulate real fly neuronal firing, nor traverse every
edge per chat message. Every curated neuron contributes to a whole-population
structural attention summary, whose embeddings were pretrained on the entire
official directed connection table. It is not the 256-neuron router, and its
accuracy cannot be inferred from the previous 93.2% internal score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from training.train_router import (
    load_examples, stratified_split, batchify, vectorize,
    _accuracy_from_logits, _balanced_accuracy_from_logits, _fit_temperature,
    _recommend_gate,
)


TRAIN_FILES = (
    "teacher_seed.jsonl",
    "curriculum_v0_3_4.jsonl",
    "curriculum_v0_3_5_short.jsonl",
    "curriculum_v0_3_6_context.jsonl",
    "curriculum_v0_3_7_yugeok.jsonl",
    "curriculum_v0_3_8_hard_negatives.jsonl",
    "curriculum_v0_3_9.jsonl",
    "curriculum_v0_5_realworld.jsonl",
)


def prepare_teacher(output: Path):
    """Build labeled training-only examples without importing unmerged PRs."""
    observed = {}
    total = 0
    for file in TRAIN_FILES:
        path = ROOT / "training" / file
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = " ".join(row["input"].casefold().split())
            total += 1
            if key in observed and observed[key]["route"] != row["route"]:
                raise ValueError("Conflicting training labels for a normalized prompt")
            observed.setdefault(key, {"input": row["input"], "route": row["route"]})
    output.write_text(
        "".join(json.dumps(x, ensure_ascii=False)+"\n" for x in observed.values()),
        encoding="utf-8",
    )
    return set(observed), total - len(observed)

def load_all_structural_keys(path: Path, minimum_neurons: int = 100000):
    """Reject subset/synthetic sources for real website use."""
    src = torch.load(path, map_location="cpu", weights_only=True)
    meta = src.get("metadata", {})
    if (
        meta.get("dataset") != "male-cns:v1.0"
        or meta.get("full_source_scanned_every_epoch") is not True
        or meta.get("no_topk_or_node_sampling") is not True
        or meta.get("training_kind") != "offline_self_supervised_directed_edge_reconstruction"
    ):
        raise ValueError("Need authentic, unsampled full MaleCNS structural training artifact")
    ids = src["body_ids"].to(dtype=torch.int64)
    pre = src["presynaptic_embeddings"].to(dtype=torch.float32)
    post = src["postsynaptic_embeddings"].to(dtype=torch.float32)
    n = ids.numel()
    if (
        n < minimum_neurons
        or n != int(meta.get("annotated_neurons_eligible", -1))
        or pre.ndim != 2 or post.shape != pre.shape
        or pre.shape[0] != n or pre.shape[1] < 2
        or not bool(torch.all(ids[1:] > ids[:-1]))
        or not bool(torch.isfinite(pre).all() and torch.isfinite(post).all())
        or int(meta.get("epoch_reports", [{}])[-1].get("eligible_directed_connection_rows", 0)) < n
    ):
        raise ValueError("Invalid/partial full-population neuron embeddings")
    # No sampling and no selection of particular neuron types beyond the
    # curator-defined all-neuron index. Every row contributes to the moments.
    keys = F.normalize(torch.cat((pre, post), dim=1), dim=1, eps=1e-8)
    return ids, keys.contiguous(), dict(meta)


class FullPopulationRouter(nn.Module):
    """All-neuron positive linear attention with exact cached moments.

    k_i are frozen pre/post structural embeddings for *every* curated neuron.
    Query-dependent weights w_i = 1 + 0.85 * dot(k_i, normalized(q)) are
    positive and nonzero. The attended structural vector is exactly:

       sum_i(w_i * k_i) / sum_i(w_i)

    Sufficient statistics (sum k_i and sum k_i k_i^T) are computed using the
    ENTIRE key matrix. This is mathematically exact, not top-K sampling.
    They permit cheap live requests even for >100,000 neurons.

    No per-question synaptic action potential simulation is asserted.
    """

    def __init__(self, neuron_ids: torch.Tensor, keys: torch.Tensor,
                 n_classes: int, vocab_size: int = 8192,
                 hidden: int = 256):
        super().__init__()
        if keys.ndim != 2 or neuron_ids.ndim != 1 or keys.shape[0] != neuron_ids.numel():
            raise ValueError("Every annotated neuron needs one structural key")
        if not neuron_ids.numel() or not bool(torch.all(neuron_ids[1:] > neuron_ids[:-1])):
            raise ValueError("Neuron IDs must be unique and sorted")
        if not bool(torch.isfinite(keys).all()):
            raise ValueError("Structural embeddings contain non-finite values")
        keys = F.normalize(keys.float(), dim=1, eps=1e-8).contiguous()
        self.register_buffer("neuron_ids", neuron_ids.long().clone())
        self.register_buffer("keys", keys.clone())
        self.register_buffer("population_mean", keys.mean(dim=0))
        self.register_buffer("population_gram", keys.T @ keys / len(keys))
        d = keys.shape[1]
        self.encoder = nn.Linear(vocab_size, hidden)
        self.query = nn.Linear(hidden, d)
        self.structural_projection = nn.Linear(d, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, n_classes)
        self.graph_logit = nn.Parameter(torch.tensor(math.log(.20/.80)))

    def query_vector(self, x):
        h = torch.tanh(self.encoder(x))
        q = F.normalize(torch.tanh(self.query(h)), dim=1, eps=1e-8)
        return h, q

    def full_population_state(self, q):
        mean = self.population_mean
        gram = self.population_gram
        numerator = mean[None, :] + 0.85 * (q @ gram)
        denominator = 1.0 + 0.85 * (q @ mean)[:, None]
        return numerator / denominator

    def forward(self, x):
        encoded, q = self.query_vector(x)
        all_neuron_context = self.full_population_state(q)
        share = torch.sigmoid(self.graph_logit)
        h = self.norm(encoded + share * torch.tanh(
            self.structural_projection(all_neuron_context)))
        return self.output(h)

    @torch.no_grad()
    def top_contributors(self, x, limit: int = 12):
        """For optional UI debugging only; routing ALWAYS aggregates all neurons."""
        _, q = self.query_vector(x)
        scores = 1.0 + 0.85 * (q @ self.keys.T)[0]
        vals, idx = torch.topk(scores, min(limit, scores.numel()))
        return [
            {"body_id": str(self.neuron_ids[int(i)].item()),
             "relative_attention_weight": round(float(v), 5)}
            for v, i in zip(vals.tolist(), idx.tolist())
        ]


def train_one(teacher: Path, ids, keys, seed: int, epochs: int,
              patience: int, batch_size: int, explicit_split=None):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(2)
    examples = load_examples(teacher)
    classes = sorted({x.route for x in examples})
    mapping = {label: i for i, label in enumerate(classes)}
    if explicit_split is None:
        train, val = stratified_split(examples, .22, seed)
        test = None
    else:
        train, val, test = explicit_split
        if {x.text for x in train} & {x.text for x in val} or (
            test is not None and (
                {x.text for x in train} & {x.text for x in test}
                or {x.text for x in val} & {x.text for x in test}
            )
        ):
            raise ValueError("Training/validation/test sets overlap")
    x_train, y_train = batchify(train, mapping, 8192, "v6")
    x_val, y_val = batchify(val, mapping, 8192, "v6")

    model = FullPopulationRouter(ids, keys, len(classes))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=.0015, weight_decay=.0004,
    )
    counts = torch.bincount(y_train, minlength=len(classes)).float()
    weight = counts.sum() / (len(classes) * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=.04)
    best_score, best_acc, best_epoch, patience_left, best_weights = -1., -1., 0, 0, None
    rng = torch.Generator().manual_seed(seed+777)

    for epoch in range(1, epochs+1):
        model.train()
        permutation = torch.randperm(len(train), generator=rng)
        for chunk in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(F.dropout(x_train[chunk], p=.08, training=True))
            target = logits.gather(1, y_train[chunk, None]).flatten()
            mask = F.one_hot(y_train[chunk], len(classes)).bool()
            other = logits.masked_fill(mask, float("-inf")).max(1).values
            loss = criterion(logits, y_train[chunk]) + .10*F.relu(.30-target+other).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            score = _balanced_accuracy_from_logits(val_logits, y_val, len(classes))
            acc = _accuracy_from_logits(val_logits, y_val)
        if score > best_score + 1e-9 or (
            abs(score-best_score) <= 1e-9 and acc > best_acc
        ):
            best_score, best_acc, best_epoch, patience_left = score, acc, epoch, 0
            # Do NOT duplicate the large frozen per-neuron keys each epoch.
            best_weights = {name: value.detach().cpu().clone()
                            for name, value in model.named_parameters()}
        else:
            patience_left += 1
            if patience_left >= patience:
                break
    if best_weights is None:
        raise RuntimeError("Training did not select a checkpoint")
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(best_weights[name])
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
    temperature, calibration = _fit_temperature(val_logits, y_val)
    gate = _recommend_gate(
        torch.softmax(val_logits / temperature, dim=1), y_val,
        target_precision=.97, min_coverage=.55,
    )
    test_metrics = None
    if test is not None:
        # Test data is untouched until AFTER checkpoint/temperature selection.
        x_test, y_test = batchify(test, mapping, 8192, "v6")
        with torch.no_grad():
            logits_test = model(x_test)
        predicted = logits_test.argmax(dim=1)
        confusion = {}
        recall = {}
        for route, index in mapping.items():
            mask = y_test == index
            count = int(mask.sum().item())
            recall[route] = (
                round(float((predicted[mask] == y_test[mask]).float().mean().item()), 6)
                if count else None
            )
            confusion[route] = {
                label: int(((predicted == other_index) & mask).sum().item())
                for label, other_index in mapping.items()
            }
        test_metrics = {
            "n": len(test),
            "accuracy": round(_accuracy_from_logits(logits_test, y_test), 6),
            "macro_recall": round(
                _balanced_accuracy_from_logits(logits_test, y_test, len(classes)), 6
            ),
            "per_route_recall": recall,
            "confusion": confusion,
        }
    return model, {
        "test_metrics": test_metrics,
        "seed": seed, "training_examples": len(train),
        "validation_examples": len(val),
        "best_epoch": best_epoch,
        "best_val_macro_recall": round(best_score, 6),
        "best_val_accuracy": round(best_acc, 6),
        "temperature": float(temperature),
        "calibration": {**calibration, **gate},
        "routes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", type=Path,
                        default=ROOT / "artifacts/malecns_full_structural_embeddings.pt")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/fly_router_malecns_all_neurons.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("Epochs, patience, and batch size must be positive")

    ids, keys, metadata = load_all_structural_keys(args.structural)
    with tempfile.TemporaryDirectory(prefix="flygpt-full-pop-") as tmp:
        teacher = Path(tmp) / "teacher.jsonl"
        train_keys, duplicates = prepare_teacher(teacher)
        digest = hashlib.sha256(teacher.read_bytes()).hexdigest()
        model, scores = train_one(teacher, ids, keys, args.seed,
                                  args.epochs, args.patience, args.batch_size)
    if model.neuron_ids.numel() != metadata["annotated_neurons_eligible"]:
        raise AssertionError("Any missing annotated neurons would invalidate the model")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "flygpt-full-population-v1",
        "architecture": "FullPopulationRouter",
        "model_state": model.state_dict(),
        "routes": scores["routes"], "vocab_size": 8192,
        "vectorizer_version": "v6",
        "n_nodes": len(ids),
        "neuron_ids": ids,
        "training_summary": scores,
        "scaffold_meta": {
            "kind": "male-cns:v1.0",
            "source": "Janelia MaleCNS v1.0 official weights and neuronal annotations",
            "full_connectome_trained": True,
            "full_source_scanned_every_epoch": True,
            "n_neurons": len(ids),
            "n_positive_connection_rows": metadata["epoch_reports"][-1]["eligible_directed_connection_rows"],
            "full_neuron_population_attended": True,
            "representation": "exact full-neuron linear attention over full-graph structural embeddings",
            "live_edge_propagation": False,
        },
        "training_source_sha256": digest,
        "structural_weights_sha256": hashlib.sha256(args.structural.read_bytes()).hexdigest(),
        "train_duplicate_prompts_removed": duplicates,
        "experimental_only": True,
    }, args.out)
    print(json.dumps({
        "checkpoint": str(args.out), "neuron_count": len(ids),
        "synaptic_connection_rows": metadata["epoch_reports"][-1]["eligible_directed_connection_rows"],
        "train_accuracy_not_claimed": True,
        "internal_validation_macro_recall": scores["best_val_macro_recall"],
        "notice": "Evaluation uses training-corpus validation; cannot claim 93% general accuracy.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
