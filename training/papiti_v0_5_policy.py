"""Papiti v0.5: full-MaleCNS, category-free multi-action policy (research only).

No single general/math/summarize/etc. route is predicted. Independently scored
tool capabilities may be combined or all omitted. Frozen structural keys from
the REAL fully scanned MaleCNS graph contribute to exact population moments.
This is NOT a biological firing simulation, a general language learner, or a
production response generator. Human-readable directives are an API contract,
not a claim that graph neurons authored unrestricted natural language.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from training.full_population_router import load_all_structural_keys
from training.train_router import vectorize

CAPABILITIES = ("web_search", "memory_lookup", "calculator")
EXPECTED_NEURONS = 166700
VOCAB_SIZE = 8192
STRENGTH = 0.85


def load_policy_examples(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("input")
        actions = row.get("tools")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{number}: input must be nonempty")
        if not isinstance(actions, list) or len(actions) != len(set(actions)):
            raise ValueError(f"{path}:{number}: tools must be a unique list")
        if any(tool not in CAPABILITIES for tool in actions):
            raise ValueError(f"{path}:{number}: unknown capability")
        key = " ".join(text.casefold().split())
        if key in seen:
            raise ValueError(f"{path}:{number}: duplicate input")
        seen.add(key)
        rows.append({"input": text.strip(), "tools": actions})
    if not rows:
        raise ValueError(f"{path}: no examples")
    return rows


def enforce_unseen(train: list[dict], holdout: list[dict]) -> None:
    a = {" ".join(x["input"].casefold().split()) for x in train}
    b = {" ".join(x["input"].casefold().split()) for x in holdout}
    if a & b:
        raise ValueError("Training and untouched holdout overlap")


def make_batch(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([vectorize(row["input"], VOCAB_SIZE, "v6") for row in examples])
    y = torch.tensor([
        [float(capability in row["tools"]) for capability in CAPABILITIES]
        for row in examples
    ], dtype=torch.float32)
    return x, y


class FullMaleCNSPolicy(nn.Module):
    """Continuous intent representation and independent tool-use decisions.

    Every curated neuron enters the full-population mean and Gram matrix.
    Each neuron's attention weight is positive for every query, so no top-K
    pruning or single-choice class gate can remove a neuron from the summary.
    """
    def __init__(self, neuron_ids: torch.Tensor, keys: torch.Tensor,
                 hidden: int = 128):
        super().__init__()
        if (neuron_ids.ndim != 1 or keys.ndim != 2
                or keys.shape[0] != len(neuron_ids)
                or keys.shape[1] < 2):
            raise ValueError("One structural vector is required per neuron")
        if not bool(torch.all(neuron_ids[1:] > neuron_ids[:-1])):
            raise ValueError("Neuronal body IDs must be unique and sorted")
        if not bool(torch.isfinite(keys).all()):
            raise ValueError("Non-finite full-population embeddings")

        normalized = F.normalize(keys.float(), dim=1, eps=1e-8)
        self.register_buffer("neuron_ids", neuron_ids.long().clone())
        self.register_buffer("keys", normalized.clone())
        # Every neuron has an individually trainable positive contribution,
        # including the 217 official neurons without eligible directed edges.
        # Structural keys stay anchored to genuine graph pretraining; action
        # supervision updates these population gains for ALL 166700 neurons.
        self.neuron_gain_logits = nn.Parameter(torch.zeros(len(neuron_ids)))
        dimension = normalized.shape[1]

        self.encoder = nn.Sequential(
            nn.Linear(VOCAB_SIZE, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.query = nn.Linear(hidden, dimension)
        self.structural_projection = nn.Linear(dimension, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.graph_logit = nn.Parameter(torch.tensor(math.log(0.5)))
        self.action_head = nn.Linear(hidden, len(CAPABILITIES))

    def full_population_state(self, q: torch.Tensor) -> torch.Tensor:
        query = F.normalize(q, dim=1, eps=1e-8)
        gains = 2.0 * torch.sigmoid(self.neuron_gain_logits)
        weighted_keys = self.keys * gains[:, None]
        total_gain = gains.sum()
        # All-neuron weighted moments, with autograd reaching every gain.
        mean = weighted_keys.sum(dim=0) / total_gain
        gram = self.keys.T @ weighted_keys / total_gain
        numerator = mean[None, :] + STRENGTH * (query @ gram)
        denominator = 1 + STRENGTH * (query @ mean)[:, None]
        return numerator / denominator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        query = torch.tanh(self.query(encoded))
        context = self.full_population_state(query)
        fused = self.norm(
            encoded + torch.sigmoid(self.graph_logit)
            * torch.tanh(self.structural_projection(context))
        )
        # Sigmoid action heads, not a softmax category classifier.
        return self.action_head(fused)

    @torch.no_grad()
    def decide(self, text: str, thresholds: list[float]) -> dict:
        self.eval()
        logits = self(vectorize(text, VOCAB_SIZE, "v6").unsqueeze(0))
        probabilities = torch.sigmoid(logits)[0].tolist()
        scores = {name: round(float(p), 5)
                  for name, p in zip(CAPABILITIES, probabilities)}
        selected = [name for name, p, threshold in
                    zip(CAPABILITIES, probabilities, thresholds)
                    if p >= threshold]
        instructions = [
            "Answer the user's actual question, not a generic self-introduction.",
            "Use reliable knowledge or provided evidence; do not invent facts or tool results.",
            "Never claim the MaleCNS network independently solved the question.",
        ]
        if "web_search" in selected:
            instructions.append("Verify time-sensitive claims with retrieved evidence first.")
        if "memory_lookup" in selected:
            instructions.append("Retrieve relevant conversation context; never invent memories.")
        if "calculator" in selected:
            instructions.append("Use a trusted calculation result; do not invent a numeric answer.")
        return {
            "contract": "papiti_v0_5_tool_policy_v1",
            "request": text,
            "capability_scores": scores,
            "selected_capabilities": selected,
            "language_model_instructions": instructions,
            "tools_executed": False,
            "experimental": True,
        }


@torch.no_grad()
def measure(model: FullMaleCNSPolicy, x: torch.Tensor, y: torch.Tensor,
            thresholds: list[float]) -> dict:
    model.eval()
    prob = torch.sigmoid(model(x))
    chosen = prob >= torch.tensor(thresholds)[None, :]
    expected = y.bool()
    tp = (chosen & expected).sum(0).tolist()
    fp = (chosen & ~expected).sum(0).tolist()
    fn = (~chosen & expected).sum(0).tolist()
    f1 = {}
    for i, action in enumerate(CAPABILITIES):
        denominator = 2 * tp[i] + fp[i] + fn[i]
        f1[action] = round(2 * tp[i] / denominator if denominator else 1.0, 4)
    exact = float((chosen == expected).all(dim=1).float().mean().item())
    n_none = (~expected.any(dim=1)).sum().item()
    false_call = ((chosen.any(dim=1)) & (~expected.any(dim=1))).sum().item()
    return {
        "n": len(y),
        "exact_action_set_accuracy": round(exact, 4),
        "macro_action_f1": round(sum(f1.values()) / len(f1), 4),
        "per_action_f1": f1,
        "unnecessary_tool_rate_on_no_tool_requests": round(false_call / n_none, 4) if n_none else None,
    }


def tune_thresholds(model: FullMaleCNSPolicy, x: torch.Tensor, y: torch.Tensor):
    # Pick thresholds using validation only, never the test/holdout set.
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x))
    thresholds = []
    for i in range(len(CAPABILITIES)):
        true = y[:, i].bool()
        best = (-1.0, 0.5)
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pred = probabilities[:, i] >= threshold
            tp = int((pred & true).sum())
            fp = int((pred & ~true).sum())
            fn = int((~pred & true).sum())
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom else 1.0
            if f1 > best[0]:
                best = (f1, threshold)
        thresholds.append(best[1])
    return thresholds


def train(args: argparse.Namespace) -> dict:
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience and batch-size must be positive")

    ids, keys, structural_metadata = load_all_structural_keys(args.structural)
    if len(ids) != EXPECTED_NEURONS:
        raise ValueError(f"Need exactly {EXPECTED_NEURONS} official curated MaleCNS neurons; got {len(ids)}")
    raw_train = load_policy_examples(args.train)
    holdout = load_policy_examples(args.holdout)
    enforce_unseen(raw_train, holdout)
    rng = random.Random(args.seed)
    raw_train.sort(key=lambda row: (" ".join(row["tools"]), row["input"]))
    rng.shuffle(raw_train)
    split_at = max(1, round(len(raw_train) * 0.8))
    training, validation = raw_train[:split_at], raw_train[split_at:]
    if not validation:
        raise ValueError("Insufficient samples for validation")

    x_train, y_train = make_batch(training)
    x_val, y_val = make_batch(validation)
    x_test, y_test = make_batch(holdout)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    model = FullMaleCNSPolicy(ids, keys)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0015, weight_decay=0.0004,
    )
    # Make rare capabilities count, but keep the independent Bernoulli targets.
    positive = y_train.sum(0)
    weight = ((len(y_train) - positive) / positive.clamp_min(1)).clamp(1, 6)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    sampler = torch.Generator().manual_seed(args.seed + 11)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in torch.randperm(len(training), generator=sampler).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(F.dropout(x_train[batch], p=0.06, training=True))
            loss = F.binary_cross_entropy_with_logits(
                logits, y_train[batch], pos_weight=weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(
                model(x_val), y_val, pos_weight=weight
            ).item()
        if val_loss < best - 1e-5:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {name: value.detach().clone()
                          for name, value in model.named_parameters()}
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} val_bce={val_loss:.5f}", flush=True)

    if best_state is None:
        raise RuntimeError("Training failed to select a checkpoint")
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(best_state[name])
    thresholds = tune_thresholds(model, x_val, y_val)
    validation_metrics = measure(model, x_val, y_val, thresholds)
    # Holdout is accessed only after model and thresholds are fixed.
    test_metrics = measure(model, x_test, y_test, thresholds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "papiti-v0.5-category-free-full-malecns",
        "model_state": model.state_dict(),
        "n_neurons": len(ids),
        "capabilities": CAPABILITIES,
        "thresholds": thresholds,
        "structural_metadata": structural_metadata,
        "source_structural_sha256": hashlib.sha256(args.structural.read_bytes()).hexdigest(),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "holdout_rows": len(holdout),
        "best_epoch": best_epoch,
        "validation_metrics": validation_metrics,
        "holdout_metrics": test_metrics,
        "experimental_only": True,
        "production_enabled": False,
        "language_model_trained": False,
    }, args.out)
    result = {
        "checkpoint": str(args.out),
        "official_curated_neurons": len(ids),
        "positive_neuronal_connection_rows": structural_metadata["epoch_reports"][-1]["eligible_directed_connection_rows"],
        "full_population_exact_attention": True,
        "training": len(training), "validation": len(validation), "holdout": len(holdout),
        "best_epoch": best_epoch, "thresholds": dict(zip(CAPABILITIES, thresholds)),
        "validation_metrics": validation_metrics, "holdout_metrics": test_metrics,
        "not_deployed": True,
        "warning": "Teacher-labeled tool selection only; not measured full conversational intelligence.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "training/papiti_v0_5_policy_train.jsonl")
    parser.add_argument("--holdout", type=Path, default=ROOT / "training/papiti_v0_5_policy_holdout.jsonl")
    parser.add_argument("--out", type=Path, default=Path("artifacts/papiti_v0_5_category_free_166700.pt"))
    parser.add_argument("--report", type=Path, default=Path("reports/papiti_v0_5_policy.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
