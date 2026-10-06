from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

TOKEN_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9_]+")


def stable_bucket(token: str, size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def _add_feature(vec: torch.Tensor, key: str, weight: float, size: int) -> None:
    vec[stable_bucket(key, size)] += weight


def vectorize(text: str, size: int, version: str = "v1") -> torch.Tensor:
    vec = torch.zeros(size, dtype=torch.float32)
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]

    for token in tokens:
        token_len = len(token)

        if version in ("v4", "v5", "v6"):
            # Very short unseen words are fragile when the whole-token hash is
            # allowed to dominate. Reduce that feature and lean more heavily on
            # reusable character/jamo structure. Longer tokens keep the old
            # whole-word strength.
            if token_len <= 2:
                word_weight = 0.42
                bigram_weight = 0.50
                char_weight = 0.42
                jamo1_weight = 0.24
                jamo2_weight = 0.24
                jamo3_weight = 0.12
            elif token_len == 3:
                word_weight = 0.68
                bigram_weight = 0.42
                char_weight = 0.32
                jamo1_weight = 0.18
                jamo2_weight = 0.21
                jamo3_weight = 0.10
            else:
                word_weight = 1.0
                bigram_weight = 0.35
                char_weight = 0.22
                jamo1_weight = 0.10
                jamo2_weight = 0.18
                jamo3_weight = 0.08
        else:
            word_weight = 1.0
            bigram_weight = 0.35
            char_weight = 0.22
            jamo1_weight = 0.0
            jamo2_weight = 0.18
            jamo3_weight = 0.08

        _add_feature(vec, "w:" + token, word_weight, size)

        if token_len >= 2:
            for i in range(token_len - 1):
                _add_feature(vec, "b:" + token[i : i + 2], bigram_weight, size)

        use_jamo = version in ("v2", "v3", "v4")
        if version in ("v5", "v6"):
            # v5+ keeps Hangul decomposition where it is meaningful, but does
            # not run Latin/code tokens through the jamo feature namespace.
            use_jamo = bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", token))

        if use_jamo:
            # Decomposed Hangul features let related forms such as
            # "반가워" and "반갑다" share more sub-character structure.
            decomposed = unicodedata.normalize("NFKD", token)

            if version in ("v4", "v5", "v6"):
                for char in decomposed:
                    _add_feature(vec, "j1:" + char, jamo1_weight, size)

            for i in range(len(decomposed) - 1):
                _add_feature(vec, "j2:" + decomposed[i : i + 2], jamo2_weight, size)
            for i in range(len(decomposed) - 2):
                _add_feature(vec, "j3:" + decomposed[i : i + 3], jamo3_weight, size)

        if version in ("v3", "v4", "v5", "v6"):
            # Syllable/character unigrams let colloquial variants share signal
            # without mapping any literal phrase directly to a route.
            for char in token:
                _add_feature(vec, "c:" + char, char_weight, size)
        elif version not in ("v1", "v2"):
            raise ValueError(f"Unknown vectorizer version: {version}")

        if version == "v6":
            # Boundary-aware character n-grams make typos and morphological
            # variants less brittle without needing a tokenizer vocabulary.
            padded = "^" + token + "$"
            for n, weight in ((2, 0.17), (3, 0.14), (4, 0.09)):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        _add_feature(vec, f"cg{n}:{padded[i:i+n]}", weight, size)

            # Prefix/suffix cues are especially useful for Korean endings and
            # English command morphology while still being language-agnostic.
            if token_len >= 3:
                for n in (2, 3):
                    if token_len >= n:
                        _add_feature(vec, f"pre{n}:{token[:n]}", 0.14, size)
                        _add_feature(vec, f"suf{n}:{token[-n:]}", 0.16, size)

            shape = []
            if re.search(r"[0-9]", token):
                shape.append("digit")
            if re.search(r"[A-Za-z]", token):
                shape.append("latin")
            if re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", token):
                shape.append("hangul")
            if "_" in token:
                shape.append("underscore")
            for item in shape:
                _add_feature(vec, "shape:" + item, 0.16, size)
            _add_feature(vec, f"len:{min(token_len, 8)}", 0.08, size)

    if version in ("v5", "v6"):
        # Add phrase context. This lets the router distinguish the same word
        # used in different intents, e.g. "공식 사이트" (research) from
        # "넓이 공식" (math), without hard-coding either route.
        for left, right in zip(tokens, tokens[1:]):
            _add_feature(vec, f"wb:{left}|{right}", 0.60, size)

    if version == "v6":
        # A one-token skip preserves useful local context when particles,
        # fillers, or typos interrupt an otherwise familiar phrase.
        for i in range(len(tokens) - 2):
            _add_feature(vec, f"ws:{tokens[i]}|{tokens[i+2]}", 0.24, size)

        if tokens:
            _add_feature(vec, "first:" + tokens[0], 0.20, size)
            _add_feature(vec, "last:" + tokens[-1], 0.24, size)

        # Operators and punctuation disappear from TOKEN_RE but are valuable
        # evidence for math/code intent. These are deliberately coarse so a
        # single symbol cannot dominate the classifier.
        punctuation_groups = {
            "question": r"[?？]",
            "equation": r"=",
            "mathop": r"[+×✕÷*/%^]",
            "codebrace": r"[{}\[\]]",
            "codepunct": r"[;:]",
            "quote": r"[\"']",
        }
        for name, pattern in punctuation_groups.items():
            count = len(re.findall(pattern, text))
            if count:
                _add_feature(vec, f"punct:{name}", 0.16 + min(count, 3) * 0.05, size)

        if re.search(r"https?://|www\.", text, re.IGNORECASE):
            _add_feature(vec, "shape:url", 0.45, size)
        if re.search(r"\b(?:git|npm|pip|python|javascript|typescript|sql|react|fastapi)\b", text, re.IGNORECASE):
            _add_feature(vec, "shape:devterm", 0.28, size)

    norm = torch.linalg.vector_norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@dataclass
class Example:
    text: str
    route: str


def load_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row.get("input"), str) or not isinstance(row.get("route"), str):
                raise ValueError(f"Invalid row at line {line_no}: input and route are required strings")
            examples.append(Example(row["input"], row["route"]))
    if len(examples) < 12:
        raise ValueError("Need at least 12 teacher examples for a useful smoke test")
    return examples


def stratified_split(examples: list[Example], val_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    by_route: dict[str, list[Example]] = {}
    for ex in examples:
        by_route.setdefault(ex.route, []).append(ex)
    rng = random.Random(seed)
    train: list[Example] = []
    val: list[Example] = []
    for group in by_route.values():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio)) if len(group) >= 3 else 1
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_scaffold(path: Path | None, synthetic_nodes: int, seed: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if path is None:
        rng = random.Random(seed)
        n_nodes = synthetic_nodes
        edges: set[tuple[int, int]] = set()
        # Ring guarantees connectivity; random shortcuts make it nontrivial.
        for i in range(n_nodes):
            edges.add((i, (i + 1) % n_nodes))
            edges.add(((i + 3) % n_nodes, i))
        while len(edges) < n_nodes * 8:
            a = rng.randrange(n_nodes)
            b = rng.randrange(n_nodes)
            if a != b:
                edges.add((a, b))
        ordered = sorted(edges)
        src = torch.tensor([a for a, _ in ordered], dtype=torch.long)
        dst = torch.tensor([b for _, b in ordered], dtype=torch.long)
        base = torch.ones(len(ordered), dtype=torch.float32)
        meta = {"kind": "synthetic-smoke-test", "nodes": n_nodes, "edges": len(ordered)}
        return n_nodes, src, dst, base, meta

    payload = json.loads(path.read_text(encoding="utf-8"))
    n_nodes = int(payload["n_nodes"])
    rows = payload["edges"]
    if not rows:
        raise ValueError("Scaffold contains no edges")
    src = torch.tensor([int(row["src"]) for row in rows], dtype=torch.long)
    dst = torch.tensor([int(row["dst"]) for row in rows], dtype=torch.long)
    base = torch.tensor([float(row.get("weight", 1.0)) for row in rows], dtype=torch.float32)
    if src.min() < 0 or dst.min() < 0 or src.max() >= n_nodes or dst.max() >= n_nodes:
        raise ValueError("Scaffold edge index out of range")
    meta = {k: v for k, v in payload.items() if k != "edges"}
    meta["kind"] = str(payload.get("dataset") or "malecns")
    return n_nodes, src, dst, base, meta


class FlyGraphRouter(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_nodes: int,
        n_classes: int,
        src: torch.Tensor,
        dst: torch.Tensor,
        base_weight: torch.Tensor,
        steps: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(vocab_size, n_nodes)
        self.edge_gain = nn.Parameter(torch.zeros(base_weight.numel()))
        self.output = nn.Linear(n_nodes, n_classes)
        self.norm = nn.LayerNorm(n_nodes)
        self.steps = steps
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("base_weight", base_weight)
        degree = torch.zeros(n_nodes, dtype=torch.float32)
        degree.index_add_(0, dst, torch.ones_like(base_weight))
        self.register_buffer("in_degree", degree.clamp_min(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.encoder(x))
        weights = self.base_weight * torch.sigmoid(self.edge_gain) * 2.0
        for _ in range(self.steps):
            messages = h[:, self.src] * weights.unsqueeze(0)
            agg = torch.zeros_like(h)
            agg.index_add_(1, self.dst, messages)
            agg = agg / self.in_degree.unsqueeze(0)
            h = self.norm(h + torch.tanh(agg))
        return self.output(h)


def batchify(
    examples: Iterable[Example],
    route_to_idx: dict[str, int],
    vocab_size: int,
    vectorizer_version: str = "v1",
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = list(examples)
    x = torch.stack([vectorize(ex.text, vocab_size, vectorizer_version) for ex in rows])
    y = torch.tensor([route_to_idx[ex.route] for ex in rows], dtype=torch.long)
    return x, y


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def _balanced_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> float:
    predictions = logits.argmax(dim=1)
    scores: list[float] = []
    for class_idx in range(n_classes):
        mask = y == class_idx
        if bool(mask.any()):
            scores.append(float((predictions[mask] == y[mask]).float().mean().item()))
    return sum(scores) / len(scores) if scores else 0.0


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return _accuracy_from_logits(model(x), y)


def _expected_calibration_error(probabilities: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(y)
    ece = torch.tensor(0.0)
    for i in range(bins):
        low = i / bins
        high = (i + 1) / bins
        if i == bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if bool(mask.any()):
            bin_conf = confidences[mask].mean()
            bin_acc = correct[mask].float().mean()
            ece += mask.float().mean() * (bin_conf - bin_acc).abs()
    return float(ece.item())


def _fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, dict[str, float]]:
    logits = logits.detach().to(dtype=torch.float32)
    y = y.detach()
    before_nll = float(F.cross_entropy(logits, y).item())

    # A deterministic 1-D search is extremely stable for a tiny router and
    # avoids adding another optimizer state to checkpoint creation.
    candidates = torch.logspace(math.log10(0.25), math.log10(6.0), steps=181)
    losses = torch.stack([F.cross_entropy(logits / t, y) for t in candidates])
    best_index = int(losses.argmin().item())
    temperature = float(candidates[best_index].item())

    calibrated = torch.softmax(logits / temperature, dim=1)
    uncalibrated = torch.softmax(logits, dim=1)
    return temperature, {
        "nll_before": before_nll,
        "nll_after": float(losses[best_index].item()),
        "ece_before": _expected_calibration_error(uncalibrated, y),
        "ece_after": _expected_calibration_error(calibrated, y),
    }


def _recommend_gate(
    probabilities: torch.Tensor,
    y: torch.Tensor,
    *,
    target_precision: float,
    min_coverage: float,
) -> dict[str, float]:
    top2 = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1)
    confidence = top2.values[:, 0]
    second = top2.values[:, 1] if probabilities.shape[1] > 1 else torch.zeros_like(confidence)
    margin = confidence - second
    predictions = top2.indices[:, 0]
    correct = predictions.eq(y)

    best: tuple[float, float, float, float] | None = None
    fallback: tuple[float, float, float, float] | None = None

    confidence_grid = [0.40 + i * 0.025 for i in range(19)]
    margin_grid = [i * 0.025 for i in range(13)]

    for min_conf in confidence_grid:
        for min_margin in margin_grid:
            accepted = (confidence >= min_conf) & (margin >= min_margin)
            count = int(accepted.sum().item())
            if count == 0:
                continue
            coverage = count / len(y)
            precision = float(correct[accepted].float().mean().item())

            candidate = (coverage, precision, min_conf, min_margin)
            if coverage >= min_coverage and precision >= target_precision:
                if best is None or (coverage, precision) > (best[0], best[1]):
                    best = candidate

            if coverage >= max(0.30, min_coverage * 0.65):
                if fallback is None or (precision, coverage) > (fallback[1], fallback[0]):
                    fallback = candidate

    selected = best or fallback
    if selected is None:
        return {
            "min_confidence": 0.55,
            "min_margin": 0.10,
            "validation_precision": 0.0,
            "validation_coverage": 0.0,
        }

    coverage, precision, min_conf, min_margin = selected
    return {
        "min_confidence": round(min_conf, 4),
        "min_margin": round(min_margin, 4),
        "validation_precision": round(precision, 6),
        "validation_coverage": round(coverage, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlyGPT's connectome-inspired task router")
    parser.add_argument("--dataset", type=Path, default=Path("training/teacher_seed.jsonl"))
    parser.add_argument("--scaffold", type=Path, help="JSON scaffold built from MaleCNS; omit only for smoke tests")
    parser.add_argument("--out", type=Path, default=Path("artifacts/fly_router_v0_2.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--synthetic-nodes", type=int, default=96)
    parser.add_argument("--vectorizer-version", choices=("v1", "v2", "v3", "v4", "v5", "v6"), default="v1")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many epochs without val improvement; 0 disables early stopping")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.0, help="Desired target-vs-runner-up logit margin")
    parser.add_argument("--margin-weight", type=float, default=0.0, help="Weight for the multiclass margin regularizer")
    parser.add_argument("--feature-dropout", type=float, default=0.0, help="Input feature dropout used only during training")
    parser.add_argument("--selection-metric", choices=("accuracy", "balanced_accuracy"), default="balanced_accuracy")
    parser.add_argument("--gate-target-precision", type=float, default=0.97)
    parser.add_argument("--gate-min-coverage", type=float, default=0.55)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.feature_dropout < 1.0:
        raise ValueError("--feature-dropout must be in [0, 1)")
    if args.margin < 0 or args.margin_weight < 0:
        raise ValueError("--margin and --margin-weight must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_examples(args.dataset)
    routes = sorted({ex.route for ex in examples})
    route_to_idx = {route: i for i, route in enumerate(routes)}
    train, val = stratified_split(examples, args.val_ratio, args.seed)

    n_nodes, src, dst, base, scaffold_meta = load_scaffold(args.scaffold, args.synthetic_nodes, args.seed)
    model = FlyGraphRouter(args.vocab_size, n_nodes, len(routes), src, dst, base, args.steps)

    train_x, train_y = batchify(train, route_to_idx, args.vocab_size, args.vectorizer_version)
    val_x, val_y = batchify(val, route_to_idx, args.vocab_size, args.vectorizer_version)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_counts = torch.bincount(train_y, minlength=len(routes)).to(dtype=torch.float32)
    class_weights = class_counts.sum() / (len(routes) * class_counts.clamp_min(1.0))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_state = None
    best_score = -1.0
    best_val = -1.0
    best_balanced = -1.0
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_input = (
            F.dropout(train_x, p=args.feature_dropout, training=True)
            if args.feature_dropout > 0
            else train_x
        )
        logits = model(train_input)
        ce_loss = loss_fn(logits, train_y)
        loss = ce_loss

        margin_loss_value = torch.tensor(0.0)
        if args.margin_weight > 0 and args.margin > 0 and len(routes) > 1:
            target_logits = logits.gather(1, train_y.unsqueeze(1)).squeeze(1)
            target_mask = F.one_hot(train_y, num_classes=len(routes)).bool()
            strongest_other = logits.masked_fill(target_mask, float("-inf")).max(dim=1).values
            margin_loss_value = F.relu(args.margin - target_logits + strongest_other).mean()
            loss = loss + args.margin_weight * margin_loss_value

        loss.backward()
        if args.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_acc = _accuracy_from_logits(val_logits, val_y)
            val_balanced = _balanced_accuracy_from_logits(val_logits, val_y, len(routes))

        score = val_acc if args.selection_metric == "accuracy" else val_balanced
        improved = score > best_score + 1e-9 or (
            math.isclose(score, best_score, abs_tol=1e-9) and val_acc > best_val
        )
        if improved:
            best_score = score
            best_val = val_acc
            best_balanced = val_balanced
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                train_logits = model(train_x)
                train_acc = _accuracy_from_logits(train_logits, train_y)
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                f"margin={margin_loss_value.item():.4f} train_acc={train_acc:.3f} "
                f"val_acc={val_acc:.3f} val_bal={val_balanced:.3f}"
            )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                f"best_{args.selection_metric}={best_score:.3f}"
            )
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(val_x)

    temperature, calibration_metrics = _fit_temperature(val_logits, val_y)
    calibrated_probabilities = torch.softmax(val_logits / temperature, dim=1)
    gate = _recommend_gate(
        calibrated_probabilities,
        val_y,
        target_precision=args.gate_target_precision,
        min_coverage=args.gate_min_coverage,
    )

    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    calibration = {
        **calibration_metrics,
        **gate,
        "target_precision": args.gate_target_precision,
        "minimum_coverage": args.gate_min_coverage,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "routes": routes,
            "vocab_size": args.vocab_size,
            "n_nodes": n_nodes,
            "steps": args.steps,
            "src": src,
            "dst": dst,
            "base_weight": base,
            "scaffold_meta": scaffold_meta,
            "teacher_dataset": str(args.dataset),
            "teacher_dataset_sha256": dataset_sha256,
            "train_examples": len(train),
            "validation_examples": len(val),
            "best_val_accuracy": best_val,
            "best_val_balanced_accuracy": best_balanced,
            "selection_metric": args.selection_metric,
            "best_epoch": best_epoch,
            "seed": args.seed,
            "vectorizer_version": args.vectorizer_version,
            "temperature": temperature,
            "calibration": calibration,
            "training_config": {
                "label_smoothing": args.label_smoothing,
                "margin": args.margin,
                "margin_weight": args.margin_weight,
                "feature_dropout": args.feature_dropout,
                "weight_decay": args.weight_decay,
            },
        },
        args.out,
    )

    print(
        f"calibration temperature={temperature:.3f} "
        f"nll={calibration_metrics['nll_before']:.4f}->{calibration_metrics['nll_after']:.4f} "
        f"ece={calibration_metrics['ece_before']:.3f}->{calibration_metrics['ece_after']:.3f}"
    )
    print(
        f"gate min_confidence={gate['min_confidence']:.3f} min_margin={gate['min_margin']:.3f} "
        f"val_precision={gate['validation_precision']:.1%} val_coverage={gate['validation_coverage']:.1%}"
    )
    print(
        f"saved={args.out} best_val_accuracy={best_val:.3f} "
        f"best_val_balanced_accuracy={best_balanced:.3f} best_epoch={best_epoch} routes={routes}"
    )


if __name__ == "__main__":
    main()
