from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9_]+")


def stable_bucket(token: str, size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def vectorize(text: str, size: int) -> torch.Tensor:
    vec = torch.zeros(size, dtype=torch.float32)
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    for token in tokens:
        vec[stable_bucket("w:" + token, size)] += 1.0
        if len(token) >= 2:
            for i in range(len(token) - 1):
                vec[stable_bucket("b:" + token[i : i + 2], size)] += 0.35
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
    meta["kind"] = "flywire"
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


def batchify(examples: Iterable[Example], route_to_idx: dict[str, int], vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = list(examples)
    x = torch.stack([vectorize(ex.text, vocab_size) for ex in rows])
    y = torch.tensor([route_to_idx[ex.route] for ex in rows], dtype=torch.long)
    return x, y


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(dim=1) == y).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlyGPT's first connectome-inspired task router")
    parser.add_argument("--dataset", type=Path, default=Path("training/teacher_seed.jsonl"))
    parser.add_argument("--scaffold", type=Path, help="JSON scaffold built from FlyWire; omit only for smoke tests")
    parser.add_argument("--out", type=Path, default=Path("artifacts/fly_router_v0_1.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--synthetic-nodes", type=int, default=96)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_examples(args.dataset)
    routes = sorted({ex.route for ex in examples})
    route_to_idx = {route: i for i, route in enumerate(routes)}
    train, val = stratified_split(examples, args.val_ratio, args.seed)

    n_nodes, src, dst, base, scaffold_meta = load_scaffold(args.scaffold, args.synthetic_nodes, args.seed)
    model = FlyGraphRouter(args.vocab_size, n_nodes, len(routes), src, dst, base, args.steps)

    train_x, train_y = batchify(train, route_to_idx, args.vocab_size)
    val_x, val_y = batchify(val, route_to_idx, args.vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

        val_acc = accuracy(model, val_x, val_y)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_acc = accuracy(model, train_x, train_y)
            print(f"epoch={epoch:03d} loss={loss.item():.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    assert best_state is not None
    model.load_state_dict(best_state)
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
            "best_val_accuracy": best_val,
            "seed": args.seed,
        },
        args.out,
    )
    print(f"saved={args.out} best_val_accuracy={best_val:.3f} routes={routes}")


if __name__ == "__main__":
    main()
