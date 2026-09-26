from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router_runtime import FlyRouterRuntime


def fail(message: str) -> None:
    print(f"FAIL {message}")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a FlyGPT checkpoint is using the FlyWire scaffold normally")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    args = parser.parse_args()

    if not args.model.is_file():
        fail(f"model not found: {args.model}")
    if not args.scaffold.is_file():
        fail(f"scaffold not found: {args.scaffold}")

    runtime = FlyRouterRuntime(args.model)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    scaffold = json.loads(args.scaffold.read_text(encoding="utf-8"))

    source = str(scaffold.get("source", ""))
    if "FlyWire" not in source or "v783" not in source:
        fail(f"unexpected scaffold source: {source!r}")

    if runtime.scaffold_meta.get("kind") != "flywire":
        fail(f"checkpoint scaffold kind is {runtime.scaffold_meta.get('kind')!r}, expected 'flywire'")

    n_nodes = int(scaffold.get("n_nodes", 0))
    edges = scaffold.get("edges") or []
    root_ids = scaffold.get("root_ids") or []

    if n_nodes < 16 or len(root_ids) != n_nodes:
        fail(f"invalid node metadata: n_nodes={n_nodes}, root_ids={len(root_ids)}")
    if not edges:
        fail("scaffold has no edges")
    if runtime.n_nodes != n_nodes:
        fail(f"checkpoint node count {runtime.n_nodes} != scaffold node count {n_nodes}")
    if len(runtime.src) != len(edges):
        fail(f"checkpoint edge count {len(runtime.src)} != scaffold edge count {len(edges)}")

    scaffold_src = torch.tensor([int(row["src"]) for row in edges], dtype=torch.long)
    scaffold_dst = torch.tensor([int(row["dst"]) for row in edges], dtype=torch.long)
    scaffold_weight = torch.tensor([float(row.get("weight", 1.0)) for row in edges], dtype=torch.float32)

    if not torch.equal(runtime.src, scaffold_src) or not torch.equal(runtime.dst, scaffold_dst):
        fail("checkpoint topology does not match the scaffold file")
    if not torch.allclose(runtime.base_weight, scaffold_weight, atol=1e-7, rtol=1e-6):
        fail("checkpoint base edge weights do not match the scaffold file")

    if not torch.isfinite(runtime.base_weight).all() or float(runtime.base_weight.min()) <= 0:
        fail("base edge weights contain invalid values")

    edge_gain = checkpoint["model_state"]["edge_gain"].detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(edge_gain).all():
        fail("learned edge gains contain non-finite values")

    with torch.no_grad():
        learned = runtime.base_weight * torch.sigmoid(edge_gain) * 2.0
    if not torch.isfinite(learned).all() or float(learned.min()) <= 0:
        fail("learned edge weights contain invalid values")

    probes = [
        "안녕하세요",
        "37 곱하기 14 계산해줘",
        "파이썬 리스트 정렬 방법 알려줘",
        "아까 내가 고른 옵션 기억나?",
        "오늘 최신 기술 뉴스 찾아줘",
        "이 문단을 두 문장으로 요약해줘",
    ]

    predicted_routes: set[str] = set()
    propagated = False

    for probe in probes:
        pred = runtime.predict(probe, top_k=3)
        predicted_routes.add(str(pred["route"]))
        trace = pred.get("trace") or []
        if len(trace) != runtime.steps + 1:
            fail(f"trace length mismatch for probe {probe!r}")

        first = trace[0].get("activations") or []
        last = trace[-1].get("activations") or []
        if len(first) != runtime.n_nodes or len(last) != runtime.n_nodes:
            fail(f"activation width mismatch for probe {probe!r}")

        delta = sum(abs(float(a) - float(b)) for a, b in zip(first, last))
        propagated = propagated or delta > 1e-4

        if not math.isfinite(float(pred["confidence"])):
            fail(f"non-finite confidence for probe {probe!r}")

    if not propagated:
        fail("graph propagation did not change activations")
    if len(predicted_routes) < 3:
        fail(f"router appears collapsed; only predicted routes: {sorted(predicted_routes)}")

    print("PASS FlyWire scaffold metadata")
    print(f"PASS topology match: nodes={runtime.n_nodes} edges={len(runtime.src)} steps={runtime.steps}")
    print(f"PASS graph propagation changed activations")
    print(f"PASS router is not collapsed: routes_seen={sorted(predicted_routes)}")
    print(
        "PASS learned edge gains are finite "
        f"(min={float(edge_gain.min()):.5f}, max={float(edge_gain.max()):.5f})"
    )
    print(
        f"PASS checkpoint={args.model.name} vectorizer={runtime.vectorizer_version} "
        f"best_val_accuracy={runtime.best_val_accuracy:.3f}"
    )
    print("RESULT: fly brain runtime looks structurally healthy")


if __name__ == "__main__":
    main()
