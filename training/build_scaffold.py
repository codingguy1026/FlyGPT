from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectome import MaleCNSConnectome


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact directed Janelia MaleCNS v1.0 scaffold for FlyGPT training"
    )
    # Accepted only so old shell commands do not fail. MaleCNS data is queried
    # through neuPrint and this path is intentionally ignored.
    parser.add_argument("--data-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("training/malecns_scaffold.json"))
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--edges", type=int, default=4096)
    parser.add_argument("--sample-rows", type=int, default=200_000, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-edges", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.nodes < 16:
        raise ValueError("--nodes must be at least 16")
    if args.edges < args.nodes:
        raise ValueError("--edges should be at least as large as --nodes")
    if args.candidate_edges < args.edges:
        raise ValueError("--candidate-edges must be >= --edges")

    with MaleCNSConnectome() as mcns:
        sampled = mcns.scaffold_edges(candidate_edges=args.candidate_edges)
        dataset = mcns.dataset

    strength: dict[int, int] = {}
    for pre, post, weight in sampled:
        strength[pre] = strength.get(pre, 0) + weight
        strength[post] = strength.get(post, 0) + weight

    root_ids = [
        root_id
        for root_id, _ in sorted(
            strength.items(),
            key=lambda item: (-item[1], item[0]),
        )[: args.nodes]
    ]
    index = {root_id: i for i, root_id in enumerate(root_ids)}

    filtered = [
        (pre, post, weight)
        for pre, post, weight in sampled
        if pre in index and post in index and pre != post
    ]
    filtered.sort(key=lambda edge: (-edge[2], edge[0], edge[1]))
    filtered = filtered[: args.edges]

    if len(root_ids) < args.nodes:
        raise RuntimeError(
            f"Only {len(root_ids)} nodes were available. Increase --candidate-edges."
        )
    if not filtered:
        raise RuntimeError("No usable MaleCNS edges after node remapping")

    max_log = max(math.log1p(weight) for _, _, weight in filtered)
    edges = [
        {
            "src": index[pre],
            "dst": index[post],
            "weight": round(math.log1p(weight) / max_log, 6),
            "syn_count": weight,
        }
        for pre, post, weight in filtered
    ]

    payload = {
        "source": "Janelia MaleCNS v1.0 via neuPrint",
        "dataset": dataset,
        "seed": args.seed,
        "candidate_edges": args.candidate_edges,
        "n_nodes": len(root_ids),
        "n_edges": len(edges),
        "root_ids": root_ids,
        "edges": edges,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.out} nodes={len(root_ids)} edges={len(edges)} dataset={dataset}")


if __name__ == "__main__":
    main()
