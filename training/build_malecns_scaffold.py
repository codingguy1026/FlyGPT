from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malecns import MaleCNSConnectome


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact directed MaleCNS v1.0 scaffold for FlyGPT training"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns/parts"))
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/malecns/metadata"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training/malecns_scaffold.json"),
    )
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--edges", type=int, default=4096)
    parser.add_argument("--sample-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.nodes < 16:
        raise ValueError("--nodes must be at least 16")
    if args.edges < args.nodes:
        raise ValueError("--edges should be at least as large as --nodes")
    if args.sample_rows < args.edges:
        raise ValueError("--sample-rows should be at least as large as --edges")

    with MaleCNSConnectome(args.data_dir, args.metadata_dir) as mcns:
        sampled = mcns.connection.execute(
            f"""
            WITH sampled AS (
                SELECT pre_pt_root_id, post_pt_root_id, syn_count
                FROM connections
                USING SAMPLE reservoir({int(args.sample_rows)} ROWS)
                REPEATABLE ({int(args.seed)})
            ), node_strength AS (
                SELECT root_id, SUM(weight) AS strength
                FROM (
                    SELECT pre_pt_root_id AS root_id, syn_count AS weight FROM sampled
                    UNION ALL
                    SELECT post_pt_root_id AS root_id, syn_count AS weight FROM sampled
                )
                GROUP BY root_id
                ORDER BY strength DESC
                LIMIT {int(args.nodes)}
            )
            SELECT
                s.pre_pt_root_id,
                s.post_pt_root_id,
                s.syn_count
            FROM sampled AS s
            JOIN node_strength AS a ON a.root_id = s.pre_pt_root_id
            JOIN node_strength AS b ON b.root_id = s.post_pt_root_id
            WHERE s.pre_pt_root_id <> s.post_pt_root_id
            ORDER BY s.syn_count DESC
            LIMIT {int(args.edges)}
            """
        ).fetchall()

    if len(sampled) < args.nodes:
        raise RuntimeError(
            f"Only {len(sampled)} internal edges survived the sample. "
            "Increase --sample-rows or reduce --nodes."
        )

    body_ids = sorted(
        {int(a) for a, _, _ in sampled}
        | {int(b) for _, b, _ in sampled}
    )
    body_ids = body_ids[: args.nodes]
    index = {body_id: i for i, body_id in enumerate(body_ids)}

    filtered = [
        (int(a), int(b), int(w))
        for a, b, w in sampled
        if int(a) in index and int(b) in index and int(a) != int(b)
    ]
    if not filtered:
        raise RuntimeError("No usable MaleCNS edges after node remapping")

    max_log = max(math.log1p(w) for _, _, w in filtered)
    edges = [
        {
            "src": index[a],
            "dst": index[b],
            "weight": round(math.log1p(w) / max_log, 6),
            "syn_count": w,
        }
        for a, b, w in filtered[: args.edges]
    ]

    payload = {
        "source": "MaleCNS v1.0 curated whole-CNS connectome",
        "seed": args.seed,
        "sample_rows": args.sample_rows,
        "n_nodes": len(body_ids),
        "n_edges": len(edges),
        "body_ids": body_ids,
        "edges": edges,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved={args.out} nodes={len(body_ids)} edges={len(edges)}")


if __name__ == "__main__":
    main()
