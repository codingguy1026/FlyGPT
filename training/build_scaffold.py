from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectome import FlyWireConnectome


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact directed FlyWire scaffold for FlyGPT training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/flywire_parts"))
    parser.add_argument("--out", type=Path, default=Path("training/flywire_scaffold.json"))
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--edges", type=int, default=4096)
    parser.add_argument("--sample-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.nodes < 16:
        raise ValueError("--nodes must be at least 16")
    if args.edges < args.nodes:
        raise ValueError("--edges should be at least as large as --nodes")

    with FlyWireConnectome(args.data_dir) as fw:
        # A reservoir sample keeps the one-time scaffold build bounded while still
        # using real v783 directed connections and synapse counts.
        sampled = fw.connection.execute(
            f"""
            WITH sampled AS (
                SELECT pre_pt_root_id, post_pt_root_id, syn_count
                FROM connections
                USING SAMPLE reservoir({int(args.sample_rows)} ROWS) REPEATABLE ({int(args.seed)})
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
            SELECT s.pre_pt_root_id, s.post_pt_root_id, s.syn_count
            FROM sampled s
            JOIN node_strength a ON a.root_id = s.pre_pt_root_id
            JOIN node_strength b ON b.root_id = s.post_pt_root_id
            ORDER BY s.syn_count DESC
            LIMIT {int(args.edges)}
            """
        ).fetchall()

    if len(sampled) < args.nodes:
        raise RuntimeError(
            f"Only {len(sampled)} internal edges survived the sample. Increase --sample-rows or reduce --nodes."
        )

    root_ids = sorted({int(a) for a, _, _ in sampled} | {int(b) for _, b, _ in sampled})
    # Keep the model size bounded even if the selected edges mention slightly more nodes.
    root_ids = root_ids[: args.nodes]
    index = {root_id: i for i, root_id in enumerate(root_ids)}
    filtered = [(int(a), int(b), int(w)) for a, b, w in sampled if int(a) in index and int(b) in index and int(a) != int(b)]
    if not filtered:
        raise RuntimeError("No usable edges after node remapping")

    max_log = max(math.log1p(w) for _, _, w in filtered)
    edges = [
        {"src": index[a], "dst": index[b], "weight": round(math.log1p(w) / max_log, 6), "syn_count": w}
        for a, b, w in filtered[: args.edges]
    ]

    payload = {
        "source": "FlyWire v783 proofread connections",
        "seed": args.seed,
        "sample_rows": args.sample_rows,
        "n_nodes": len(root_ids),
        "n_edges": len(edges),
        "root_ids": root_ids,
        "edges": edges,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.out} nodes={len(root_ids)} edges={len(edges)}")


if __name__ == "__main__":
    main()
