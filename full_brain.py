"""Experimental directed signal propagation over ALL annotated MaleCNS neurons.

Unlike the 256-node trained task router, this traverses the entire official
Feather connection file once per step. The graph is not loaded into RAM.
This is an offline structural experiment, not a biological firing simulation,
not an LLM, and not part of the live /api/chat latency budget.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from full_connectome import DATASET, DEFAULT_PATH, FullMaleCNS

DEFAULT_ANNOTATIONS = Path(
    "data/malecns_full/body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
ANNOTATIONS_SOURCE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


def load_annotated_neurons(path: str | Path = DEFAULT_ANNOTATIONS) -> tuple[int, ...]:
    """Retain curated neuronal bodyIds, including untyped/TBC subclasses.

    The full weights table contains untraced segments and glia. Merely
    counting unique endpoint IDs would NOT define a whole-neuron graph.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Neuron annotations not found: {path}\n"
            f"Download the official 13 MB Feather file from {ANNOTATIONS_SOURCE}"
        )
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise RuntimeError(
            "Install pyarrow: python -m pip install -r requirements-full-connectome.txt"
        ) from exc
    table = feather.read_table(path, columns=["bodyId", "superclass"])
    body_ids = table.column("bodyId").to_pylist()
    superclasses = table.column("superclass").to_pylist()
    neurons: set[int] = set()
    for body_id, superclass in zip(body_ids, superclasses):
        name = str(superclass or "").strip()
        # A nonempty superclass is the dataset's neuron-classification gate.
        # Exclude explicit glia even if an upstream classification changes.
        if body_id is not None and name and "glia" not in name.lower():
            neurons.add(int(body_id))
    if not neurons:
        raise ValueError("No annotated neuronal body IDs in annotations file")
    return tuple(sorted(neurons))


class FullBrainPropagator:
    """Stream all directed edges, with eligibility from official annotations.

    Seeds are body IDs, never row indices. At each step:
      next[post] = sum(active[pre] * weight(pre, post))
    Then max-normalize within the step so numerical values remain bounded.
    All eligible weighted edges are considered: no top-K edge sampling or
    hidden synapse threshold. A safety limit causes a loud error rather than
    silently dropping activated neurons.
    """

    def __init__(
        self,
        *,
        weights_path: str | Path = DEFAULT_PATH,
        annotations_path: str | Path = DEFAULT_ANNOTATIONS,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.annotations_path = Path(annotations_path)

    def propagate(
        self,
        seed_id: int,
        *,
        steps: int = 3,
        min_weight: int = 1,
        max_active: int = 200_000,
        top: int = 10,
        progress_every: int = 0,
    ) -> dict[str, Any]:
        if not 1 <= steps <= 4:
            raise ValueError("steps must be between 1 and 4")
        if min_weight < 1:
            raise ValueError("min_weight must be >= 1")
        if not 1 <= top <= 100:
            raise ValueError("top must be between 1 and 100")
        if max_active < 1:
            raise ValueError("max_active must be positive")
        if progress_every < 0:
            raise ValueError("progress_every must be nonnegative")

        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc

        neuron_ids = load_annotated_neurons(self.annotations_path)
        neuron_set = set(neuron_ids)
        seed_id = int(seed_id)
        if seed_id not in neuron_set:
            raise ValueError(
                f"body ID {seed_id} is not a curated neuronal body in this annotation file"
            )
        if len(neuron_ids) > max_active:
            raise ValueError(
                f"max_active={max_active} is smaller than the full annotated graph "
                f"({len(neuron_ids)} neurons). Raise it for an uncapped whole-graph run."
            )

        neuron_array = pa.array(neuron_ids, type=pa.int64())
        active: dict[int, float] = {seed_id: 1.0}
        stages: list[dict[str, Any]] = []
        total_rows_scanned = 0

        with FullMaleCNS(self.weights_path) as graph:
            for step in range(1, steps + 1):
                active_ids = tuple(sorted(active))
                active_arr = pa.array(active_ids, type=pa.int64())
                amplitudes = np.asarray(
                    [active[body] for body in active_ids], dtype=np.float64
                )
                accumulated: dict[int, float] = defaultdict(float)
                contributing_rows = 0
                scanned = 0

                for batch_index, batch in enumerate(graph._batches(), start=1):
                    scanned += batch.num_rows
                    pre = batch.column(batch.schema.get_field_index("body_pre"))
                    post = batch.column(batch.schema.get_field_index("body_post"))
                    weight = batch.column(batch.schema.get_field_index("weight"))

                    eligible = pc.and_(
                        pc.and_(
                            pc.is_in(pre, value_set=active_arr),
                            pc.is_in(post, value_set=neuron_array),
                        ),
                        pc.greater_equal(weight, min_weight),
                    )
                    selected = batch.filter(eligible)
                    if selected.num_rows:
                        contributing_rows += selected.num_rows
                        source = selected.column(
                            selected.schema.get_field_index("body_pre")
                        )
                        destination = selected.column(
                            selected.schema.get_field_index("body_post")
                        )
                        source_indices = pc.index_in(
                            source, value_set=active_arr
                        ).to_numpy(zero_copy_only=False)
                        destinations = destination.to_numpy(zero_copy_only=False)
                        strengths = selected.column(
                            selected.schema.get_field_index("weight")
                        ).to_numpy(zero_copy_only=False)
                        contributions = amplitudes[source_indices] * strengths
                        unique_posts, inverse = np.unique(
                            destinations, return_inverse=True
                        )
                        grouped = np.bincount(
                            inverse, weights=contributions, minlength=len(unique_posts)
                        )
                        for body, value in zip(unique_posts, grouped):
                            accumulated[int(body)] += float(value)
                        if len(accumulated) > max_active:
                            raise RuntimeError(
                                f"Step {step} exceeded max_active={max_active}. "
                                "No signals were truncated; aborting."
                            )

                    if progress_every and batch_index % progress_every == 0:
                        print(
                            f"[full-brain] step={step}/{steps} "
                            f"batches={batch_index}/{graph._reader.num_record_batches} "
                            f"rows={scanned:,} reached={len(accumulated):,}",
                            file=sys.stderr,
                            flush=True,
                        )

                total_rows_scanned += scanned
                if not accumulated:
                    active = {}
                    stages.append({
                        "step": step, "rows_scanned": scanned,
                        "contributing_connection_rows": contributing_rows,
                        "active_neurons": 0, "top": [],
                    })
                    break

                scale = max(accumulated.values())
                if scale <= 0:
                    raise RuntimeError("Unexpected nonpositive propagation signal")
                active = {node: value / scale for node, value in accumulated.items()}
                ranking = sorted(active.items(), key=lambda item: (-item[1], item[0]))
                stages.append({
                    "step": step,
                    "rows_scanned": scanned,
                    "contributing_connection_rows": contributing_rows,
                    "active_neurons": len(active),
                    "normalization": "per-step maximum = 1",
                    "top": [
                        {"body_id": body, "relative_activity": round(score, 8)}
                        for body, score in ranking[:top]
                    ],
                })

        return {
            "dataset": DATASET,
            "weights_path": str(self.weights_path),
            "annotations_path": str(self.annotations_path),
            "annotated_neurons_eligible": len(neuron_ids),
            "seed_body_id": seed_id,
            "steps_requested": steps,
            "steps_completed": len(stages),
            "min_edge_weight": min_weight,
            "safety_max_active": max_active,
            "rows_scanned_total": total_rows_scanned,
            "stages": stages,
            "full_source_file_scanned_each_completed_step": True,
            "uses_all_eligible_neurons": True,
            "method": "read-only streamed weighted directed propagation",
            "biological_firing_model": False,
            "trained_language_reasoning_model": False,
            "integrated_into_chat": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline full-neuron MaleCNS signal propagation (not /api/chat)."
    )
    parser.add_argument("seed_id", type=int, help="Curated neuron bodyId")
    parser.add_argument("--weights", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument("--max-active", type=int, default=200_000)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    brain = FullBrainPropagator(
        weights_path=args.weights,
        annotations_path=args.annotations,
    )
    result = brain.propagate(
        args.seed_id,
        steps=args.steps,
        min_weight=args.min_weight,
        max_active=args.max_active,
        top=args.top,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
