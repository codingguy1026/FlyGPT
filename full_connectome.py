"""Read the full MaleCNS v1.0 connection table without loading it into RAM.

Phase 1: immutable, memory-mapped Feather access for dataset inspection, exact
full-table audits, and bounded neighborhood lookups. This is NOT the trained
FlyGraphRouter and does not silently replace its 256-node checkpoint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterator

DEFAULT_PATH = Path(
    "data/malecns_full/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
DATASET = "male-cns:v1.0"
SOURCE = "https://male-cns.janelia.org/download/"
EXPECTED_COLUMNS = ("body_pre", "body_post", "weight")


def _arrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError(
            "Full MaleCNS access requires pyarrow. "
            "Install: python -m pip install -r requirements-full-connectome.txt"
        ) from exc
    return pa, pc, ipc


def _value(scalar: Any) -> int:
    return int(scalar.as_py() or 0)


class FullMaleCNS:
    """Lazy, read-only view over every row of the official weights file.

    Full weights include untraced segments as well as annotated neurons.
    Do not present distinct endpoint IDs as '166,691 annotated neurons'.
    Neighborhood lookups perform full scans; use an indexed/sparse engine
    before wiring this into the latency-sensitive chat endpoint.
    """

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._mapped: Any = None
        self._reader: Any = None
        self._pc: Any = None

    def __enter__(self) -> "FullMaleCNS":
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Missing MaleCNS Feather file: {self.path}. "
                f"Download from {SOURCE} after checking available disk space."
            )
        pa, pc, ipc = _arrow()
        mapped = pa.memory_map(str(self.path), "r")
        try:
            reader = ipc.open_file(mapped)
            missing = set(EXPECTED_COLUMNS) - set(reader.schema.names)
            if missing:
                raise ValueError(f"Missing required Feather columns: {sorted(missing)}")
            for name in EXPECTED_COLUMNS:
                if not pa.types.is_integer(reader.schema.field(name).type):
                    raise ValueError(f"{name} must contain integer values")
        except Exception:
            mapped.close()
            raise
        self._mapped = mapped
        self._reader = reader
        self._pc = pc
        return self

    def __exit__(self, *_args: Any) -> None:
        self._reader = None
        if self._mapped is not None:
            self._mapped.close()
            self._mapped = None
        self._pc = None

    def _batches(self) -> Iterator[Any]:
        if self._reader is None:
            raise RuntimeError("Use FullMaleCNS inside a 'with' block")
        for index in range(self._reader.num_record_batches):
            yield self._reader.get_batch(index)

    def inspect(self) -> dict[str, Any]:
        """Read schema and batch count, without scanning the 1+ GB data file."""
        if self._reader is None:
            raise RuntimeError("The Feather file is not open")
        return {
            "dataset": DATASET,
            "source": SOURCE,
            "path": str(self.path),
            "file_bytes": self.path.stat().st_size,
            "record_batches": self._reader.num_record_batches,
            "columns": {f.name: str(f.type) for f in self._reader.schema},
            "scope": "all segment-to-segment weight rows, not only traced neurons",
            "full_scan_performed": False,
        }

    def audit(self) -> dict[str, Any]:
        """Visit every batch and report counts; never sample or truncate."""
        pc = self._pc
        rows = 0
        synapses = 0
        self_edges = 0
        nonpositive_weights = 0
        rows_with_nulls = 0
        for batch in self._batches():
            pre = batch.column(batch.schema.get_field_index("body_pre"))
            post = batch.column(batch.schema.get_field_index("body_post"))
            weight = batch.column(batch.schema.get_field_index("weight"))
            rows += batch.num_rows
            synapses += _value(pc.sum(weight))
            self_edges += _value(pc.sum(pc.equal(pre, post).cast("int64")))
            nonpositive_weights += _value(
                pc.sum(pc.less_equal(weight, 0).cast("int64"))
            )
            bad = pc.or_(
                pc.or_(pc.is_null(pre), pc.is_null(post)),
                pc.is_null(weight),
            )
            rows_with_nulls += _value(pc.sum(bad.cast("int64")))
        return {
            "dataset": DATASET,
            "source": SOURCE,
            "rows_scanned": rows,
            "synapse_weight_sum": synapses,
            "self_connections": self_edges,
            "nonpositive_weights": nonpositive_weights,
            "rows_with_nulls": rows_with_nulls,
            "full_scan_performed": True,
            "unique_annotated_neurons": None,
            "note": "This is a segment-level source audit, not a neuron annotation count.",
        }

    def neighbors(
        self,
        body_id: int,
        *,
        direction: str = "both",
        min_weight: int = 1,
        limit: int = 20,
        partner_id: int | None = None,
    ) -> dict[str, Any]:
        """Aggregate matching rows from ALL batches; return strongest partners.

        This intentionally scans the full file and is for offline exploration.
        The result limit applies only after aggregation, never to data scanned.
        """
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be incoming, outgoing, or both")
        if body_id < 0 or (partner_id is not None and partner_id < 0):
            raise ValueError("body IDs must be nonnegative")
        if min_weight < 1 or not 1 <= limit <= 1000:
            raise ValueError("min_weight must be >=1 and limit must be 1..1000")

        pc = self._pc
        selections = (
            (("outgoing", "body_pre", "body_post"),)
            if direction == "outgoing"
            else (("incoming", "body_post", "body_pre"),)
            if direction == "incoming"
            else (
                ("outgoing", "body_pre", "body_post"),
                ("incoming", "body_post", "body_pre"),
            )
        )
        weights: dict[tuple[str, int], int] = defaultdict(int)
        matched_rows = 0
        for batch in self._batches():
            columns = {key: batch.column(batch.schema.get_field_index(key))
                       for key in EXPECTED_COLUMNS}
            for label, own, partner in selections:
                mask = pc.and_(
                    pc.equal(columns[own], body_id),
                    pc.greater_equal(columns["weight"], min_weight),
                )
                if partner_id is not None:
                    mask = pc.and_(mask, pc.equal(columns[partner], partner_id))
                selected = batch.filter(mask)
                if selected.num_rows == 0:
                    continue
                matched_rows += selected.num_rows
                for other, count in zip(
                    selected.column(selected.schema.get_field_index(partner)).to_pylist(),
                    selected.column(selected.schema.get_field_index("weight")).to_pylist(),
                ):
                    weights[(label, int(other))] += int(count)
        ranked = sorted(weights.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        return {
            "dataset": DATASET,
            "body_id": body_id,
            "direction": direction,
            "min_weight": min_weight,
            "matched_rows": matched_rows,
            "distinct_directional_partners": len(weights),
            "full_scan_performed": True,
            "returned": [
                {"direction": label, "partner_body_id": other, "synapses": weight}
                for (label, other), weight in ranked[:limit]
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only, memory-mapped access to the full MaleCNS Feather weights table."
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Show schema without scanning the file")
    sub.add_parser("audit", help="Scan ALL rows and validate data totals")
    neighbors = sub.add_parser("neighbors", help="Scan the full table for one body")
    neighbors.add_argument("body_id", type=int)
    neighbors.add_argument("--direction", choices=("incoming", "outgoing", "both"),
                           default="both")
    neighbors.add_argument("--partner", type=int)
    neighbors.add_argument("--min-weight", type=int, default=1)
    neighbors.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    with FullMaleCNS(args.file) as full:
        if args.command == "inspect":
            result = full.inspect()
        elif args.command == "audit":
            result = full.audit()
        else:
            result = full.neighbors(
                args.body_id, direction=args.direction,
                partner_id=args.partner, min_weight=args.min_weight,
                limit=args.limit,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
