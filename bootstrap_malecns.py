from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    "connectome-data/flat-connectome"
)
WEIGHTS_NAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANNOTATIONS_NAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_NAME = "body-neurotransmitters-male-cns-v1.0.feather"

WEIGHTS_URL = f"{BASE_URL}/{WEIGHTS_NAME}"
ANNOTATIONS_URL = f"{BASE_URL}/{ANNOTATIONS_NAME}"
NEUROTRANSMITTERS_URL = f"{BASE_URL}/{NEUROTRANSMITTERS_NAME}"

# Independently reproduced from the public MaleCNS v1.0 object.
EXPECTED_WEIGHTS_SHA256 = (
    "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"
)

DEFAULT_ROOT = Path("data/malecns")
DEFAULT_RAW = DEFAULT_ROOT / "raw"
DEFAULT_PARTS = DEFAULT_ROOT / "parts"
DEFAULT_METADATA = DEFAULT_ROOT / "metadata"


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{value:.1f} {units[i]}"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "FlyGPT-MaleCNS/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"↩️  Resuming {target.name} at {human_bytes(existing)}")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None)
        content_length = response.headers.get("Content-Length")
        remaining = int(content_length) if content_length else None

        if existing and status != 206:
            print("Server did not accept resume; restarting this file.")
            existing = 0
            mode = "wb"
        else:
            mode = "ab" if existing else "wb"

        total = existing + remaining if remaining is not None else None
        downloaded = existing
        started = time.monotonic()
        last_print = 0.0

        with partial.open(mode) as out:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_print >= 1.0:
                    elapsed = max(now - started, 0.001)
                    speed = (downloaded - existing) / elapsed
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"⬇️  {target.name}: {pct:6.2f}%  "
                            f"{human_bytes(downloaded)} / {human_bytes(total)}  "
                            f"{human_bytes(speed)}/s",
                            flush=True,
                        )
                    else:
                        print(
                            f"⬇️  {target.name}: {human_bytes(downloaded)}  "
                            f"{human_bytes(speed)}/s",
                            flush=True,
                        )
                    last_print = now

    partial.replace(target)
    print(f"✅ Downloaded: {target} ({human_bytes(target.stat().st_size)})")


def ensure_download(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 0:
        print(f"✅ Already present: {target} ({human_bytes(target.stat().st_size)})")
        return
    download(url, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def curated_body_ids(annotations_path: Path) -> set[int]:
    print("🧬 Reading curated MaleCNS neuron IDs...")
    table = feather.read_table(annotations_path, memory_map=True)
    names = set(table.column_names)

    if "bodyId" not in names:
        raise ValueError("MaleCNS annotation file is missing bodyId")

    if "superclass" in names:
        superclass = pc.cast(table["superclass"], pa.string())
        mask = pc.and_(
            pc.is_valid(superclass),
            pc.not_equal(pc.fill_null(superclass, ""), ""),
        )
        filtered = table.filter(mask)
        rule = "non-empty superclass"
    elif "status" in names:
        mask = pc.equal(pc.fill_null(table["status"], ""), "Traced")
        filtered = table.filter(mask)
        rule = "status == Traced"
    else:
        raise ValueError(
            "MaleCNS annotation file has neither superclass nor status; "
            "cannot identify curated neurons safely"
        )

    ids = {int(value) for value in filtered["bodyId"].to_pylist() if value is not None}
    print(f"🪰 Curated neurons: {len(ids):,} ({rule})")
    return ids


def convert_metadata(source: Path, output: Path, compression_level: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = feather.read_table(source, memory_map=True)
    pq.write_table(
        table,
        output,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=True,
        write_statistics=True,
    )
    print(f"✅ Metadata: {output} ({human_bytes(output.stat().st_size)})")


def split_weights(
    source: Path,
    output_dir: Path,
    *,
    parts: int,
    compression_level: int,
    keep_ids: set[int] | None,
    min_weight: int,
    force: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("malecns_weights_part_*.parquet"))

    if existing and not force:
        raise FileExistsError(
            f"{len(existing)} MaleCNS part(s) already exist in {output_dir}. "
            "Use --force to rebuild them."
        )

    if force:
        for path in existing:
            path.unlink()

    print("🧠 Opening MaleCNS Feather file as Arrow record batches...")
    with pa.memory_map(str(source), "r") as mapped:
        reader = ipc.open_file(mapped)
        columns = set(reader.schema.names)
        required = {"body_pre", "body_post", "weight"}
        missing = required - columns
        if missing:
            raise ValueError(
                "Unexpected MaleCNS weight schema. Missing: "
                + ", ".join(sorted(missing))
            )

        if reader.num_record_batches < 1:
            raise ValueError("MaleCNS weight file contains no record batches")

        actual_parts = min(parts, reader.num_record_batches)
        batches_per_part = math.ceil(reader.num_record_batches / actual_parts)
        keep_values = (
            pa.array(sorted(keep_ids), type=pa.int64()) if keep_ids is not None else None
        )

        total_rows = 0
        total_synapses = 0
        created = 0

        print(
            f"Record batches: {reader.num_record_batches:,} | "
            f"requested parts: {parts} | actual parts: {actual_parts}"
        )
        if keep_ids is not None:
            print(f"Filter: {len(keep_ids):,} curated neurons at both endpoints")
        print(f"Minimum connection weight: {min_weight}")

        for part_index in range(actual_parts):
            start_batch = part_index * batches_per_part
            end_batch = min(
                reader.num_record_batches,
                start_batch + batches_per_part,
            )
            if start_batch >= end_batch:
                break

            output = output_dir / f"malecns_weights_part_{part_index + 1:03d}.parquet"
            writer: pq.ParquetWriter | None = None
            part_rows = 0
            part_synapses = 0

            try:
                for batch_index in range(start_batch, end_batch):
                    batch = reader.get_batch(batch_index)

                    mask = pc.greater_equal(batch["weight"], min_weight)
                    if keep_values is not None:
                        mask = pc.and_(
                            mask,
                            pc.is_in(batch["body_pre"], value_set=keep_values),
                        )
                        mask = pc.and_(
                            mask,
                            pc.is_in(batch["body_post"], value_set=keep_values),
                        )

                    batch = batch.filter(mask)
                    if batch.num_rows == 0:
                        continue

                    if writer is None:
                        writer = pq.ParquetWriter(
                            output,
                            batch.schema,
                            compression="zstd",
                            compression_level=compression_level,
                            use_dictionary=True,
                            write_statistics=True,
                        )

                    writer.write_batch(batch)
                    part_rows += batch.num_rows
                    weight_sum = pc.sum(batch["weight"]).as_py()
                    part_synapses += int(weight_sum or 0)
            finally:
                if writer is not None:
                    writer.close()

            if writer is None:
                empty = pa.Table.from_batches([], schema=reader.schema)
                pq.write_table(
                    empty,
                    output,
                    compression="zstd",
                    compression_level=compression_level,
                )

            created += 1
            total_rows += part_rows
            total_synapses += part_synapses
            print(
                f"✅ {created:02d}/{actual_parts}: {output.name} | "
                f"{part_rows:,} rows | {part_synapses:,} synapses | "
                f"{human_bytes(output.stat().st_size)}"
            )

    return {
        "parts": created,
        "rows": total_rows,
        "synapses": total_synapses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official MaleCNS v1.0 flat connectome, keep the curated "
            "neuron graph, and split it into Parquet pieces for FlyGPT."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parts", type=int, default=32)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument(
        "--all-segments",
        action="store_true",
        help="Keep every segment in the raw weights table instead of curated neurons only.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the known SHA-256 integrity check for the 1.1 GB weights file.",
    )
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete the three downloaded Feather files after successful conversion.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing MaleCNS Parquet parts.",
    )
    args = parser.parse_args()

    if args.parts < 1:
        parser.error("--parts must be at least 1")
    if args.min_weight < 1:
        parser.error("--min-weight must be at least 1")
    if not 1 <= args.compression_level <= 22:
        parser.error("--compression-level must be between 1 and 22")

    raw_dir = args.root / "raw"
    parts_dir = args.root / "parts"
    metadata_dir = args.root / "metadata"
    manifest_path = args.root / "manifest.json"

    weights_path = raw_dir / WEIGHTS_NAME
    annotations_path = raw_dir / ANNOTATIONS_NAME
    neurotransmitters_path = raw_dir / NEUROTRANSMITTERS_NAME

    print("🌐 MaleCNS v1.0 bootstrap")
    ensure_download(ANNOTATIONS_URL, annotations_path)
    ensure_download(NEUROTRANSMITTERS_URL, neurotransmitters_path)
    ensure_download(WEIGHTS_URL, weights_path)

    if not args.skip_sha256:
        print("🔎 Checking MaleCNS weights SHA-256...")
        actual = sha256(weights_path)
        if actual != EXPECTED_WEIGHTS_SHA256:
            print(
                "❌ SHA-256 mismatch. The download may be incomplete or corrupted.\n"
                f"Expected: {EXPECTED_WEIGHTS_SHA256}\n"
                f"Actual:   {actual}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print("✅ SHA-256 matches.")

    keep_ids = None if args.all_segments else curated_body_ids(annotations_path)

    result = split_weights(
        weights_path,
        parts_dir,
        parts=args.parts,
        compression_level=args.compression_level,
        keep_ids=keep_ids,
        min_weight=args.min_weight,
        force=args.force,
    )

    convert_metadata(
        annotations_path,
        metadata_dir / "annotations.parquet",
        args.compression_level,
    )
    convert_metadata(
        neurotransmitters_path,
        metadata_dir / "neurotransmitters.parquet",
        args.compression_level,
    )

    manifest = {
        "dataset": "MaleCNS v1.0",
        "source": BASE_URL,
        "license": "CC-BY 4.0",
        "weights_sha256": None if args.skip_sha256 else EXPECTED_WEIGHTS_SHA256,
        "curated_only": not args.all_segments,
        "curated_neurons": None if keep_ids is None else len(keep_ids),
        "min_weight": args.min_weight,
        **result,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_size = sum(
        path.stat().st_size for path in parts_dir.glob("malecns_weights_part_*.parquet")
    )
    print("")
    print("🎉 MaleCNS is assembled for FlyGPT.")
    print(f"📦 Parquet parts: {result['parts']}")
    print(f"🔗 Connection rows: {result['rows']:,}")
    print(f"🧠 Synaptic weight total: {result['synapses']:,}")
    print(f"💾 Parquet size: {human_bytes(total_size)}")
    print(f"📁 Data directory: {parts_dir.resolve()}")
    print(f"🧾 Manifest: {manifest_path.resolve()}")

    if args.delete_raw:
        for path in (weights_path, annotations_path, neurotransmitters_path):
            path.unlink(missing_ok=True)
        print("🧹 Deleted raw Feather files.")


if __name__ == "__main__":
    main()
