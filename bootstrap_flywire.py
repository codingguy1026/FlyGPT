from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.feather as feather
import pyarrow.parquet as pq


ZENODO_URL = (
    "https://zenodo.org/records/10676866/files/"
    "proofread_connections_783.feather?download=1"
)
EXPECTED_MD5 = "f48f972d262323a102aed49af1396b8a"
DEFAULT_SOURCE = Path("data/proofread_connections_783.feather")
DEFAULT_OUTPUT = Path("data/flywire_parts")


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
    headers = {"User-Agent": "FlyGPT/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"↩️  Resuming at {human_bytes(existing)}")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", None)
        content_length = response.headers.get("Content-Length")
        remaining = int(content_length) if content_length else None

        if existing and status != 206:
            print("Server did not accept resume; restarting download.")
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
                            f"⬇️  {pct:6.2f}%  "
                            f"{human_bytes(downloaded)} / {human_bytes(total)}  "
                            f"{human_bytes(speed)}/s",
                            flush=True,
                        )
                    else:
                        print(
                            f"⬇️  {human_bytes(downloaded)}  "
                            f"{human_bytes(speed)}/s",
                            flush=True,
                        )
                    last_print = now

    partial.replace(target)
    print(f"✅ Downloaded: {target} ({human_bytes(target.stat().st_size)})")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, output_dir: Path, parts: int, compression_level: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🧠 Opening Feather file with memory mapping...")
    table = feather.read_table(source, memory_map=True)
    rows = table.num_rows
    rows_per_part = math.ceil(rows / parts)

    print(f"Rows: {rows:,}")
    print(f"Columns: {table.num_columns}")
    print(f"Output parts: {parts}")

    for i in range(parts):
        start = i * rows_per_part
        length = min(rows_per_part, rows - start)
        if length <= 0:
            break

        output = output_dir / f"proofread_connections_783_part_{i + 1:02d}.parquet"

        if output.exists() and output.stat().st_size > 0:
            print(
                f"⏭️  {i + 1}/{parts} already exists "
                f"({human_bytes(output.stat().st_size)}), skipping"
            )
            continue

        print(
            f"🗜️  {i + 1}/{parts}: rows "
            f"{start:,}..{start + length - 1:,}",
            flush=True,
        )

        pq.write_table(
            table.slice(start, length),
            output,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
        )

        print(
            f"✅ {output.name}: {human_bytes(output.stat().st_size)}",
            flush=True,
        )

    created = sorted(output_dir.glob("proofread_connections_783_part_*.parquet"))
    total_size = sum(path.stat().st_size for path in created)

    print("")
    print(f"🎉 Ready: {len(created)} Parquet files")
    print(f"📦 Total Parquet size: {human_bytes(total_size)}")
    print(f"📁 Directory: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the public FlyWire v783 proofread connection table from "
            "Zenodo and split it into Parquet files for FlyGPT."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parts", type=int, default=9)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip the official Zenodo MD5 integrity check.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the downloaded Feather file after successful conversion.",
    )
    args = parser.parse_args()

    if args.parts < 1:
        parser.error("--parts must be at least 1")

    if not args.source.exists():
        print("🌐 Downloading FlyWire v783 proofread connections from Zenodo...")
        download(ZENODO_URL, args.source)
    else:
        print(
            f"✅ Source already exists: {args.source} "
            f"({human_bytes(args.source.stat().st_size)})"
        )

    if not args.skip_md5:
        print("🔎 Checking MD5...")
        actual = md5(args.source)
        if actual != EXPECTED_MD5:
            print(
                "❌ MD5 mismatch. The download may be incomplete or corrupted.\n"
                f"Expected: {EXPECTED_MD5}\n"
                f"Actual:   {actual}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print("✅ MD5 matches the official Zenodo file.")

    convert(
        args.source,
        args.output,
        parts=args.parts,
        compression_level=args.compression_level,
    )

    if args.delete_source:
        args.source.unlink(missing_ok=True)
        print(f"🧹 Deleted source Feather file: {args.source}")


if __name__ == "__main__":
    main()
