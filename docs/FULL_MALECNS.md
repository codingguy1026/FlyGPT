# Full MaleCNS integration: phase 1 (read-only)

FlyGPT's current trained graph router is a **256-node** MaleCNS-derived subset.
This phase adds an independent, **full-row** source-data reader. It does not
automatically replace the graph router or make the language model think with
all the neurons. The original `fly_router_malecns_v0_4.pt` stays unchanged.

## Source and scope

Official Janelia MaleCNS v1.0 (CC-BY):
https://male-cns.janelia.org/download/

The official file below is approximately **1.1 GB** and contains the complete
segment-to-segment connection weight table (including untraced/non-neuronal
segments). This is NOT the same as a cleaned graph of only the approximately
166,691 annotated neurons. Do not confuse total row count, unique segment IDs,
unique annotated neurons, and distinct neuron-to-neuron connections.

- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- Columns: `body_pre`, `body_post`, `weight`
- Presynaptic and postsynaptic segment IDs are **not** arbitrary sequential
  indices. Preserve original 64-bit IDs and directed weights.

**No automatic download** occurs at import time, during normal chat, or when
running tests. An explicit download and sufficient disk space are required.

## Set up

Check available storage before downloading. Codespaces may lack enough space
for a 1.1 GB raw file plus separate environments and temporary caches.

```bash
df -h .
python -m pip install -r requirements-full-connectome.txt
mkdir -p data/malecns_full
```

After verifying disk space, explicitly download the official source file.
Use `-C -` to resume an interrupted transfer:

```bash
curl -fL --retry 3 -C - \
  -o data/malecns_full/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
```

If it fails, do not continue with a partial file. The script validates Feather
format and required column names, but does **not** establish a cryptographic
match to the official release. Use a trusted official checksum if available.

## Inspect without loading the whole dataset

```bash
python full_connectome.py inspect
```

Reports the source, schema, number of Feather record batches, and file size.
Inspection does not scan all edges and does not count annotated neurons.

## Perform an exact full-table audit

```bash
python full_connectome.py audit
```

Streams **every batch**, including the last, and reports total table rows,
total synapse weight, self-connections, null rows, and nonpositive weights.
This may take a while; unlike `inspect`, it must traverse the whole dataset.

## Inspect connections for one body ID

```bash
python full_connectome.py neighbors 12781 --direction outgoing --limit 10
python full_connectome.py neighbors 12781 --direction incoming --limit 10
python full_connectome.py neighbors 12781 --direction outgoing --partner 85165
```

Queries scan every row and aggregate across all record batches before applying
the result limit. This preserves correctness without loading the 1.1 GB table
as a single pandas DataFrame, but full scans are **not interactive-speed**.
Do not call them synchronously from `/api/chat`.

For a custom fixture/file, place `--file FILE` **before** the subcommand.

## Tests

```bash
python -m unittest tests.test_full_connectome -v
```

Tests use a tiny synthetic Feather with multiple batches and do not access the
network or require a neuPrint token.

## What remains before full-brain routing

1. Validate downloaded source and cross-check counts against official exports.
2. Load neuron annotations and decide how to distinguish traced/annotated
   neurons from arbitrary segments.
3. Build a persistent disk-backed/sparse adjacency index with measured disk
   and RAM requirements. The phase-1 scan is not suitable for per-chat use.
4. Train/evaluate a **new** hierarchical/full-graph router as a separate
   checkpoint. Do not overwrite the 256-node baseline or claim full-graph
   inference merely because the raw data file is present.
5. Compare latency, task accuracy, memory, and failure handling before switching
   the default model.

The existing router checkpoint and Ollama timeout issues are independent of
this data import path.
