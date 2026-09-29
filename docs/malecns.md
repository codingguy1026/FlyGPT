# MaleCNS v1.0 backend

FlyGPT can use the 2026 MaleCNS v1.0 whole-central-nervous-system
connectome in addition to the original FlyWire v783 backend.

## What the bootstrap does

The official bulk release stores the main graph as one large Feather file.
FlyGPT does not load that entire file into Python memory.

Instead, `bootstrap_malecns.py`:

1. downloads the official MaleCNS v1.0 weights, annotations, and
   neurotransmitter tables;
2. verifies the known SHA-256 for the weights file unless disabled;
3. keeps curated neurons using the annotation table;
4. streams Arrow record batches from the large Feather file;
5. writes the surviving graph into many Zstandard-compressed Parquet parts;
6. stores annotations and neurotransmitter metadata as Parquet;
7. writes `data/malecns/manifest.json`.

DuckDB later exposes all of those Parquet pieces as one logical
`connections` view, so FlyGPT can query the graph without stitching the
files into one giant in-memory object.

## Build the local dataset

The default build creates 32 Parquet parts and preserves every curated
connection with weight >= 1.

```bash
python bootstrap_malecns.py --parts 32
```

To discard the downloaded raw Feather files after a successful conversion:

```bash
python bootstrap_malecns.py --parts 32 --delete-raw
```

If you want a smaller, stronger-edge graph for experiments:

```bash
python bootstrap_malecns.py --parts 32 --min-weight 5
```

To rebuild existing parts:

```bash
python bootstrap_malecns.py --parts 32 --force
```

## Run FlyGPT with MaleCNS

```bash
export FLYGPT_CONNECTOME=malecns
export MALECNS_DATA_DIR=data/malecns/parts
export MALECNS_METADATA_DIR=data/malecns/metadata
bash scripts/run_all.sh
```

The original backend remains the default. To switch back:

```bash
export FLYGPT_CONNECTOME=flywire
```

## CLI examples

```bash
python cli.py --backend malecns stats
python cli.py --backend malecns neuron 12781 --limit 10
python cli.py --backend malecns neuron 12781 --context
python cli.py --backend malecns pair 12781 556329
python cli.py --backend malecns top --limit 20
```

## Build a MaleCNS training scaffold

```bash
python training/build_malecns_scaffold.py \
  --data-dir data/malecns/parts \
  --metadata-dir data/malecns/metadata \
  --nodes 256 \
  --edges 4096
```

The resulting `training/malecns_scaffold.json` uses real MaleCNS directed
connectivity and synapse-count weights.

## Data layout

```text
data/malecns/
├── manifest.json
├── raw/
│   ├── connectome-weights-male-cns-v1.0-minconf-0.5.feather
│   ├── body-annotations-male-cns-v1.0-minconf-0.5.feather
│   └── body-neurotransmitters-male-cns-v1.0.feather
├── parts/
│   ├── malecns_weights_part_001.parquet
│   ├── ...
│   └── malecns_weights_part_032.parquet
└── metadata/
    ├── annotations.parquet
    └── neurotransmitters.parquet
```

The MaleCNS v1.0 data is released under CC-BY 4.0. Keep dataset attribution
when redistributing derived data or results.
