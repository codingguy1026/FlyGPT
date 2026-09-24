# FlyGPT

FlyGPT is an experimental interface for querying the **FlyWire v783 whole-brain
Drosophila connectome** without loading the entire connectivity table into RAM.

The current loader targets `proofread_connections_783.feather` after conversion
to one or more Parquet parts. DuckDB scans the Parquet files lazily, so queries
such as "what are this neuron's strongest outputs?" only read the columns and
row groups they need.

## Dataset

Source: FlyWire Whole-brain Connectome Connectivity Data, release 783  
https://zenodo.org/records/10676866

The proofread connection table contains one row per
presynaptic-neuron / postsynaptic-neuron / neuropil combination and includes
synapse counts plus average neurotransmitter probabilities.

Expected columns:

- `pre_pt_root_id`
- `post_pt_root_id`
- `neuropil`
- `syn_count`
- `gaba_avg`
- `ach_avg`
- `glut_avg`
- `oct_avg`
- `ser_avg`
- `da_avg`

## Setup

```bash
python -m pip install -r requirements.txt
```

Put the split Parquet files in:

```text
data/flywire_parts/
  proofread_connections_783_part_01.parquet
  proofread_connections_783_part_02.parquet
  ...
  proofread_connections_783_part_09.parquet
```

The large data files are intentionally ignored by Git.

You can also keep the data anywhere else and point FlyGPT at it with:

```bash
export FLYWIRE_DATA_DIR=/path/to/flywire_parts
```

## CLI

Dataset statistics:

```bash
python cli.py stats
```

Strongest input/output partners for a neuron:

```bash
python cli.py neuron 720575940000000000 --limit 10
```

Only outputs in one neuropil:

```bash
python cli.py neuron 720575940000000000 \
  --direction outputs \
  --neuropil AL_L \
  --min-synapses 5
```

Inspect one directed neuron pair:

```bash
python cli.py pair 720575940000000000 720575940000000001
```

Generate compact text suitable for passing to an LLM:

```bash
python cli.py neuron 720575940000000000 --context --limit 10
```

## Python API

```python
from connectome import FlyWireConnectome

with FlyWireConnectome("data/flywire_parts") as fw:
    print(fw.stats())

    partners = fw.neuron(
        720575940000000000,
        direction="both",
        min_synapses=5,
        limit=20,
    )
    print(partners)

    context = fw.context_for_neuron(720575940000000000)
    print(context)
```

## Why DuckDB?

The v783 proofread connectivity table is large enough that eagerly converting
the whole dataset into a pandas DataFrame is wasteful for interactive queries.
DuckDB can query all Parquet parts as one logical table while keeping the
working set much smaller.

## Tests

```bash
python -m unittest discover -s tests -v
```
