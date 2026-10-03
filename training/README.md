# FlyGPT router training

FlyGPT uses a compact graph router for task routing. The graph topology can be
built from real **Janelia MaleCNS v1.0** directed connectivity.

This does not mean a fruit-fly connectome replaces a language model. The
connectome supplies a constrained communication graph for the router, while the
router is trained on labeled task-routing examples.

## 1. Configure neuPrint

```bash
export NEUPRINT_TOKEN=your-token-here
export NEUPRINT_SERVER=https://neuprint.janelia.org
export NEUPRINT_DATASET=male-cns:v1.0
```

## 2. Build the MaleCNS scaffold

```bash
python training/build_scaffold.py \
  --out training/malecns_scaffold.json \
  --nodes 256 \
  --edges 4096 \
  --candidate-edges 32768
```

The builder fetches only a bounded set of strong edges from neuPrint. It does
not download the full MaleCNS connection table.

The scaffold records:

- source: Janelia MaleCNS v1.0 via neuPrint
- dataset: `male-cns:v1.0`
- selected MaleCNS body IDs
- directed edge weights derived from synapse counts

## 3. Train

All `train_v0_3*.sh` scripts now point at
`training/malecns_scaffold.json`.

For a direct training run:

```bash
python training/train_router.py \
  --dataset training/teacher_seed.jsonl \
  --scaffold training/malecns_scaffold.json \
  --out artifacts/fly_router_malecns_v0_4.pt
```

## 4. Diagnose

```bash
python training/diagnose_brain.py \
  --model artifacts/fly_router_malecns_v0_4.pt \
  --scaffold training/malecns_scaffold.json
```

The integrity check verifies that the checkpoint topology and weights match the
MaleCNS scaffold and that the checkpoint metadata identifies
`male-cns:v1.0`.
