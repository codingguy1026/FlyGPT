# FlyGPT router training

FlyGPT uses a compact graph router for task routing. The graph topology can be
built from real **Janelia MaleCNS v1.0** directed connectivity.

This does not mean a fruit-fly connectome replaces a language model. The
connectome supplies a constrained communication graph for the router, while the
router is trained on labeled task-routing examples.

## 1. Configure neuPrint

You only need neuPrint credentials when building or refreshing the scaffold.

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

## 3. Train v0.5

First prepare the isolated CPU PyTorch environment once:

```bash
make train-setup
```

Then train the v0.5 brain:

```bash
make train-v0.5
```

v0.5 reuses the committed scaffold by default so model comparisons are not
confounded by a changing graph. To deliberately rebuild it:

```bash
REFRESH_SCAFFOLD=1 make train-v0.5
```

The v0.5 pipeline adds:

- vectorizer v6 with boundary n-grams, skip context, token-shape and punctuation features
- input feature dropout for typo/noise robustness
- label smoothing and multiclass logit-margin regularization
- balanced-accuracy checkpoint selection
- post-training temperature scaling for better confidence calibration
- validation-selected confidence and margin gates stored inside the checkpoint
- a held-out v0.5 real-world regression suite

The output checkpoint is:

```
artifacts/fly_router_malecns_v0_5.pt
```

## 4. Diagnose

```bash
python training/diagnose_brain.py \
  --model artifacts/fly_router_malecns_v0_5.pt \
  --scaffold training/malecns_scaffold.json
```

The integrity check verifies that the checkpoint topology and weights match the
MaleCNS scaffold and that the checkpoint metadata identifies
`male-cns:v1.0`.

## 5. Evaluate

The v0.5 training script automatically runs both the new held-out real-world
suite and the older regression suites. You can also run the new suite directly:

```bash
.venv-train/bin/python training/eval_router.py \
  --model artifacts/fly_router_malecns_v0_5.pt \
  --dataset training/regression_v0_5_realworld.jsonl
```

By default the evaluator uses the confidence and margin thresholds stored in
the checkpoint after calibration.
