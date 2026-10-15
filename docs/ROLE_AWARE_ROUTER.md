# FlyGPT biological role-aware router experiment

This is an **opt-in, offline 256-neuron experiment** using the SAME official
MaleCNS v1.0 directed scaffold and teacher curriculum as v0.5. It does not
alter the production router by default and it is **not full-160k training**.

## Purpose and scientific limits

The experimental mapping uses the official neuPrint `superclass` annotation
for each checkpoint's actual body ID. We map it conservatively to:
`sensory`, `motor`, `interneuron` (intrinsic), `projection`,
`other`, or `unclassified`.

Biological **sensory** neurons don't literally parse text, biological
**motor** neurons don't literally speak, and MaleCNS neurons are not
biologically trained to solve mathematics. Their class is used only as an
experimental inductive bias in a text-task router.

The role-aware graph applies **fixed nonzero** multipliers to:
- encoder input: sensory x1.25, motor x0.9 (all unclassified x1);
- graph message relay: intrinsic x1.2, projection x1.15 (other x1);
- final task-category readout: motor x1.25, sensory x0.9,
  projection x1.05 (other x1).

These numeric gains are hypotheses, **not learned or measured biological
parameters**. All 256 nodes and the exact directed edge set remain connected.
All unknown/unannotated cell roles stay neutral at x1. There is no
hardcoded classification like "all math must go through motor cells".
Output is a **task-category logit** from a trained neural model, not a
biological muscle command. Real synapses and text-feature vectors are
distinct objects and never conflated.

## Setup and commands

The existing `make run` stays on its prior checkpoint. Never silently
replace or rename the current production artifact.

```bash
# On the experiment branch, in Codespaces:
make train-setup
# Use your .env NEUPRINT_TOKEN privately; do not commit/share this secret.
make train-role-experiment
# Produces: artifacts/fly_router_malecns_role_experiment.pt
# Also produces: artifacts/malecns_roles_v1.json
```

The trainer obtains fresh official annotations for the 256 scaffold body IDs.
If a token is missing, the dataset is wrong, the annotations don't align
with the frozen scaffold, or there is no useful verified classification,
training stops rather than substituting guesses. The training job may take
time and CPU; it is never triggered by `make run`.

To benchmark fairly, first have the unchanged baseline checkpoint on
disk (`artifacts/fly_router_malecns_v0_5.pt`, generated with
`make train-v0.5` if necessary) and then:

```bash
make compare-role-experiment
```

The benchmark requires identical scaffold ID order, edges, weights, class
set, vectorizer version and propagation step count for both candidates.
It reports per-regression-dataset accuracy, balanced accuracy,
decision-gate precision/coverage, and warmed CPU median/p95 inference
latency. Benchmark environment and sample sizes must be considered.
Results are saved to `artifacts/role_aware_comparison.json`.
Compare accuracy *and* runtime, not node count or model size alone.

**Release gate:** keep the experiment out of production until it meets
pre-agreed thresholds on independent held-out sets, latency,
non-regression, and reproducibility, preferably across multiple seeds.
A role prior can reduce accuracy or increase inference time.

## Test

```bash
python -m unittest tests.test_role_aware_router -v
python -m py_compile training/train_router.py training/build_role_labels.py training/compare_role_aware.py router_runtime.py
bash -n training/train_malecns_v0_5.sh
```

Without installed torch, only the pure mapping tests run and
the model-dependent tests are skipped. Test the training and
`router_runtime.FlyRouterRuntime.predict` forward path in a
PyTorch-equipped Codespace before treating the experiment as verified.

The full annotated MaleCNS connectome is not involved in this experiment:
it is a 256-node hypothesis test. It can inform a later larger graph
experiment, but does not establish generalizable biological intelligence.
