# FlyGPT training v0.1

This is FlyGPT's first actual learning loop: **teacher-output distillation for task routing**.

The goal is deliberately small. Instead of pretending the whole language model can be replaced by a fruit-fly brain, v0.1 teaches a compact router to choose the right kind of processing path for a prompt. The router's hidden communication graph can be built from real **FlyWire v783** directed connections.

## What is being distilled?

`teacher_seed.jsonl` contains examples authored by the teacher model with:

- `input`: the user prompt
- `route`: the target route the student should learn
- `teacher_answer`: a concise target answer or handling rule
- `teacher`: teacher model identifier
- `dataset_version`: dataset version

v0.1 trains only the `route` target. The answer field is intentionally kept so later versions can add response-level imitation or preference training.

Current routes:

- `math`
- `code`
- `memory`
- `summarize`
- `research`
- `general`

## 1. Install training dependency

```bash
python -m pip install -r requirements.txt
python -m pip install -r training/requirements.txt
```

## 2. Build a real FlyWire scaffold

Make sure the nine Parquet parts are under `data/flywire_parts/`, then run:

```bash
python training/build_scaffold.py \
  --data-dir data/flywire_parts \
  --out training/flywire_scaffold.json \
  --nodes 256 \
  --edges 4096 \
  --sample-rows 200000
```

The script samples real directed FlyWire edges, selects strong sampled neurons, remaps their root IDs to a compact graph, and stores normalized synapse strengths as the fixed topology prior.

## 3. Train the first Fly router

```bash
python training/train_router.py \
  --dataset training/teacher_seed.jsonl \
  --scaffold training/flywire_scaffold.json \
  --out artifacts/fly_router_v0_1.pt \
  --epochs 60
```

The model keeps the graph topology fixed while learning:

1. how text activates the compact fly-neuron state,
2. how strongly the allowed FlyWire edges contribute during propagation,
3. which final route should be selected.

## Smoke test without the dataset

To verify the trainer itself before building the FlyWire scaffold:

```bash
python training/train_router.py \
  --dataset training/teacher_seed.jsonl \
  --out artifacts/smoke_router.pt \
  --epochs 20
```

Omitting `--scaffold` intentionally uses a deterministic synthetic graph. This is only a software smoke test and **must not be reported as a FlyWire result**.

## Important scientific boundary

This does not prove that a Drosophila connectome makes an AI smarter. It creates the experiment needed to test that idea. The next proper comparison is to train the same teacher dataset with:

- the FlyWire scaffold router,
- a parameter-matched ordinary MLP or randomly wired graph,

then compare held-out accuracy, sample efficiency, robustness, and compute cost.

That A/B experiment is the point where Project FDF starts producing evidence instead of vibes. 🪰
