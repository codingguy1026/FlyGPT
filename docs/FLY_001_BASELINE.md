# FLY-001 — Frozen Router Baseline

## Scope and safety

The FLY-001 script measures the existing router without changing its weights, routing gate or production path:

1. **raw:** the unchanged FlyRouterRuntime.predict classification and acceptance gate.
2. **effective:** the raw prediction after applying the existing semantic route guard.

It does **not** evaluate per-user personalization, the LLM, neuPrint, downstream reasoning, or end-to-end chat. It does not import app.py or access the user databases.

## Actual source structure

- app.py: picks FLYGPT_MODEL_PATH if set, otherwise the first available checkpoint (v0.5, v0.4, then older models).
- router_runtime.py: loads calibrated graph-router checkpoint, exposes predict.
- semantic_route_guard.py: a small set of known intents can override the route to general without overwriting raw model provenance.
- training/eval_router.py and training/evaluate_router.py: pre-existing console evaluators, left unchanged.
- training/grouped_corpus.py: near-duplicate grouping for training splits.

The repository contains a v0.4 checkpoint, but v0.5 can exist locally. Pass the **actual** checkpoint used by the deployment, rather than guessing from docs.

## Run from the repository root

A Python environment containing CPU PyTorch is required. To create the project's training environment when missing, run make train-setup.

~~~bash
.venv-train/bin/python training/fly001_baseline.py \
  --model artifacts/fly_router_malecns_v0_4.pt \
  --dataset training/fly001_diagnostic.jsonl \
  --output reports/fly001_baseline.json
~~~

If the app currently uses v0.5 or any opt-in experiment, replace the model argument with its exact checkpoint path. This is an offline diagnostic; no backend, API key, or neuPrint connection is required. The first example is run twice as warmup before recording inference times.

## Diagnostic corpus

The committed training/fly001_diagnostic.jsonl contains exactly 100 labeled prompts: 20 general and 16 each of math, code, memory, summarize and research.

**This is a diagnostic regression set, not a demonstrably independent held-out test set.** It may contain paraphrases close to old training examples. Do not use it for both fitting the new model and claiming out-of-sample accuracy. New test sets should be frozen, held out and checked for near-duplicate overlap. Avoid inserting private transcripts into committed datasets or reports.

## JSON report schema v1

The generated report has the following shape. **The numbers below are illustrations, not measured FlyGPT scores.**

~~~json
{
  "schema_version": "fly001-baseline-v1",
  "created_at_utc": "<ISO8601 timestamp>",
  "scope": "offline_router_and_semantic_guard_no_personalization",
  "limitations": ["<scope caveats>"],
  "provenance": {
    "git_commit": "<revision or null>",
    "model_path": "<selected checkpoint path>",
    "model_sha256": "<sha256>",
    "dataset_path": "<dataset path>",
    "dataset_sha256": "<sha256>",
    "python": "<version>",
    "torch": "<version>"
  },
  "model": {
    "name": "<checkpoint filename>",
    "routes": ["code", "general", "math", "memory", "research", "summarize"],
    "n_nodes": 0,
    "n_edges": 0,
    "steps": 0,
    "vectorizer_version": "<checkpoint value>",
    "scaffold_kind": "<checkpoint value>",
    "temperature": 1.0,
    "min_confidence": 0.55,
    "min_margin": 0.1,
    "checkpoint_best_val_accuracy": 0.0
  },
  "benchmark": {
    "dataset_size": 100,
    "label_counts": {"general": 20, "math": 16, "code": 16, "memory": 16, "summarize": 16, "research": 16},
    "warmup_iterations": 2,
    "raw": {
      "total": 100,
      "correct": 0,
      "accuracy": 0.0,
      "macro_f1": 0.0,
      "accepted": 0,
      "coverage": 0.0,
      "accepted_correct": 0,
      "accepted_precision": null,
      "usable_rate": 0.0,
      "rejected": 100,
      "by_route": {"<label>": {"support": 0, "predicted": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}},
      "confusion": {"<expected>": {"<predicted>": 0}}
    },
    "effective": {"<same shape as raw>": "..."},
    "semantic_overrides": {"total": 0, "corrected": 0},
    "latency_ms": {
      "inference": {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0},
      "semantic_guard": {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    }
  },
  "cases": [
    {
      "id": 1, "input": "hi", "expected": "general",
      "raw": {"route": "general", "accepted": true, "confidence": 0.8, "margin": 0.6},
      "effective": {"route": "general", "accepted": true, "confidence": 0.8, "margin": 0.6, "override": false, "override_source": null, "speech_act": null},
      "latency_ms": {"inference": 1.1, "semantic_guard": 0.1}
    }
  ]
}
~~~

All checkpoint parameters, labels, predictions and aggregate scores in an actual report come from the measured model and dataset. The SHA-256 fingerprints allow exact model/dataset comparisons.

**Accuracy** is correct-route fraction ignoring the gate. **Coverage** is the accepted fraction. **Usable rate** requires both a correct route and acceptance. **Accepted precision** is the correct fraction *among accepted requests*. **Macro-F1** is an unweighted mean of per-route F1.

## Acceptance and next steps

- Verify dataset validation and metric tests in tests/test_fly001_baseline.py.
- Run the offline command with the local production checkpoint and preserve the first report outside git.
- Inspect per-case failures, but do not treat the same diagnostic data as new training and held-out proof.
- Record a separate truly held-out suite before training an improved model.
