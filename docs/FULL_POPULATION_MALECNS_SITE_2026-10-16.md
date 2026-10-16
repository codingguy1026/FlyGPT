# Full-population MaleCNS router, experimental site integration

## Status

**Code prepared; no full-neuron model has yet been trained or deployed from this branch.**

The previous 93.2% figure refers to an internal test of a **256-neuron** model, not a full-brain score. The full-neuron model is a DIFFERENT model and must be trained and evaluated anew. Neither model accuracy nor production activation can be inferred from a PR being opened.

## What 'all neurons' means here

- Official **Janelia MaleCNS v1.0** annotations define which body IDs are curated neurons. Explicit glia, unannotated segments and segment fragments are excluded (they are not neurons in this task).
- The official full directed weights Feather table is approximately 1.1 GB and includes many segment-to-segment rows; only edges with *both* endpoints among curated neuron IDs are used for neuronal structural training.
- Structural pretraining uses every eligible positive directed edge in the entire table without top-K filtering and retains a separate pre/postsynaptic embedding for **every curated annotated neuron**, including neurons with no observed eligible edge (their embeddings can remain weak/untrained).
- The chat route classifier subsequently uses **all** of these neuron embeddings via mathematically exact *positive linear attention* population moments. No top-K neuron or edge sampling is performed in the population representation.
- This method is an offline full-graph structural learning stage followed by online full-population **compressed** attention. It is **NOT** a biophysical simulation and does **NOT** propagate through 150 million edges per user message.
- The current Brain panel deliberately shows summary-only status, not fabricated live calcium traces, when the full-population model is active.

Official data and provenance: https://male-cns.janelia.org/download/

## Offline procedure (Codespaces / machine with adequate disk)

1. Merge/review the stacked work: research PRs #60-63 and optional 256-site checkpoint compatibility PR #64, then this PR. Nothing here runs automatically on site startup.
2. Use `make download-full-neuron-data`. This explicitly downloads official ~1.1 GB weights and ~13 MB annotations. The download script checks for at least 4GB free disk.
3. `make train-setup` then `.venv-train/bin/python -m pip install -r requirements-full-connectome.txt` to install PyArrow; confirm PyTorch is available.
4. Run `make train-full-neuron-router`. It first learns 16-dimensional pre/post structural embeddings over *all* official eligible directed connection rows, then supervises a six-route classifier using the complete neuron population. The new result is written to `artifacts/fly_router_malecns_all_neurons.pt`.
5. Perform an independent evaluation and verify performance before opting in. Its internal validation score is NOT a reliable estimate of general-user chat quality.
6. Make sure `.venv/bin/python` has CPU PyTorch, then `make enable-full-neuron-router` and restart `make stable` (or `make run`). Health reports `full_population_router.enabled`, `.active`, and `.all_annotated_neurons` separately.
7. If problems arise, `make disable-full-neuron-router` and restart. The earlier site model and trained artifact remain intact.

## Architecture

`training/train_full_connectome.py` already implements read-only streaming over official Feather rows and all-neuron unsupervised structure learning. New `training/full_population_router.py` converts its full population embeddings into an exact query-dependent attention summary using population mean and Gram matrix. Identity: for normalized query `q`, all normalized neuron keys `k_i`, and weights `w_i = 1 + 0.85 * (k_i · q)` (always positive), the attended state is `Σ_i w_i k_i / Σ_i w_i`. Precompute `mean(k)` and `mean(k kᵀ)` over all neuron keys so requests do not require 150M-edge scans or 160k-dot UI renderings. Learning updates classifier/query projection from labeled routes; no answer text or benchmark labels are injected into code.

`full_population_runtime.py` rejects checkpoints with missing all-neuron metadata, a count below 100,000, or a non-`male-cns:v1.0` provenance. It only activates when `FLYGPT_ENABLE_FULL_POPULATION_ROUTER=1` and a full checkpoint is pointed to by `FLYGPT_MODEL_PATH`. The router reports the full curated neuron count and eligible graph row count; it never emits fake activation traces.

## Scope and limitations

- The full graph has **not been downloaded/trained/benchmarked in this PR**. The available GitHub CI tests mathematical properties on small synthetic keys only; it cannot prove true Janelia data accuracy or operation speed.
- Training requires substantial CPU, storage and time. If the current Codespaces cannot fit the data, a separate machine/CI job is necessary; do not silently sample fewer neurons and label the run as full.
- The full population summary mixes structural embeddings, not the real fly's behavior or spike dynamics.
- Even after obtaining a valid trained full checkpoint, the previously seen 93.2% is NOT its expected accuracy. Evaluate this new architecture on *new* independent questions and compare with an ordinary text classifier and randomized-connectome control.
- The default deployed site stays on its existing model unless explicitly switched. A private local `.env` marker permits rollback without writing credentials to the repository.

## Unit tests

`python -m unittest tests.test_full_population_router -v` (PyTorch required). The tests assert that exact cached population attention matches directly summing every neuron, removing one neuron changes the summary, and incomplete provenance fails. Standard repository tests skip the optional PyTorch-only unit suite if PyTorch is not installed.

## Next scientific milestone

Actually run the full official-data two-stage training, audit counts and repeatability, and **independently evaluate** route classification before putting it on the public domain. Distinguish full-population *structural use* from live all-neuron *electrophysiology* in marketing/UI.
