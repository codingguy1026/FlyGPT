# Full-population MaleCNS router, experimental site integration

## Status

**Actual full-neuron models trained and verified in GitHub Actions; code is merged into `main`, but no live Codespaces or external website has been switched automatically.**

The previous 93.16% internal group-heldout score was achieved by a **256-neuron** model. We have now trained **three separate full-population models** on genuine official MaleCNS data and obtained **93.2349% average macro recall**, versus **93.1649%** for the archived 256-neuron references on the same group-heldout folds. **This is a 3-seed mean, not the accuracy of a single deployed model or an end-to-end chatbot score.** The difference is only +0.0700 percentage points and is not evidence that all-neuron connectivity is more effective.

## What 'all neurons' means here

- Official **Janelia MaleCNS v1.0** annotations define which body IDs are curated neurons. Explicit glia, unannotated segments and segment fragments are excluded (they are not neurons in this task).
- The official full directed weights Feather table is approximately 1.1 GB and includes many segment-to-segment rows; only edges with *both* endpoints among curated neuron IDs are used for neuronal structural training.
- Structural pretraining uses every eligible positive directed edge in the entire table without top-K filtering and retains a separate pre/postsynaptic embedding for **every curated annotated neuron**, including neurons with no observed eligible edge (their embeddings can remain weak/untrained).
- The chat route classifier subsequently uses **all** of these neuron embeddings via mathematically exact *positive linear attention* population moments. No top-K neuron or edge sampling is performed in the population representation.
- This method is an offline full-graph structural learning stage followed by online full-population **compressed** attention. It is **NOT** a biophysical simulation and does **NOT** propagate through 150 million edges per user message.
- The current Brain panel deliberately shows summary-only status, not fabricated live calcium traces, when the full-population model is active.

Official data and provenance: https://male-cns.janelia.org/download/

## Offline procedure (Codespaces / machine with adequate disk)

1. On the FlyGPT Codespace: `git checkout main && git pull --ff-only` after dealing with any local work normally.
2. Run **`make install-full-neuron-router`**. This fetches the already-trained full-neuron model from [successful GitHub Actions run #38915242351](https://github.com/codingguy1026/FlyGPT/actions/runs/38915242351), verifies the SHA-256 checksum for seed 44, ensures local PyTorch is available, verifies actual full-neuron inference, and opts in through local `.env`. GitHub Actions retains the model artifact for 30 days.
3. Restart the FlyGPT backend (`make stable` or `make run`). Health reports `full_population_router.enabled`, `.active`, and `.all_annotated_neurons`.
4. If problems arise, `make disable-full-neuron-router` then restart to use the previous checkpoint.

To **retrain independently from the official full dataset** instead of installing the archive: `make download-full-neuron-data`, `make train-setup`, `.venv-train/bin/python -m pip install -r requirements-full-connectome.txt`, and `make train-full-neuron-router`.

## Architecture

`training/train_full_connectome.py` already implements read-only streaming over official Feather rows and all-neuron unsupervised structure learning. New `training/full_population_router.py` converts its full population embeddings into an exact query-dependent attention summary using population mean and Gram matrix. Identity: for normalized query `q`, all normalized neuron keys `k_i`, and weights `w_i = 1 + 0.85 * (k_i · q)` (always positive), the attended state is `Σ_i w_i k_i / Σ_i w_i`. Precompute `mean(k)` and `mean(k kᵀ)` over all neuron keys so requests do not require 150M-edge scans or 160k-dot UI renderings. Learning updates classifier/query projection from labeled routes; no answer text or benchmark labels are injected into code.

`full_population_runtime.py` rejects checkpoints with missing all-neuron metadata, a count below 100,000, or a non-`male-cns:v1.0` provenance. It only activates when `FLYGPT_ENABLE_FULL_POPULATION_ROUTER=1` and a full checkpoint is pointed to by `FLYGPT_MODEL_PATH`. The router reports the full curated neuron count and eligible graph row count; it never emits fake activation traces.

## Scope and limitations

- The real official full dataset was downloaded and processed in [GitHub Actions run #38915242351](https://github.com/codingguy1026/FlyGPT/actions/runs/38915242351); the entire source table was streamed. **166,700 annotated neurons** were retained; **166,483** had eligible positive connections and **217** were included without a positive structural-training edge. **25,582,938** eligible directed edge rows were trained on. A separately [successful real-checkpoint site integration test](https://github.com/codingguy1026/FlyGPT/actions/runs/38916038675) loaded the actual 166,700-neuron model and verified exact all-neuron attention.
- Training requires substantial CPU, storage and time. If the current Codespaces cannot fit the data, a separate machine/CI job is necessary; do not silently sample fewer neurons and label the run as full.
- The full population summary mixes structural embeddings, not the real fly's behavior or spike dynamics.
- Full-neuron macro recall was **92.3961%, 94.6347%, 92.6739%** over three independently trained seeds, giving mean **93.2349%** on grouped internal teacher-corpus heldout questions (102 questions per seed). The actual site-installable seed-44 checkpoint was selected using *validation accuracy*, not its test results. **Its own held-out macro recall was 92.6739%, not 93.2349%.** Further independent external testing and a randomized-connectome control are required before stronger claims.
- The default deployed site stays on its existing model unless explicitly switched. A private local `.env` marker permits rollback without writing credentials to the repository.

## Unit tests

`python -m unittest tests.test_full_population_router -v` (PyTorch required). The tests assert that exact cached population attention matches directly summing every neuron, removing one neuron changes the summary, and incomplete provenance fails. Standard repository tests skip the optional PyTorch-only unit suite if PyTorch is not installed.

## Next scientific milestone

Conduct new external blind route-classification evaluations, compare with an equally trained all-neuron rewired connectivity control, and separately test real chat quality before a public performance claim. Distinguish full-population *structural use* from live all-neuron *electrophysiology* in the UI. The original real-model training and tests have already succeeded; **live website activation still requires a running application environment**.
