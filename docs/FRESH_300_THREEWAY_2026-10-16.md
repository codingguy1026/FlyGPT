# FlyGPT Experiment 2: Fresh 300-Question Three-Way Evaluation

Date: 2026-10-16 (KST)

**Status: exploratory experiment. On this developer-authored held-out set, the conventional text-only model achieved the highest mean accuracy. Authentic MaleCNS connectivity did not show a consistent advantage.**

## Reproducibility

- GitHub Actions run: https://github.com/codingguy1026/FlyGPT/actions/runs/38856008421
- Full JSON output: artifact `flygpt-threeway-fresh300` attached to the run
- Evaluation dataset: `training/fresh_holdout_300_20261016.jsonl`; first frozen in commit `b5738c99b5a81d169d9dadfc1cf776fa2fa4928d` **before the model runner was created or any scores were inspected**
- Dataset: **300 new prompts; 50 questions each** for general, code, math, memory, summarize, research; mixed Korean/English
- Evaluator: `training/fresh_threeway.py`
- Three paired random seeds: `42, 43, 44`; same teacher examples, same vectorizer (v6), class labels, seed-matched train/validation splits and training settings
- Maximum 90 epochs, patience 18; same basic loss/optimizer configuration, trained from scratch in each condition
- All three model variants are compared on the same untouched 300 prompts after each checkpoint is selected on *training-time validation*, not the 300-test set
- Evaluation aborts for normalized exact overlaps between teacher and holdout data

## Architectures

1. **MaleCNS**: FlyGraphRouter with the committed 256-node, 948-directed-edge subset of Janelia MaleCNS v1.0.
2. **Rewired**: same graph architecture, same edge/weight counts and per-node directed in/out edge *counts*, with randomly swapped edge destinations.
3. **Text-only**: conventional text feature MLP without graph message passing, with the same 8192-dimensional text features and 256 hidden channels. Trainable parameter count differs by less than 1% from the graph model. Note: **computation depth differs**, so model-size matching does not establish perfect architecture equivalence.

No LLM output quality was evaluated, and no models were deployed.

## Results: all 300 fixed test prompts, route classification accuracy

| Paired seed | MaleCNS | Rewired | Text-only |
|---|---:|---:|---:|
| 42 | **80.0%** | 78.0% | 77.3% |
| 43 | 69.0% | 74.3% | **76.0%** |
| 44 | 70.3% | 68.7% | **76.3%** |
| **Mean** | **73.11%** | **73.67%** | **76.56%** |

Because the six classes are **exactly balanced** (50 questions per class), macro recall and overall accuracy coincide in this test.

Mean accuracy differences (MaleCNS minus comparison model):

- MaleCNS vs rewired: **-0.56 percentage points**.
- MaleCNS vs text-only: **-3.44 percentage points**.

The original MaleCNS model wins both competitors on seed 42, but the text-only model wins on seeds 43 and 44. There is no consistent MaleCNS benefit in these three runs.

## Comparison to Experiment 1

Experiment 1 used 60 older real-world regression questions and found MaleCNS mean 91.7% vs rewired 89.4%. This experiment's new 300 prompts yield a smaller and reversed MaleCNS-rewired gap, with substantially lower absolute accuracy. The benchmarks are different; the numbers are **not** a before/after deterioration for one continuously trained model.

## Scientific interpretation

- The honest finding is **mixed/inconclusive** with a mild text-only advantage on this particular test.
- Rewiring changes graph motifs and weight-to-node assignment even while preserving degree counts and global edge-weight distribution. This is not a full connectome reconstruction.
- Only three seeds: insufficient to claim statistical significance; no confidence intervals or p-values were calculated.
- Developer-authored test set, not an independently blinded third-party benchmark. Exact overlap is checked; semantic paraphrase leakage still may occur.
- The 300 questions are fixed and evaluated together; using them to tune hyperparameters would invalidate their status as an untouched holdout.
- Text-only baseline is nearly parameter-matched but differs in computation graph. Compare additional conventional baselines before claiming superiority or inferiority based on architecture alone.
- **Routing accuracy does not equal chatbot comprehension, autonomous reasoning, or answer correctness.**
- Do not claim that "the fruit fly brain is smarter" or "the connectome proved superior" on this evidence.

## Next steps

1. Have independent reviewers blind-label and quality-check ambiguous prompts without changing this dataset.
2. Assemble a *new* external held-out benchmark with distinct wording and sources; freeze before any additional tuning.
3. Compare more seeds, per-route confusion matrices, and controlled graph ablations with matched optimization budgets.
4. Measure end-to-end correctness and tool choice separately from model routing.
5. Keep current FlyGPT production checkpoint unmodified until release safety checks are complete.

**Outcome: Experiment 2 successfully challenged the tentative advantage suggested by Experiment 1. The claim that authentic MaleCNS topology improves task routing remains unproven.**
