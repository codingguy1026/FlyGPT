# FlyGPT Experiment 4: Natural Supervised Learning Curves

Date: 2026-10-16 (KST)

**Status: research only; no production checkpoint or inference endpoint changed.**

## Purpose

The goal is **learning from examples**, not hardcoding a fixed answer for the previously tested benchmark. The experiment progressively increases the amount of *existing training-only labeled questions* while holding out separate validation and test examples. This remains supervised classification over six route labels; it is not a biological simulation of fruit-fly synaptic learning.

## Protocol

- Existing FlyGPT teacher and curriculum routing examples are de-duplicated by normalized prompt, with no teacher answer texts fed into the model.
- Close paraphrases are clustered using a **character-trigram Jaccard threshold of 0.86**, before allocating groups to train/validation/test. This reduces obvious prompt overlap; it **does not guarantee semantic independence**.
- Three paired seeds: 42, 43, 44; separate split each seed, same split for all models and learning fractions within a seed.
- Fixed **25%, 50%, and 100% nested route-stratified training subsets**. Actual training rows per seed: **119, 237, 473**.
- Model selection/early stopping uses **validation only**; final test predictions are computed **once per trained model** after selection.
- Architecture comparison: (a) authentic 256-node/948-edge MaleCNS residual FlyGraphRouter; (b) same graph model with directed degree-preserving rewiring; (c) no-graph text-only model with near-matched parameter count.
- Same text features (v6), optimizer, loss, training schedule (90 maximum epochs, patience 18) across models.
- Nine conditions per seed, **27 trained and evaluated models** total.
- Existing 300-question evaluation benchmark from Experiment 2 is **not used for training, validation, or testing here**.

## Measured held-out test macro recall (balanced accuracy)

| Fraction of available train split | Real MaleCNS residual | Rewired residual | Text-only |
|---|---:|---:|---:|
| 25% (119 examples) | **81.02%** | 80.30% | 80.74% |
| 50% (237 examples) | 86.89% | **88.65%** | 88.46% |
| 100% (473 examples) | 93.16% | 92.04% | **93.72%** |

All values are the **mean of three paired seeds**, not a result on the previously exposed 300 questions. Exact reproducible per-seed results and route-specific confusion counts are in the GitHub Actions JSON artifact.

For 100% training split, MaleCNS test macro recall by seed was **91.9%, 92.8%, 94.8%** (rounded), while text-only was **91.9%, 92.3%, 97.0%** (rounded). The mean gap from text-only to MaleCNS was **0.56 percentage points** on this **teacher-corpus holdout**, but the seed-to-seed outcomes are mixed.

## Evidence

- [First successful full run](https://github.com/codingguy1026/FlyGPT/actions/runs/38858030657)
- Full score tables and per-class confusion data: GitHub Actions artifact `flygpt-natural-learning-curves`.
- The training script is `training/natural_learning_curves.py`.
- The follow-on run archives **research-only MaleCNS checkpoints** trained on each seed's entire 473-example train partition as `flygpt-research-only-trained-malecns`. These are **not production-ready**, and the application loader is not changed to load them.

## What the findings show and do not show

- In this experiment, increasing the amount of labeled training examples consistently increased the mean score for the authentic MaleCNS model on unseen held-out questions from the **same source corpus**.
- Test scores differ from and are substantially lower than training-set accuracy (generally near 100%). That supports **some within-corpus generalization**, but does not exclude memorization of patterns or linguistic near-duplicate leakage.
- **We have not yet shown reliable superiority of real biological wiring**: rewired networks remain competitive and text-only marginally leads at 100% training.
- This evaluation is **not comparable in difficulty** to the previously authored 300-question test; the much higher scores here **do not mean FlyGPT suddenly reached 93% on general new real-world queries**.
- The biological scaffold represents a compact subset of MaleCNS, not the whole connectome. Learned weights result from ordinary backpropagation on route labels, not actual synaptic plasticity.
- Only three seeds, no confidence interval/statistical significance. Similar language patterns could leak across groups despite the heuristic.

## Research principles

1. Do not hardcode which answer a test question must produce.
2. Keep all test prompts away from training and checkpoint selection.
3. Freeze architectures/hyperparameters before comparing them on final held-out data.
4. Inspect *both* training and test performance and per-route error patterns.
5. Never claim biological superiority or world-first proof without independent validation.
6. Do not deploy a research model merely because it scored well on an internal benchmark.

**Next validation milestone:** Independently sourced questions, ideally by external reviewers, frozen before model/parameter changes. Quantify route-wise generalization and the authentic-vs-rewired connection contribution, with end-to-end dialogue quality evaluated separately.
