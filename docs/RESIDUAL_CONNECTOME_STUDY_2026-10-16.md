# FlyGPT Experiment 3: Residual MaleCNS task router

Date: 2026-10-16 (KST)

**Outcome: an architectural change improved the original graph model's average on the *reused diagnostic* 300-item set (73.1% -> 75.1%) but did not match the 76.6% conventional text-only baseline. Both graph variants were near 75%, so an authentic biological wiring advantage remains unproven.**

## Experiment and controls

- [Completed GitHub Actions run](https://github.com/codingguy1026/FlyGPT/actions/runs/38856718415)
- Structured JSON artifact: \`residual-connectome-study\`
- Prespecified candidate: graph/text residual mixing. For encoder state \`h0\` and 3-step graph state \`hgraph\`, predict from \`(1-alpha)*LayerNorm(h0) + alpha*hgraph\`, where \`alpha = sigmoid(trainable scalar)\` initially 0.20.
- 256 nodes, authentic 948 MaleCNS edges. Neither brain wiring nor the graph processing pipeline was removed.
- Matched controls: the **same residual architecture using degree-preserving rewired edges**, and a conventional text-only classifier of similar parameter count.
- Seeds 42, 43, 44. Original teacher examples and validation splits, same base text vectorizer v6 and training setup as experiment 2.
- Model checkpoints and gate thresholds selected using only training-derived validation. No deployed checkpoint was changed.
- Existing 300-question suite previously tested in Experiment 2: **diagnostic only, not a genuinely fresh test**. No further hyperparameter tuning on these questions was performed in this run.
- All five experimental architecture tests and GitHub Actions workflow succeeded.

## Diagnostic results (300 questions, 50 per class)

| Seed | Residual MaleCNS | Residual rewired | Text-only |
|---|---:|---:|---:|
| 42 | 78.00% | 78.00% | 77.33% |
| 43 | 75.67% | 76.67% | 76.00% |
| 44 | 71.67% | 72.00% | 76.33% |
| **Mean** | **75.11%** | **75.56%** | **76.56%** |

The prior ungated MaleCNS candidate from Experiment 2 averaged **73.11%** with the same three seeds, training pool and 300-question evaluator. The new residual topology **averaged 2.00 percentage points higher in this retrospective diagnostic comparison**, but the new control with randomly rewired edges also improved. This is **not an independent improvement confirmation**.

## Biological graph contribution checks

| Seed | Learned MaleCNS graph alpha | Residual MaleCNS accuracy | Same trained model, graph gate approximately zero |
|---|---:|---:|---:|
| 42 | 0.2023 | 78.00% | 77.67% |
| 43 | 0.2014 | 75.67% | 75.67% |
| 44 | 0.2004 | 71.67% | 70.67% |

A nonzero graph alpha shows the graph was *included*, but its contribution to accuracy was small on these diagnostics. The gate barely moved from the initial 0.20 under the current training learning rate. The gate=0 check is an **inference-only ablation of the trained residual model**, NOT a separately trained baseline.

The rewired residual control has mean 75.56%, slightly above authentic wiring's 75.11%, so this result does **not** support any specific superiority of anatomical connectivity.

## Honest interpretation

1. We have not reached the ~76.6% baseline average in this 3-seed comparison.
2. A residual skip path may mitigate information loss in graph propagation, but the benefit has not been independently validated.
3. The work remains a six-way **task routing** classifier experiment, not an LLM or full fly-brain simulation.
4. The current 300 prompts are no longer blind; repeatedly improving this metric would constitute benchmark overfitting.
5. Three seeds are insufficient for statistical conclusions. The text-only model differs in computation depth even if similar in parameter count.
6. **The correct next experiment** is to use only training-side validation to study per-route errors and tune architecture, pre-register a brand-new *externally labeled* holdout before the final comparison, and report all attempted candidates including those that fail. A fair parity target is matching the mean text-only accuracy on that same new holdout, not retroactively gaming the 300-question score.

## Scientific goal

**Achieve reproducible generalization parity or better against a conventional text-only router, while demonstrating a nontrivial contribution from the authentic MaleCNS topology relative to an equally trained rewired control.**

This outcome must be established with frozen data, comparable training and parameter budgets, multiple paired seeds, and separately measured routing accuracy versus end-to-end chat quality.
