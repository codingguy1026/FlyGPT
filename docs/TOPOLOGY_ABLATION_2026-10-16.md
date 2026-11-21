# FlyGPT MaleCNS topology ablation: initial findings

Date: 2026-10-16 (KST)

**Status: EXPLORATORY. Does not establish a biological wiring advantage.**
The experimental checkpoints never replace the production router.

## Reproducible run

- Code: training/topology_ablation.py (PR #60)
- GitHub Actions run: https://github.com/codingguy1026/FlyGPT/actions/runs/38855002038
- Artifact: topology-ablation-replicated (download from run page, includes JSON metrics)
- Workflow: .github/workflows/topology-ablation-pilot.yml
- 3 paired seeds: 42, 43, 44
- Training schedule: at most 90 epochs, patience 18, learning rate 0.0015, vectorizer v6, vocabulary 8192
- Per model: trained from scratch with identical text examples, train/validation split, same model architecture and initial seed
- Original frozen Janelia MaleCNS v1.0 scaffold: 256 neurons, 948 edges
- Control: directed double-edge swaps preserving each node's in/out edge counts, overall edge-count and original edge-weight multiset
- Three existing regression datasets; exact teacher-prompt overlaps excluded from evaluation

## Real-world regression: 60 questions

| Seed | MaleCNS routing accuracy | Degree-preserving rewired accuracy | MaleCNS minus rewired |
|---|---:|---:|---:|
| 42 | 90.0% | 83.3% | +6.7 percentage points |
| 43 | 91.7% | 93.3% | -1.7 percentage points |
| 44 | 93.3% | 91.7% | +1.7 percentage points |
| **Mean** | **91.7%** | **89.4%** | **+2.2 percentage points** |

Route label distributions for this set are balanced across six classes; the macro-recall differences are likewise +6.67, -1.67, +1.67 pp per seed (mean +2.22 pp).

## Additional existing regression sets

| Set (after exact overlap removal) | MaleCNS mean accuracy | Rewired mean accuracy | Mean MaleCNS minus rewired macro recall |
|---|---:|---:|---:|
| Paraphrases (20; 5 overlapping questions excluded) | 95.0% | 95.0% | +0.0 pp |
| Hard negatives (21; 9 overlapping questions excluded) | 100.0% | 98.4% | +1.1 pp |

The third seed's rewired model missed one hard-negative example, while the other seed pairs tied.

Internal validation macro recalls (original vs rewired):
- seed 42: 93.3% vs 93.8%
- seed 43: 92.6% vs 93.4%
- seed 44: 93.4% vs 93.0%

These validation results do not show a consistent original-topology advantage either.

## Interpretation (not for inflated marketing claims)

**A small, mixed difference:** The actual MaleCNS topology slightly improved mean route accuracy on one existing regression set under this particular protocol. One of three seeds favored the control, and two other regression sets showed ties or nearly tied results. This is NOT enough to conclude that biological connectivity is causally superior to a conventional task router, let alone that a fruit-fly brain reasons about language.

Important caveats:

1. **Only three seeds:** Too few for strong statistical conclusions. No p-value or confidence interval is claimed.
2. **Not a blind benchmark:** Existing project regressions could overlap with development decisions or contain paraphrases of training prompts. Normalized exact duplicates (14 total across the latter two sets) were removed, but semantic/near-duplicate contamination is possible.
3. **Not a conventional neural-router comparison:** Both models use the same FlyGraphRouter architecture. This is an **anatomical versus rewired topology ablation** only.
4. **Not full MaleCNS:** The original experiment uses a **256-node, 948-edge subset**, not the full connectome.
5. **Controlled degree counts, not everything:** Swaps preserve incoming and outgoing *edge counts* and the global weight multiset; in-strength distributions, biological motifs and full degree correlations can change.
6. **Task routing, not chatbot understanding:** No end-to-end response quality or autonomous reasoning was measured.
7. **Selection and calibration:** Validation sets select stopping points and thresholds separately, using the same procedure across models; actual best epochs differed.

## Next study

Pre-register a new **independently authored, never-trained-on holdout set**. Add a conventional text-only classifier and a topology-free fully connected or degree-matched synthetic baseline with parameter counts and training budget controlled. Evaluate more paired seeds and compare accuracy, macro recall, latency, and calibrated gate precision. Keep all development and experimental checkpoints isolated from the deployed FlyGPT service.

## Verdict

The experiment successfully shows that changing connectivity can alter predictions. **Whether authentic fruit-fly wiring itself provides a reproducible advantage remains unanswered.** This finding should be presented as an honest negative-or-inconclusive pilot, not a breakthrough.
