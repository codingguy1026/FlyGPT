# Papiti v0.6.4: factorized bilingual clause learning (experimental)

## Why this experiment exists
The v0.6.3 36-item *previously unpaired* cross-style transfer set had:
- 13/36 verbatim training replies (36.1%); 23/36 non-copied.
- Manual strict whole-response review: 13/36 natural, **all 13 copied**.
- 0/23 natural among the non-copied responses, and 17/36 conveyed the basic intent.
- No 36-item probe has been added to training. Neither style-pair-specific
  answers nor generated model failures are copied back as training targets.
- Historical 94-item test: v0.6.3 char NLL 0.47175; 62/94 copied a training reply.
  (v0.6.2 inquiry: 0.42579 and 41/94 copied.)

**Human review is essential:** the model's character NLL and exact-match rate
alone do not measure naturalness. The 94-item set is a repeatedly inspected
regression suite, not an untouched blind test.

## Learning change
The current 0..3 styles and new 4..5 styles are taught only within their own
style families; the transfer set requests 4+0 and 0+5. A new optional
`--composition-curriculum` trains from scratch using two auxiliary lessons
per unique **training-side** social response:

1. **Opening focus**: Mask the *follow-up style cue* with `?`; teacher-force
   the original two-part response, but compute loss only through the opening
   **and its newline**, not the follow-up or premature EOS.
2. **Follow-up focus**: Mask the *opening style cue* with `?`; teacher-force
   the original two-part response, but compute loss only on the follow-up
   **and EOS**. The opening remains actual context, not an inference-time
   template.
3. First six epochs focus on both clause types. Subsequent epochs optimize
   all full training responses plus a rotating third of the auxiliary lessons.
   This discourages learning an inseparable joint style pair.
4. All prompts/targets are authored or derived from **training split only**.
   Neither inference-time regex rewrites nor prohibited-word lists, answer
   tables, frozen probe answers, pretrained weights, or arbitrary knowledge
   injection are introduced.

This is a hypothesis about compositional generalization, **not a guaranteed
improvement**.

## Targets on the unchanged 36-item transfer exam
A person must label whole-response `grammar_ok`, `meaning_ok`,
`severe_fact_error`, and `language_mix` (1=yes, 0=no).

| Goal | Threshold |
| --- | --- |
| Broken / unnatural whole responses | At most **7 of 36** (19.4%) |
| Natural whole responses | At least **29 of 36** |
| Semantic intent retained | At least **33 of 36** (91.7%) |
| Exact training-reply copies | At most **10 of 36** (27.8%) |
| Natural among non-copied outputs | At least **70%** |
| Korean-English leakage | **0 of 36** |
| Severe fact errors | **0 of 36** |

Secondary regression guardrail: report char NLL and copy count on the same 94
examples, and prefer no deterioration beyond 5% relative to v0.6.3 **without
overriding the primary human-reviewed transfer targets**. This is an experiment
acceptance target, not a claim of having achieved it.

## Run locally on Codespaces
Use the new branch `experiment/papiti-v064-factorized-clauses`.

```bash
.venv-train/bin/python -m unittest discover -s tests -p 'test_papiti_v0_6_4_composition.py'
make papiti-lm-train-v0.6.4-composition PAPITI_PYTHON=.venv-train/bin/python
.venv-train/bin/python -m papiti_lm.v0_6_1 eval \
  --checkpoint artifacts/papiti_language_v0_6_4_composition.pt \
  --report artifacts/papiti_v0_6_4_eval.json
make papiti-lm-transfer-v0.6.4 PAPITI_PYTHON=.venv-train/bin/python
make papiti-lm-review-v0.6.4 PAPITI_PYTHON=.venv-train/bin/python
# Fill grammar_ok / meaning_ok / severe_fact_error / language_mix manually.
make papiti-lm-review-score-v0.6.4 PAPITI_PYTHON=.venv-train/bin/python
```

Recommended training experiment uses 128 hidden units x 3 layers, 36 epochs,
batch 24, learning rate 0.0007; isolated checkpoint (no deployment).
For a fairer curriculum-only comparison with v0.6.3's original architecture
and training budget:

```bash
make papiti-lm-train-v0.6.4-control PAPITI_PYTHON=.venv-train/bin/python
```

**Do not tune on the same 36 items indefinitely.** Their repeated inspection
will gradually turn them into a development set. Before any release or robust
generalization claim, collect separately authored, truly unseen prompts and
evaluate responses against an independent human rubric.

## Non-goals
This does not change the MaleCNS router, native fact-checking policy, online
runtime, production checkpoint, or the source of verified factual answers.
It does not make Papiti an IB-certified student or a general-purpose LLM.
