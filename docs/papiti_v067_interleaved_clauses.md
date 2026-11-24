# Papiti v0.6.7: interleaved clause rehearsal (scratch experiment)

## Motivation

v0.6.6 fine-tuning of v0.6.5 reduced validation Char NLL to ~0.404,
but in a newly opened 36-item transfer test training-reply copying rose
from 5/36 (v0.6.5) to 16/36 (v0.6.6 fluency) and 17/36 (control).
Human reading found awkward invented words, unstable Korean, and poor
style-cue compliance. Extra fine-tuning alone is not the solution.

## Hypothesis, not a claim of success

Rehearsing *independent opening and follow-up clauses throughout training*
may help the model follow new style combinations while also preserving
complete-sentence supervision.

- Both models are **random-initialized** (no v0.6.6 fine-tuning).
- Both use v0.6.5's original TRAIN-only tokenizer and same model capacity,
  264 unique original training examples and the 72 action pairs.
- Epochs 1-6 are identical to the v0.6.5 factorized foundation.
- Epochs 7+ always use the SAME original + rotating action examples.
  Each arm adds 20% more TRAIN-only items per epoch:
  - **Candidate:** rotating opening-only and follow-up-only masked-label
    exercises built from those original complete training sentences.
  - **Control:** the corresponding complete original training sentences.
- Same number of sampled rows, minibatches and optimizer steps, seed,
  learning rate, early stopping, and existing validation split.
- **Important:** supervised *token* counts differ between candidate/control
  because masking deliberately reduces their target spans. Train NLL
  between arms is not directly comparable.
- There is NO model-ranked loss-based reflection in v0.6.7. That feature is
  reserved for the original v0.6.5 experiment: this ablation isolates a
  different continuous rehearsal technique. It is therefore not an
  apples-to-apples reproduction of v0.6.5's entire training schedule.

The data volume and tiny model size make fluent novel Korean/English
utterances unlikely from this single change alone. If transfer performance
does not improve without memorizing complete replies, reject the technique.

## Run in Codespaces

    .venv-train/bin/python -m unittest tests.test_papiti_v067_interleaved -v

Train both arms independently from scratch (different checkpoints):

    make papiti-lm-train-v0.6.7-interleaved PAPITI_PYTHON=.venv-train/bin/python
    make papiti-lm-train-v0.6.7-control PAPITI_PYTHON=.venv-train/bin/python

They use the existing 36 epoch budget and early stopping patience 7,
with seed 42, 128d / 3 layers, 448 context, learning rate 0.0007.
Neither changes the existing v0.6.5/v0.6.6 checkpoints or the app's
default speaker.

## One-time previously unseen exam

ONLY AFTER both new models complete:

    make papiti-lm-eval-v0.6.7-blind PAPITI_PYTHON=.venv-train/bin/python

This compares:
1. Existing v0.6.5 reference
2. v0.6.7 full sentence replay control
3. v0.6.7 interleaved clause candidate

Result: artifacts/papiti_v067_blind_transfer_eval.json

The 36 predeclared cue pairs are (4,2) and (2,5), 18 social
act/language packets x 2. They are not included in training or in the
previously opened tests: v0.6.3/v0.6.4 (4,0)/(0,5), another inspected
36-probe (4,1)/(1,5), v0.6.5 training (2,4)/(3,5)/(5,2)/(5,3),
and v0.6.6 sealed probes (5,0)/(0,4).

A probe can contain *already taught individual clauses*, which means
this is a **recombination** exam, not a test of open-world general
language understanding. Unmodified 94 historical tests remain untouched.

Compare complete natural grammatical replies, meaning, style adherence,
copying, repetition, and strange cross-language tokens through human
review. Neither copied-reply counts nor NLL nor correct 2-clause shapes
prove meaningful generalization. Do not tune on the newly exposed
probe set after viewing the outputs.

## Integrity

All new auxiliary examples are derived solely from original training
rows, with unchanged human-authored texts. The experiment creates no
runtime string replacement, banned-word blacklist, or fixed reply
lookup. No user conversation data is collected.
