# Papiti v0.6.6: train-only fluency recovery (draft experiment)

## Why

v0.6.5 IB finished at epoch 36 with validation char NLL 0.43353.
Its repeatedly inspected 36-item mixed-style transfer diagnostic reported
1/36 exact training-reply copies, but visibly broken completions such as
"Talk you next cgood time." and concatenated word endings. Lower copying is
not yet evidence of fluent or general-purpose text generation.

## Hypothesis (unproven)

The tiny from-scratch bilingual speaker may have underlearned how to finish
a complete response, even when its opening is good. Keep all human-authored,
train-side sentences and reweight NEXT-TOKEN losses on second clauses (1.6x)
and the final EOS token (1.4x). This is a training-only objective, not a
runtime filter, dictionary, postprocessor or canned response.

A *strictly matched control* starts from the same v0.6.5 model weights and
sees the same deterministic training rows, order, steps, learning rate,
validation set and training budget, with all tokens weighted 1x.

Both models still learn only from unique original + training-only action
examples. They use the original v0.6.5 tokenizer and vocabulary.

The intended outcome is improved **naturalness**, but it is not guaranteed.
Higher suffix/EOS weights may worsen flow, encourage repetition, or cause
premature ending. If so, reject the candidate.

## Training (run locally in Codespaces)

First make sure your completed v0.6.5 checkpoint is present locally:

    ls -lh artifacts/papiti_language_v0_6_5_ib.pt

Then test:

    .venv-train/bin/python -m unittest tests.test_papiti_v066_fluency -v

Train candidate and matched control (both warm-start from the same local
checkpoint; separate output paths):

    make papiti-lm-train-v0.6.6-fluency PAPITI_PYTHON=.venv-train/bin/python
    make papiti-lm-train-v0.6.6-control PAPITI_PYTHON=.venv-train/bin/python

Default: 12 additional epochs, early stopping patience 4, learning rate
0.00015, AdamW, batch size 24, original seed 42 and model capacity.

A baseline epoch 0 is saved before fine-tuning. An epoch-0 selected checkpoint
means the fine-tuning run did NOT beat v0.6.5 validation NLL.

Both candidate and control checkpoints are separate from v0.6.5. The 94 old
frozen tests and validation rows are unchanged. No production deployment.

## Preregistered sealed 36-pair test

After BOTH runs have finished, run ONCE:

    make papiti-lm-eval-v0.6.6-sealed PAPITI_PYTHON=.venv-train/bin/python

Report:
    artifacts/papiti_v066_sealed_fluency_eval.json

This predeclares 36 new style cue pairs (5,0) and (0,4), 18
act/language packets x two combinations. The constituent opening and
follow-up texts appear in the training split, but the composed prompts
and answers are NOT passed to the optimizer. These pairs do not overlap
v0.6.5 action training ((2,4),(3,5),(5,2),(5,3)) or earlier
36-item diagnostics ((4,0),(0,5)) and ((4,1),(1,5)).
The script tests baseline v0.6.5, control and candidate on the SAME pairs.

The automatic report tracks:
- teacher-forced Char NLL and perplexity
- full-reply copies of training examples
- exact-match counts
- two-clause shape completion (necessary, never sufficient)

**Review full generated text manually** for grammar, fluency, meaning,
style following and repetition. Do not claim human-quality speech from
low NLL, lower copying or two-clause shape alone.

Once the 36 outputs have been viewed, they are no longer blind. Never
train directly on those prompt-answer pairs or tune hyperparameters against
them. More genuinely new evaluations are required for later iterations.

## Scientific and engineering limitations

- Only suffix and EOS loss weights change in this ablation. Both arms
  continue the same pretrained v0.6.5 optimizer-independent weights,
  with fresh AdamW optimizer states.
- Both have the same number of examples and optimizer steps, but the
  weighted training loss is NOT numerically comparable across arms.
- Checkpoint selection uses the unchanged validation Char NLL, not any
  sealed transfer or frozen test labels.
- The original content is a narrow, human-authored 18-packet bilingual
  social/prose corpus, NOT open-domain conversational training.
- This is not an official IB curriculum or an IB-endorsed model.
