# Papiti v0.5: model-only Korean conversation and supervised addition

The v0.5 goal is to evaluate what the independently initialized, small
Transformer learned rather than to make an answer look correct by routing
it to a built-in calculator or a dictionary of canned responses.

## Strict runtime rule

- There is no calculator, regex math parser, fixed output string,
  dictionary mapping, question-based fallback, search, external AI,
  or pretrained model anywhere in the v0.5 chat command.
- The model only predicts tokens autoregressively from learned weights.
- The v0.5 chat command is a separate, explicitly opted-in mode. The
  former v0.4 training and chat experiments remain in the repository
  for comparison. v0.4's old optional calculator is NOT used in v0.5.
- FlyGPT's normal Native/Hybrid chat backend and MaleCNS router stay
  unchanged. The separately trained fly-connectome-derived router is
  not the language model.

## Training data vs runtime hardcoding

In supervised learning, the training data must contain expected
answers. v0.5 generates 3,000 deterministic arithmetic training
examples before training, by choosing addition operand pairs (0..99)
and computing their target labels. This generation runs only in the
training/evaluation code path. The runtime inference command does not
import the arithmetic-data generator and cannot call a calculator.

We reserve 160 randomly selected unordered operand pairs plus these
explicit unseen probes from the training set: 1+1, 23+47, 8+9,
39+56, 44+44, 2+7, 12+32, and 58+29. Evaluation checks exact
matches against independent ground truth, and failures are shown
verbatim. Reverse-order variants of a held-out pair are also excluded
from training. This is a limited test of interpolation, not proof
of general arithmetic or logical reasoning.

For Korean, v0.5 retains the v0.4 721 original human-authored pairs
and the same group-disjoint language validation split. The training
groups alone receive punctuation variation augmentation. We mix the
Korean examples and synthetic mathematics to help the model learn both
kinds of responses, and compare their validation losses independently.

## Tokenizer and model

DigitSubwordTokenizer builds its vocabulary and frequent BPE merges
from the training split only, inherits the Korean word-boundary
logic of v0.4, and treats every ASCII numeral as an individual token
to enable digit composition. No numerical answer is inferred by the
tokenizer. The model uses random-initialized Transformer weights;
the v0.4 checkpoint cannot be directly resumed.

## Reproduce in Codespaces

After merging and pulling the repository:

    git pull origin main
    make papiti-lm-train-v0.5
    make papiti-lm-eval-v0.5
    make papiti-lm-chat-v0.5

The CPU PyTorch training run creates
artifacts/papiti_language_v0_5.pt. Alternately, the successful GitHub
Actions workflow uploads papiti-v0-5-model-only-trained-weights
with the same checkpoint for download into artifacts/.

The v0.5 chat command defaults to greedy output
(temperature=0) so the same inputs give reproducible answers.
To test another sampling configuration, run
python -m papiti_lm.v0_5 chat --temperature 0.65.
This entrypoint deliberately has no --math-tool option.

## Evaluation and limitations

The experiment logs independent language and arithmetic validation
character-normalized negative log-likelihoods and the exact-match
rate on 32 never-trained addition questions, including 1+1은?
and 23+47은?. More than one output may be plausible for language,
so inspect generated dialogue responses too.

The model is very small; 721 language prompts and 3,000 arithmetic
examples are not enough to guarantee grammar, understanding or
generalization to unseen arithmetic pairs. If exact-match arithmetic
accuracy is low, v0.5 genuinely failed the test; we will not conceal
that failure by substituting fixed answers.
