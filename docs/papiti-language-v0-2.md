# Papiti Language v0.2: Korean conversation training experiment 🪰

**Not a fluent conversational model.** This is an original, randomly
initialized character-level Transformer, trained with PyTorch. It is separate
from MaleCNS and does **not** load Qwen, GPT, Ollama or any pretrained weights.

## Dataset & evaluation

- v0.1 demonstration: **48 pairs, 12 groups**.
- v0.2 experiment: **225 manually authored Korean pairs, 22 groups** in
  `training/papiti_dialogues_v0_2.jsonl`.
- Conversations were written as new, rights-cleared examples; no private
  messages, personal chat export, scraped forum, or external dataset is used.
- Each `group` stays exclusively in train **or** validation. The tokenizer
  vocabulary comes from training only. These examples are enough for a
  repeatable learning **experiment**, not trustworthy general-purpose dialogue.
- The held-out NLL and perplexity are meaningful only when comparing runs on
  the **same data and split**. Do **not** directly compare v0.1 and v0.2
  perplexity because their data and character vocabularies are different.

## Train and evaluate from scratch

After pulling the branch and installing CPU PyTorch in `.venv-papiti`:

```bash
# Train fresh weights on the larger corpus. Creates a separate v0.2 checkpoint.
make papiti-lm-train-v0.2

# Test on held-out topic groups.
make papiti-lm-eval-v0.2

# Test sampled chat, with no fixed response templates.
make papiti-lm-chat-v0.2
```

Explicit commands:

```bash
.venv-papiti/bin/python -m papiti_lm train \
  --data training/papiti_dialogues_v0_2.jsonl \
  --checkpoint artifacts/papiti_language_v0_2.pt \
  --epochs 24 --batch-size 16 --seed 42
.venv-papiti/bin/python -m papiti_lm eval \
  --data training/papiti_dialogues_v0_2.jsonl \
  --checkpoint artifacts/papiti_language_v0_2.pt
.venv-papiti/bin/python -m papiti_lm chat \
  --checkpoint artifacts/papiti_language_v0_2.pt --message "안녕!" --seed 42
```

The original v0.1 `make papiti-lm-train`, `make papiti-lm-chat`
targets are unchanged. **v0.2 is freshly initialized, NOT a continuation of
the v0.1 weights.** Character vocabularies differ. In v0.2, gradients still
update randomly initialized parameters, never the original v0.1 checkpoint.

## Actual GitHub Actions training and checkpoint

The `Papiti v0.2 training experiment` GitHub workflow trains **two**
separate models with the exact same v0.2 data, seed and architecture:
**1 epoch** (baseline) versus **24 epochs** (longer training).
It evaluates both on the same held-out topic groups and samples the same
user prompt. This is not guaranteed to improve unseen-dialogue quality;
an increase in training time can overfit.

The workflow publishes `papiti-v0-2-trained-weights`, a temporary
downloadable **GitHub Actions artifact** containing the best 24-epoch
checkpoint, and logs for evaluation/sample inspection. To try the model
in Codespaces, open the successful workflow run, download the ZIP from
**Artifacts**, extract its checkpoint to `artifacts/papiti_language_v0_2.pt`
and run `make papiti-lm-chat-v0.2`. Check artifact expiry before using it.
GitHub Actions checkpoints are not committed to the repository.

## Optional local API, still opt-in

In a separate terminal, after loading a trained checkpoint locally:

```bash
make papiti-lm-serve-v0.2
```

The API remains bound only to `127.0.0.1:8765`. It has no public
authentication or production safety system; don't expose it to the internet.
As in v0.1, `FLYGPT_HYBRID_URL` can optionally point at this experimental
API. The model identifier remains `papiti-char-transformer-v0.1` because the
wire format and architecture are unchanged; only the local trained
checkpoint differs. The separate MaleCNS router and default Native reply
behavior are not changed.

## Recommended acceptance criteria

1. The 225-pair dataset validates and keeps groups disjoint.
2. CPU PyTorch tests, fresh training, held-out evaluation and chat pass.
3. Record **actual** held-out metrics and generated samples from both runs.
4. Do not claim fluent or meaningful conversation based only on lower loss.
   Test unseen human-authored prompts, confusion, repetition and safety
   before considering actual product integration.
