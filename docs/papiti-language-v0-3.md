# Papiti v0.3: Korean jamo + expanded conversations (research-only)

An independently **random-initialized** causal Transformer, not Qwen,
GPT, Ollama, or other pretrained weights. The MaleCNS router is a
**separate model**. v0.3 is opt-in, does not replace v0.1/v0.2, and
does not change the main FlyGPT backend's default response path.

## Changes

1. **721 original, manually written example pairs from 53 group clusters**,
   compared with v0.2's 225 pairs. The source is
   `training/papiti_dialogues_v0_3.jsonl`. It incorporates v0.2 examples
   plus new Korean question/answer examples. These are artificial training
   examples, not real private user chats or scraped texts. They do not
   constitute sufficient data for a fluent general-purpose assistant.
2. Optional **Unicode NFD jamo tokenizer**: Korean syllables are
   decomposed to their jamo for learning and reassembled via NFC for
   display. This reduces unseen-syllable failures compared with a
   train-only precomposed Hangul-syllable vocabulary. As before, the jamo
   vocabulary is built ONLY from training examples, without held-out
   leakage or pretrained vocabulary. NFD characters may be emitted in
   ungrammatical combinations; this method does not guarantee good Korean.
3. **Repetition penalty** only for recent generated tokens. v0.1/v0.2
   defaults remain exactly 1.0 to preserve historical behavior.
4. Optional **validation patience** for early stopping. The best
   held-out NLL checkpoint is selected automatically.
5. **Unit tests** cover Hangul roundtrip, jamo checkpoint loading,
   seeded generation, and group-level evaluation split.

## Run in Codespaces

After `git pull origin main` and installation of CPU PyTorch into
`.venv-papiti` (if not already installed):

```bash
make papiti-lm-train-v0.3
make papiti-lm-eval-v0.3
make papiti-lm-chat-v0.3
```

The new independent checkpoint is `artifacts/papiti_language_v0_3.pt`.
The original `artifacts/papiti_language_v0_2.pt` remains untouched.

Full training command:

```bash
.venv-papiti/bin/python -m papiti_lm train \
  --data training/papiti_dialogues_v0_3.jsonl \
  --checkpoint artifacts/papiti_language_v0_3.pt \
  --epochs 30 --patience 7 --batch-size 16 --tokenizer jamo \
  --d-model 128 --heads 4 --layers 2 --context 192
```

This is **fresh training**, not finetuning v0.2 weights. Its
tokenizer and dimensions differ from v0.2, so resuming older
weights directly is not supported.

## Real checkpoint from GitHub Actions

`Papiti v0.3 Korean language training` workflow trains on GitHub's
CPU PyTorch runner using the full dataset and emits an artifact called
`papiti-v0-3-trained-weights`. From a successful run, use the Artifacts
section to download the ZIP and extract the single checkpoint file to
`artifacts/papiti_language_v0_3.pt`, then run
`make papiti-lm-chat-v0.3` without retraining locally.

A one-epoch baseline runs against the **same corpus, seed, architecture,
and grouped held-out split**. We compare its NLL/perplexity against
the full best-checkpoint result. We also print actual generation for
greeting, thanks, a paraphrased greeting and sports. These are
qualitative probes, not proof of conversational understanding.
Because v0.2 uses another corpus and syllable tokenizer, v0.2 and
v0.3 raw NLL/perplexity are **not directly comparable**.

## Boundaries and next step

- This is still **one-turn text generation**. No history, external tools,
  web retrieval, system instruction following or safety policy learned.
- It may hallucinate, repeat fragments or output isolated jamo.
- An unseen human-written evaluation set and larger diverse licensed
  corpus are necessary to establish real Korean comprehension.
- Do not publish the local test HTTP service to the internet;
  `make papiti-lm-serve-v0.3` is bound to localhost only.
- No actual biological neurons have been trained to speak. MaleCNS is the
  separately trained task router.
