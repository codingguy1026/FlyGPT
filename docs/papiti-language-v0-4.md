# Papiti v0.4: Korean subword language model + optional calculator

**Experimental training, not a production chat model.** Papiti's language
model is still an independently initialized, small causal Transformer
without weights from Qwen/GPT/Ollama. It remains separate from the MaleCNS
connectome-derived router and does not replace `make run`.

## Why v0.4?

v0.3 learned patterns using decomposed Hangul jamo but generated malformed
syllables and vague conversations. When asked `1+1은?`, it produced
incoherent text instead of doing arithmetic. v0.4 tackles those as two
different concerns:

1. **Subword BPE learned only on the training split.** The tokenizer learns
   common Korean syllable combinations and phrases as unified tokens,
   never merging across whitespace boundaries. Its vocabulary and merge
   rules are stored with the trained checkpoint. It is not pretrained and
   uses no external tokenizer model. Compared with jamo-level generation,
   it avoids producing independent decomposed jamo, but it can still
   generate incoherent sequences of otherwise valid words.
2. **Explicit and optional deterministic calculator.** In v0.4 chat mode,
   `--math-tool` handles narrowly formatted short expressions with
   `+ - * /` and parentheses. It parses only a safe subset of Python's
   arithmetic syntax; **no eval, code execution or external service**.
   The calculator, **not the Transformer**, answers `1+1은?` with `2야.`.
   Malformed, unsupported or unsafe input does not execute any code.
3. **Char-normalized held-out evaluation.** v0.3 jamo and v0.4 BPE
   tokenize differently, so raw token perplexity is not comparable.
   The extra `char_nll` and `char_perplexity` divide each sequence's
   total target token loss by its count of original Unicode codepoints
   plus EOS. The *same* 721-example corpus, seed, grouped held-out
   split, architecture and training budget is used for an ablation.
   The result might favor the original v0.3; report it either way.
4. **Fully opt-in.** Existing v0.1/v0.2/v0.3 training, chat, serving,
   native routing, checkpoints and behavior remain available.

## Run in Codespaces

If `.venv-papiti` already has CPU PyTorch and the repository is current:

```bash
git pull origin main
make papiti-lm-train-v0.4
make papiti-lm-eval-v0.4
make papiti-lm-chat-v0.4
```

The checkpoint is `artifacts/papiti_language_v0_4.pt`. **Training
starts fresh** with new BPE tokenization; v0.3 weights cannot be
directly resumed. You can instead download the successful GitHub
workflow's `papiti-v0-4-trained-weights` artifact ZIP and extract the
checkpoint into `artifacts/` before running the chat command.

```
you> 안녕
papiti> (new model-generated text, results vary)
you> 1+1은?
papiti> 2야.
```

To see raw language-model output for the arithmetic question, deliberately
disable `--math-tool` in a direct `python -m papiti_lm chat` call.
To enable the tool on the loopback API instead, use
`make papiti-lm-serve-v0.4`. It sets `PAPITI_LM_ARITHMETIC=1` only in
that explicitly opted-in serve process; older serve targets do not enable it.
The API still binds to `127.0.0.1` and is **not authenticated**.
Never expose it directly to the public internet.

## Limitations / next experiments

721 examples are **not enough** to establish fluent Korean, reliable
interpretation or safe instruction following. The calculator does not
improve neural mathematical reasoning and handles only strictly matched,
bounded expressions; multi-turn dialogue, retrieval, facts, and general
problem solving remain unsupported. Compare actual unseen prompt
responses, not just loss, before treating v0.4 as better than v0.3.
The evaluation split remains group-disjoint; neither held-out examples
nor the user's private conversations are included in training.
