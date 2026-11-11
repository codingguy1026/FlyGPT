# Papiti's first **own** language model (v0.1, research-only)

This is an **independently initialized, character-level causal Transformer**,
not Ollama/Qwen, a prompt template, a rule-based response lookup or a model
trained from proprietary chat history. It uses PyTorch to learn next-character
distributions with a masked answer-only objective. It does **not** claim that
the MaleCNS connectome is generating language or that it is comparable to
a production conversational model.

**Status: implementation, not a trained release.** The repository does not
ship weights. The 48 hand-authored sample dialogues cover 12 topical groups
and are for pipeline smoke tests, not usable general-conversation quality.
Free-form replies are genuinely sampled from the trained weights, but will
probably be nonsensical or repetitive at this tiny data scale.

## 1. Explicit, offline training

Use a separate Python 3.12 virtual environment to avoid disturbing the
existing FlyGPT training and serving environments:

```bash
python3 -m venv .venv-papiti
.venv-papiti/bin/python -m pip install --upgrade pip
# Use the PyTorch CPU installation instructions appropriate for your system.
.venv-papiti/bin/python -m pip install torch fastapi uvicorn
.venv-papiti/bin/python -m papiti_lm train \
  --data training/papiti_demo_dialogues.jsonl \
  --epochs 12 --context 192 --d-model 96 --heads 4 --layers 2
```

The experiment uses random initialization, an autoregressive causal attention
mask, explicit teacher-forced answer-token labels, AdamW and gradient clipping.
It splits *groups*, not individual lines, into train/validation deterministically.
The best held-out negative-log-likelihood checkpoint is stored **locally** at
`artifacts/papiti_language_v0_1.pt`. The report includes held-out perplexity.
No texts or model checkpoints are committed automatically, and no training
happens during website requests.

Your own training corpus is newline-delimited JSON:

```json
{"group":"sports-a","user":"오늘 축구했어","assistant":"어땠어? 재미있었어?"}
```

Add many **rights-cleared and consented** conversations without identifiable
private details. A shared `group` clusters related examples so they cannot
appear on both sides of the validation split. Use independent evaluation topics,
not copied prompt/answer pairs, and consider a stricter external human
evaluation before promoting a model. Do not scrape private chats or mix
hold-out examples back into training. Character vocabulary is built from the
training partition **only**; unseen characters become unknown tokens.

## 2. Evaluate and chat with learned weights

```bash
.venv-papiti/bin/python -m papiti_lm eval
.venv-papiti/bin/python -m papiti_lm chat --message "오늘 학교에서 축구 이겼어!"
.venv-papiti/bin/python -m papiti_lm chat
```

The model **generates** characters autoregressively rather than selecting
hard-coded response strings. The `--seed` option can reproduce a sample.
It handles only one user turn at a time; the v0.1 corpus does **not** teach
system prompts, instruction following, conversation history, tool use,
source citation or safety behavior. Do not treat its claims as reliable.

## 3. Optional link to Papiti Hybrid

In terminal A, start the experimental OpenAI-compatible local server:

```bash
.venv-papiti/bin/python -m papiti_lm serve --port 8765
```

It binds to **127.0.0.1 only**, with a small request/token budget and **no
streaming**. Do not expose it to the internet: there is no authentication or
production-grade safety enforcement. It accepts the most recent user message;
system/developer/history content is not learned or used by this demo.

In terminal B, on the **same host or network namespace**, configure the
existing Hybrid bridge explicitly:

```bash
export FLYGPT_HYBRID_URL=http://127.0.0.1:8765/v1/chat/completions
export FLYGPT_HYBRID_MODEL=papiti-char-transformer-v0.1
make hybrid
```

Only supported bounded open-ended plans reach the model. Verified math and
native social acts still use the deterministic path. The Hybrid output guard
may reject poor, empty or invalid model text and return a limitation.
No pretrained model is silently substituted. If your frontend/backend and
the local server are in different containers, loopback will not connect them.

## 4. Testing and limitations

```bash
python -m unittest tests.test_papiti_lm -v
python -m unittest tests.test_native_voice tests.test_hybrid_voice -v
```

Corpus tests run in the normal CI environment; PyTorch-specific tests are
skipped if Torch is not installed. Train/inference must be smoke-tested in a
Torch-enabled environment before treating this as an operational model.
Train on a tiny clean corpus first, then enlarge it legally, and evaluate
held-out examples and qualitative novelty before suggesting it can converse.

The MaleCNS trained router and this language model are **separate modules**.
No end-to-end connectome-language learning has been demonstrated.
