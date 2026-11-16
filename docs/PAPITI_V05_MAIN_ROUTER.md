# Papiti v0.5 chat with the local v0.6.1 speaker

The 166,700-neuron MaleCNS policy independently selects web search, session
memory and calculator. Python builds a `mouth_only_v1` utterance plan from the
supported conversational act or tool evidence. The local Papiti v0.6.1
Transformer renders that plan, and the existing speech validator checks its
output. Normal v0.5 chat uses this checkpoint exclusively.

## Run in Codespaces

From the FlyGPT checkout:

```bash
make enable-papiti-v05
make run
```

Enablement verifies the pinned v0.5 policy checkpoint from
[Actions run 43168156480](https://github.com/codingguy1026/FlyGPT/actions/runs/43168156480),
then updates the local settings. If v0.5 is already active, restart with
`make run` after updating the checkout. Existing hosted generator settings in
an older Codespace do not apply to active v0.5.

If the v0.6.1 checkpoint is missing, use an existing trusted checkpoint at
`artifacts/papiti_language_v0_6_1.pt` or train one:

```bash
make papiti-lm-train-v0.6.1
```

Use `FLYGPT_HYBRID_SPEECH_CHECKPOINT` to select a custom checkpoint. See
[v0.6.1 setup](papiti-language-v0-6-1.md) for the Python environment.

## Verify a reply

Send `안녕` in a new chat. A successful reply shows
`Router v0.5 · 166,700 neurons` and `GEN · papiti-language-v0.6.1`.
`/api/health` reports the speaker model, whether the checkpoint is present,
and `api_key_configured: false` for language generation.

- The speaker is trained on basic social acts, FlyGPT project facts and exact
  calculator results. A calculator result is accepted only if the speaker
  preserves it.
- Open-ended questions such as the apple and banana comparison, web and memory
  summaries, and requests combining multiple tools report an unsupported speech
  plan with the current checkpoint. Selected tools and their statuses still
  appear in the response metadata.
- Web search still requires its own research configuration. Memory retrieval
  stays scoped to the authenticated session.

## Roll back

```bash
make disable-papiti-v05
make run
```

This restores the saved generator settings and returns to the previous router.

## Validation

```bash
python -m unittest tests.test_papiti_v05_active_chat tests.test_papiti_v0_5_policy
```

The policy checkpoint was trained with 128 labeled examples and selected the
exact tool subset for 33 of 36 held-out questions. This measures tool routing.
It provides no measure of open-domain answer quality or biological firing.
