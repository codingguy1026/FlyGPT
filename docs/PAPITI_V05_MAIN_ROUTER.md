# Papiti v0.5 MAIN chat policy: switching from six-way routing

**Status:** opt-in application integration. Checkpoint originally trained in
[Actions run 43168156480](https://github.com/codingguy1026/FlyGPT/actions/runs/43168156480).
This is a 166,700-curated-neuron full structural attention model with a small
three-capability, multi-label tool policy. It is not a general language model or
biological firing simulation.

## Turn on in GitHub Codespaces

1. Sync the updated application branch / merged main.
2. Run:

```bash
cd /workspaces/FlyGPT
make enable-papiti-v05
```

The command downloads the *trained policy weights* if missing from the trusted
GitHub Actions artifact, verifies the pinned SHA-256
`75bfe6703e5eccc0d13b5ef030ddb631addbdfe4be81470375c72ce32615e28f`,
checks exact 166,700-neuron provenance/inference and updates the ignored
private `.env` file. It configures **HTTP external language generation**, NOT
Ollama / Qwen. It preserves the previous switch settings in
`data/papiti_v05_previous_env.json`.

3. In private `.env` add a real `GEMINI_API_KEY=...` value **if one is not
already configured**. NEVER paste the key into a chat, commit, or pull request.
Keep `FLYGPT_GENERATOR_URL` unset for the normal Gemini preset in
`scripts/run_all.sh`; an existing explicit provider URL continues to take
precedence. Valid provider model names and API quotas must be confirmed against
the provider's current documentation.

4. Restart FlyGPT with `make run` and send a normal chat request.

The chat UI should display `Router v0.5 · 166,700 neurons` plus the selected
capabilities instead of `Router vmalecns.v0.4`. Example:

- `사과와 바나나의 공통점이 뭐야?` should ask the external speaker with no
  extra tools when the v0.5 policy selects none.
- `오늘 한국 야구 경기 결과를 알려줘` can select `web_search`; real search
  requires a separately configured Brave research service.
- `전에 말한 계산을 다시 해줘` can select `memory_lookup` and `calculator`;
  each capability is independent and is attempted within the authenticated
  user session.

**Note:** the three tool scores reflect experimental teacher-labeled
multi-label classification, not independent proof that MaleCNS understands the
question. The language model handles substantive reasoning and wording.

## Precise runtime boundaries

- **Active (`FLYGPT_V05_ACTIVE=1`):** ordinary messages go through
  `PapitiV05PolicyRuntime.decide`, execute the *selected* tools, and deliver
  evidence + instructions to an explicit external HTTP language-model API.
  The old six-category `predict_route()` is **not invoked** for these requests.
- **Disabled:** the entire existing v0.4 / old-model chat pipeline works as
  before, unmodified. Historic connectome-specific inspection shortcuts,
  graph endpoints and non-chat routes remain supported. V0.5 does not lie
  about the older graph's firing representation.
- **Missing / corrupt checkpoint:** fail **closed** with a clear v0.5
  checkpoint error, never silently claim that a v0.4 reply came from v0.5.
- **Missing API key / HTTP mode:** the v0.5 policy still runs and reports
  that a separate external language model is unavailable; it does not install
  or use any local language model.
- **Search unavailable:** the evidence message forbids fabricating fresh
  news, prices or citations. Exact math is only claimed when the deterministic
  math parser succeeds. Memory retrieval remains scoped to the authenticated
  user's session.
- No raw user messages are written to the v0.5 route diagnostics. The
  existing conversation memory and knowledge stores preserve their own
  pre-existing behavior.

## Roll back in one command

```bash
make disable-papiti-v05
make run
```

This restores `.env` switch variables from before enablement and leaves the
trained artifact intact. Avoid manually passing a v0.5 checkpoint to the
legacy `FLYGPT_MODEL_PATH`: that loader expects six fixed categories and
is incompatible with the category-free policy contract.

## Validation

```bash
python -m unittest tests.test_papiti_v05_active_chat -v
python -m unittest tests.test_papiti_v0_5_policy -v
```

Tests cover "no tool", combined tools, session-scoped memory, API missing,
web retrieval unavailable, and a direct app request that raises if the
old categorical router is called.

## Remaining work

The policy checkpoint was trained on 128 labeled examples and evaluated on
36 held-out examples (33/36 exact tool subsets). This is NOT a generalization
guarantee. A larger human-reviewed corpus, independent real-dialogue blind
tests, tool call reliability, safeguards for prompt injection, provider quotas,
and comparison to a no-connectome model must precede claims about production
readiness. This v0.5 work is an explicit, reversible integration milestone,
not automatic deployment or a completed autonomous brain.
