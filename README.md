# FlyGPT

FlyGPT is an experimental Drosophila connectome interface and connectome-routed
task router.

## Papiti language v0.6 (opt-in research experiment)

v0.6 adds a staged, source-attributed mathematics curriculum, Korean sentence
and situation examples, contrastive training, and separate validation/final
tests. Its chat and local API generate answers only with the scratch
Transformer, even when the legacy calculator environment variable is set.
See [the Korean experiment guide](docs/papiti-language-v0-6.md) for commands,
data provenance, evaluation criteria and measured limitations. This does not
change `make run`, Native/Hybrid behavior, or earlier version commands.

## Papiti Native (experimental, no LLM)

`make native` starts **Papiti Native v1** explicitly. It keeps the MaleCNS
task router and verified symbolic tools, but **does not start or download
Ollama or call a hosted language model**. The new `native_realizer.py`
renders validated conversation plans with deterministic Python logic.

```bash
make run                # Hybrid neural speech by default
make native             # explicitly force native deterministic mode
python -m unittest tests.test_native_voice -v
```

The first native implementation can express greetings, Papiti/project facts,
verified tool results, limited stored context and missing-evidence messages.
It **cannot yet write free-form creative answers, arbitrary code, or explain
unsolved questions**. Those requests receive a transparent limitation notice.
MaleCNS routes task types; the fly connectome itself does not learn Korean or
produce human language.

The optional HTTP language generator remains available only when explicitly
configured. Native and Hybrid modes do not select or download pretrained
models. Previously installed model files must be removed separately.

## Hybrid evidence-first conversation (new experimental option)

In Hybrid, the router's category/confidence is diagnostic only. An
evidence-need policy, not a category, determines whether a public factual
question requires Brave Search. Personal announcements are never searched
automatically, and private memory stays private. Search excerpts carry
source links and are not misrepresented as trained language-model knowledge.
This does not automatically teach new weights to the speaker. See
[the evidence-first design and limits](docs/hybrid-evidence-first.md).

## Papiti Hybrid v0.6.1 neural speech (experimental)

In Hybrid, the **MaleCNS router and Python reasoning engine still decide
WHAT to say**. The new from-scratch **v0.6.1 Transformer only converts a
completed semantic packet into a response**. The model has no independent
reasoning or calculator authority and is trained only on approved brain
speech plans, not fixed answers at runtime.

```bash
make train-setup
make papiti-lm-train-v0.6.1
make papiti-lm-eval-v0.6.1
make hybrid
```

Hybrid selects v0.6.1 as its preferred experimental speaker **only if a
compatible local checkpoint is present**. Its outputs are labeled
`papiti-hybrid-v0.6.1` when genuinely generated and accepted by semantic
checks. Incorrect exact results, corrupted factual content, repeated text,
missing checkpoints and model execution errors are never presented as
Native language replies in Hybrid. A separately configured, authorized
loopback language model may answer eligible tasks; otherwise the request
returns a visible language-model-unavailable status. No model, formula, or answer is
downloaded or invented at startup. Unsupported open-ended tasks require an explicitly configured local model.
v0.6.1 alone cannot reason about them. To disable this experiment, set
`FLYGPT_HYBRID_SPEAKER=legacy`. See
[the v0.6.1 experiment guide](docs/papiti-language-v0-6-1.md).

### Historical Hybrid v0.1 behavior

## Papiti Hybrid v0.1 (experimental, opt-in)

`make hybrid` combines the native reply engine with an **optional local
language model**. In the historical v0.1 mode, verified plans (greetings, exact calculations,
checked FlyGPT facts and retrieved memory) answered deterministically.
The newer v0.6.1 mode attempts a conditioned neural realization first and
retains deterministic Native fallback.
Only a validated open-ended `bounded_task_v1` plan may be sent to a language
model. Unsupported requests still receive the native limitation message if no
local model is configured.

```bash
make hybrid
python -m unittest tests.test_native_voice tests.test_hybrid_voice -v
```

To experiment with a separately prepared locally hosted language model,
set these variables in your private `.env`:

```bash
FLYGPT_HYBRID_URL=http://127.0.0.1:PORT/v1/chat/completions
FLYGPT_HYBRID_MODEL=YOUR_LOCAL_MODEL_NAME
```

The model endpoint must be loopback. **No model is downloaded or started**
automatically by `make hybrid`; no hosted endpoint, API key, or previous
legacy generator configuration is implicitly reused. This is a routing bridge; a separate experimental from-scratch Transformer
implementation is documented below. No language weights are shipped or
trained automatically. It reuses the existing bounded-task and output
validation gates for local model responses. An optional model may be slow
or hallucinate, so test before relying on its answers.

## Papiti v0.5: **model-only** Korean + learned arithmetic experiment

Papiti v0.5 eliminates the v0.4 calculator from this version's inference
path entirely. Every response, **including math answers**, is predicted
directly by a fresh, random-initialized Transformer. The model receives
separate Korean dialogue training and 3,000 generated, supervised
two-digit addition examples; 168 operand pairs (including `1+1`
and `23+47`) are **excluded from training** so its arithmetic
generalization can be tested honestly. Correct results are **not
guaranteed**. This uses no Qwen/GPT/Ollama pretrained weights.

```bash
make papiti-lm-train-v0.5
make papiti-lm-eval-v0.5
make papiti-lm-chat-v0.5
```

This independently trained checkpoint and launcher do **not** call any
runtime arithmetic parser, hardcoded response dictionary, tool router or
pretrained model. Its output may be wrong, and the eval command prints
every correct and incorrect held-out probe. The older v0.1-v0.4
experiments and the FlyGPT Native/Hybrid default entrypoints remain
available for historical comparison. See
[the v0.5 experiment notes](docs/papiti-language-v0-5.md).

## Papiti v0.4: Korean subword model + explicit arithmetic calculator

Papiti's v0.4 is a separate **from-scratch** Transformer experiment that learns
Korean multi-syllable **BPE tokens only from the training partition**, rather
than emitting isolated decomposed jamo. It uses the exact same 721-dialogue,
group-held-out corpus as v0.3 so character-normalized held-out loss can be
compared fairly. This is still too small to prove conversational fluency.

The optional standalone v0.4 chat and local API also provide a **clearly
separate deterministic arithmetic calculator** for simple expressions such
as `1+1은?`. This does not mean the language model learned arithmetic.

```bash
make papiti-lm-train-v0.4
make papiti-lm-eval-v0.4
make papiti-lm-chat-v0.4
```

The GitHub Actions experiment compares new training with the v0.3 jamo
baseline on the same split and uploads its best checkpoint separately.
See [Papiti v0.4 setup, metrics and limitations](docs/papiti-language-v0-4.md).
Older models and FlyGPT's normal Native/Hybrid defaults are unchanged.

## Papiti v0.3: Korean syllable composition and 721 dialogue examples

Papiti v0.3 remains a self-trained **randomly initialized** Transformer,
separate from the MaleCNS router. It now offers an optional Unicode
Hangul-jamo tokenizer, more varied Korean training examples, an opt-in
output repetition penalty and validation early stopping. No pretrained
Qwen/GPT model is downloaded or used.

```bash
make papiti-lm-train-v0.3
make papiti-lm-eval-v0.3
make papiti-lm-chat-v0.3
```

The GitHub Actions CPU workflow trains a one-epoch control and a separately
trained best-checkpoint candidate on **the same** grouped held-out split,
then uploads a short-lived trained-weight artifact. Read
[the v0.3 experiment guide](docs/papiti-language-v0-3.md) for results,
limitations, and setup. New data do not establish fluency by themselves;
v0.1/v0.2 checkpoints, commands, and production defaults remain available.

## Papiti independent language-model v0.2 (225 dialogues)

The separate Papiti character-level Transformer now has a larger **225-example,
22-topic Korean training experiment**. It is still randomly initialized from
scratch, not pretrained, and remains too small for reliable general chat.
The original v0.1 model and Native/Hybrid defaults are unchanged.

```bash
make papiti-lm-train-v0.2
make papiti-lm-eval-v0.2
make papiti-lm-chat-v0.2
```

A dedicated GitHub Actions workflow also trains and compares fresh **one-epoch**
and **24-epoch** models on the *same* held-out split, and uploads the learned
v0.2 checkpoint as a short-lived artifact. See
[Papiti v0.2 experiment and checkpoint guide](docs/papiti-language-v0-2.md).

## Papiti scratch Transformer v0.1 (experimental)

A separately initialized character-level causal Transformer can now be trained
from scratch with an explicitly supplied rights-cleared conversation corpus.
The bundled 48 original demo pairs are for pipeline checks only and **will not
produce a reliable conversational assistant**.

See [Papiti's original Transformer training guide](docs/papiti-own-transformer.md).

```bash
# First install CPU PyTorch, fastapi and uvicorn in .venv-papiti
make papiti-lm-train
make papiti-lm-eval
make papiti-lm-chat
make papiti-lm-serve
```

The tiny model is not a MaleCNS language generator. Website Native mode remains
the default, and Hybrid needs an explicit local endpoint and checkpoint.

To return to the pure deterministic experiment, use `make native`.

As of v0.10.0, interactive connectome queries target the **Janelia MaleCNS v1.0**
dataset through the official neuPrint service instead of requiring local
FlyWire v783 Parquet files.

## Dataset

Primary dataset:

- Janelia FlyEM Male CNS Connectome v1.0
- neuPrint dataset: `male-cns:v1.0`
- neuPrint server: `https://neuprint.janelia.org`
- project/download page: `https://male-cns.janelia.org/download/`

The MaleCNS release covers the male Drosophila brain and ventral nerve cord.
FlyGPT uses targeted neuPrint queries for neuron partners, directed pairs,
dataset metadata, and bounded scaffold construction.

The full flat connection graph is available from Janelia for bulk analysis, but
FlyGPT does **not** require that 1.1 GB file for normal app/CLI usage.

## Setup

```bash
python -m pip install -r requirements.txt
```

Create a neuPrint account/token, then configure:

```bash
export NEUPRINT_TOKEN=your-token-here
export NEUPRINT_SERVER=https://neuprint.janelia.org
export NEUPRINT_DATASET=male-cns:v1.0
```

Do not commit real tokens.

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## More stable Codespaces sessions

Codespaces is a development environment, not an always-on hosting service.
For a longer demo or to investigate crashes caused by development watchers,
use:

```bash
make stable
```

This first builds the Next.js frontend **before** starting the backend, then
runs the built frontend with `next start` and Uvicorn **without `--reload`**.
Your usual `make run` keeps hot reload for editing. `make stable` supports
the same `.env` config; for local-only generation use
`bash scripts/run_all.sh --stable --local-only`.

In a **second terminal** while the stack is running, sample all three local
endpoints to detect intermittent failures:

```bash
make stability
```

This checks frontend `:3000`, backend `:8000/api/health`, and frontend
API proxy `:3000/api/health` six times. It reports failures by component.
Use `make doctor` to diagnose the current Codespaces forwarding URL separately.
The checks do not prove that a private forwarded port is reachable from an
unauthenticated client or that a language-model provider is available.

**Important:** Stable mode does not keep a Codespace running after GitHub
suspends/stops it. For a public, always-on release, deploy to a hosting service
rather than relying on a Codespaces preview.

## Named-neuron connection partner ranking

Questions such as **"MaleCNS에서 뉴런 10009와 가장 강하게 연결된 뉴런
10개"** are resolved via `MaleCNSConnectome.neuron(body_id)`, not the
global neuron site-count leaderboard.

Input connections are `partner → selected neuron`, and output connections
are `selected neuron → partner`. When the user requests both directions,
FlyGPT fetches the strongest candidates from each direction, merges them
by **directed synapse weight**, and returns the requested overall top N.
The reported weights are neuron-pair connections, **not** the
`pre + post` site totals used for global neuron rankings. Data is
retrieved from neuPrint; query and total timings are shown separately.

This distinction fixes the prior parser issue where generic words
`뉴런 + 연결 + 가장 강` took priority over a selected body ID,
causing a requested specific-neuron ranking to return global TOP 10.

### Fast named-neuron ranking (v0.12.6)

The specialized strongest-partner chat fast path now queries
`ConnectsTo` edges using **neuPrint Cypher** with server-side ranking
and `LIMIT N`. For a mixed input/output TOP N request, FlyGPT makes
**one** neuPrint request with an undirected edge match anchored to the
selected neuron. The query retains the real presynaptic endpoint using
`startNode(edge)`, labels each returned connection as an input or
output, and selects the actual top N across both directions on the server.
For input-only or output-only rankings it still uses the narrower directed
match. This avoids the previous **two sequential remote requests** for
mixed-direction rankings, without fetching a full ROI breakdown.

The response still reports directed IDs, synapse weights, presynaptic NT,
and optionally the number of **ROI entries** reported on each returned
edge (which must not be mistaken for a count of primary neuropils).
The regular `neuron()` detail path and ROI-filtered calls continue using
`fetch_adjacencies()` when a full per-ROI breakdown is needed.

Identical ranked-neuron requests are cached in each running backend process
for **120 seconds** by default (at most 64 entries). Set
`FLYGPT_PARTNER_CACHE_TTL=0` to disable caching or a value in seconds
up to 3600 to tune it. The cached answer can be up to the TTL old, and the
cache is cleared by a process restart. The existing `QUERY` and `TOTAL`
timings remain useful for comparing a cold lookup against a repeated query.

**Important:** Runtime improvements are expected but not guaranteed. Real
neuPrint latency and performance must be measured from your Codespace;
unit tests mock neuPrint and only verify logic/query shape, not actual
server response times. To assess **cold** performance, query a neuron ID
not recently used, note `QUERY` and `TOTAL`, then repeat the exact
question within the 120-second TTL for a **warm** cache measurement.
A backend restart clears the in-process cache. The aspiration is a
roughly 1-second cold query, not a promised latency guarantee.

## Verified positional logic (experimental)

For explicitly formatted Korean line-up puzzles (3 to 7 named participants),
FlyGPT has a small **deterministic** solver for constraints it recognizes:
fixed positions from the left, immediately left/right, not being at an
endpoint, adjacent participants, and specified position differences.

The solver tests all candidate permutations and reports **no solution**,
**one solution**, or **multiple solutions**, along with the candidate counts
after each rule. This prevents small generation models from inventing seating
orders that violate the given constraints. It appears as `VERIFIED · ALL
ARRANGEMENTS` in chat and does not require the LLM to reason about the answer.

This is not a general-purpose natural language or logic prover. If a
participant-specific clause is unfamiliar, the solver declines to claim a
verified result and the request proceeds through the normal experimental LLM
path. Important unsolved questions should still be checked independently.

## Bounded generator recovery

FlyGPT now distinguishes the generation stage from the fly-connectome
router. Transient HTTP errors (429/500/502/503/504), transport disconnects,
body-read interruptions and invalid JSON responses are retried **at most
once** when time remains. If a fallback model is configured, retryable
failures use that model instead of repeatedly calling the same one.

A response marked with `finish_reason=length` is incomplete, not a valid
answer. When budget remains, FlyGPT retries once with a larger output-token
limit. It never shows the truncated fragment as if it were complete.

Generation metadata includes `error_code` and `attempts`, and the interface
can show a recovered request or a failure category without exposing raw
provider error details in the chat text. These retries cannot rescue an
unavailable Ollama process, an exhausted request budget, invalid credentials,
or resource exhaustion on Codespaces. Running a bigger model may also require
more memory and a longer budget.

## Open-ended task generation and semantic gating

FlyGPT uses a MaleCNS-derived router to **classify** user requests. The
connectome does not solve natural-language puzzles itself. For a familiar,
preplanned response (such as the project introduction), the original
`mouth_only_v1` fact-only gate remains in place.

For **accepted requests with no preplanned answer**, a separate
`bounded_task_v1` execution plan now sends general reasoning, code,
summarization and non-tool-solvable math requests to the configured language
model, rather than returning `mouth-only semantic plan is incomplete` before
calling that model. Research requests are delegated only when the web search
actually returned source evidence. Unavailable web results still cannot
authorize invented current events or citations, and all existing grounding
checks for atomic project facts remain active.

This removes a structural refusal, **not** the limits of the underlying model.
A small local language model can still make incorrect deductions in
multi-constraint logic puzzles. Use a more capable configured generator for
reliable complex reasoning, and independently check important results.

## Codespaces preview: HTTP 404 despite local HTTP 200

FlyGPT runs its Next.js frontend on **3000** and its FastAPI backend on **8000**.
The frontend forwards `/api/:path*` to the backend using
`frontend/next.config.mjs`. Start both with:

```bash
make run
```

The launcher now reserves **exactly** ports 3000 and 8000, refuses to start
when either port is already in use, waits for the frontend and its API proxy to
respond, and prints the current Codespaces URL based on `CODESPACE_NAME`.

If the browser reports **HTTP 404** but both local ports returned 200,
leave the existing server running and open a *second terminal*:

```bash
make doctor
```

The doctor checks the page, backend and same-origin API proxy separately;
compares the forwarded host against the local frontend; then tests the
**unauthenticated** forwarded URL. It never prints or transmits the
`GITHUB_TOKEN`. An unauthenticated 404 from a private forwarded port may
be caused by GitHub's forwarding/authentication layer and does **not**
automatically mean Next.js returned 404.

If all three local paths work but your browser still shows 404:

1. In the same running Codespace, open VS Code **PORTS**; confirm that
   port **3000** is forwarded, and open its **current** forwarded address via
   **Open in Browser**. Do not reuse old bookmarks after a restart.
2. If the entry is stale, remove the *forwarding entry* for port 3000 and
   add it again (this does not delete any source files).
3. Use a GitHub-authenticated browser session for a **private** forwarded
   port. Keep the port private rather than publishing your personal account
   interface simply for troubleshooting.
4. If PORTS shows a healthy current address but the gateway still returns
   404, fully stop/restart the Codespace and forward 3000 again. This is
   a GitHub-hosted access-layer problem; changes to the FlyGPT Python model
   or MaleCNS dataset will not fix it.

Never share your GitHub token, `.env` credentials, neuPrint token or private
session cookie while diagnosing a port.

## CLI

Dataset statistics:

```bash
python cli.py stats
```

Strongest partners for a MaleCNS body:

```bash
python cli.py neuron 12781 --limit 10
```

Inspect one directed pair:

```bash
python cli.py pair 12781 85165
```

Show strongest total directed edges:

```bash
python cli.py top --limit 10
```

## Router scaffold

The task router can now build its compact graph scaffold from real MaleCNS
connections via neuPrint:

```bash
python training/build_scaffold.py \
  --out training/malecns_scaffold.json \
  --nodes 256 \
  --edges 4096 \
  --candidate-edges 32768
```

The resulting checkpoint records `male-cns:v1.0` in its scaffold metadata.
Existing FlyWire-trained checkpoints remain readable, but they must be retrained
with `training/malecns_scaffold.json` before the router itself is genuinely
MaleCNS-backed.

## Local accounts and login

FlyGPT includes a self-hosted email/password login flow. Accounts are stored in
`data/flygpt_auth.sqlite3`. Password plaintext is never stored. Browser
sessions use HttpOnly, SameSite=Lax cookies.

Optional settings:

```bash
export FLYGPT_AUTH_PATH=data/flygpt_auth.sqlite3
export FLYGPT_COOKIE_SECURE=0
```

Behind HTTPS, use `FLYGPT_COOKIE_SECURE=1`.

Chat memory is scoped by authenticated user plus conversation session.


## Connectome brain state

FlyGPT v0.12 adds a deterministic decision layer between the MaleCNS router
and the language model. The goal is to keep task choice and evidence policy
upstream of prose generation:

```text
user input
  -> MaleCNS graph router
  -> fly brain_state
  -> memory / research / exact-tool execution
  -> finalized brain_state
  -> LLM language realization
```

The `brain_state` records the selected route, response objective, retrieval
action, evidence policy, confidence/margin, and a compact signature of the
strongest activations from the final connectome graph-propagation step. The
generator is instructed to follow that state rather than silently choosing a
different route or retrieval policy.

This does not mean a biological fly is composing sentences. The current
MaleCNS-backed network still learns task-level decisions rather than a full
semantic answer plan. v0.12 establishes the boundary needed for future
multi-head training where the connectome model can predict richer answer-plan
slots and leave the LLM primarily responsible for wording.

## Compositional Fly Brain response planning

The first bounded fact planner lives in `semantic_planner.py` and is called by
`brain_runtime.py` **after MaleCNS route selection**. For a small set of
recognized project-specific questions (identity, capabilities, and MaleCNS),
the planner selects **individual, bilingual repository-grounded facts**, their
order, and a language/length objective. It does **not** store a complete
self-introduction to repeat for every question.

For example, `너 누구야?`, `너 이름이 뭐야?`, and `네 뇌가 뭐야?`
select different combinations of identity, architecture, and limitation facts.
In normal operation, `generator_runtime.py` sends a compact Korean/English
fact brief to the configured language model, without showing it raw plan JSON.
The LLM may change wording and grammar but may not invent new facts.

A language mismatch or leaked prompt structure triggers a last-resort,
deterministic response assembled from the **validated fact atoms**. Tampered
facts or forged ordering fail closed. This fallback is a safety net, not the
normal conversation path. A small local model can still be awkward, repetitive,
or inaccurate; run manual conversations to assess quality.

**Limits:** This is a deterministic Python fact-selection layer, **not new
training of the MaleCNS connectome**. The connectome still classifies tasks;
semantic composition is rule-based and restricted to known project facts.
Unknown open-ended questions remain gated instead of being answered with
fabricated information. neuPrint and router checkpoints remain unchanged.

Run focused tests without model downloads or neuPrint access:

```bash
python -m unittest tests.test_semantic_composition -v
```

## Answer generation

### Explicit, strictly local generation

`make run` defaults to Papiti Hybrid, with real neural speech required.
`make hybrid` uses the local trained v0.6.1 speaker when a completed
meaning packet is available. Neither launcher downloads or silently selects
a pretrained language model. For unsupported open-ended prompts, configure
a separate authorized loopback language model using
`FLYGPT_HYBRID_URL` and `FLYGPT_HYBRID_MODEL`. Without one,
Hybrid reports that generation is unavailable rather than returning
Native fallback prose.

To use a model already installed in Ollama, configure your ignored local
`.env` with its exact name:

```dotenv
FLYGPT_GENERATOR_MODE=http
FLYGPT_LOCAL_ONLY=1
FLYGPT_LOCAL_MODEL=YOUR_INSTALLED_MODEL_NAME
```

Run `make local`. For `make fast`, set `FLYGPT_FAST_MODEL` to an
installed model name first. A missing model fails with a clear error;
the app never pulls model weights at startup. The optional
`scripts/setup_ollama_generator.sh` helper likewise uses only an
explicitly chosen, already-installed model.

Local-only generation rejects hosted URLs and hosted fallbacks. MaleCNS
queries and optional research still may use external services, and the
semantic gate cannot solve arbitrary open-ended problems. `make native`
restores the deterministic path. Unused Ollama weights must be removed
separately with `ollama rm`, not by changing GitHub code.

FlyGPT can forward generative routes to an OpenAI-compatible
chat-completions endpoint.

### Gemini 3.8 Flash

For explicit hosted HTTP mode, FlyGPT has a Gemini preset. Set
`FLYGPT_GENERATOR_MODE=http` and put the real key in the ignored local
`.env` file:

```bash
GEMINI_API_KEY=your-key-here
```

When `GEMINI_API_KEY` is present and no explicit generator URL is set,
FlyGPT automatically selects:

```text
provider: gemini
model: gemini-3.8-flash
endpoint: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
reasoning effort: low
fallback model: gemini-3.7-flash
```

For transient `502`, `503`, or `504` responses, FlyGPT retries Gemini 3.8
Flash once after a short delay. If the second attempt is still unavailable, it
tries `gemini-3.7-flash` once before surfacing an error. Both the fallback
model and retry delay can be overridden with
`FLYGPT_GENERATOR_FALLBACK_MODEL` and `FLYGPT_GENERATOR_RETRY_DELAY`.

Hosted generation skips the local Ollama startup and preload path. The API key
is forwarded only by the backend in the Authorization header and is never
included in generator status responses.

### Other OpenAI-compatible providers

Override the generator settings when needed:

```bash
export FLYGPT_GENERATOR_URL=http://127.0.0.1:11434/v1/chat/completions
export FLYGPT_GENERATOR_MODEL=llama3.2:3b
export FLYGPT_GENERATOR_PROVIDER=ollama
```

Hosted providers can additionally use `FLYGPT_GENERATOR_API_KEY`.
`FLYGPT_GENERATOR_REASONING_EFFORT` accepts `low`, `medium`, or `high`
for compatible providers. Keep real API keys out of Git.

## Python API

```python
from connectome import MaleCNSConnectome

with MaleCNSConnectome() as cns:
    print(cns.stats())
    print(cns.neuron(12781, direction="both", limit=10))
    print(cns.context_for_neuron(12781))
```

## Tests

```bash
python -m unittest discover -s tests -v
```


## Account-adaptive routing

FlyGPT v0.9 adds a bounded online personalization layer on top of the frozen
MaleCNS router checkpoint. The global checkpoint is never rewritten by normal
chat traffic.

For authenticated users, FlyGPT can:

- learn a high-confidence routing example automatically when the frozen router
  and personalized route agree with a comfortable confidence/margin
- give stronger weight to explicit route confirmations from the UI
- reuse similar phrasing later to gently bias the route distribution
- keep each account's adaptation completely separate
- decay old examples over time and cap each account at 240 examples
- clear the learned routing profile independently from conversation memory

The adaptive-routing database stores a SHA-256 prompt digest plus hashed
word/character/context features and the route label. It does **not** store the
raw prompt text in the adaptive-learning database. Conversation memory remains a
separate feature with its own storage and clear controls.

Optional path override:

```bash
export FLYGPT_ROUTE_LEARNING_PATH=data/flygpt_route_learning.sqlite3
```

Useful API endpoints:

```text
GET  /api/router/learning/status
POST /api/router/learning/feedback
POST /api/router/learning/clear
```


## Provenance-aware long-term knowledge

FlyGPT v0.11 separates conversation memory from durable knowledge. The
`KnowledgeStore` keeps account-scoped items with provenance, confidence,
verification state, timestamps, optional expiry, and correction history.

Normal chat traffic follows a deliberately conservative policy:

- first-person profile/project statements can be stored as `user-asserted`
  context
- general claims supplied by a user remain unverified claims and are **not**
  used as factual evidence
- trusted evidence can promote an item to `verified` when it comes from a
  repository, official source, verified manual source, or web retrieval layer
- lower-trust contradictory claims cannot override a stronger verified item
- corrected structured facts supersede old values without deleting history
- expired verified facts are excluded from retrieval
- secret-like strings such as passwords, API keys, access tokens, and private
  keys are rejected from automatic long-term learning
- knowledge failures are non-critical; chat can continue without the optional
  context layer

The generator receives provenance labels separately from conversation history.
It may use `[verified]` items as factual evidence. `[user-asserted]` items
are only the user's own profile/project context and must not be presented as
independently verified external facts.

Default storage:

```bash
data/flygpt_knowledge.sqlite3
```

Override it with:

```bash
export FLYGPT_KNOWLEDGE_PATH=data/flygpt_knowledge.sqlite3
```

Account APIs:

```text
GET  /api/knowledge/status
GET  /api/knowledge/items
POST /api/knowledge/remember
POST /api/knowledge/reject
POST /api/knowledge/clear
```

Objective facts are intentionally **not** exposed to the client as an arbitrary
"mark verified" action. Verification is reserved for trusted retrieval/source
code paths so the UI cannot turn an unsupported claim into a fact.


## Live web verification and learning

FlyGPT v0.11 can optionally ground `research` routes with Brave Search's
LLM Context API. The search layer returns extracted source text plus URL/date
metadata that can be passed directly to the generator.

Configure it server-side:

```bash
BRAVE_SEARCH_API_KEY=your-key-here
FLYGPT_SEARCH_LANG=ko
FLYGPT_SEARCH_SAFESEARCH=strict
```

The web-learning pipeline is intentionally conservative:

1. retrieve relevant web evidence with strict relevance filtering
2. answer from that evidence and expose numbered source labels to the generator
3. ask the generator for atomic fact candidates and exact supporting URLs
4. reject any URL that was not part of the current retrieval result
5. require either two independent root domains or one very high-authority
   primary source
6. store only facts that pass the deterministic gate as `verified`
7. assign an expiry window based on volatility:
   - rapid-changing facts: 2 days
   - medium-changing facts such as versions/policies/pricing: 30 days
   - slow-changing facts: 180 days
8. automatically exclude expired facts from future knowledge retrieval

A language model cannot promote a fact merely by claiming high confidence.
Its output is only a candidate. Source membership, domain independence, and
authority checks are enforced in Python before `KnowledgeStore` sees it.

If Brave search is unavailable or unconfigured, research mode fails closed:
FlyGPT may still answer from non-live context where appropriate, but it is told
not to fabricate current search results or citations.

The default research endpoint is:

```text
https://api.search.brave.com/res/v1/llm/context
```

Research telemetry includes source count, search latency, verification latency,
and the number of web facts promoted to verified long-term knowledge.
