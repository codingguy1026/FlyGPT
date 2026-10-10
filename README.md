# FlyGPT

FlyGPT is an experimental Drosophila connectome interface and connectome-routed
task router.

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

## Answer generation

### Strictly local LLM generation (first independence milestone)

FlyGPT can run its language realization layer **without a hosted LLM API**.
Install Ollama and its small starter model once (initial downloads need network):

```bash
bash scripts/setup_ollama_generator.sh
make local
```

Or set `FLYGPT_LOCAL_ONLY=1` in `.env`, then run `make run`.
The mode defaults to `http://127.0.0.1:11434/v1/chat/completions`
and model `qwen2.5:0.5b-instruct` (override with `FLYGPT_LOCAL_MODEL`
or `FLYGPT_GENERATOR_MODEL`). The generator refuses non-loopback URLs,
ignores hosted API keys, disables HTTP proxies and redirects, and never
falls back to a hosted language model. A broken local model returns an error
rather than using a cloud service. Check `/api/generator/status` for
`local_only` and `local_endpoint_allowed`.

**Scope:** Only LLM generation is local. MaleCNS data queries can still use
neuPrint, and optional Brave Search still uses its online API. This is **not**
a fully offline app. The current fly-brain `mouth_only_v1` semantic gate also
rejects open-ended conversations it cannot plan; installing a local LLM does
not, by itself, make the fly router ChatGPT-equivalent. The default 0.5B
model is an inexpensive starting point, not a ChatGPT-quality model. Memory,
planning, and local-model quality need separate evaluation.

FlyGPT can forward generative routes to an OpenAI-compatible
chat-completions endpoint.

### Gemini 3.8 Flash

`make run` has a Gemini preset. Put the real key in the ignored local
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
