# FlyGPT

FlyGPT is an experimental Drosophila connectome interface and connectome-routed
task router.

As of v0.8.0, interactive connectome queries target the **Janelia MaleCNS v1.0**
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

## Answer generation

FlyGPT can use both OpenAI and Gemini at the same time. The MaleCNS router's
task route also selects the preferred language-model provider.

Put the real keys in the ignored local `.env` file:

```bash
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
```

Default route plan:

```text
general   -> OpenAI / gpt-6-luna
memory    -> OpenAI / gpt-6-luna
summarize -> OpenAI / gpt-6-luna
math      -> local math fast-path first, then OpenAI if generation is needed

code      -> Gemini / gemini-3.8-flash
research  -> Gemini / gemini-3.8-flash

connectome queries -> MaleCNS direct handlers, no LLM when a direct query matches
```

OpenAI uses the Responses API by default:

```text
https://api.openai.com/v1/responses
```

Gemini uses Google's OpenAI-compatible chat-completions endpoint:

```text
https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
```

If the preferred provider returns a transient `429`, `502`, `503`, `504`,
or a network timeout, FlyGPT immediately switches to the other configured
provider instead of retrying the same slow request. If both providers are
temporarily unavailable, `gemini-3.7-flash` remains the last-resort Gemini
fallback.

Optional overrides:

```bash
FLYGPT_OPENAI_MODEL=gpt-6-luna
FLYGPT_OPENAI_URL=https://api.openai.com/v1/responses

FLYGPT_GENERATOR_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
FLYGPT_GENERATOR_MODEL=gemini-3.8-flash
FLYGPT_GENERATOR_PROVIDER=gemini
FLYGPT_GENERATOR_FALLBACK_MODEL=gemini-3.7-flash
FLYGPT_GENERATOR_REASONING_EFFORT=low
```

API keys are read only by the backend and are never included in generator
status responses. Keep real API keys out of Git.

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
