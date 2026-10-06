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
```

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
