# FlyGPT v0.7 · Live Brain + Long Memory

The Next.js frontend for FlyGPT.

## Development

Run the FastAPI backend from the repository root:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api/*` to `http://127.0.0.1:8000` by default.

To use another backend URL:

```bash
FLYGPT_BACKEND_URL="http://127.0.0.1:8000" npm run dev
```

Open the Next.js forwarded port, normally port 3000.

## v0.7

- NDJSON live chat event stream
- route decision emitted before the final answer
- FlyGraph propagation frames streamed step by step
- React Brain HUD driven by live server frames
- SQLite short-term conversation memory
- compressed long-term memory capsules every 8 user turns
- recall searches both recent messages and memory capsules
- memory counters in the frontend
- math fast-path and existing v0.5 dispatcher behavior preserved

### Important

The Brain HUD visualizes FlyGPT model activations over the FlyWire-derived
scaffold. It is not a measurement of biological neural firing.

The v0.7 memory capsule compressor is deterministic and lightweight. It
compresses user turns into searchable durable snippets; it is not yet an
LLM-generated semantic memory system.
