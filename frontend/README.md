# FlyGPT v0.6 Frontend Alpha

This directory is the Next.js frontend for FlyGPT.

## Development

Run the existing FastAPI backend from the repository root:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend defaults to proxying `/api/*` to `http://127.0.0.1:8000`.

To use another backend URL:

```bash
FLYGPT_BACKEND_URL="http://127.0.0.1:8000" npm run dev
```

Open the Next.js forwarded port, usually port 3000.

## v0.6 scope

- React chat UI
- existing v0.5 browser memory session compatibility
- memory clear control
- router confidence panel
- FlyGraph SVG HUD with activation trace animation
- FastAPI health status
- responsive desktop/mobile layout

The legacy Jinja/JavaScript frontend is intentionally kept during the migration.
