# FlyGPT · Frontend

The Next.js 15 / React 19 UI for FlyGPT. This folder is the visual workspace; the real connectome queries, routing, session memory, research, and response generation remain owned by the FastAPI backend.

## Develop

Start the backend from the repository root:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then, in a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the forwarded Next.js port, normally 3000. By default, `/api/*` is proxied to `http://127.0.0.1:8000`. Override via `FLYGPT_BACKEND_URL` when required.

## Frontend structure

- `app/page.tsx`: entry point through the authentication gate.
- `components/AuthGate.tsx`: account login/signup, session check, and connection-error recovery.
- `components/FlyGPTApp.tsx`: chat workspace orchestration and telemetry interactions.
- `components/BrainPanel.tsx`: backend router graph and trace visualization, plus an explicitly labeled calcium-style demo.
- `components/RoutePanel.tsx`: router decision summary.
- `components/FlightIdentity.tsx`: reusable FlyGPT visual identity elements.
- `lib/chat-history.ts`: account-scoped local chat records, validation, and storage.
- `lib/chat-presentation.ts`: pure response-to-UI metadata formatting.
- `lib/flight-theme.ts`: named themes without duplication in the app component.
- `lib/http.ts`: shared JSON request/response error handling.
- `lib/api.ts`: typed backend API contract; `lib/auth.ts`: account API.
- `app/globals.css`: established visual system; `app/refinements.css`: focused accessibility and recovery styling.

## Quality checks

```bash
cd frontend
npm run typecheck
npm run build
```

A pull-request workflow runs the same checks automatically. Keep frontend requests bound to the existing FastAPI APIs and do not replace actual route/brain outputs with browser-generated examples.

## Compatibility and safety

The existing `flygpt-chat-history-v0.7:<user-id>` keys and account-scoped memory session IDs remain unchanged. The interface keeps its established palette, sidebar, conversation view, telemetry, and graph UI. Backend outages must not be presented as successful model inference. Browser chat records remain local to the device and may not persist in private-browsing sessions.

The legacy Jinja/JavaScript frontend is intentionally retained during migration.
