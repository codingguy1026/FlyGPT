# Papiti Orbit 2.0 · Brand & UI redesign

## Intent
FlyGPT is the repository/technical project name. **papiti.** is the human-facing assistant brand.

**Brand line:** Curiosity in motion.  
**Core principle:** Conversation first, explainability on demand.

The existing MaleCNS router, streaming chat responses, session memory, local conversation search/export, and neuron observatory continue to use their current APIs. No fake inference states or fabricated router results are introduced.

## Brand tokens

| Token | Color | Usage |
| --- | --- | --- |
| Deep Pine | `#173D37` | Brand sidebar, send button, headline |
| Signal Lime | `#C5EF8B` | New chat, attention, brand mark |
| Soft Canvas | `#F6F7F1` | Main workspace background |
| Sky Mist | `#B5DBEC` | Tiny signal dot |
| Pine Ink | `#213C36` | Body copy |
| Muted Leaf | `#6C7B72` | Secondary UI labels |

The Papiti Mark is one continuous curved thought path and a small independent spark. This replaces the visible FLY emblem in user-facing conversation and authentication surfaces. Repository names, persisted theme IDs, existing assets, and server version strings remain compatible.

## Information hierarchy

1. **Navigation**: brand, new chat, conversation view, optional neuron map/observability, recent local chats, account/preferences.
2. **Conversation**: title, truthful connection state, messages, composer.
3. **Welcome**: three starter prompts that **prefill** the composer, not auto-submit a request.
4. **Observatory**: a closed-by-default drawer. It only displays real backend route and trace events.
5. **Account**: same authentication logic and account/session separation with a matching visual style.

The previous Ask / Code / Research chips were only input-prefix shortcuts. Removing those chips does **not** change backend routing or add new search functionality. Search/tool selection remains a backend capability/decision.

## Implementation

- `frontend/app/papiti-orbit.css`: new last-loaded brand layer.
- `frontend/components/FlightIdentity.tsx`: PapitiMark visual component.
- `frontend/components/FlyGPTApp.tsx`: conversational labels, opt-in observatory, starter prompts, composer copy.
- `frontend/components/AuthGate.tsx`: branded login, signup and connection states.
- `frontend/public/papiti-mark.svg`: independent favicon.
- `frontend/app/layout.tsx`: import and page metadata.
- `frontend/lib/flight-theme.ts`: default color values with unchanged `bio` theme ID.

## Verification checklist

Automated (run under `frontend/`):
- `npm install` (or `npm ci` if a lockfile is added)
- `npm run typecheck`
- `npm run build`

Manual (with FastAPI backend running):
- Verify login, signup, logout and offline recovery.
- Verify the three starter cards only prefill and focus the composer.
- Send a question, receive streamed status, route, and assistant content; confirm no sudden observatory popup.
- Open the observatory, verify real route/trace values, close via button and backdrop.
- Verify new chat, local history search, history reopen and Markdown export.
- Verify router confirmation, settings and memory clearing still work.
- Verify desktop, tablet (approx. 900px), and narrow mobile (approx. 375px) layouts.
- Verify keyboard focus, reduced motion and legible dark/light colors.

## Deployment caution

This branch is a visual/UI candidate, not a backend replacement. Review and test before merging into `main`. Do not claim production readiness without tests. Avoid deleting legacy CSS until dependencies are audited.
