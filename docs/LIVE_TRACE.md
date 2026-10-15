# FlyGPT real-time neural activity monitor (experimental)

This adds a POST Server-Sent Events endpoint, `/api/chat/stream`, which runs the
**same authenticated chat request once** and delivers the router's genuine
input/propagation activations as each inference step finishes.

The old `/api/chat` JSON endpoint is preserved. Events have a single
`data: <JSON>` line and a blank-line delimiter:

- `status` with `processing`, `router_loading`, or `generation`.
- `trace` with `frame: {stage, activations}`, emitted **inside** the actual
  PyTorch router forward pass. Values match `router.trace` returned in the
  final answer, and are absolute activations normalized to the maximum within
  each frame.
- `route` with the selected category, confidence, accepted flag, and whether
  a semantic override was used.
- Exactly one terminal `result` (the ordinary chat response) or `error`.

No second router prediction, simulated calcium data, or replay-derived
timestamps are generated. Events may happen within milliseconds; browser
rendering, proxy buffering and the separate generator duration can affect
how long a human sees live activity. No artificial delays are inserted.

## Browser behavior

Sending a question opens the monitor immediately. When real frames arrive,
its status reads **LIVE · ROUTER FRAME**. After the route is complete, the
monitor reads **ROUTER COMPLETE · GENERATING**, and after the answer it
reads **REPLAY · RECORDED TRACE**. The slider only navigates **recorded**
frames: replay never pretends to be new neural activity.

The schematic node positions retain the existing green aesthetic. They are
**not real biological spatial coordinates**. Node IDs come from the checkpoint
scaffold and graph edges come from the trained directed topology. Clicking a
node highlights its incident edges and shows its normalized router activity.
A route is the model's task-category selection, not a measured biological
path through specific neurons.

When a request bypasses the router (e.g. the deterministic `1972^2`
math fast path), the monitor says **EXACT TOOL · NO ROUTER** and shows no
fabricated activity. The labels `GCaMP` and `ΔF/F` have been removed
because no real calcium-imaging data is measured.

## Test and rollout

In your Codespace, on the feature branch:

```bash
python -m unittest tests.test_live_trace_stream tests.test_router_live_frames -v
cd frontend && npm run build
cd .. && make run
```

While the backend is running, test with the normal, authenticated site
(and the browser Network tab's `/api/chat/stream` request):

1. Type `안녕`. With a working router checkpoint, events should contain
   `trace` frames for input and successive propagation steps; the live frame
   equals the recorded replay frame after completion.
2. Type `1972^2는?`. No `trace` or `route` events should be emitted
   because the exact arithmetic fast path bypasses the model.
3. Click a node, scrub the replay slider, and confirm body ID/relative
   activity change by frame.
4. Check that `/api/chat` still returns a normal JSON response.

Do not call a lack of visibly animated frames a failure without inspecting
Network event chunks: very fast inference and network/proxy buffering may
make the real trace appear as one batch. No experimental data is uploaded
to a third party by this feature, but normal FlyGPT auth rules still apply.

The tests are designed to run locally; a real Codespaces live-stream/browser
test has not yet been performed by this change. Do not merge/deploy to
production before checking the build and the full test suite.
