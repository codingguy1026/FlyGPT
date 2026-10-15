# Live flying-fly thinking indicator 🪰

The chat's previous `route tracing` ellipsis is replaced by a small
animated pair of flies and a **real SSE stage-aware** status card.

The animated flies, dotted trail, and sparks are decorative. They **do not**
prove that all MaleCNS neurons are currently calculating, nor do they
represent a measured progress percentage.

## Honest stage labels

| Incoming event | Display |
| --- | --- |
| before any router event (`processing`) | 질문 처리 중… |
| `status: router_loading` | 초파리 신경망 준비 중… |
| `trace: input` | 뉴런 입력 신호 계산 중… |
| `trace: step_N` | 뉴런 신호 전파 중… with actual `activations.length` node count and step |
| `status: generation` | 답변을 문장으로 만드는 중… |

Every message shows a playful secondary line:
`파리가 날아다니고 있습니다… 🪰`.

The backend still performs **exactly one** chat inference and emits its real
trace. This change does not alter model weights, biological annotations,
the tool bypass, chat history, or the backend SSE API.

The fly animation continues while the generator runs, but the headline
explicitly stops claiming neuron computation after the router stage.
The live router stage can be very short on fast hardware, so browsers may
briefly render or coalesce it naturally. The app does not slow down inference
just to make the animation look longer.

Only the router trace's actual node count is shown. The ~140k MaleCNS
connectome is *not* presented as the active live chat router.

## Accessibility / review

- Text labels remain visible if the fly emoji cannot render.
- `role=status` announces the current actual processing stage.
- `prefers-reduced-motion: reduce` disables the orbit and sparkle animation.
- Phone screens get a narrower layout.

### Manual smoke checks

1. Ask `안녕`: flies animate while waiting; status changes based on
   the incoming SSE phase and the answer replaces the card when finished.
2. Inspect the live trace event: `routing` displays the actual node count
   and propagation stage, not a made-up progress percentage.
3. Test a slow generator: the card says "답변을 문장으로 만드는 중…",
   not "뉴런 계산 중…".
4. Simulate an unreachable server: error replaces the loading card.
5. Enable OS reduced motion, narrow browser width, and screen reader.
