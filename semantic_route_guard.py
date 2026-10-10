"""Conservative bridge between the unchanged MaleCNS router and Fly Brain plans.

A precise, already-supported conversational intent can be dispatched through
GENERAL even if the learned classifier guesses MATH or another route. Keep
the model's predictions untouched in separate provenance fields and never
misrepresent a deterministic rule as model probability.
"""

from __future__ import annotations

from typing import Any

from brain_runtime import FlyBrainRuntime


# Restrict to low-risk, anchored patterns defined by Fly Brain. Never use
# open-ended questions, summaries, code or mathematical expressions here.
_SUPPORTED_GENERAL_INTENTS = frozenset({
    "introduce_flygpt",
    "describe_supported_features",
    "explain_malecns",
    "acknowledge_presence",
    "request_specific_task",
    "return_welcome",
})


def apply_semantic_route_override(
    message: str,
    route_info: dict[str, Any],
    *,
    brain: FlyBrainRuntime | None = None,
) -> dict[str, Any]:
    """Return an effective route for known intents without losing raw telemetry."""
    if not isinstance(route_info, dict):
        raise TypeError("route_info must be a dict")

    planner = brain if brain is not None else FlyBrainRuntime()
    plan = planner._initial_utterance_plan(message, "general")
    speech_act = str(plan.get("speech_act") or "")
    if not plan.get("ready") or speech_act not in _SUPPORTED_GENERAL_INTENTS:
        return route_info

    if route_info.get("route") == "general" and route_info.get("accepted") is True:
        return route_info

    raw_route = str(route_info.get("route") or "unknown")
    raw_confidence = float(route_info.get("confidence") or 0.0)
    raw_margin = float(route_info.get("margin") or 0.0)
    general_probability = next(
        (
            float(item.get("confidence", 0.0))
            for item in (route_info.get("top_routes") or [])
            if isinstance(item, dict) and item.get("route") == "general"
        ),
        0.0,
    )

    updated = dict(route_info)
    updated.update({
        "route": "general",
        # This is the model's GENERAL probability, not rule confidence.
        "confidence": general_probability,
        "margin": 0.0,
        "accepted": True,
        "semantic_override": {
            "source": "fly_brain_known_intent",
            "speech_act": speech_act,
            "model_route": raw_route,
            "model_confidence": raw_confidence,
            "model_margin": raw_margin,
            "model_accepted": bool(route_info.get("accepted", False)),
        },
    })
    return updated
