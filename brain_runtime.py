from __future__ import annotations

from copy import deepcopy
from typing import Any


class FlyBrainRuntime:
    """Turn connectome-router activity into a compact upstream decision state.

    This layer is intentionally deterministic. It does not call an LLM and it
    does not author prose. The generator receives the resulting state and acts
    only as the language-realization layer.
    """

    _ROUTE_PLANS: dict[str, dict[str, Any]] = {
        "general": {
            "objective": "respond_to_current_message",
            "retrieval": "none",
            "response_mode": "conversation",
            "evidence_policy": "Use the current message, conversation continuity, and supplied verified account knowledge only.",
            "directives": [
                "Answer the current intent directly.",
                "Do not invent external facts that require live verification.",
            ],
        },
        "code": {
            "objective": "solve_programming_request",
            "retrieval": "none",
            "response_mode": "practical_code_help",
            "evidence_policy": "Use the user's code/context and supplied verified knowledge; state assumptions when required.",
            "directives": [
                "Prefer a small correct implementation or concrete fix.",
                "Keep assumptions explicit instead of silently changing requirements.",
            ],
        },
        "summarize": {
            "objective": "summarize_supplied_material",
            "retrieval": "none",
            "response_mode": "summary",
            "evidence_policy": "Use only material supplied in the request or conversation context.",
            "directives": [
                "Preserve the source meaning.",
                "Do not add unsupported facts.",
            ],
        },
        "math": {
            "objective": "solve_math_request",
            "retrieval": "exact_math_if_available",
            "response_mode": "math_explanation",
            "evidence_policy": "Prefer deterministic math-tool output when present.",
            "directives": [
                "Use exact tool evidence when available.",
                "Keep the explanation proportional to the user's request.",
            ],
        },
        "memory": {
            "objective": "recall_relevant_prior_context",
            "retrieval": "memory",
            "response_mode": "memory_recall",
            "evidence_policy": "Use retrieved prior-session messages as the evidence for claims about earlier conversation.",
            "directives": [
                "Ground recalled details in retrieved memory.",
                "If retrieval finds nothing relevant, say so instead of fabricating a memory.",
            ],
        },
        "research": {
            "objective": "answer_with_current_external_evidence",
            "retrieval": "research",
            "response_mode": "grounded_research",
            "evidence_policy": "Use live retrieval evidence for current/external factual claims and preserve source labels.",
            "directives": [
                "Ground time-sensitive factual claims in supplied web evidence.",
                "If live retrieval is unavailable, do not fabricate current facts or citations.",
            ],
        },
    }

    def _neural_signature(
        self,
        route_info: dict[str, Any],
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        trace = route_info.get("trace") or []
        if not isinstance(trace, list) or not trace:
            return []

        final_frame = trace[-1] if isinstance(trace[-1], dict) else {}
        activations = final_frame.get("activations") or []
        if not isinstance(activations, list):
            return []

        ranked: list[tuple[int, float]] = []
        for index, value in enumerate(activations):
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            ranked.append((index, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return [
            {"node_index": index, "activation": round(score, 4)}
            for index, score in ranked[: max(1, int(limit))]
        ]

    def plan(
        self,
        *,
        route_info: dict[str, Any],
        dispatch: dict[str, Any],
    ) -> dict[str, Any]:
        route = str(dispatch.get("route") or route_info.get("route") or "general")
        base = deepcopy(self._ROUTE_PLANS.get(route, self._ROUTE_PLANS["general"]))

        confidence = float(route_info.get("confidence", dispatch.get("confidence", 0.0)) or 0.0)
        margin = float(route_info.get("margin", dispatch.get("margin", 0.0)) or 0.0)
        accepted = bool(route_info.get("accepted", dispatch.get("status") != "uncertain"))

        if confidence >= 0.85 and margin >= 0.25:
            certainty = "high"
        elif confidence >= 0.65 and margin >= 0.12:
            certainty = "medium"
        else:
            certainty = "low"

        state = {
            "source": "connectome_router",
            "route": route,
            "objective": base["objective"],
            "retrieval": base["retrieval"],
            "response_mode": base["response_mode"],
            "evidence_policy": base["evidence_policy"],
            "directives": list(base["directives"]),
            "confidence": round(confidence, 6),
            "margin": round(margin, 6),
            "certainty": certainty,
            "accepted": accepted,
            "neural_signature": self._neural_signature(route_info),
            "evidence": {
                "available": None,
                "count": 0,
                "knowledge_hits": 0,
            },
        }

        if route == "math" and dispatch.get("tool_context"):
            state["retrieval"] = "exact_math"
        return state

    def finalize(
        self,
        state: dict[str, Any],
        *,
        memory_hits: list[dict[str, Any]] | None = None,
        research_result: dict[str, Any] | None = None,
        tool_context: str | None = None,
        knowledge_hits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        final = deepcopy(state)
        retrieval = str(final.get("retrieval", "none"))
        evidence = dict(final.get("evidence") or {})
        evidence["knowledge_hits"] = len(knowledge_hits or [])

        if retrieval == "memory":
            count = len(memory_hits or [])
            evidence["count"] = count
            evidence["available"] = count > 0
            if count == 0:
                final["directives"].append(
                    "Explicitly report that no relevant prior-session memory was retrieved."
                )
        elif retrieval == "research":
            result = research_result or {}
            sources = result.get("sources") or []
            count = len(sources) if isinstance(sources, list) else 0
            used = bool(result.get("used"))
            evidence["count"] = count
            evidence["available"] = used and count > 0
            if not evidence["available"]:
                final["directives"].append(
                    "Explicitly report that live external evidence was unavailable."
                )
        elif retrieval in {"exact_math", "exact_math_if_available"}:
            available = bool((tool_context or "").strip())
            evidence["count"] = 1 if available else 0
            evidence["available"] = available
        else:
            evidence["available"] = bool(
                (tool_context or "").strip() or knowledge_hits
            )
            evidence["count"] = 1 if (tool_context or "").strip() else 0

        final["evidence"] = evidence
        return final
