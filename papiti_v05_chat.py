"""Papiti v0.5 tool policy and local v0.6.1 speech pipeline."""
from __future__ import annotations

import time
from typing import Any

from brain_runtime import FlyBrainRuntime
from dispatcher import run_math_tool

_TOOL_ORDER = ("memory_lookup", "web_search", "calculator")
_brain = FlyBrainRuntime()


def _tool_results(
    message: str, selected: list[str], *,
    memory: Any, research: Any, session_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, str | None]:
    """Execute each selected tool independently, with bounded evidence."""
    reports: list[dict[str, Any]] = []
    memory_hits: list[dict[str, Any]] = []
    research_data = None
    calculation = None

    for name in _TOOL_ORDER:
        if name not in selected:
            continue
        if name == "memory_lookup":
            try:
                memory_hits = (
                    memory.search(session_id, message, limit=5, exclude_content=message)
                    if session_id else []
                )
                if memory_hits:
                    reports.append({"tool": name, "status": "retrieved", "hits": len(memory_hits)})
                else:
                    reports.append({"tool": name, "status": "no_evidence", "hits": 0})
            except Exception:
                reports.append({"tool": name, "status": "unavailable"})
        elif name == "web_search":
            try:
                retrieved = research.search(message)
                research_data = retrieved.to_dict()
                reports.append({
                    "tool": name,
                    "status": "retrieved" if retrieved.used else "no_evidence",
                })
            except Exception:
                reports.append({"tool": name, "status": "unavailable"})
        elif name == "calculator":
            try:
                calculation = run_math_tool(message)
                reports.append({
                    "tool": name,
                    "status": "verified" if calculation else "unsupported",
                })
            except (ArithmeticError, ValueError, OverflowError):
                reports.append({"tool": name, "status": "failed"})
    return reports, memory_hits, research_data, calculation


def _utterance_plan(
    message: str, route: dict[str, Any], memory_hits: list[dict[str, Any]],
    research: dict[str, Any] | None, calculation: str | None,
) -> dict[str, Any]:
    selected = route["selected_capabilities"]
    if len(selected) > 1:
        # The current speech checkpoint cannot combine independent answers.
        return {}
    # These names select an existing plan builder, based only on the v0.5
    # decision. The old category classifier is not called.
    plan_routes = {"calculator": "math", "memory_lookup": "memory", "web_search": "research"}
    plan_route = plan_routes[selected[0]] if selected else "general"
    if plan_route == "math" and calculation is None:
        # A calculator false positive must not swallow a recognized greeting
        # or other social act. Open-ended requests still produce no ready plan.
        plan_route = "general"
    state = _brain.plan(route_info=route, dispatch={"route": plan_route}, message=message)
    return _brain.finalize(
        state, memory_hits=memory_hits, research_result=research,
        tool_context=calculation,
    )["utterance_plan"]


def run_v05_chat(
    message: str, *,
    policy: Any, memory: Any, research: Any, generator: Any,
    session_id: str | None, budget_seconds: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    decision = policy.decide(message)
    selected = list(decision.get("selected_capabilities") or [])
    scores = dict(decision.get("capability_scores") or {})
    if any(tool not in policy.capabilities for tool in selected):
        raise ValueError("Policy suggested an unsupported tool")

    # Standalone social turns need no tools even if a score exceeds its threshold.
    if selected and _brain.is_standalone_social(message):
        selected = []

    # The new path NEVER calls the old six-category router or dispatch().
    reports, memory_hits, research_data, calculation = _tool_results(
        message, selected, memory=memory, research=research,
        session_id=session_id,
    )
    tools_ms = round((time.perf_counter() - start) * 1000)
    model_name = getattr(policy, "name", "papiti-v0.5-category-free-166700")
    route = {
        "model": model_name,
        "model_version": "v0.5",
        "route": "multi_action",
        "confidence": 0.0,  # Not a task-classification confidence!
        "confidence_kind": "not_applicable",
        "accepted": True,
        "category_free": True,
        "capability_scores": scores,
        "selected_capabilities": selected,
        "top_routes": [{"route": key, "confidence": value}
                       for key, value in scores.items()],
        "n_nodes": policy.n_neurons,
        "all_curated_neurons_attended": True,
        "trace": [],  # Never fabricate neuronal firing.
    }

    plan = _utterance_plan(message, route, memory_hits, research_data, calculation)
    remaining = budget_seconds - (time.perf_counter() - start)
    generation = generator.generate_v061(plan, budget_seconds=remaining)
    error = generation.error_code
    answer = generation.answer
    if error == "hybrid_speaker_missing":
        answer = (
            "🪰 파피티 v0.6.1 발화 모델의 체크포인트가 없어. "
            "터미널에서 make papiti-lm-train-v0.6.1로 학습하거나 "
            "기존 체크포인트 경로를 설정해 줘."
        )
    elif error == "hybrid_speaker_unsupported":
        answer = (
            "🪰 MaleCNS v0.5는 도구 선택을 마쳤어. "
            "파피티 v0.6.1은 아직 이 질문에 답할 수 없어. "
            "인사나 간단한 계산을 물어봐 줘."
        )
    elif not generation.used:
        answer = "🪰 파피티 v0.6.1이 발화 계획에 맞는 답변을 생성하지 못했어."

    total_ms = round((time.perf_counter() - start) * 1000)
    return {
        "answer": answer,
        "response_type": "papiti_v05",
        "route": route,
        "data": {
            "brain_state": {
                "contract": "papiti_v0_5_multi_action_v1",
                "category_free": True,
                "selected_capabilities": selected,
                "capability_scores": scores,
                "tool_reports": reports,
                "speech_act": plan.get("speech_act", "unresolved"),
                "is_biological_firing_simulation": False,
            },
            "generation": {
                "provider": generation.provider,
                "model": generation.model,
                "used": generation.used,
                "error_code": error,
                "finish_reason": generation.finish_reason,
                "latency_ms": generation.latency_ms,
            },
            "memory_hits": memory_hits if "memory_lookup" in selected else None,
            "research": research_data,
            "timings": {
                "tools_ms": tools_ms,
                "generation_ms": generation.latency_ms,
                "total_ms": total_ms,
            },
            "ui_meta": {
                "mode": "v0.5 · multi-action",
                "router_model": model_name,
                "app_version": "0.12.6",
            },
            "v05_error_code": error,
        },
        "timings": {"router_ms": tools_ms, "total_ms": total_ms},
    }
