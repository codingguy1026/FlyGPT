"""Small, deterministic Papiti response realizer.

This is Python language realization, NOT a new trained language model.
It can only express previously validated brain plans or accurately report
that an open-ended request is not yet supported. No network or model calls.
"""
from __future__ import annotations

import re
from typing import Any

from semantic_planner import trusted_fact_fallback


_SOCIAL: dict[str, tuple[str, str]] = {
    "return_greeting": ("안녕! 무엇을 이야기해 볼까?", "Hello! What would you like to talk about?"),
    "acknowledge_thanks": ("천만에! 도움이 됐다면 좋겠어.", "You're welcome! Glad I could help."),
    "return_farewell": ("잘 가! 좋은 하루 보내.", "Goodbye! Have a good day."),
    "acknowledge_presence": ("응, 이야기할 준비가 됐어.", "I'm ready to chat."),
    "request_specific_task": ("어떤 도움이 필요한지 알려줘.", "What would you like help with?"),
    "return_welcome": ("나도 반가워!", "Nice to meet you too!"),
    "acknowledge_personal_sports_win": (
        "축하해! 몇 대 몇으로 이겼어?",
        "Congrats! What was the score?",
    ),
}


def _language(message: str, plan: dict[str, Any]) -> str:
    style = plan.get("style")
    if isinstance(style, dict) and style.get("language") in {"ko", "en"}:
        return style["language"]
    return "ko" if re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", message) else "en"


def _units(plan: dict[str, Any], kind: str) -> list[str] | None:
    units = plan.get("content_units")
    if not isinstance(units, list) or not 1 <= len(units) <= 5:
        return None
    values: list[str] = []
    for unit in units:
        if not isinstance(unit, dict) or unit.get("kind") != kind:
            return None
        value = unit.get("value")
        if not isinstance(value, str) or not value.strip():
            return None
        values.append(value.strip())
    return values


def realize_native(
    message: str,
    route: str,
    *,
    brain_state: dict[str, Any] | None,
    tool_context: str | None = None,
) -> str | None:
    """Express an already planned answer without inventing task content."""
    state = brain_state if isinstance(brain_state, dict) else {}
    plan = state.get("utterance_plan")
    if not isinstance(plan, dict) or plan.get("contract") != "mouth_only_v1":
        return None
    lang = _language(message, plan)
    ko = lang == "ko"
    act = plan.get("speech_act")

    if plan.get("ready") is True:
        if act in {"introduce_flygpt", "describe_supported_features", "explain_malecns"}:
            # This validates fact IDs, exact text, provenance, ordering and language.
            return trusted_fact_fallback(plan)

        if act in _SOCIAL and _units(plan, "communicative_act"):
            return _SOCIAL[act][0 if ko else 1]

        if act == "mirror_social_reaction" and _units(plan, "communicative_act"):
            if re.search(r"[ㅠㅜ]", message):
                return "ㅠㅠ" if ko else "Aw."
            if re.search(r"[ㅋㅎ]", message):
                return "ㅋㅋㅋ" if ko else "Haha!"
            return "응!" if ko else "Yep!"

        if act == "report_no_memory" and _units(plan, "status"):
            return ("관련된 이전 대화를 찾지 못했어." if ko
                    else "I couldn't find a relevant earlier conversation.")

        if act == "report_no_external_evidence" and _units(plan, "status"):
            return ("지금은 확인된 최신 자료가 없어서 답을 단정할 수 없어." if ko
                    else "I don't have verified current sources for that question.")

        if act == "state_exact_math_result":
            values = _units(plan, "exact_result")
            prefix = "Exact math tool result: "
            if (values and len(values) == 1 and tool_context
                    and values[0] == tool_context.strip()
                    and values[0].startswith(prefix)):
                return "🧮 " + values[0][len(prefix):].strip()
            return None

        if act == "verbalize_retrieved_memory":
            values = _units(plan, "retrieved_memory")
            if values:
                excerpts = [re.sub(r"\s+", " ", value)[:250] for value in values[:3]]
                header = "이전 대화에서 찾은 내용이야:" if ko else "Relevant earlier messages:"
                return header + "\n" + "\n".join(f"• {v}" for v in excerpts)

        if act == "report_relevant_stored_knowledge":
            units = plan.get("content_units")
            if isinstance(units, list) and 1 <= len(units) <= 3:
                rendered: list[str] = []
                for unit in units:
                    if not isinstance(unit, dict):
                        return None
                    kind = unit.get("kind")
                    source = unit.get("provenance")
                    value = unit.get("value")
                    if (kind == "verified_knowledge" and source == "verified"):
                        label = "확인된 정보" if ko else "Verified"
                    elif kind == "user_asserted_context" and source == "user-asserted":
                        label = "사용자가 알려준 내용" if ko else "User-provided"
                    else:
                        return None
                    if not isinstance(value, str) or not value.strip():
                        return None
                    rendered.append(f"• {label}: {value.strip()[:300]}")
                return "\n".join(rendered)

        return None

    # A MaleCNS task label is not a solved answer. Be clear about the
    # capabilities of this deliberately limited no-LLM prototype.
    task = state.get("task_plan")
    if (isinstance(task, dict)
            and task.get("contract") == "bounded_task_v1"
            and task.get("ready") is True
            and task.get("route") == route):
        return (
            "지금은 파피티 자체 문장 생성 모드야. 이 자유형 질문은 아직 직접 풀 수 없어. "
            "검증된 계산, 간단한 인사, 파피티 소개처럼 지원하는 요청부터 처리하고 있어."
            if ko else
            "Papiti is using its own deterministic response mode. I can't solve "
            "this open-ended request yet. Verified calculations, greetings and "
            "Papiti information are supported first."
        )
    return None
