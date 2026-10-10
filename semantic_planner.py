"""Small, evidence-bounded semantic planner for FlyGPT's self-description.

This is a deterministic Python planner, not learned reasoning in the MaleCNS
connectome. It selects atomic, locally grounded project facts and a speaking
order; the language model chooses *wording*, never facts or objectives.
"""

from __future__ import annotations

from typing import Any

PROJECT_FACTS: dict[str, dict[str, str]] = {
    "identity.name": {
        "ko": "프로젝트 이름은 FlyGPT이며, 파피티는 한국어 애칭이다.",
        "en": "The project is named FlyGPT, and Papiti is its Korean nickname.",
    },
    "identity.role": {
        "ko": "파피티는 실험 단계의 AI 대화 도우미다.",
        "en": "Papiti is an experimental AI conversation assistant.",
    },
    "architecture.router": {
        "ko": "MaleCNS 기반 작업 라우터가 사용자 질문의 처리 유형을 분류한다.",
        "en": "A MaleCNS-backed task router classifies the type of user request.",
    },
    "architecture.language": {
        "ko": "별도의 언어 모델이 주어진 응답 계획을 자연어 문장으로 표현한다.",
        "en": "A separate language model turns a supplied answer plan into natural language.",
    },
    "architecture.limit": {
        "ko": "초파리 커넥톰 자체가 인간 언어를 만들거나 의식을 가진다는 뜻은 아니다.",
        "en": "The fly connectome itself does not compose human language or imply consciousness.",
    },
    "capability.neurons": {
        "ko": "FlyGPT는 MaleCNS의 뉴런 및 신경 연결을 조회할 수 있다.",
        "en": "FlyGPT can look up MaleCNS neurons and neural connections.",
    },
    "capability.math": {
        "ko": "FlyGPT는 별도의 정확한 수학 계산 도구를 사용할 수 있다.",
        "en": "FlyGPT can use a separate tool for exact mathematical calculations.",
    },
    "capability.memory": {
        "ko": "FlyGPT는 로그인한 계정별로 대화 기억을 분리하여 관리한다.",
        "en": "FlyGPT separates conversation memory by authenticated user account.",
    },
    "capability.limits": {
        "ko": "자유로운 복잡한 추론은 아직 실험 단계이며 실시간 검색은 별도 설정이 필요하다.",
        "en": "Open-ended complex reasoning remains experimental, and live search requires separate configuration.",
    },
    "dataset.definition": {
        "ko": "MaleCNS v1.0은 수컷 초파리 뇌와 배쪽 신경삭의 뉴런 연결 정보를 담은 데이터셋이다.",
        "en": "MaleCNS v1.0 is a connectome dataset covering neurons in the male fruit fly brain and ventral nerve cord.",
    },
    "dataset.access": {
        "ko": "FlyGPT는 neuPrint 서비스를 통해 필요한 MaleCNS 연결 정보를 조회한다.",
        "en": "FlyGPT retrieves selected MaleCNS connection information through neuPrint.",
    },
}

INTENT_FACT_IDS: dict[str, tuple[str, ...]] = {
    "introduce_flygpt": (
        "identity.name",
        "identity.role",
        "architecture.router",
        "architecture.language",
        "architecture.limit",
    ),
    "describe_supported_features": (
        "capability.neurons",
        "capability.math",
        "capability.memory",
        "capability.limits",
    ),
    "explain_malecns": (
        "dataset.definition",
        "dataset.access",
        "architecture.router",
    ),
}

_GOALS = {
    "ko": {
        "introduce_flygpt": "누구인지 설명",
        "describe_supported_features": "지원하는 기능 설명",
        "explain_malecns": "MaleCNS 데이터셋 설명",
    },
    "en": {
        "introduce_flygpt": "introduce the assistant",
        "describe_supported_features": "describe supported features",
        "explain_malecns": "explain the MaleCNS dataset",
    },
}


def answer_goal(speech_act: str, language: str) -> str | None:
    return _GOALS.get(language, _GOALS["ko"]).get(speech_act)


def compose_fact_plan(
    message: str, speech_act: str, language: str
) -> dict[str, Any] | None:
    """Choose a small ordered set of facts, not a memorized final sentence."""
    if speech_act not in INTENT_FACT_IDS:
        return None

    lang = language if language in {"ko", "en"} else "ko"
    text = message.casefold()

    if speech_act == "introduce_flygpt":
        if "이름" in text or "name" in text:
            selection = ("identity.name", "identity.role")
        elif any(term in text for term in ("뇌", "신경", "brain", "neuron")):
            selection = (
                "identity.name",
                "architecture.router",
                "architecture.language",
                "architecture.limit",
            )
        else:
            selection = ("identity.name", "identity.role", "architecture.router")
    elif speech_act == "describe_supported_features":
        selection = (
            "capability.neurons",
            "capability.math",
            "capability.memory",
            "capability.limits",
        )
    else:
        selection = ("dataset.definition", "dataset.access", "architecture.router")

    units = [
        {
            "kind": "grounded_project_fact",
            "fact_id": fact_id,
            "value": PROJECT_FACTS[fact_id][lang],
            "provenance": "repository_project_description",
        }
        for fact_id in selection
    ]

    return {
        "content_units": units,
        "semantic_steps": [
            {"order": index, "fact_id": unit["fact_id"]}
            for index, unit in enumerate(units, start=1)
        ],
        "style": {
            "language": lang,
            "length": "short" if len(units) <= 3 else "medium",
            "tone": "natural",
        },
    }


def trusted_fact_fallback(plan: dict[str, Any]) -> str | None:
    """Recover from unsafe LLM output using *only* original project fact atoms.

    This is a last-resort safety output, not the normal response generator.
    A malformed or fabricated fact ID/text cannot be surfaced through it.
    """
    speech_act = str(plan.get("speech_act") or "")
    allowed = INTENT_FACT_IDS.get(speech_act)
    units = plan.get("content_units")
    if not allowed or not isinstance(units, list) or not 1 <= len(units) <= 4:
        return None
    style = plan.get("style") or {}
    if not isinstance(style, dict):
        return None
    lang = style.get("language")
    if lang not in {"ko", "en"}:
        return None
    verified: list[str] = []
    for unit in units:
        if not isinstance(unit, dict):
            return None
        key = unit.get("fact_id")
        if (
            key not in allowed
            or key in [u.get("fact_id") for u in units[:len(verified)]]
            or unit.get("kind") != "grounded_project_fact"
            or unit.get("provenance") != "repository_project_description"
            or unit.get("value") != PROJECT_FACTS[key][lang]
        ):
            return None
        verified.append(PROJECT_FACTS[key][lang])
    return " ".join(verified)
