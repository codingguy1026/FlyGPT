from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from semantic_planner import compose_fact_plan


class FlyBrainRuntime:
    """Turn connectome-router activity into an upstream semantic decision state.

    This layer is intentionally deterministic. It does not call an LLM and it
    does not author surface prose. It produces the meaning/intent packet that
    the downstream generator is allowed to verbalize.
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

    _GREETING_RE = re.compile(
        r"^(?:안녕(?:하세요|하세용|하세여)?|ㅎㅇ|하이|hello|hi|hey)[!?.~\s]*$",
        re.IGNORECASE,
    )
    _THANKS_RE = re.compile(
        r"^(?:고마워(?:요)?|감사(?:해|합니다|해요)?|thanks?|thx)[!?.~\s]*$",
        re.IGNORECASE,
    )
    _FAREWELL_RE = re.compile(
        r"^(?:잘\s*가|잘자|안녕히\s*(?:가세요|계세요)|바이|bye|goodbye)[!?.~\s]*$",
        re.IGNORECASE,
    )
    _EMOTICON_RE = re.compile(r"^[\sㅋㅎㅠㅜㅇㅅ^._;:()<>/=+\-]{1,24}$")

    # Conservative, anchored phrases: do not guess the meaning of arbitrary
    # general questions or let the downstream LLM invent an answer plan.
    _IDENTITY_RE = re.compile(
        r"^(?:(?:파피티(?:야)?)[,\s]*(?:(?:너|넌|너는|당신은)\s*)?|(?:너|넌|너는|당신은)\s*)"
        r"(?:누구(?:야|니|세요|인가요|입니까)?|뭐(?:야|니|예요|에요)?)"
        r"[?!。\.\s]*$|^(?:who are you|what are you)[?!\.\s]*$"
        r"|^(?:(?:파피티야?\s*)?(?:너(?:는)?\s*)?)?이름(?:이|은)?\s*"
        r"(?:뭐야|뭐니|뭐예요|뭐에요)[?!\.\s]*$"
        r"|^(?:(?:파피티야\s*)?(?:너의|네|니)\s*)"
        r"뇌(?:는|가)?\s*(?:뭐야|뭐니|뭐예요)[?!\.\s]*$"
        r"|^what is your name[?!\.\s]*$"
        r"|^what is your brain[?!\.\s]*$",
        re.IGNORECASE,
    )
    _CAPABILITIES_RE = re.compile(
        r"^(?:(?:너|넌|너는|파피티(?:야)?)\s*)?"
        r"(?:뭘|무엇을|뭐)\s*할\s*수\s*있(?:어|니|나요|어요)"
        r"[?!\.\s]*$|^(?:what can you do|how can you help)[?!\.\s]*$",
        re.IGNORECASE,
    )
    _MALECNS_RE = re.compile(
        r"^malecns(?:가|는|란|이란)?\s*(?:뭐야|뭔데|뭐예요|무엇인가요|가\s*뭐야)?"
        r"[?!\.\s]*$|^what is malecns[?!\.\s]*$",
        re.IGNORECASE,
    )
    _HOW_ARE_YOU_RE = re.compile(
        r"^(?:잘\s*지내|(?:오늘\s*)?기분\s*어때|how are you)"
        r"[?!\.\s]*$",
        re.IGNORECASE,
    )
    _HELP_RE = re.compile(
        r"^(?:도와줘|도와주세요|도움이\s*필요해|help me|i need help)"
        r"[?!\.\s]*$",
        re.IGNORECASE,
    )
    _MEET_RE = re.compile(
        r"^(?:반가워(?:요)?|만나서\s*반가워(?:요)?|nice to meet you)"
        r"[!\.\s]*$",
        re.IGNORECASE,
    )

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

    @staticmethod
    def _language_hint(message: str) -> str:
        if re.search(r"[가-힣]", message):
            return "ko"
        if re.search(r"[A-Za-z]", message):
            return "en"
        return "match_user"

    def _initial_utterance_plan(self, message: str, route: str) -> dict[str, Any]:
        """Build a strict semantic packet.

        ready=False means the fly brain has not yet produced enough semantic
        content for a mouth-only generator. The LLM is not allowed to fill in
        that missing thought on its own.
        """
        text = " ".join((message or "").strip().split())
        plan: dict[str, Any] = {
            "contract": "mouth_only_v1",
            "semantic_authority": "fly_brain",
            "ready": False,
            "speech_act": "unresolved",
            "content_units": [],
            "style": {
                "language": self._language_hint(text),
                "length": "short",
                "tone": "natural",
            },
            "permissions": {
                "infer_new_meaning": False,
                "add_new_facts": False,
                "add_new_questions": False,
                "change_objective": False,
            },
            "forbidden_additions": [
                "unspecified facts",
                "unspecified questions",
                "offers to help unless explicitly planned",
                "self-introduction unless explicitly planned",
            ],
        }

        if route != "general" or not text:
            return plan

        if self._GREETING_RE.fullmatch(text):
            plan.update(
                {
                    "ready": True,
                    "speech_act": "return_greeting",
                    "content_units": [
                        {
                            "kind": "communicative_act",
                            "value": "Return the user's greeting briefly and warmly.",
                        }
                    ],
                }
            )
            return plan

        if self._THANKS_RE.fullmatch(text):
            plan.update(
                {
                    "ready": True,
                    "speech_act": "acknowledge_thanks",
                    "content_units": [
                        {
                            "kind": "communicative_act",
                            "value": "Acknowledge the user's thanks briefly and warmly.",
                        }
                    ],
                }
            )
            return plan

        if self._FAREWELL_RE.fullmatch(text):
            plan.update(
                {
                    "ready": True,
                    "speech_act": "return_farewell",
                    "content_units": [
                        {
                            "kind": "communicative_act",
                            "value": "Return the farewell briefly and warmly.",
                        }
                    ],
                }
            )
            return plan

        if self._EMOTICON_RE.fullmatch(text):
            plan.update(
                {
                    "ready": True,
                    "speech_act": "mirror_social_reaction",
                    "content_units": [
                        {
                            "kind": "communicative_act",
                            "value": "Give a very short social reaction matching the user's visible emotion; add no factual content.",
                        }
                    ],
                    "style": {
                        "language": self._language_hint(text),
                        "length": "very_short",
                        "tone": "casual",
                    },
                }
            )

        # Select facts and their order upstream; do not store an English
        # paragraph to be quoted by the downstream local language model.
        for pattern, speech_act in (
            (self._IDENTITY_RE, "introduce_flygpt"),
            (self._CAPABILITIES_RE, "describe_supported_features"),
            (self._MALECNS_RE, "explain_malecns"),
        ):
            if pattern.fullmatch(text):
                composed = compose_fact_plan(
                    text, speech_act, self._language_hint(text)
                )
                if composed is not None:
                    plan.update({
                        "ready": True,
                        "speech_act": speech_act,
                        **composed,
                    })
                return plan

        if self._HOW_ARE_YOU_RE.fullmatch(text):
            plan.update({
                "ready": True,
                "speech_act": "acknowledge_presence",
                "content_units": [
                    {
                        "kind": "communicative_act",
                        "value": (
                            "Reply briefly that the assistant is ready to chat. "
                            "Do not assert human emotions or subjective experience."
                        ),
                    }
                ],
            })
            return plan

        if self._HELP_RE.fullmatch(text):
            plan.update({
                "ready": True,
                "speech_act": "request_specific_task",
                "content_units": [
                    {
                        "kind": "communicative_act",
                        "value": (
                            "Ask the user what specific task they need help with. "
                            "This clarification question is explicitly planned."
                        ),
                    }
                ],
            })
            return plan

        if self._MEET_RE.fullmatch(text):
            plan.update({
                "ready": True,
                "speech_act": "return_welcome",
                "content_units": [
                    {
                        "kind": "communicative_act",
                        "value": "Warmly acknowledge the user's pleasure in meeting.",
                    }
                ],
            })
            return plan

        return plan

    def plan(
        self,
        *,
        route_info: dict[str, Any],
        dispatch: dict[str, Any],
        message: str = "",
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
            "utterance_plan": self._initial_utterance_plan(message, route),
            "evidence": {
                "available": None,
                "count": 0,
                "knowledge_hits": 0,
            },
        }

        if route == "math" and dispatch.get("tool_context"):
            state["retrieval"] = "exact_math"
        return state

    @staticmethod
    def _set_evidence_plan(
        final: dict[str, Any],
        *,
        speech_act: str,
        content_units: list[dict[str, Any]],
        length: str = "short",
    ) -> None:
        plan = dict(final.get("utterance_plan") or {})
        style = dict(plan.get("style") or {})
        style["length"] = length
        plan.update(
            {
                "ready": bool(content_units),
                "speech_act": speech_act,
                "content_units": content_units,
                "style": style,
            }
        )
        final["utterance_plan"] = plan

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
            hits = memory_hits or []
            count = len(hits)
            evidence["count"] = count
            evidence["available"] = count > 0
            if count == 0:
                final["directives"].append(
                    "Explicitly report that no relevant prior-session memory was retrieved."
                )
                self._set_evidence_plan(
                    final,
                    speech_act="report_no_memory",
                    content_units=[
                        {
                            "kind": "status",
                            "value": "No relevant prior-session memory was retrieved.",
                        }
                    ],
                )
            else:
                units = []
                for item in hits[:5]:
                    role = str(item.get("role", "user"))
                    content = str(item.get("content", "")).strip()
                    if content:
                        units.append(
                            {
                                "kind": "retrieved_memory",
                                "role": role,
                                "value": content[:700],
                            }
                        )
                self._set_evidence_plan(
                    final,
                    speech_act="verbalize_retrieved_memory",
                    content_units=units,
                    length="medium",
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
                self._set_evidence_plan(
                    final,
                    speech_act="report_no_external_evidence",
                    content_units=[
                        {
                            "kind": "status",
                            "value": "Live external evidence was unavailable for this request.",
                        }
                    ],
                )
            else:
                # Raw search context is evidence, not a finished thought. Keep strict
                # mouth-only mode closed until an upstream semantic extractor turns it
                # into atomic claims.
                plan = dict(final.get("utterance_plan") or {})
                plan["speech_act"] = "research_semantics_pending"
                plan["ready"] = False
                final["utterance_plan"] = plan
        elif retrieval in {"exact_math", "exact_math_if_available"}:
            text = (tool_context or "").strip()
            available = bool(text)
            evidence["count"] = 1 if available else 0
            evidence["available"] = available
            if available:
                self._set_evidence_plan(
                    final,
                    speech_act="state_exact_math_result",
                    content_units=[{"kind": "exact_result", "value": text}],
                )
        else:
            evidence["available"] = bool(
                (tool_context or "").strip() or knowledge_hits
            )
            evidence["count"] = 1 if (tool_context or "").strip() else 0

            # General questions may report relevant account-scoped evidence.
            # An asserted user preference is not an independently verified fact.
            # Never promote unverified general claims to an answer.
            plan = dict(final.get("utterance_plan") or {})
            if final.get("route") == "general" and not plan.get("ready"):
                units: list[dict[str, Any]] = []
                for item in (knowledge_hits or [])[:3]:
                    status = str(item.get("status", ""))
                    kind = str(item.get("kind", ""))
                    statement = str(item.get("statement") or "").strip()
                    if not statement:
                        continue
                    if status == "verified":
                        unit_kind = "verified_knowledge"
                    elif status == "asserted" and kind in {"personal", "project"}:
                        unit_kind = "user_asserted_context"
                    else:
                        continue
                    units.append({
                        "kind": unit_kind,
                        "value": statement[:700],
                        "provenance": (
                            "verified" if unit_kind == "verified_knowledge"
                            else "user-asserted"
                        ),
                    })
                if units:
                    self._set_evidence_plan(
                        final,
                        speech_act="report_relevant_stored_knowledge",
                        content_units=units,
                        length="medium",
                    )
                    evidence["available"] = True
                    evidence["count"] = len(units)

        final["evidence"] = evidence
        return final
