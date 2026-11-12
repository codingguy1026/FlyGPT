"""Evidence-first conversation policy for Hybrid.

An upstream MaleCNS classification is retained for diagnostics, but it does
NOT veto conversational requests. Decide whether retrieval is needed from
the user's words and available *grounded* evidence, not the route label or
a speculative claim about the language model's hidden knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_BROWSE_REQUEST = re.compile(
    r"(?:검색|찾아\s*봐|찾아\s*줘|웹에서|인터넷에서|최신|최근|실시간|"
    r"뉴스|현재|오늘\s*(?:날씨|경기|환율|가격)|"
    r"\bsearch\b|\blook\s*up\b|\bweb\b|\blatest\b|\brecent\b|\btoday'?s\b)",
    re.I,
)
_QUESTION = re.compile(
    r"(?:[?？]|(?:뭐|무엇|누구|언제|어디|왜|어떻게|어느|"
    r"몇|알려\s*줘|설명해\s*줘|정리해\s*줘|정보|"
    r"\bwho\b|\bwhat\b|\bwhere\b|\bwhen\b|\bwhy\b|\bhow\b|\bwhich\b))",
    re.I,
)
_PAST_CHAT = re.compile(
    r"(?:내가\s*(?:전에|예전에|지난번|말했던|정했던)|"
    r"우리가\s*(?:전에|예전에|지난번)|"
    r"이전\s*대화|지난\s*대화|기억해\s*\?|"
    r"\b(?:we|i)\s+(?:previously|earlier)\b)",
    re.I,
)
_PERSONAL = re.compile(
    r"(?:내\s*(?:비밀번호|암호|전화번호|집\s*주소|계정|학교생활|성적)|"
    r"나\s*(?:누구|어때|어디\s*살)|"
    r"\b(?:my|mine|our|ours|private|password|address)\b)",
    re.I,
)
_NO_WEB = re.compile(r"(?:검색\s*하지\s*마|검색\s*없이|인터넷\s*쓰지\s*마|오프라인|no\s+(?:web|search|browsing))", re.I)
_SENSITIVE_IDENTIFIER = re.compile(
    r"(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|"
    r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b)",
    re.I,
)
_CODE_TASK = re.compile(r"(?:코드|코딩|파이썬|오류|버그|프로그램|스크립트|정렬|함수|알고리즘|\b(?:python|javascript|debug|code|bug)\b)", re.I)
_LOCAL_PROMPT = re.compile(r"(?:^|[\s,])(?:안녕|반가워|고마워|잘가|잘\s*가)(?:[!?.~\s]|$)", re.I)


@dataclass(frozen=True)
class EvidenceDecision:
    search: bool
    memory: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"search": self.search, "memory": self.memory, "reason": self.reason}


def decide_evidence(
    message: str,
    *,
    ready_plan: bool = False,
    verified_local_evidence: bool = False,
) -> EvidenceDecision:
    """Conservative external search for unanswered *public factual* requests.

    No arbitrary personal statements, user secrets or personal memories go
    to the search provider. An unknown fact does NOT imply that an offline
    language model needs to be retrained.
    """
    text = " ".join(message.split())
    if not text:
        return EvidenceDecision(False, False, "empty")
    if _PAST_CHAT.search(text):
        return EvidenceDecision(False, True, "prior_conversation")
    if _NO_WEB.search(text):
        return EvidenceDecision(False, False, "user_disallowed_search")
    if (_PERSONAL.search(text) or _SENSITIVE_IDENTIFIER.search(text)
            or re.search(r"(?:내가|나는|내s*친구|우리s*(?:집|가족|학교)|"
                         r"엄마|아빠|할머니|선생님)", text)):
        return EvidenceDecision(False, False, "private_context")
    if ready_plan:
        return EvidenceDecision(False, False, "upstream_semantic_plan_ready")
    if verified_local_evidence:
        return EvidenceDecision(False, False, "verified_local_evidence")
    if _CODE_TASK.search(text) and not _BROWSE_REQUEST.search(text):
        return EvidenceDecision(False, False, "local_programming_request")
    if _LOCAL_PROMPT.fullmatch(text):
        return EvidenceDecision(False, False, "social")
    explicit = bool(_BROWSE_REQUEST.search(text))
    is_question = bool(_QUESTION.search(text))
    # Statements such as "오늘 야구에서 역전승했어!" are not search queries.
    if not (explicit or is_question):
        return EvidenceDecision(False, False, "personal_statement_or_nonquestion")
    # A bare instruction to summarize or rewrite supplied content is local.
    if re.search(r"(?:요약해\s*줘|번역해\s*줘|다듬어\s*줘|고쳐\s*줘)", text) and not explicit:
        return EvidenceDecision(False, False, "transform_local_content")
    if len(text) > 200:
        return EvidenceDecision(False, False, "query_too_long_or_private")
    return EvidenceDecision(True, False, "unanswered_public_question")


def no_gate_route(advisory: dict[str, Any]) -> dict[str, Any]:
    """Return a GENERAL execution context without mutating MaleCNS scores.

    'general' is only a legacy dispatch identifier, never an inferred
    categorical judgment. The raw predictions remain visible as *advisory*
    diagnostic data, not fabricated probabilities.
    """
    result = dict(advisory)
    result["advisory_route"] = {
        "route": advisory.get("route"),
        "confidence": advisory.get("confidence"),
        "margin": advisory.get("margin"),
        "accepted": advisory.get("accepted"),
    }
    result["route"] = "general"
    result["confidence"] = next(
        (float(row.get("confidence", 0.0))
         for row in (advisory.get("top_routes") or [])
         if isinstance(row, dict) and row.get("route") == "general"),
        0.0,
    )
    result["margin"] = 0.0
    result["accepted"] = True
    result["semantic_override"] = {
        "source": "hybrid_evidence_first",
        "model_route": advisory.get("route"),
        "model_confidence": advisory.get("confidence"),
        "model_accepted": advisory.get("accepted"),
    }
    # Deliberately do not replace a low general score with artificial 100%.
    return result
