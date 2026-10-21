"""Bounded, auditable three-speaker truth-table reasoning.

Extracts explicit role premises and simple first-person propositions from Korean
text. This is deterministic symbolic inference, not learned MaleCNS reasoning.
Unsupported language is intentionally left unanswered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LogicResult:
    possible_speakers: tuple[str, ...]
    truth_values: dict[str, bool]
    speaker_cases: dict[str, bool]
    explanation: str
    proof_steps: tuple[str, ...]


def _roles(text: str) -> dict[str, str] | None:
    result: dict[str, str] = {}
    for name in "ABC":
        start = re.search(rf"(?<![A-Za-z]){name}[는은]\s*", text)
        if start is None:
            return None
        # Look only at this speaker's own clause, not other speakers' roles.
        rest = text[start.end():]
        bound = re.search(r"(?:[.!?]|[ABC][는은]|(?:이고|이며|하며)\s*[ABC][는은])", rest)
        clause = rest[:bound.start()] if bound else rest[:70]
        if re.search(r"(?:항상\s*)?(?:진실|참)(?:만)?\s*(?:을\s*)?말", clause):
            role = "truth"
        elif re.search(r"(?:항상\s*)?거짓말(?:만)?\s*(?:을\s*)?(?:하|말)|(?:항상\s*)?거짓(?:만)?\s*(?:을\s*)?말", clause):
            role = "lie"
        elif re.search(r"(?:진실과\s*거짓|참과\s*거짓).{0,12}자유롭게\s*말|자유롭게\s*(?:말|참과\s*거짓)", clause):
            role = "free"
        else:
            return None
        result[name] = role
    return result if sorted(result.values()) == ["free", "lie", "truth"] else None


def _clauses(utterance: str) -> tuple[tuple[str, bool], ...] | None:
    compact = re.sub(r"\s+", "", utterance.strip()).strip(".!?")
    compact = compact.replace("그리고", "|").replace("또한", "|")
    parts = [p.strip(".,") for p in compact.split("|")]
    if not 1 <= len(parts) <= 3:
        return None
    claims: list[tuple[str, bool]] = []
    for part in parts:
        m = re.fullmatch(r"(?:나는|난)([ABC])(?:가|이)(아니다|아니야|아니에요|맞다|맞아|이다|야)", part)
        if not m:
            # 'B도 아니다' is a natural continuation of '나는 A가 아니다'.
            m2 = re.fullmatch(r"([ABC])도(아니다|아니야|아니에요)", part)
            if not m2 or not claims:
                return None
            claims.append((m2.group(1), False))
        else:
            claims.append((m.group(1), m.group(2) in ("맞다", "맞아", "이다", "야")))
    return tuple(claims)


def solve_truth_teller_puzzle(message: str) -> LogicResult | None:
    text = " ".join(message.strip().split())
    if not text or len(text) > 2000:
        return None
    roles = _roles(text)
    if roles is None:
        return None
    if not re.search(r"(?:누군가|한\s*명|세\s*사람\s*중\s*한\s*명).{0,25}(?:말했|말했다|이렇게\s*말|라고\s*말)", text):
        return None
    quoted = re.findall(r'["“‘\']([^"”’\']+)["”’\']', text)
    if len(quoted) != 1:
        return None
    claims = _clauses(quoted[0])
    if claims is None:
        return None
    if not re.search(r"(?:누구|발언자|알아낼\s*수\s*있|확정|가능|identify|determine)", text):
        return None
    truth_values = {
        candidate: all((candidate == name) == positive for name, positive in claims)
        for candidate in "ABC"
    }
    possible = tuple(
        name for name in "ABC"
        if roles[name] == "free"
        or (roles[name] == "truth" and truth_values[name])
        or (roles[name] == "lie" and not truth_values[name])
    )
    labels = {"truth": "항상 참을 말하는", "lie": "항상 거짓을 말하는", "free": "참과 거짓을 자유롭게 말하는"}
    steps = tuple(
        f"{name}가 발언했다면 문장은 {'참' if truth_values[name] else '거짓'}이다. "
        f"{labels[roles[name]]} {name}는 {'가능' if name in possible else '불가능'}하다."
        for name in "ABC"
    )
    if len(possible) == 1:
        intro = f"발언자는 {possible[0]}로 확정할 수 있어."
    elif possible:
        intro = f"발언자를 한 명으로 확정할 수 없어. 가능한 사람은 {', '.join(possible)}야."
    else:
        intro = "모든 조건을 만족하는 발언자는 없어. 주어진 조건들이 모순돼."
    explanation = intro + "\n\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
    return LogicResult(possible, truth_values, {name: name in possible for name in "ABC"}, explanation, steps)
