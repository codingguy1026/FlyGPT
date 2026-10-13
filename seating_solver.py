"""Small, deterministic solver for explicitly constrained linear seating puzzles.

This is NOT a general natural-language understanding engine. It recognizes a
strict subset of Korean ordering rules, and refuses to produce a verified
answer when any participant-subject constraint is unrecognized. That prevents
a partial parse from being mistaken for a proof.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass


_ORDINALS = {
    "첫": 1, "두": 2, "세": 3, "네": 4,
    "다섯": 5, "여섯": 6, "일곱": 7,
    "한": 1,
}

_PARTICIPANTS = re.compile(
    r"(?P<names>[A-Z](?:\s*,\s*[A-Z]){2,6})\s+"
    r"(?P<count>[3-7])\s*명(?:이|은|은|가|은)?\s*일렬로\s*서"
)

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fixed", re.compile(
        r"\b([A-Z])(?:가|이|는|은)\s+왼쪽에서\s*"
        r"(첫|두|세|네|다섯|여섯|일곱|[1-7])\s*번째"
    )),
    ("direct", re.compile(
        r"\b([A-Z])(?:는|은|가|이)\s+([A-Z])의\s+"
        r"바로\s*(오른쪽|왼쪽)"
    )),
    ("not_end", re.compile(
        r"\b([A-Z])(?:는|은|가|이)\s+양\s*끝(?:에)?\s*"
        r"서지\s*않"
    )),
    ("neighbor", re.compile(
        r"\b([A-Z])(?:는|은|가|이)\s+([A-Z])와\s+"
        r"(?:서로\s*)?이웃"
    )),
    ("distance", re.compile(
        r"\b([A-Z])(?:는|은|가|이)\s+([A-Z])와\s+"
        r"(한|두|세|네|다섯|여섯|[1-7])\s*칸\s*떨어져"
    )),
)

_SUBJECT = re.compile(r"\b([A-Z])(?:는|은|가|이)\s+")


@dataclass(frozen=True)
class Rule:
    kind: str
    first: str
    second: str | None
    distance: int | None
    explanation: str

    def satisfied(self, seating: tuple[str, ...]) -> bool:
        positions = {name: index for index, name in enumerate(seating)}
        a = positions[self.first]
        b = positions[self.second] if self.second else None

        if self.kind == "fixed":
            return a == (self.distance or 1) - 1
        if self.kind == "direct":
            assert b is not None
            return a == b + (self.distance or 0)
        if self.kind == "not_end":
            return 0 < a < len(seating) - 1
        if self.kind == "neighbor":
            assert b is not None
            return abs(a - b) == 1
        if self.kind == "distance":
            assert b is not None
            return abs(a - b) == self.distance
        raise ValueError("unknown rule")


@dataclass(frozen=True)
class SeatingResult:
    answer: str
    participants: tuple[str, ...]
    possible_count: int
    checked_count: int
    rule_count: int


def solve_linear_seating(message: str) -> SeatingResult | None:
    """Return a proved solution, or None when parsing is incomplete.

    Position rules are applied to *every* permutation. No language-model
    guess is ever used as the verdict.
    """
    introduction = _PARTICIPANTS.search(message)
    if introduction is None:
        return None

    participants = tuple(re.findall(r"[A-Z]", introduction["names"]))
    if (
        len(participants) != int(introduction["count"])
        or len(set(participants)) != len(participants)
    ):
        return None

    rules: list[Rule] = []
    matches: list[tuple[int, int]] = []

    for kind, pattern in _RULES:
        for match in pattern.finditer(message):
            groups = match.groups()
            first = groups[0]
            second: str | None = None
            distance: int | None = None

            if first not in participants:
                return None
            if kind == "fixed":
                distance = _ORDINALS.get(groups[1], int(groups[1]) if groups[1].isdigit() else 0)
                explanation = f"{first}는 왼쪽에서 {distance}번째"
            elif kind == "direct":
                second = groups[1]
                distance = 1 if groups[2] == "오른쪽" else -1
                explanation = f"{first}는 {second}의 바로 {groups[2]}"
            elif kind == "not_end":
                explanation = f"{first}는 양 끝이 아님"
            elif kind == "neighbor":
                second = groups[1]
                explanation = f"{first}와 {second}는 이웃"
            else:
                second = groups[1]
                distance = _ORDINALS.get(groups[2], int(groups[2]) if groups[2].isdigit() else 0)
                explanation = f"{first}와 {second}의 자리 차이는 {distance}칸"

            if second is not None and (second not in participants or second == first):
                return None
            if distance is not None and (distance == 0 or abs(distance) >= len(participants)):
                # A direction is +/-1; long/fixed distances beyond the line
                # would have zero candidates, but avoid "verified" bad parsing.
                if not (kind == "fixed" and distance == len(participants)):
                    return None
            rules.append(Rule(kind, first, second, distance, explanation))
            matches.append(match.span())

    if len(rules) < 3:
        return None

    # Every participant-subject clause must have been recognized. Otherwise
    # a missing constraint could make an invalid arrangement look valid.
    for subject in _SUBJECT.finditer(message, introduction.end()):
        if subject.group(1) in participants and not any(
            start <= subject.start() < end for start, end in matches
        ):
            return None

    # Present fixed positions first so pruning can be followed by a reader.
    order = {"fixed": 0, "direct": 1, "not_end": 2, "neighbor": 3, "distance": 4}
    rules.sort(key=lambda item: order[item.kind])
    total = len(list(itertools.permutations(participants)))
    candidates = list(itertools.permutations(participants))
    steps = [f"총 {total}가지 순서를 모두 검사합니다."]

    for index, rule in enumerate(rules, 1):
        candidates = [seating for seating in candidates if rule.satisfied(seating)]
        steps.append(
            f"{index}. {rule.explanation}: 남은 배치 {len(candidates)}가지"
        )

    if len(candidates) == 0:
        verdict = (
            "결론: 모든 조건을 동시에 만족하는 배치는 없습니다. "
            "따라서 주어진 조건은 서로 모순됩니다."
        )
    elif len(candidates) == 1:
        verdict = "결론: 유일한 배치는 " + " → ".join(candidates[0]) + " 입니다."
    else:
        shown = [" → ".join(seating) for seating in candidates[:8]]
        verdict = (
            f"결론: 가능한 배치가 {len(candidates)}개라서 하나로 확정할 수 없습니다. "
            "예: " + "; ".join(shown)
        )

    return SeatingResult(
        answer="\n".join(steps) + "\n" + verdict,
        participants=participants,
        possible_count=len(candidates),
        checked_count=total,
        rule_count=len(rules),
    )
