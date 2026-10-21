"""Auditable bounded integer constraint reasoning, with exhaustive assignments.

Supports explicit equality, offsets and inequalities for two named variables.
Never infer unstated ranges: finite search is permitted only with an explicit
domain bound in the user's question. This is NOT connectome neural reasoning.
"""
from __future__ import annotations
import itertools
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintResult:
    answer: str
    solutions: tuple[dict[str, int], ...]
    checked_count: int
    constraints: tuple[str, ...]


def solve_integer_constraints(message: str) -> ConstraintResult | None:
    text = " ".join(message.strip().split())
    if not text or len(text) > 1200:
        return None
    # A known, finite search universe is a mathematical requirement,
    # not a convenience that should be hidden from the user.
    match = re.search(r"([A-Z])와\s*([A-Z])는\s*(-?\d+)부터\s*(-?\d+)까지의\s*정수(?:다|이다)?", text)
    if not match:
        return None
    left, right, lo, hi = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
    if left == right or hi < lo or hi - lo > 60 or lo < -100 or hi > 100:
        return None
    names = (left, right)
    domain = match.group(0)
    fragments = [part.strip() for part in re.split(r"[.。]\s*", text) if part.strip()]
    if not fragments or fragments[0] != domain:
        return None
    query = fragments[-1]
    q = re.fullmatch(rf"([{left}{right}])(?:의\s*값은|는)\s*(?:얼마|무엇)(?:인가|일까|인가요|야|입니까)?[?？]?", query)
    if not q:
        return None
    target = q.group(1)
    rules: list[tuple[str,str,str,int]] = []
    for frag in fragments[1:-1]:
        # Exact equality of either named variable to a constant.
        eq = re.fullmatch(rf"([{left}{right}])는\s*(-?\d+)(?:이다|이야)", frag)
        if eq:
            rules.append(("eq", eq.group(1), "", int(eq.group(2))))
            continue
        offset = re.fullmatch(rf"([{left}{right}])는\s*([{left}{right}])보다\s*(\d+)\s*(크다|작다)", frag)
        if offset and offset.group(1) != offset.group(2):
            k = int(offset.group(3))
            rules.append(("offset", offset.group(1), offset.group(2), k if offset.group(4) == "크다" else -k))
            continue
        compare = re.fullmatch(rf"([{left}{right}])는\s*([{left}{right}])보다\s*(크다|작다)", frag)
        if compare and compare.group(1) != compare.group(2):
            rules.append(("gt" if compare.group(3) == "크다" else "lt", compare.group(1), compare.group(2), 0))
            continue
        # Reject every unknown fragment. No partial-problem solutions.
        return None
    if not rules or len(rules) > 8:
        return None

    def valid(env: dict[str,int]) -> bool:
        for op, a, b, v in rules:
            if op == "eq" and env[a] != v: return False
            if op == "offset" and env[a] != env[b] + v: return False
            if op == "gt" and env[a] <= env[b]: return False
            if op == "lt" and env[a] >= env[b]: return False
        return True

    solutions = tuple(
        dict(zip(names, pair))
        for pair in itertools.product(range(lo, hi + 1), repeat=2)
        if valid(dict(zip(names, pair)))
    )
    if len(solutions) > 200:
        return None
    values = sorted({row[target] for row in solutions})
    if not solutions:
        intro = "모든 조건을 동시에 만족하는 정수 조합이 없어."
    elif len(values) == 1:
        intro = f"{target}의 값은 {values[0]}이야."
    else:
        intro = f"{target}의 값은 하나로 확정할 수 없어. 가능한 값은 {', '.join(map(str,values))}야."
    summary = "검증된 조합: " + (
        ", ".join(f"({left}={s[left]}, {right}={s[right]})" for s in solutions[:12])
        if solutions else "없음"
    )
    if len(solutions) > 12:
        summary += f" 외 {len(solutions)-12}개"
    return ConstraintResult(
        answer=f"{intro}\n\n{summary}\n{(hi-lo+1)**2}개 조합을 검사했어.",
        solutions=solutions,
        checked_count=(hi-lo+1)**2,
        constraints=tuple(fragments[1:-1]),
    )
