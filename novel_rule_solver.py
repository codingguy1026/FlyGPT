"""Execute *explicitly defined*, bounded arithmetic rules instead of echoing them.

This is a narrow deterministic interpreter, not general language reasoning.
Return None on ambiguity. Never run user expressions with eval/exec.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import re
from typing import Any

# Untrusted user text only supplies an arithmetic operation and a bounded value.
_DEFINITION = re.compile(
    r"(?P<name>[가-힣A-Za-z]{2,12})['’\"]?\s*(?:은|는)\s*"
    r"숫자에\s*(?P<amount>-?\d{1,8}(?:\.\d{1,4})?)\s*(?:을|를)\s*"
    r"(?P<verb>더하는|빼는|곱하는|나누는)\s*연산"
)
_START = re.compile(r"(?<![\d.])(?P<value>-?\d{1,8}(?:\.\d{1,4})?)\s*에\s*")
_CALL = re.compile(
    r"(?P<name>[가-힣A-Za-z]{2,12})(?:을|를)\s*"
    r"(?:(?:먼저|나중에)\s*)?적용"
)
_OP = {"더하는": "+", "빼는": "-", "곱하는": "×", "나누는": "÷"}
_MAX_ABS = Decimal("1000000000000")


@dataclass(frozen=True)
class NovelRuleResult:
    answer: str
    start: str
    result: str
    steps: tuple[dict[str, str], ...]


def _display(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _definitions(text: str) -> dict[str, tuple[str, Decimal]] | None:
    rules: dict[str, tuple[str, Decimal]] = {}
    for match in _DEFINITION.finditer(text[:1200]):
        name = match.group("name")
        definition = (_OP[match.group("verb")], Decimal(match.group("amount")))
        if name in rules and rules[name] != definition:
            return None  # conflicting definitions must not silently overwrite
        rules[name] = definition
        if len(rules) > 8:
            return None
    return rules


def solve_defined_operations(
    message: str, prior_messages: list[dict[str, Any]] | None = None,
) -> NovelRuleResult | None:
    """Evaluate ordered 'A를 적용한 다음 B를 적용하면?' questions.

    Definitions may come from a recent *user* message for a follow-up; no
    assistant text, external retrieved document or model guess can define rules.
    All parsing is conservative. Unsupported grammar falls through to the
    normal pipeline rather than producing a made-up numeric answer.
    """
    text = message.strip()
    if not text or len(text) > 1200 or "적용" not in text:
        return None
    if not re.search(r"하면|결과|얼마|값|계산|구해", text):
        return None

    rules = _definitions(text)
    if rules is None:
        return None
    if not rules:
        for row in reversed((prior_messages or [])[-10:]):
            if row.get("role") != "user":
                continue
            prior_text = str(row.get("content") or "")
            if len(prior_text) > 1200:
                continue
            rules = _definitions(prior_text)
            if rules is None:
                return None
            if rules:
                break
    if not rules:
        return None

    # The last explicit numeric 'N에' before operation instructions is the
    # input value. Avoid reading constants from earlier rule definitions.
    starts = list(_START.finditer(text))
    if len(starts) != 1:
        return None
    start = starts[0]
    expression = text[start.end():]
    calls = list(_CALL.finditer(expression))
    if not 1 <= len(calls) <= 8 or any(m.group("name") not in rules for m in calls):
        return None
    # Only a clear sequence of named operation applications is eligible.
    # Unknown named operations must not be silently ignored.
    if len(re.findall(r"적용", expression)) != len(calls):
        return None

    try:
        with localcontext() as ctx:
            ctx.prec = 28
            current = Decimal(start.group("value"))
            step_rows: list[dict[str, str]] = []
            for match in calls:
                name = match.group("name")
                operator, operand = rules[name]
                before = _display(current)
                if operator == "+":
                    current += operand
                elif operator == "-":
                    current -= operand
                elif operator == "×":
                    current *= operand
                elif operator == "÷":
                    if not operand:
                        return NovelRuleResult(
                            answer=f"'{name}' 규칙은 0으로 나누게 되어 계산할 수 없어.",
                            start=_display(Decimal(start.group("value"))),
                            result="undefined",
                            steps=tuple(step_rows),
                        )
                    current /= operand
                if not current.is_finite() or abs(current) > _MAX_ABS:
                    return None
                step_rows.append({
                    "name": name,
                    "before": before,
                    "operator": operator,
                    "operand": _display(operand),
                    "after": _display(current),
                })
    except (InvalidOperation, OverflowError, ValueError):
        return None

    rendered = [
        f"{index}. {step['name']}: {step['before']} {step['operator']} "
        f"{step['operand']} = {step['after']}"
        for index, step in enumerate(step_rows, 1)
    ]
    answer = "주어진 규칙을 순서대로 적용하면:\n" + "\n".join(rendered)
    answer += f"\n\n**정답: {_display(current)}**"
    return NovelRuleResult(
        answer=answer,
        start=_display(Decimal(start.group("value"))),
        result=_display(current),
        steps=tuple(step_rows),
    )
