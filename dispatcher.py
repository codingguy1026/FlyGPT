from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Callable


MIN_CONFIDENCE = 0.55
MIN_MARGIN = 0.10


@dataclass
class DispatchResult:
    route: str
    handler: str
    status: str
    answer: str
    confidence: float
    margin: float
    tool_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_number(value: float) -> float:
    if not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("계산 결과가 너무 큽니다.")
    return value


def _eval_ast(node: ast.AST, variables: dict[str, float] | None = None) -> float:
    variables = variables or {}

    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, variables)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _safe_number(float(node.value))

    if isinstance(node, ast.Name) and node.id in variables:
        return _safe_number(float(variables[node.id]))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)

        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("지수의 절댓값은 12 이하만 지원합니다.")

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("0으로 나눌 수 없습니다.")

        return _safe_number(float(_BINARY_OPS[type(node.op)](left, right)))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _safe_number(float(_UNARY_OPS[type(node.op)](_eval_ast(node.operand, variables))))

    raise ValueError("지원하지 않는 수식입니다.")


def _safe_eval(expression: str, variables: dict[str, float] | None = None) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_ast(tree, variables)


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-10):
        return str(int(round(value)))
    return f"{value:.10g}"


def _normalize_math(text: str) -> str:
    return (
        text.replace("×", "*")
        .replace("✕", "*")
        .replace("÷", "/")
        .replace("^", "**")
        .replace("−", "-")
    )


def _solve_linear_equation(left: str, right: str) -> float:
    def normalize_variable(expr: str) -> str:
        expr = expr.replace("X", "x")
        expr = re.sub(r"(?<=\d)x", "*x", expr)
        expr = re.sub(r"x(?=\d)", "x*", expr)
        expr = re.sub(r"(?<=\))x", "*x", expr)
        expr = re.sub(r"x(?=\()", "x*", expr)
        return expr

    left = normalize_variable(left)
    right = normalize_variable(right)

    def difference(x: float) -> float:
        return _safe_eval(left, {"x": x}) - _safe_eval(right, {"x": x})

    f0 = difference(0.0)
    f1 = difference(1.0)
    coefficient = f1 - f0

    if math.isclose(coefficient, 0.0, abs_tol=1e-12):
        raise ValueError("x의 일차방정식으로 풀 수 없습니다.")

    f2 = difference(2.0)
    if not math.isclose(f2, f0 + 2 * coefficient, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("현재는 일차방정식만 지원합니다.")

    return _safe_number(-f0 / coefficient)


def run_math_tool(message: str) -> str | None:
    """Return deterministic math evidence when the small exact tool can solve it."""
    text = _normalize_math(message)

    fraction_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:을|를)?\s*분수", text)
    if fraction_match:
        value = Fraction(fraction_match.group(1)).limit_denominator(1_000_000)
        return f"Exact math tool result: {value.numerator}/{value.denominator}"

    root_patterns = (
        r"(?:루트|sqrt\s*\(?)\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*의\s*제곱근",
    )
    for pattern in root_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if value < 0:
                return "Exact math tool result: no real square root for a negative input."
            return f"Exact math tool result: {_format_number(math.sqrt(value))}"

    square_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:의\s*제곱|squared)",
        text,
        re.IGNORECASE,
    )
    if square_match:
        value = float(square_match.group(1))
        return f"Exact math tool result: {_format_number(value ** 2)}"

    cube_match = re.search(r"(-?\d+(?:\.\d+)?)\s*의\s*세제곱", text)
    if cube_match:
        value = float(cube_match.group(1))
        return f"Exact math tool result: {_format_number(value ** 3)}"

    equation_match = re.search(
        r"([0-9xX+\-*/().\s*]+)=([0-9xX+\-*/().\s*]+)",
        text,
    )
    if equation_match and re.search(r"[xX]", equation_match.group(0)):
        try:
            value = _solve_linear_equation(
                equation_match.group(1).strip(),
                equation_match.group(2).strip(),
            )
            return f"Exact math tool result: x = {_format_number(value)}"
        except (SyntaxError, ValueError, ZeroDivisionError):
            return None

    candidates = re.findall(r"[0-9.+\-*/()%\s]+", text)
    candidates = [candidate.strip() for candidate in candidates if re.search(r"\d", candidate)]
    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        if not re.search(r"[+\-*/%]", candidate):
            continue
        try:
            value = _safe_eval(candidate)
            return f"Exact math tool result: {_format_number(value)}"
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue

    return None


def is_math_fast_path(message: str) -> bool:
    """Return True for short expressions that are obviously arithmetic."""
    raw = message.strip()
    raw = re.sub(
        r"\s*(?:은|는)?\s*(?:얼마(?:야|인가|지)?|몇(?:이야|인가)?)?\s*\?\s*$",
        "",
        raw,
    ).strip()
    raw = raw.rstrip("?").strip()

    if not raw or len(raw) > 120:
        return False

    normalized = _normalize_math(raw).replace(" ", "")

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", normalized):
        return False

    if not re.fullmatch(r"[0-9.()+\-*/%]+", normalized):
        return False

    return bool(re.search(r"[+*/%]|\*\*|(?<!^)-", normalized))


def dispatch_math_fast_path(message: str) -> DispatchResult:
    if not is_math_fast_path(message):
        raise ValueError("message is not eligible for math fast-path")

    tool_context = run_math_tool(message)
    if tool_context is None:
        raise ValueError("math fast-path tool could not solve expression")

    answer = tool_context.removeprefix("Exact math tool result: ").strip()
    return DispatchResult(
        route="math",
        handler="math_fast_path",
        status="completed",
        answer=f"🧮 {answer}",
        confidence=1.0,
        margin=1.0,
        tool_context=tool_context,
    )


def dispatch(message: str, route_info: dict[str, Any]) -> DispatchResult:
    """Gate the router and prepare tool context. It does not author canned answers."""
    route = str(route_info.get("route", "general"))
    confidence = float(route_info.get("confidence", 0.0))

    ranked = route_info.get("top_routes") or []
    second_confidence = (
        float(ranked[1].get("confidence", 0.0))
        if len(ranked) > 1
        else 0.0
    )
    margin = float(route_info.get("margin", max(0.0, confidence - second_confidence)))

    # v0.5 checkpoints carry thresholds selected on held-out validation data.
    # Older checkpoints do not, so the long-standing defaults remain the
    # compatibility fallback.
    min_confidence = max(
        0.0,
        min(float(route_info.get("min_confidence", MIN_CONFIDENCE)), 1.0),
    )
    min_margin = max(
        0.0,
        min(float(route_info.get("min_margin", MIN_MARGIN)), 1.0),
    )

    feature_signal = float(route_info.get("feature_signal", 1.0))
    accepted = route_info.get("accepted")
    if accepted is None:
        accepted = (
            feature_signal > 1e-8
            and confidence >= min_confidence
            and margin >= min_margin
        )

    if not bool(accepted):
        top = ", ".join(
            f"{item.get('route', '?')} {float(item.get('confidence', 0.0)):.1%}"
            for item in ranked[:3]
        )
        return DispatchResult(
            route=route,
            handler="confidence_gate",
            status="uncertain",
            answer=(
                "🤔 요청 의도를 충분히 확신하지 못했어요. "
                "조금 더 구체적으로 말해 주세요.\n\n"
                f"후보: {top}\n"
                f"게이트: confidence≥{min_confidence:.0%}, margin≥{min_margin:.0%}"
            ),
            confidence=confidence,
            margin=margin,
        )

    tool_context = run_math_tool(message) if route == "math" else None
    handler = {
        "math": "math_tool+generator" if tool_context else "generator",
        "memory": "memory_context+generator",
        "research": "research_context+generator",
    }.get(route, "generator")

    return DispatchResult(
        route=route,
        handler=handler,
        status="ready",
        answer="",
        confidence=confidence,
        margin=margin,
        tool_context=tool_context,
    )
