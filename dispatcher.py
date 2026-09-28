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


def _math_handler(message: str) -> tuple[str, str]:
    text = _normalize_math(message)

    fraction_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:을|를)?\s*분수", text)
    if fraction_match:
        value = Fraction(fraction_match.group(1)).limit_denominator(1_000_000)
        return "math", f"🧮 {value.numerator}/{value.denominator}"

    root_patterns = (
        r"(?:루트|sqrt\s*\(?)\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*의\s*제곱근",
    )
    for pattern in root_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if value < 0:
                return "math", "🧮 실수 범위에서는 음수의 제곱근을 계산할 수 없습니다."
            return "math", f"🧮 {_format_number(math.sqrt(value))}"

    square_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:의\s*제곱|squared)",
        text,
        re.IGNORECASE,
    )
    if square_match:
        value = float(square_match.group(1))
        return "math", f"🧮 {_format_number(value ** 2)}"

    cube_match = re.search(r"(-?\d+(?:\.\d+)?)\s*의\s*세제곱", text)
    if cube_match:
        value = float(cube_match.group(1))
        return "math", f"🧮 {_format_number(value ** 3)}"

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
            return "math", f"🧮 x = {_format_number(value)}"
        except (SyntaxError, ValueError, ZeroDivisionError) as exc:
            return "math", f"🧮 이 방정식은 아직 계산하지 못했어요. ({exc})"

    candidates = re.findall(r"[0-9.+\-*/()%\s]+", text)
    candidates = [candidate.strip() for candidate in candidates if re.search(r"\d", candidate)]
    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        if not re.search(r"[+\-*/%]", candidate):
            continue
        try:
            value = _safe_eval(candidate)
            return "math", f"🧮 {_format_number(value)}"
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue

    return (
        "math",
        "🧮 Math handler까지 도착했어요. v0.3 계산기는 사칙연산, 제곱/세제곱, "
        "제곱근, 간단한 일차방정식을 우선 지원합니다.",
    )


def _code_handler(message: str) -> tuple[str, str]:
    lower = message.lower()

    if "파이썬" in message and ("리스트" in message or "list" in lower) and (
        "정렬" in message or "sort" in lower
    ):
        return (
            "code",
            "💻 Python 리스트 정렬\n\n"
            "새 리스트가 필요하면: sorted(values)\n"
            "원본 리스트를 직접 바꾸려면: values.sort()",
        )

    if ("javascript" in lower or "자바스크립트" in message) and (
        "길이" in message or "length" in lower
    ):
        return "code", "💻 JavaScript 배열 길이는 array.length로 구합니다."

    if ("javascript" in lower or "자바스크립트" in message) and "json" in lower:
        return "code", "💻 JSON 문자열 파싱: const data = JSON.parse(text);"

    if "git" in lower and ("브랜치" in message or "branch" in lower):
        return "code", "💻 새 브랜치 생성: git switch -c <branch-name>"

    if "sql" in lower and ("필터" in message or "where" in lower):
        return "code", "💻 SQL에서 행 필터링은 WHERE 절을 사용합니다."

    if "react" in lower and ("상태" in message or "state" in lower):
        return "code", "💻 React의 기본 상태 훅은 useState입니다."

    return (
        "code",
        "💻 Code handler까지 정상적으로 도착했습니다. "
        "아직 v0.3에는 범용 코드 생성 모델이 연결되지 않아, 현재는 등록된 코드 패턴만 직접 답합니다.",
    )


def _extract_summary_source(message: str) -> str:
    if "\n" in message:
        _, tail = message.split("\n", 1)
        if len(tail.strip()) >= 40:
            return tail.strip()

    if ":" in message:
        _, tail = message.split(":", 1)
        if len(tail.strip()) >= 40:
            return tail.strip()

    return ""


def _summarize_handler(message: str) -> tuple[str, str]:
    source = _extract_summary_source(message)
    if not source:
        return (
            "summarize",
            "📝 Summarize handler까지 왔어요. 요약할 본문을 다음 줄이나 ':' 뒤에 붙여주세요.",
        )

    pieces = [
        part.strip()
        for part in re.split(r"(?<=[.!?。])\s+|\n+", source)
        if part.strip()
    ]

    if not pieces:
        return "summarize", "📝 요약할 문장을 찾지 못했습니다."

    selected = pieces[: min(3, len(pieces))]
    summary = "\n".join(f"• {piece}" for piece in selected)
    return (
        "summarize",
        "📝 v0.3 기본 추출 요약\n\n"
        f"{summary}\n\n"
        "※ 아직 생성형 요약 모델이 아니라 앞부분의 핵심 문장을 추출하는 실험적 처리기입니다.",
    )


def _general_handler(message: str) -> tuple[str, str]:
    lower = message.lower()

    if "하늘" in message and ("파랗" in message or "파란" in message):
        return (
            "general",
            "🌤️ 하늘이 파랗게 보이는 주된 이유는 대기 분자가 짧은 파장의 빛을 더 강하게 산란시키는 레일리 산란 때문입니다.",
        )

    if "초파리" in message and ("연구" in message or "왜" in message):
        return (
            "general",
            "🪰 초파리는 세대가 짧고 번식이 빠르며, 유전학 자료가 풍부해서 생물학 연구에 널리 쓰입니다.",
        )

    if "ram" in lower:
        return (
            "general",
            "🧠 RAM은 실행 중인 프로그램과 데이터를 잠시 보관하는 빠른 작업 메모리입니다.",
        )

    if "얼음" in message and "물" in message and "떠" in message:
        return (
            "general",
            "🧊 얼음은 액체 물보다 밀도가 낮아서 물 위에 뜹니다.",
        )

    if "photosynthesis" in lower or "광합성" in message:
        return (
            "general",
            "🌱 광합성은 식물 등이 빛 에너지를 이용해 이산화탄소와 물로부터 화학 에너지를 저장하는 과정입니다.",
        )

    return (
        "general",
        "💬 General handler까지 정상적으로 도착했습니다. "
        "범용 답변 생성기는 아직 연결 전이라 v0.3에 등록된 기본 지식만 답할 수 있어요.",
    )


def _pending_handler(route: str) -> tuple[str, str]:
    if route == "research":
        return (
            "research",
            "🔎 Research route까지 도착했습니다. 아직 FlyGPT 서버에 웹 검색 엔진이 연결되지 않아 "
            "v0.3에서는 실제 검색을 실행하지 않습니다.",
        )

    return (
        "memory",
        "🧠 Memory route까지 도착했습니다. 아직 대화 기록 저장소가 연결되지 않아 "
        "v0.3에서는 이전 대화를 실제로 불러오지는 않습니다.",
    )


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

    # Avoid treating common ISO-like dates as subtraction.
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", normalized):
        return False

    if not re.fullmatch(r"[0-9.()+\-*/%]+", normalized):
        return False

    # A lone number is not enough. Require an actual arithmetic operator.
    return bool(re.search(r"[+*/%]|\*\*|(?<!^)-", normalized))


def dispatch_math_fast_path(message: str) -> DispatchResult:
    if not is_math_fast_path(message):
        raise ValueError("message is not eligible for math fast-path")

    handler_name, answer = _math_handler(message)
    return DispatchResult(
        route="math",
        handler="math_fast_path",
        status="completed",
        answer=answer,
        confidence=1.0,
        margin=1.0,
    )


def dispatch(message: str, route_info: dict[str, Any]) -> DispatchResult:
    route = str(route_info.get("route", "general"))
    confidence = float(route_info.get("confidence", 0.0))

    ranked = route_info.get("top_routes") or []
    second_confidence = (
        float(ranked[1].get("confidence", 0.0))
        if len(ranked) > 1
        else 0.0
    )
    margin = max(0.0, confidence - second_confidence)

    if confidence < MIN_CONFIDENCE or margin < MIN_MARGIN:
        top = ", ".join(
            f"{item.get('route', '?')} {float(item.get('confidence', 0.0)):.1%}"
            for item in ranked[:3]
        )
        return DispatchResult(
            route=route,
            handler="confidence_gate",
            status="uncertain",
            answer=(
                "🤔 FlyGPT가 요청 경로를 충분히 확신하지 못했어요. "
                "잘못된 처리기를 강제로 실행하지 않고 보류했습니다.\n\n"
                f"후보: {top}"
            ),
            confidence=confidence,
            margin=margin,
        )

    handlers: dict[str, Callable[[str], tuple[str, str]]] = {
        "math": _math_handler,
        "code": _code_handler,
        "summarize": _summarize_handler,
        "general": _general_handler,
    }

    if route in ("research", "memory"):
        handler_name, answer = _pending_handler(route)
        status = "pending"
    else:
        handler = handlers.get(route, _general_handler)
        handler_name, answer = handler(message)
        status = "completed"

    return DispatchResult(
        route=route,
        handler=handler_name,
        status=status,
        answer=answer,
        confidence=confidence,
        margin=margin,
    )
