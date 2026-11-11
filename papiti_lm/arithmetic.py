"""Explicit, opt-in calculator for simple arithmetic, NOT neural reasoning.

The standalone scratch Transformer is not a trustworthy arithmetic solver.
This module accepts only short, narrowly formatted numeric expressions.
It NEVER executes arbitrary Python or user-provided code.
"""
from __future__ import annotations

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext


# Only plain decimal arithmetic, with an optional short Korean question suffix.
# No identifiers, code, powers, function calls, or hidden evaluator.
_ARITHMETIC = re.compile(
    r"^\s*(?P<expr>[0-9+\-*/().\s]{3,64}?)\s*"
    r"(?:(?:은|는)\s*)?(?:(?:얼마(?:야|인가요|입니까)?|계산해\s*줘)\s*)?"
    r"(?:[?？]|=\s*[?？]?)?\s*$"
)
_MAX_ABS = Decimal("1000000000")


class _Unsupported(Exception):
    pass


def _compute(node: ast.AST, depth: int = 0) -> Decimal:
    if depth > 12:
        raise _Unsupported("expression too deeply nested")
    if isinstance(node, ast.Expression):
        return _compute(node.body, depth + 1)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        result = Decimal(str(node.value))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        term = _compute(node.operand, depth + 1)
        result = -term if isinstance(node.op, ast.USub) else term
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _compute(node.left, depth + 1)
        right = _compute(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        else:
            if not right:
                raise ZeroDivisionError
            result = left / right
    else:
        raise _Unsupported("unsupported operation")
    if not result.is_finite() or abs(result) > _MAX_ABS:
        raise _Unsupported("result exceeds limit")
    return result


def try_arithmetic(message: str) -> str | None:
    """Return an explicitly computed answer or None for non-math input.

    A matched but invalid arithmetic expression gets a clear error instead
    of silently handing the expression to a language model that may bluff.
    """
    match = _ARITHMETIC.fullmatch(message)
    if match is None:
        return None
    expr = match.group("expr").strip()
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "계산식을 다시 확인해 줘."
    # Do not treat a bare number or unary sign as a calculation request.
    if not any(isinstance(n, ast.BinOp) for n in ast.walk(parsed)):
        return None
    try:
        with localcontext() as ctx:
            ctx.prec = 24
            value = _compute(parsed)
            text = format(value.normalize(), "f")
    except ZeroDivisionError:
        return "0으로 나눌 수 없어."
    except (InvalidOperation, DivisionByZero, _Unsupported):
        return "이 계산식은 아직 지원하지 않아."
    return f"{text}야."
