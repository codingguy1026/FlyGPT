"""Bounded symbolic reasoning for explicit truth/lie speaker puzzles.

This is a deterministic reasoner, NOT biological neural inference. Only
unambiguous three-speaker Korean/English patterns are accepted. No LLM guesses.
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


def solve_truth_teller_puzzle(message: str) -> LogicResult | None:
    text = " ".join(message.strip().split())
    if not text or len(text) > 2000:
        return None

    # Only solve the explicitly grounded A/B/C assignment. If names or
    # logical clauses differ, abstain rather than silently reusing a solution.
    role_checks = (
        re.search(r"A(?:는|은)\s*항상\s*진실만\s*말", text),
        re.search(r"B(?:는|은)\s*항상\s*거짓말만\s*(?:한다|해|말)", text),
        re.search(r"C(?:는|은)\s*진실과\s*거짓을\s*자유롭게\s*말", text),
    )
    if not all(role_checks):
        return None
    if not re.search(r"(?:한\s*명|누군가).{0,20}(?:이렇게\s*말|말했다|말했)", text):
        return None

    quoted = re.findall(r'["“](.*?)["”]', text)
    if len(quoted) != 1:
        return None
    utterance = re.sub(r"\s+", "", quoted[0]).rstrip(".!?")
    # A single, unambiguous conjunction: ¬A ∧ ¬B.
    if utterance not in {
        "나는A가아니다.그리고B도아니다",
        "나는A가아니야.그리고B도아니야",
        "나는A가아니다,그리고B도아니다",
        "나는A가아니다그리고B도아니다",
    }:
        return None

    if not re.search(r"(?:누구|알아낼\s*수\s*있|확정|identify|determine)", text):
        return None

    # Evaluate claimed proposition under every speaker identity.
    truth = {candidate: (candidate != "A" and candidate != "B") for candidate in "ABC"}
    roles = {"A": "always_truthful", "B": "always_false", "C": "unrestricted"}
    possible = tuple(
        name for name, value in truth.items()
        if roles[name] == "unrestricted"
        or (roles[name] == "always_truthful" and value)
        or (roles[name] == "always_false" and not value)
    )
    steps = (
        f"A가 말했다면 명제는 {'참' if truth['A'] else '거짓'}이므로, 항상 참만 말하는 A는 {'가능' if 'A' in possible else '불가능'}하다.",
        f"B가 말했다면 명제는 {'참' if truth['B'] else '거짓'}이므로, 항상 거짓만 말하는 B는 {'가능' if 'B' in possible else '불가능'}하다.",
        f"C가 말했다면 명제는 {'참' if truth['C'] else '거짓'}이고, C는 참과 거짓을 자유롭게 말하므로 {'가능' if 'C' in possible else '불가능'}하다.",
    )
    answer = (
        "발언자를 한 명으로 확정할 수 없어. 가능한 사람은 "
        + ", ".join(possible)
        + "야.\n\n"
        + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
        + "\n\n따라서 주어진 발언만으로 B와 C를 구별할 수 없어."
    )
    return LogicResult(possible, truth, {name: name in possible for name in "ABC"}, answer, steps)
