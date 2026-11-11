"""Offline audit of LLM-free reasoning, without making neural-reasoning claims.

Run from the repository root:
    python -m scripts.audit_native_reasoning

This exercises bounded production solvers and the semantic planner without
starting FastAPI, downloading a checkpoint, or contacting an LLM. These are
fixed regression probes, NOT a model accuracy benchmark. MaleCNS graph
propagation is deliberately out of scope; the result must not be presented as
proof that the connectome learned these solutions.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

from brain_runtime import FlyBrainRuntime
from chain_reasoner import solve_chain
from dispatcher import dispatch_math_fast_path, is_math_fast_path
from reasoning_v4 import reason_v4
from seating_solver import solve_linear_seating
from symbolic_reasoner import solve_truth_teller_puzzle
from verified_realizer import realize_verified


SPEAKER_PUZZLE = (
    "어느 섬에 세 사람이 살고 있다. A는 항상 진실만 말하고, "
    "B는 항상 거짓말만 한다. C는 진실과 거짓을 자유롭게 말한다. "
    '세 사람 중 한 명이 이렇게 말했다. "나는 A가 아니다. 그리고 B도 아니다." '
    "이 말을 한 사람이 누구인지 반드시 알아낼 수 있을까? "
    "가능하다면 누구인지, 불가능하다면 이유를 설명해 줘."
)
SEATING_PUZZLE = (
    "A, B, C 3명이 일렬로 서 있습니다. "
    "B가 왼쪽에서 첫 번째에 있습니다. "
    "A는 B의 바로 오른쪽에 있고 "
    "C가 왼쪽에서 세 번째에 있습니다. 정확한 순서는?"
)


class ForbiddenGenerator:
    """An assertion trap to ensure verified output never silently uses a language generator."""

    model = "disabled-in-audit"
    configured = True
    local_only = False

    def __init__(self) -> None:
        self.calls = 0

    def _generate_once(self, **_kwargs):
        self.calls += 1
        raise AssertionError("The offline audit must not invoke a generator")


def _general_plan(text: str) -> dict:
    return FlyBrainRuntime().plan(
        route_info={"route": "general", "confidence": 0.95,
                    "margin": 0.4, "accepted": True, "trace": []},
        dispatch={"route": "general", "status": "ready"},
        message=text,
    )["utterance_plan"]


def run_audit() -> dict:
    generator = ForbiddenGenerator()
    cases: list[dict] = []

    def record(name: str, category: str, passed: bool, answer: str = "") -> None:
        cases.append({
            "name": name,
            "category": category,
            "passed": bool(passed),
            "origin": "deterministic_solver" if category == "proof"
                      else "semantic_planner" if category == "semantic_only"
                      else "unsupported_input_guard",
            "is_connectome_inference": False,
            "generator_used": False,
            "answer": answer if category == "proof" else None,
        })

    def verified(name: str, result, condition: bool, answer: str) -> None:
        # Even if a developer enables FLYGPT_VERIFIED_REALIZER_MODE=always,
        # the audit must keep the generator completely disabled.
        with patch.dict(os.environ, {"FLYGPT_VERIFIED_REALIZER_MODE": "off"}):
            text, meta = realize_verified(answer, generator)
        record(
            name, "proof",
            result is not None and condition and text == answer
            and not meta["attempted"] and meta["reason"] == "realizer_disabled",
            text,
        )

    math = dispatch_math_fast_path("17*23")
    verified("exact_arithmetic", math, math.status == "completed" and "391" in math.answer, math.answer)

    quad = reason_v4("다음 2차방정식의 모든 실근을 구해. x² - 5x + 6 = 0")
    verified("quadratic_roots", quad, quad is not None and quad.evidence["roots"] == ["2", "3"],
             quad.answer if quad else "")

    changed = reason_v4("다음 2차방정식의 모든 실근을 구해. x² - 7x + 10 = 0")
    verified("changed_coefficients_change_roots", changed,
             changed is not None and changed.evidence["roots"] == ["2", "5"],
             changed.answer if changed else "")

    chain_true = solve_chain("A이면 B이다. B이면 C이다. A는 참이다. C는 참인가?")
    verified("forward_chain_entails", chain_true,
             chain_true is not None and chain_true.evidence["entailed"] is True,
             chain_true.answer if chain_true else "")

    chain_unknown = solve_chain("A이면 B이다. B이면 C이다. D는 참이다. C는 참인가?")
    verified("missing_premise_does_not_entail", chain_unknown,
             chain_unknown is not None and chain_unknown.evidence["entailed"] is False,
             chain_unknown.answer if chain_unknown else "")

    syllogism = reason_v4(
        "모든 루미는 파랑이다. 일부 파랑은 동그라미이다. "
        "모든 루미가 동그라미라고 결론 내릴 수 있을까?"
    )
    verified("countermodel_rejects_conclusion", syllogism,
             syllogism is not None and syllogism.evidence.get("entails") is False
             and syllogism.evidence.get("counterexample") is not None,
             syllogism.answer if syllogism else "")

    speakers = solve_truth_teller_puzzle(SPEAKER_PUZZLE)
    verified("speaker_cases", speakers,
             speakers is not None and speakers.possible_speakers == ("B", "C"),
             speakers.explanation if speakers else "")

    seating = solve_linear_seating(SEATING_PUZZLE)
    verified("unique_seating", seating,
             seating is not None and seating.possible_count == 1
             and "B → A → C" in seating.answer,
             seating.answer if seating else "")

    unknown = "왜 하늘은 파래?"
    # Not having a bounded proof is a *correct abstention*, not a solved case.
    no_solver = (
        not is_math_fast_path(unknown)
        and solve_chain(unknown) is None
        and reason_v4(unknown) is None
        and solve_truth_teller_puzzle(unknown) is None
        and solve_linear_seating(unknown) is None
    )
    record("open_ended_not_proven", "abstention",
           no_solver and _general_plan(unknown)["ready"] is False)

    greeting = _general_plan("안녕")
    record("greeting_is_only_a_semantic_plan", "semantic_only",
           greeting["ready"] is True and greeting["speech_act"] == "return_greeting"
           and greeting["permissions"]["infer_new_meaning"] is False)

    passed = all(case["passed"] for case in cases) and generator.calls == 0
    return {
        "audit": "native_reasoning_v1",
        "status": "PASS" if passed else "FAIL",
        "kind": "offline_fixed_regression_probes",
        "llm_available": False,
        "generator_calls": generator.calls,
        "malecns_checkpoint_loaded": False,
        "neural_reasoning_demonstrated": False,
        "proof_cases": sum(case["category"] == "proof" for case in cases),
        "proof_cases_passed": sum(case["category"] == "proof" and case["passed"] for case in cases),
        "cases": cases,
        "limitations": [
            "Deterministic solvers, not the MaleCNS neural graph, derive the audited proofs.",
            "Semantic greeting plans do not count as solved reasoning problems.",
            "Fixed regression probes are not representative accuracy estimates.",
            "This is module-level testing; the live API and real router checkpoint are not measured.",
        ],
    }


def main() -> int:
    report = run_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
