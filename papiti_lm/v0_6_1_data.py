"""v0.6.1 semantic-packet corpus. Labels are training-only; no canned replies at runtime.

The MaleCNS/Python brain remains the authority for what to communicate.
The Transformer learns to verbalize a bounded packet instead of deciding
new facts or retrieving answers. Evaluation packets are excluded from train.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from papiti_lm.corpus import Example


def packet_text(plan: dict[str, Any]) -> str:
    """Compact, auditable brain-to-mouth contract, not a user-question lookup."""
    if (not isinstance(plan, dict) or plan.get("contract") != "mouth_only_v1"
            or plan.get("ready") is not True):
        raise ValueError("v0.6.1 requires a completed upstream mouth-only plan")
    style = plan.get("style") or {}
    units = plan.get("content_units")
    if (not isinstance(style, dict) or not isinstance(units, list)
            or not 1 <= len(units) <= 5):
        raise ValueError("invalid semantic packet")
    content = []
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get("kind"), str):
            raise ValueError("invalid content unit")
        value = unit.get("value")
        if not isinstance(value, str) or not value.strip() or len(value) > 480:
            raise ValueError("invalid unit meaning")
        content.append({"kind": unit["kind"], "value": value.strip()})
    packet = {
        "act": str(plan.get("speech_act") or "")[:80],
        "lang": str(style.get("language") or "ko")[:16],
        "tone": str(style.get("tone") or "natural")[:32],
        "content": content,
    }
    result = json.dumps(packet, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(result) > 2100:
        raise ValueError("semantic packet exceeds experimental context")
    return result


@dataclass(frozen=True)
class SpeechCorpus:
    train: list[Example]
    valid: list[Example]
    test: list[Example]


def _sample_plan(message: str) -> dict[str, Any] | None:
    from brain_runtime import FlyBrainRuntime
    brain = FlyBrainRuntime()
    state = brain.plan(
        route_info={"route": "general", "accepted": True, "confidence": 1.0, "margin": 1.0},
        dispatch={"route": "general"}, message=message,
    )
    return brain.finalize(state).get("utterance_plan")


def _math_plan(result: int) -> dict[str, Any]:
    """This is a supervised example of a value chosen by the brain, not a calculator."""
    return {
        "contract": "mouth_only_v1", "ready": True,
        "speech_act": "state_exact_math_result",
        "content_units": [{"kind": "exact_result", "value": f"Exact math tool result: {result}"}],
        "style": {"language": "ko", "tone": "natural"},
    }


def create_corpus(seed: int = 42) -> SpeechCorpus:
    """Training examples use original human prose and verified upstream content.

    An input packet is never present across different splits. The numeric
    exact-result holdout tests whether the model *copies supplied evidence*,
    not whether it independently calculates arithmetic.
    """
    from semantic_planner import trusted_fact_fallback
    authored = (
        ("안녕", "안녕! 반가워."),
        ("안녕하세요", "안녕하세요!"),
        ("고마워", "천만에!"),
        ("감사합니다", "별말씀을요."),
        ("잘 가", "잘 가!"),
        ("안녕히 가세요", "안녕히 가세요."),
        ("반가워", "반가워!"),
        ("도움이 필요해", "어떤 도움이 필요해?"),
        ("너 누구야?", "파피티에 대해 알려 줄게."),
        ("Hi", "Hello!"),
        ("Thanks", "You're welcome!"),
        ("Bye", "Goodbye!"),
    )
    originals: dict[str, Example] = {}
    for prompt, answer in authored:
        plan = _sample_plan(prompt)
        if not plan or not plan.get("ready"):
            continue
        key = packet_text(plan)
        originals.setdefault(key, Example("social:" + plan["speech_act"], key, answer))
    # Project facts are selected and grounded by Python, not fabricated by a
    # language model. Train on their approved exact meanings, then require
    # verbatim preservation in inference for fact-bearing plans.
    for prompt in ("너 누구야?", "파피티 이름은?", "파피티 뇌가 뭐야?",
                   "뭐 할 수 있어?", "MaleCNS가 뭐야?", "What is MaleCNS?"):
        plan = _sample_plan(prompt)
        if plan and plan.get("ready") and plan.get("semantic_steps"):
            answer = trusted_fact_fallback(plan)
            if answer:
                key = packet_text(plan)
                originals[key] = Example("facts:" + plan["speech_act"], key, answer)
    numeric = [Example("verified-number", packet_text(_math_plan(value)), str(value))
               for value in range(200)]
    rng = random.Random(seed)
    rng.shuffle(numeric)
    # Distinct value packets; no label leakage from held-out examples.
    valid, test, train_numbers = numeric[:20], numeric[20:40], numeric[40:]
    social = list(originals.values())
    # Held-out stylistic variations of known meanings test whether the model
    # preserves content rather than matching a complete training packet.
    varied = []
    for row in social[:min(4, len(social))]:
        payload = json.loads(row.user)
        payload["tone"] = "careful_unseen"  # NOT present in training.
        varied.append(Example(row.group + ":style", json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ), row.assistant))
    valid.extend(varied[:2])
    test.extend(varied[2:])
    if len(social) < 5:
        raise ValueError("brain did not produce enough independent speech plans")
    train = social * 12 + train_numbers
    rng.shuffle(train)
    data = SpeechCorpus(train, valid, test)
    train_prompts = {row.user for row in data.train}
    for other in (data.valid, data.test):
        if train_prompts & {row.user for row in other}:
            raise AssertionError("cross-split semantic packet leakage")
    if {row.user for row in data.valid} & {row.user for row in data.test}:
        raise AssertionError("valid/test semantic packet leakage")
    return data
