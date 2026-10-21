"""Bounded forward-chaining inference over explicit propositional rules.

A deterministic rule engine, NOT connectome reasoning. It only parses precise
'A이면 B이다', 'A는 참이다', 'B는 참인가?' Korean statements. Unknown
premises are not false; unsupported text is rejected without partial proofs.
"""
from __future__ import annotations
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ChainResult:
    answer: str
    evidence: dict


def solve_chain(message: str) -> ChainResult | None:
    text = " ".join(message.strip().split())
    if not text or len(text) > 2000:
        return None
    pieces = [p.strip() for p in re.split(r"\.\s*", text) if p.strip()]
    if len(pieces) < 3 or len(pieces) > 16:
        return None
    query = re.fullmatch(r"([A-Z])(?:가|이|는|은)\\s*참(?:이라고\\s*확정할\\s*수\\s*있을까\\?\\s*이유를\\s*설명해\\s*줘\\??|인가\\??)", pieces[-1])
    if query is None:
        return None
    target = query.group(1)
    rules: list[tuple[tuple[str, ...], str]] = []\n    unknown: set[str] = set()
    facts: set[str] = set()
    for piece in pieces[:-1]:
        r = re.fullmatch(r"([A-Z])이면\\s*([A-Z])이다", piece)
        r_and = re.fullmatch(r"([A-Z])와\\s*([A-Z])가\\s*모두\\s*참이면\\s*([A-Z])가\\s*참이다", piece)
        u = re.fullmatch(r"([A-Z])는\\s*참인지\\s*모른다", piece)
        f = re.fullmatch(r"([A-Z])(?:는|은)\s*참이다", piece)
        if r:
            rules.append(((r.group(1),),r.group(2)))\n        elif r_and:\n            rules.append(((r_and.group(1),r_and.group(2)),r_and.group(3)))\n        elif u:\n            unknown.add(u.group(1))
        elif f:
            facts.add(f.group(1))
        else:
            return None
    if not rules or not facts or (unknown & facts) or len(rules) > 12:
        return None
    known = set(facts)
    # Capture first derivation, from a strictly smaller set of previous facts.
    steps = [{"fact": fact, "source": "given"} for fact in sorted(facts)]
    rounds = 0
    while True:
        rounds += 1
        changed = False
        for src, dest in rules:
            if src in known and dest not in known:
                known.add(dest)
                steps.append({"fact": dest, "source": "modus_ponens", "because": "와 ".join(src), "rule": f"{'∧'.join(src)}→{dest}"})
                changed = True
        if not changed:
            break
        if rounds > 26:
            return None
    proven = target in known
    if proven:
        answer = f"{target}는 참이라고 도출할 수 있어.\n" + "\n".join(
            f"{step['because']}가 참이고 {step['rule']}이므로 {step['fact']}가 참이다."
            for step in steps if step["source"] == "modus_ponens"
        )
    else:
        answer = f"주어진 조건만으로 {target}가 참이라고 결론 내릴 수 없어. 참이 아니라는 뜻은 아니야."
    return ChainResult(answer, {
        "engine": "forward_chain_v1",
        "initial_facts": sorted(facts),
        "rules": [{"if":list(a),"then":b} for a,b in rules],\n        "unknown_facts": sorted(unknown),
        "derived_facts": sorted(known),
        "target": target,
        "entailed": proven,
        "proof_steps": steps,
        "rounds": rounds,
        "is_connectome_inference": False,
    })
