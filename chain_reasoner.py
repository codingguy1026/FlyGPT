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


def solve_chain(message: str, *, neural_route: dict | None = None) -> ChainResult | None:
    text = " ".join(message.strip().split())
    if not text or len(text) > 2000:
        return None
    pieces = [p.strip() for p in re.split(r"\.\s*", text) if p.strip()]
    if len(pieces) < 3 or len(pieces) > 16:
        return None
    query = re.fullmatch(r"([A-Z])(?:가|이|는|은)\s*참(?:이라고\s*확정할\s*수\s*있을까\?\s*이유를\s*설명해\s*줘\??|인가\??)", pieces[-1])
    if query is None:
        return None
    target = query.group(1)
    rules: list[tuple[tuple[str, ...], str]] = []
    unknown: set[str] = set()
    facts: set[str] = set()
    for piece in pieces[:-1]:
        r = re.fullmatch(r"([A-Z])이면\s*([A-Z])이다", piece)
        r_and = re.fullmatch(r"([A-Z])와\s*([A-Z])가\s*모두\s*참이면\s*([A-Z])가\s*참이다", piece)
        u = re.fullmatch(r"([A-Z])는\s*참인지\s*모른다", piece)
        f = re.fullmatch(r"([A-Z])(?:는|은)\s*참이다", piece)
        if r:
            rules.append(((r.group(1),),r.group(2)))
        elif r_and:
            rules.append(((r_and.group(1),r_and.group(2)),r_and.group(3)))
        elif u:
            unknown.add(u.group(1))
        elif f:
            facts.add(f.group(1))
        else:
            return None
    if not rules or not facts or (unknown & facts) or len(rules) > 12:
        return None
    ranked = (neural_route or {}).get("top_routes") or []
    scores = {r.get("route"): r.get("confidence", 0.0) for r in ranked if isinstance(r, dict)}
    neural_ready = bool(neural_route and neural_route.get("accepted") is True and scores)
    conjunction_first = neural_ready and scores.get("math", 0.0) > scores.get("general", 0.0)
    indexed_rules = list(enumerate(rules))
    indexed_rules.sort(key=lambda item: ((-1 if conjunction_first else 1) * len(item[1][0]), item[0]))
    selection = "neural_conjunction_first" if conjunction_first else ("neural_single_first" if neural_ready else "baseline")
    # Goal-directed dependency closure: rules unrelated to the queried atom
    # cannot help prove it. This is a deterministic optimization, independent
    # of the neural router, and applies equally to both ablation arms.
    needed = {target}
    while True:
        before = len(needed)
        for _, (premises, conclusion) in indexed_rules:
            if conclusion in needed:
                needed.update(premises)
        if len(needed) == before:
            break
    relevant_rules = [(i, rule) for i, rule in indexed_rules if rule[1] in needed]
    pruned_count = len(indexed_rules) - len(relevant_rules)
    known = set(facts)
    # Capture first derivation, from a strictly smaller set of previous facts.
    steps = [{"fact": fact, "source": "given"} for fact in sorted(facts)]
    rounds = 0
    rule_checks = 0
    while True:
        rounds += 1
        changed = False
        for rule_index, (src, dest) in relevant_rules:
            rule_checks += 1
            if all(value in known for value in src) and dest not in known:
                known.add(dest)
                steps.append({"fact": dest, "source": "modus_ponens", "rule_index": rule_index, "because": "와 ".join(src), "rule": f"{'∧'.join(src)}→{dest}"})
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
        "engine": "forward_chain_v2_neural_scheduler",
        "neural_selection": {
            "mode": selection,
            "accepted": neural_ready,
            "model": (neural_route or {}).get("model") if neural_ready else None,
            "rule_order": [i for i, _ in indexed_rules],
            "scope": "schedule_only_not_learned_logic",
        },
        "initial_facts": sorted(facts),
        "rules": [{"if":list(a),"then":b} for a,b in rules],
        "unknown_facts": sorted(unknown),
        "derived_facts": sorted(known),
        "target": target,
        "entailed": proven,
        "proof_steps": steps,
        "rounds": rounds,
        "rule_checks": rule_checks,
        "pruned_rules": pruned_count,
        "relevant_rules": len(relevant_rules),
        "is_connectome_inference": False,
    })
