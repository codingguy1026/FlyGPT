"""FlyGPT reasoning v4: conservative exact algebra and finite logical entailment.

Deterministic tools, NOT MaleCNS neural inference. If parsing is ambiguous,
return None and make no claims. Quadratic coefficients use exact integers.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
import re


@dataclass(frozen=True)
class ReasoningV4Result:
    answer: str
    engine: str
    evidence: dict


def _quadratic(message: str) -> ReasoningV4Result | None:
    # Preserve problem framing but only accept a single exact equation.
    text = message.strip().replace("²", "^2").replace("−", "-").replace("＋", "+")
    if not re.search(r"(?:2차방정식|이차방정식|실근|근을|모든\s*근|방정식)", text):
        return None
    # Coefficients 0, 1, and -1 are implicit; permit any order of terms.
    matches = re.findall(r"(?<![\w^])(?:[+-]?\s*\d*\s*\*?\s*x\s*(?:\^\s*2)?|[+-]?\s*\d+)\s*(?:[+-]\s*(?:\d*\s*\*?\s*x\s*(?:\^\s*2)?|\d+))*\s*=\s*0(?!\d)",text,re.I)
    if len(matches) != 1:
        return None
    equation = matches[0]
    left = equation.split("=")[0].replace(" ","").replace("*","").lower()
    if not left or not re.fullmatch(r"[0-9x^+\-]+",left):
        return None
    terms=re.findall(r"[+-]?[^+-]+",left)
    a=b=c=0
    for term in terms:
        if "x^2" in term:
            m=re.fullmatch(r"([+-]?\d*)x\^2",term)
            if not m:return None
            raw=m.group(1);a+=int(raw) if raw not in ("","+","-") else (-1 if raw=="-" else 1)
        elif "x" in term:
            m=re.fullmatch(r"([+-]?\d*)x",term)
            if not m:return None
            raw=m.group(1);b+=int(raw) if raw not in ("","+","-") else (-1 if raw=="-" else 1)
        else:
            try:c+=int(term)
            except ValueError:return None
    if a==0 or max(map(abs,(a,b,c)))>100000:
        return None
    d=b*b-4*a*c
    evidence={"coefficients":{"a":a,"b":b,"c":c},"discriminant":d,"domain":"real","method":"quadratic_formula"}
    if d<0:
        roots=[]
        message_out=f"판별식 D = {d}로 음수이므로 실근이 없어."
    elif d==0:
        root=Fraction(-b,2*a)
        roots=[str(root)]
        message_out=f"판별식 D = 0이므로 중근 x = {root}이야."
    else:
        square=isqrt(d)
        if square*square==d:
            roots=sorted(set((Fraction(-b+square,2*a),Fraction(-b-square,2*a))))
            message_out="판별식 D = "+str(d)+"로 양수이므로 서로 다른 두 실근이 있어. "+" , ".join(f"x = {r}" for r in roots)+"."
            roots=[str(r) for r in roots]
        else:
            roots=[f"({-b} + √{d})/{2*a}",f"({-b} - √{d})/{2*a}"]
            message_out=f"판별식 D = {d}로 양수이므로 서로 다른 두 실근이 있어. 근의 공식에 따라 x = ({-b} ± √{d})/{2*a}야."
    evidence["roots"]=roots
    return ReasoningV4Result(message_out,"quadratic_exact_v1",evidence)


def _syllogism(message: str) -> ReasoningV4Result | None:
    # Controlled, auditable fragment of universally quantified implications
    # and existential assertions. Evaluate finite models, not canned answers.
    text=" ".join(message.strip().split())
    pattern=r"모든\s+([가-힣A-Za-z]+)(?:는|은)\s+([가-힣A-Za-z]+)(?:이다)"
    universal=re.findall(pattern,text)
    exists=re.findall(r"일부\s+([가-힣A-Za-z]+)(?:는|은)\s+([가-힣A-Za-z]+)(?:이다)",text)
    conclusion=re.search(r"모든\s+([가-힣A-Za-z]+)(?:가|이)\s+([가-힣A-Za-z]+)(?:라고|이다고)\s+결론",text)
    if not conclusion or not universal or len(universal)>3 or len(exists)>2:
        return None
    # This deliberately supports only exactly parsed premises.
    before=text[:conclusion.start()]
    sentences=[x.strip() for x in re.split(r"[.!?]\s*",before) if x.strip()]
    if len(sentences)!=len(universal)+len(exists):
        return None
    vocab=sorted(set(sum(([x,y] for x,y in universal+exists),[])+list(conclusion.groups())))
    if len(vocab)>4:return None
    pos={word:i for i,word in enumerate(vocab)}
    antecedent, consequent=conclusion.groups()
    # Enumerate all 2-individual finite interpretations for a countermodel.
    # For these affirmative monadic syllogisms, two individuals suffice:
    # one can witness the falsity of the universal conclusion.
    checked=0;counterexample=None
    from itertools import product
    for membership in product((False,True),repeat=2*len(vocab)):
        checked+=1
        individuals=[{w:membership[k*len(vocab)+pos[w]] for w in vocab} for k in range(2)]
        if not all(all(not person[x] or person[y] for person in individuals) for x,y in universal):continue
        if not all(any(person[x] and person[y] for person in individuals) for x,y in exists):continue
        if any(person[antecedent] and not person[consequent] for person in individuals):
            counterexample=individuals;break
    valid=counterexample is None
    answer=("주어진 전제에서 그 결론은 반드시 따라와." if valid
            else "주어진 전제만으로는 그 결론을 확정할 수 없어. 조건을 만족하면서 결론이 거짓인 반례가 존재해.")
    return ReasoningV4Result(answer,"finite_entailment_v1",{
        "entails":valid,"checked_models":checked,
        "counterexample":counterexample,"premises":{"all":universal,"some":exists},
        "conclusion":{"all":[antecedent,consequent]},
    })


def reason_v4(message: str) -> ReasoningV4Result | None:
    return _quadratic(message) or _syllogism(message)
