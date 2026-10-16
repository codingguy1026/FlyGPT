"""Conservative rescue for simple, low-confidence GENERAL definitions/why questions.

The classifier remains authoritative about uncertainty. A separate cautious
generator may try a basic evergreen *explanation* without claiming the router
had high confidence or bypassing research/memory/math/code policy.
"""
from __future__ import annotations

import re
from typing import Any

# Disallow fresh/external evidence, personal records, executable requests and
# higher-stakes domains. For everything else we aren't sure of: fail closed.
_DISALLOWED = re.compile(
    r"(?:오늘|어제|내일|이번|현재|지금|최근|최신|실시간|올해|"
    r"뉴스|날씨|환율|주가|가격|검색|웹|인터넷|사이트|링크|"
    r"기억|아까|방금|예전에|저번|전에\s*말|우리\s*대화|"
    r"코드|코딩|파이썬|javascript|python|sql|react|fastapi|"
    r"계산|몇\s*(?:이야|이지|개|살)|\d+|[+*/=^]|"
    r"만들어|작성해|해줘|추천|가르쳐|사용법|방법|절차|"
    r"약|질병|치료|증상|진단|의학|자해|자살|폭탄|무기|"
    r"선거|투표|정당|법률|대출|투자|계좌|인증|비밀번호|"
    r"today|now|latest|current|recent|price|stock|weather|"
    r"search|remember|previous|before|code|write|"
    r"how to|instructions|steps|medical|health|vote|politic|law)"
    , re.IGNORECASE,
)

# Definition and WHY questions only, never generic imperatives or instructions.
_SIMPLE_QUESTION = re.compile(
    r"(?:"
    r"왜\s*.+(?:\?|？|까|나요|어|지)$|"
    r".+(?:뭐야|뭔데|무슨\s*뜻|뜻이\s*뭐|의미가\s*뭐|무엇인가요|"
    r"무엇인지|이유가\s*뭐|왜\s*그런|설명해\s*줄래)(?:\?|？|\.)?$|"
    r"(?:what is|what are|why (?:is|are|does|do)|what does .+ mean)\b.+\??$"
    r")",
    re.IGNORECASE,
)


def eligible_basic_general(
    message: str, route_info: dict[str, Any],
) -> bool:
    """Grant an *attempt*, not a factual correctness or model-acceptance badge."""
    if str(route_info.get("route")) != "general":
        return False
    if route_info.get("accepted") is not False:
        return False
    if float(route_info.get("feature_signal", 1.0) or 0.0) <= 1e-8:
        return False
    if float(route_info.get("confidence", 0.0) or 0.0) < 0.35:
        return False
    msg = " ".join(message.split())
    if not 5 <= len(msg) <= 100:
        return False
    if _DISALLOWED.search(msg):
        return False
    return bool(_SIMPLE_QUESTION.fullmatch(msg))
