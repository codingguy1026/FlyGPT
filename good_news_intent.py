"""Local intent model for short Korean good-news announcements."""

from functools import lru_cache
import json
import math
from pathlib import Path
import re

_MODEL = Path(__file__).with_name("training") / "good_news_intent_weights.json"
_POSITIVE_CUE = re.compile(r"좋[은아았]|기[쁜뻐]|반가운|신나|합격|붙었|해냈|성공|달성|이겼")
_REQUEST = re.compile(
    r"[?？]|검색|찾아\s*(?:줘|봐|주|달라)|알려\s*(?:줘|주|달라)|"
    r"보여\s*(?:줘|주)|요약|추천|알아\s*봐|있(?:니|을까|나요)|(?:뭐|무슨|어떤)\s|"
    r"(?:해|써|줘|주세요|말해\s*봐|보고\s*싶어)[!.~\s]*$",
)


def features(text: str) -> set[str]:
    spaced = " ".join(re.findall(r"[가-힣]+", text))
    return {spaced[i:i + n] for n in (2, 3) for i in range(len(spaced) - n + 1)}


@lru_cache(maxsize=1)
def _weights() -> dict:
    return json.loads(_MODEL.read_text(encoding="utf-8"))


def is_good_news(text: str) -> bool:
    if len(text) > 90 or "\n" in text or not _POSITIVE_CUE.search(text) or _REQUEST.search(text):
        return False
    model = _weights()
    weights = model["weights"]
    terms = features(text)
    score = model["bias"] + math.fsum(weights.get(term, 0) for term in terms)
    return score >= model["threshold"]
