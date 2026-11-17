"""Compose a short answer from relevant sentences in retrieved web sources."""
from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from urllib.parse import urlsplit

from research_runtime import ResearchResult


_KOREAN = re.compile(r"[가-힣]")
_WORD = re.compile(r"[가-힣]+|[a-zA-Z0-9]+")
_PARTICLE = re.compile(r"(?:에서|에게|으로|부터|까지|처럼|[은는이가을를의에도와과로])$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")
_STOP_WORDS = frozenset("""
    a an and are can could did do does for how i in is it of on or please
    the to was were what when where which who why will with would you
    왜 언제 어디 어디서 누가 누구 어떻게 무엇 뭐 몇 때문 이유 알려 알려줘 해줘 설명해
    어떤 무슨 알려주세요 설명해주세요 좀 있어 있어요 했어 했나요 시작됐어 시작됐나요
""".split())
_CAUSE = re.compile(
    r"때문|원인|계기로|요청으로|(?:피해|요청)로|막기 위해|"
    r"\b(?:because|due to|led to|caused by|in response to|in order to)\b", re.I,
)
_TIME = re.compile(
    r"\b\d{3,4}\b|\d\s*[년월일]|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.I,
)
_CASUAL_ENDINGS = (
    ("했습니다.", "했어."),
    ("하였습니다.", "했어."),
    ("되었습니다.", "됐어."),
    ("됐습니다.", "됐어."),
    ("있었습니다.", "있었어."),
)


@dataclass(frozen=True)
class AnswerSource:
    url: str
    title: str


@dataclass(frozen=True)
class _EvidenceSentence:
    text: str
    terms: set[str]
    source: AnswerSource
    score: float


def _terms(text: str) -> set[str]:
    terms = set()
    for word in _WORD.findall(text.lower()):
        if word in _STOP_WORDS:
            continue
        if len(word) > 2 and _KOREAN.match(word):
            word = _PARTICLE.sub("", word, count=1)
        if word and word not in _STOP_WORDS:
            terms.add(word)
    return terms


def _clean(text: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]*>", "", text)).split())


def _sentences(snippet: str) -> list[str]:
    # Split before collapsing whitespace so separate lines stay separate.
    return [
        sentence
        for piece in _SENTENCE_BREAK.split(snippet)
        if 14 <= len(sentence := _clean(piece)) <= 240
        and sentence.endswith((".", "。"))
        and not sentence.endswith(("...", "…"))
    ]


def compose_web_answer(
    question: str, research: ResearchResult | None,
) -> tuple[str, list[AnswerSource]] | None:
    """Select up to two on-topic statements and preserve their attribution."""
    query_terms = _terms(question)
    if not research or not research.used or not query_terms:
        return None

    korean = bool(_KOREAN.search(question))
    why = bool(re.search(r"왜|이유|\bwhy\b", question, re.I))
    when = bool(re.search(r"언제|몇\s*년|\bwhen\b", question, re.I))
    candidates: list[_EvidenceSentence] = []
    for index, source in enumerate(research.sources):
        try:
            url = urlsplit(source.url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username is not None:
                continue
        except ValueError:
            continue
        citation = AnswerSource(source.url, _clean(source.title) or url.hostname)
        title_matches = query_terms & _terms(citation.title)
        for snippet in source.snippets:
            for sentence in _sentences(snippet):
                terms = _terms(sentence)
                matches = query_terms & terms
                if not (matches or title_matches == query_terms):
                    continue
                score = 3 * len(matches) + 0.5 * len(title_matches - matches)
                score += 2 * (why and bool(_CAUSE.search(sentence)))
                score += 2 * (when and bool(_TIME.search(sentence)))
                score += 0.5 * (korean == bool(_KOREAN.search(sentence)))
                score -= 0.05 * index
                candidates.append(_EvidenceSentence(sentence, terms, citation, score))

    # A matching title can identify a follow-up sentence, but cannot establish
    # relevance when none of the retrieved sentences itself matches the question.
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    primary = next((candidate for candidate in ranked if candidate.terms & query_terms), None)
    if primary is None:
        return None
    selected = [primary]
    for candidate in ranked:
        common = primary.terms & candidate.terms
        if len(common) / max(1, min(len(primary.terms), len(candidate.terms))) >= 0.72:
            continue
        relevant = bool(candidate.terms & query_terms)
        if why:
            relevant = (relevant or candidate.source == primary.source) and bool(_CAUSE.search(candidate.text))
        if relevant and not when:
            selected.append(candidate)
            break

    citations: list[AnswerSource] = []
    lines = []
    for sentence in selected:
        if sentence.source not in citations:
            citations.append(sentence.source)
        number = citations.index(sentence.source) + 1
        text = sentence.text
        if korean:
            for formal, casual in _CASUAL_ENDINGS:
                if text.endswith(formal):
                    text = text[:-len(formal)] + casual
                    break
        lines.append(f"{text} [{number}]")
    return " ".join(lines), citations
