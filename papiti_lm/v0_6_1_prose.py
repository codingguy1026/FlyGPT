"""Shared training and decoding for novel wording across supported reply plans."""
from __future__ import annotations

from collections import defaultdict
import json
import random
from collections.abc import Callable
from typing import Any

from papiti_lm.corpus import Example
from papiti_lm.v0_6_1_data import SpeechCorpus, _sample_plan, create_corpus, packet_text

STYLES = range(4)
HELD_OUT_PROSE = "prose-combination"


def prose_packet(packet: str, wording: int, continuation: int) -> str:
    return f"{packet}\nProse {wording}/{continuation}"


# These continuations are supervised text, used only when creating the corpus.
# The live worker never receives these strings.
_FOLLOWUPS = {
    "return_greeting": (
        ("어서 와.", "편하게 이야기해.", "오늘은 뭐가 궁금해?", "무슨 얘기부터 할까?"),
        ("Welcome!", "Make yourself at home.", "What's on your mind?", "Where shall we start?"),
    ),
    "return_welcome": (
        ("편하게 얘기하자.", "어서 와.", "같이 이야기해 보자.", "뭐부터 이야기할까?"),
        ("Welcome!", "Let's chat.", "What shall we talk about?", "Make yourself at home."),
    ),
    "acknowledge_thanks": (
        ("언제든 편하게 물어봐.", "도움이 됐다니 기뻐.", "다른 궁금한 점도 말해 줘.", "필요하면 또 같이 살펴보자."),
        ("Ask anytime.", "I'm glad it was useful.", "Let me know if you need anything else.", "We can look at it together anytime."),
    ),
    "return_farewell": (
        ("좋은 하루 보내.", "또 이야기하자.", "편히 쉬어.", "다음에 만나."),
        ("Have a good day.", "Talk to you later.", "See you next time.", "Hope the rest of your day goes well."),
    ),
    "request_specific_task": (
        ("아는 것부터 말해 줘.", "같이 방법을 찾아보자.", "어디부터 막혔는지 알려 줘.", "간단하게 설명해 줘도 돼."),
        ("Tell me what you know so far.", "We can work it out together.", "Show me where you're stuck.", "A quick description is enough."),
    ),
    "acknowledge_presence": (
        ("편하게 이야기해 줘.", "뭐부터 얘기할까?", "하고 싶은 얘기부터 해.", "궁금한 게 있으면 말해 줘."),
        ("Tell me what's on your mind.", "Where would you like to start?", "Go ahead whenever you're ready.", "Ask me anything you're curious about."),
    ),
    "acknowledge_good_news": (
        ("무슨 일이 있었어?", "어떤 소식인데?", "어떻게 된 건지 궁금해.", "좀 더 듣고 싶어."),
        ("What happened?", "I'd love to hear more.", "How did it happen?", "Tell me the good news!"),
    ),
    "acknowledge_tiredness": (
        ("오늘은 좀 쉬어도 좋겠다.", "잠깐 숨 돌리는 건 어때?", "어떤 일이 있었어?", "지금은 좀 쉴 수 있어?"),
        ("I hope you get a chance to rest.", "Can you take a short break?", "What made today so tiring?", "Are you able to rest now?"),
    ),
    "respond_to_laughter": (
        ("뭐가 그렇게 웃겼어?", "웃긴 얘기 더 있어?", "어떤 부분에서 터졌어?", "나도 그 얘기 듣고 싶어."),
        ("What made you laugh?", "Got another funny story?", "Which part got you?", "I'd like to hear the story too."),
    ),
}
_FACT_FOLLOWUPS = (
    ("궁금한 부분이 있으면 더 설명할게.", "어느 부분을 더 살펴볼까?", "관련 기능도 물어봐 줘.", "원하면 자세히 풀어서 말해 줄게."),
    ("I can explain any part in more detail.", "Which part would you like to explore?", "Feel free to ask about related features.", "I can go into more detail if you'd like."),
)


def _packet_details(packet: str) -> tuple[str, str]:
    if packet.startswith("Social reply ("):
        language = packet.removeprefix("Social reply (").split(",", 1)[0]
        return packet.rsplit(": ", 1)[1], language
    data = json.loads(packet)
    return data["act"], data["lang"]


def create_prose_corpus(seed: int = 42) -> SpeechCorpus:
    from hybrid_speech import reply_key
    from semantic_planner import trusted_fact_fallback

    base = create_corpus(seed, diverse_social=True, phrasing_cues=True, everyday_dialogue=True)
    groups: dict[tuple[str, str], dict[str, None]] = defaultdict(dict)
    for row in base.train:
        if (row.group.startswith(("social:", "facts:")) and not row.user.startswith("Delivery:")
                and not (row.user.startswith("Social reply (") and ", natural):" not in row.user)):
            groups[row.group, row.user].setdefault(row.assistant)

    for prompt, answers in (
        ("I need help", ("What do you need help with?", "How can I help?", "What's the problem?", "What are you working on?")),
        ("Nice to meet you", ("Nice to meet you too!", "Glad to meet you!", "Likewise!", "It's good to meet you.")),
    ):
        plan = _sample_plan(prompt)
        groups["social:" + plan["speech_act"], packet_text(plan)].update(dict.fromkeys(answers))
    for prompt in ("Who are you?", "What can you do?"):
        plan = _sample_plan(prompt)
        facts = trusted_fact_fallback(plan)
        groups["facts:" + plan["speech_act"], packet_text(plan)].update(dict.fromkeys(
            lead + facts for lead in ("", "Here's the short version. ", "Let me explain. ", "Let's start with the basics. ")
        ))

    result = SpeechCorpus([], [], [])
    for (group, packet), replies in groups.items():
        act, language = _packet_details(packet)
        followups = (_FACT_FOLLOWUPS if group.startswith("facts:") else _FOLLOWUPS[act])[language == "en"]
        for wording, first in enumerate(replies):
            first = first if first.endswith((".", "!", "?")) else first + "."
            for continuation, second in enumerate(followups):
                if first.casefold() == second.casefold():
                    continue
                row = Example(HELD_OUT_PROSE, prose_packet(packet, wording, continuation), f"{first}\n{second}")
                offset = (continuation - wording) % len(STYLES)
                if offset == 1:
                    result.valid.append(row)
                elif offset == 2:
                    result.test.append(row)
                else:
                    result.train.extend([row] * 6)
    random.Random(seed).shuffle(result.train)
    # Alternate punctuation and spacing can make two slots realize the same
    # wording. Keep those duplicates out of the evaluation splits.
    seen = {reply_key(row.assistant) for row in result.train}
    held_out = []
    for split in (result.valid, result.test):
        unseen = []
        for row in split:
            key = reply_key(row.assistant)
            if key not in seen:
                unseen.append(row)
                seen.add(key)
        held_out.append(unseen)
    return SpeechCorpus(result.train, held_out[0], held_out[1])


def prose_fingerprints(corpus: SpeechCorpus) -> dict[str, Any]:
    from hybrid_speech import reply_key

    by_plan: dict[str, tuple[set[str], set[str]]] = defaultdict(lambda: (set(), set()))
    for row in corpus.train:
        packet, _ = row.user.rsplit("\n", 1)
        first, second = row.assistant.split("\n", 1)
        first_keys, second_keys = by_plan[packet]
        first_keys.add(reply_key(first))
        second_keys.add(reply_key(second))
    return {
        "replies": sorted({reply_key(row.assistant) for row in corpus.train}),
        "parts": {packet: [sorted(first), sorted(second)] for packet, (first, second) in by_plan.items()},
    }


def decode_novel_prose(
    plan: dict[str, Any], decode: Callable[[str, float], str],
    fingerprints: dict[str, Any], previous_keys: tuple[str, ...],
) -> str:
    from hybrid_speech import check_candidate, reply_key

    packet = packet_text(plan)
    parts = fingerprints["parts"].get(packet)
    if not parts:
        return ""
    first_keys, second_keys = map(set, parts)
    blocked = (*fingerprints["replies"], *previous_keys)
    combinations = [(wording, continuation) for wording in STYLES for continuation in STYLES]
    random.shuffle(combinations)
    attempts = [(*pair, 0.0 if index % 2 == 0 else 0.65)
                for index, pair in enumerate(combinations)]
    attempts.extend((*pair, 0.0) for pair in combinations[1::2])
    generated_first: dict[str, str] = {}
    generated_second: dict[str, str] = {}
    for wording, continuation, temperature in attempts:
        candidate = decode(prose_packet(packet, wording, continuation), temperature)
        first, separator, second = candidate.partition("\n")
        if not separator:
            continue
        first_key, second_key = reply_key(first), reply_key(second)
        if first_key in first_keys:
            generated_first[first_key] = first.strip()
        if second_key in second_keys:
            generated_second[second_key] = second.strip()
        answer = f"{first.strip()} {second.strip()}"
        if (first_key in generated_first and second_key in generated_second
                and check_candidate(plan, answer, avoid_reply_keys=blocked) is None):
            return answer
        # Use only model-generated text from this request. Checking pairs
        # here avoids exhaustively decoding the remaining cues on long plans.
        for generated_head in generated_first.values():
            for generated_tail in generated_second.values():
                combined = f"{generated_head} {generated_tail}"
                if check_candidate(plan, combined, avoid_reply_keys=blocked) is None:
                    return combined
    return ""
