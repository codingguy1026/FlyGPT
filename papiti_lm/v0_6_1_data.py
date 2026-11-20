"""v0.6.1 semantic-packet corpus. Labels are training-only; no canned replies at runtime.

The MaleCNS/Python brain remains the authority for what to communicate.
The Transformer learns to verbalize a bounded packet instead of deciding
new facts or retrieving answers. Evaluation packets are excluded from train.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any

from papiti_lm.corpus import ASSISTANT, BOS, EOS, USER, Example


EVERYDAY_SPEECH_ACTS = frozenset({
    "acknowledge_presence", "acknowledge_good_news", "acknowledge_tiredness", "respond_to_laughter",
})
SOCIAL_SPEECH_ACTS = frozenset({
    "return_greeting", "acknowledge_thanks", "return_farewell",
    "return_welcome", "request_specific_task",
}) | EVERYDAY_SPEECH_ACTS
MIRROR_GREETING_PREFIX = "Mirror greeting: "
DELIVERY_STYLES = ("direct", "brief", "warm", "casual")
GREETING_CUES = tuple((opening, followup) for opening in range(5) for followup in range(5))


def delivery_packet(packet: str, style: str) -> str:
    """A style cue changes how to speak, without changing the facts to speak."""
    return f"Delivery: {style}\n{packet}"


def greeting_packet(packet: str, opening: int, followup: int) -> str:
    return f"Greeting opening {opening}\nGreeting follow-up {followup}\n{packet}"


def is_korean_greeting(plan: dict[str, Any]) -> bool:
    return (
        plan.get("speech_act") == "return_greeting"
        and (plan.get("style") or {}).get("language", "ko") == "ko"
    )


def greeting_parts(text: str) -> tuple[str, str] | None:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else None


def greeting_fingerprints(corpus: SpeechCorpus) -> dict[str, list[str]]:
    from hybrid_speech import reply_key

    openings, followups = set(), set()
    for row in corpus.train:
        if row.group == "compositional-greeting":
            parts = greeting_parts(row.assistant)
            if parts is None:
                raise ValueError("incomplete greeting training example")
            opening, followup = parts
            openings.add(reply_key(opening))
            followups.add(reply_key(followup))
    return {
        "replies": sorted({reply_key(row.assistant) for row in corpus.train}),
        "openings": sorted(openings),
        "followups": sorted(followups),
    }


def packet_text(plan: dict[str, Any], *, mirror_greetings: bool = False) -> str:
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
    if packet["act"] in EVERYDAY_SPEECH_ACTS:
        return f"Social reply ({packet['lang']}, {packet['tone']}): {packet['act']}"
    if mirror_greetings and plan.get("speech_act") == "return_greeting":
        greeting = plan.get("greeting")
        if not isinstance(greeting, str) or not 1 <= len(greeting) <= 96:
            raise ValueError("invalid source greeting")
        return MIRROR_GREETING_PREFIX + greeting
    return result


def greeting_sequence(tokenizer, packet: str, expected: str | None = None):
    """Align source characters with supervised output positions."""
    prefix = [BOS, USER, *tokenizer.encode(MIRROR_GREETING_PREFIX)]
    greeting = list(packet.removeprefix(MIRROR_GREETING_PREFIX))
    source = prefix + [tokenizer.char_to_id[ch] for ch in greeting] + [ASSISTANT]
    if expected is None:
        return source, len(prefix)
    targets = [tokenizer.char_to_id[ch] for ch in expected]
    return source, [-100] * len(prefix) + targets + [EOS]


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


def _greeting_examples(seed: int) -> SpeechCorpus:
    """Teach one copy relation across input forms, with disjoint surface text."""
    korean = ("안녕", "안녕하세요", "안녕하세용", "안녕하세여", "ㅎㅇ", "하이")
    english = ("hi", "hello", "hey")
    stems = (*korean, *(variant for word in english
                        for variant in (word, word.title(), word.upper(), word.swapcase())))
    endings = ("", "!", "?", ".", "~", "!!", "??", "..", "~~", "!?", "?!", "!!!", "...", "~~~",
               "?!?!", "!?!", "!~", "~!", "!?!!", " !", " ?", " ~")
    # Reserve unfamiliar punctuation patterns across every greeting form.
    held_out = {"!?": "test", "?!?!": "test", "!?!!": "test", "!?!": "valid",
                "~!": "valid", " ?": "valid"}
    rows: dict[str, list[Example]] = {"train": [], "valid": [], "test": []}
    for stem in dict.fromkeys(stems):
        for ending in endings:
            greeting = stem + ending
            plan = _sample_plan(greeting)
            assert plan and plan.get("speech_act") == "return_greeting"
            rows[held_out.get(ending, "train")].append(Example(
                "mirror-greeting", packet_text(plan, mirror_greetings=True), greeting,
            ))
    random.Random(seed).shuffle(rows["train"])
    return SpeechCorpus(**rows)


def create_corpus(
    seed: int = 42, *, diverse_social: bool = False, mirror_greetings: bool = False,
    everyday_dialogue: bool = False, phrasing_cues: bool = False, compositional_greetings: bool = False,
) -> SpeechCorpus:
    """Training examples use original human prose and verified upstream content.

    An input packet is never present across different splits. The numeric
    exact-result holdout tests whether the model *copies supplied evidence*,
    not whether it independently calculates arithmetic.
    """
    from semantic_planner import trusted_fact_fallback
    authored = (
        ("안녕", ("안녕! 반가워.", "안녕하세요!", "안녕! 만나서 반가워.", "반가워! 안녕.")),
        ("고마워", ("천만에!", "별말씀을요.", "도움이 됐다니 다행이야.", "별거 아니야!")),
        ("잘 가", ("잘 가!", "안녕히 가세요.", "다음에 또 봐!", "잘 가, 또 보자.")),
        ("반가워", ("반가워!", "나도 반가워.", "만나서 반가워!", "나도 만나서 반가워.")),
        ("도움이 필요해", ("어떤 도움이 필요해?", "뭘 도와줄까?", "어떤 일인지 말해줄래?", "어떤 걸 같이 해볼까?")),
        ("Hi", ("Hello!", "Hi there!", "Hey!", "Hi, nice to meet you!")),
        ("Thanks", ("You're welcome!", "No problem!", "Glad I could help.", "Anytime!")),
        ("Bye", ("Goodbye!", "See you!", "Take care!", "Bye, see you later!")),
    )
    if not diverse_social:
        # Keep packet order and training reproduction for the original weights.
        authored = authored[:5] + (("너 누구야?", ("파피티에 대해 알려 줄게.",)),) + authored[5:]
    if everyday_dialogue:
        authored += (
            ("잘 지내?", ("응, 이야기할 준비 됐어.", "응, 여기 있어. 편하게 얘기해.", "응! 얘기하자.", "응, 대화할 수 있어.")),
            ("좋은 일이 있었어!", ("오, 좋은 소식이네!", "오! 무슨 좋은 일이 있었어?", "와, 듣기 좋다!", "좋은 일이라니 반갑다!")),
            ("오늘 너무 피곤해", ("많이 피곤했겠다.", "오늘 많이 지쳤나 봐.", "아이고, 정말 피곤했겠다.", "오늘 꽤 힘들었나 보다.")),
            ("ㅋㅋㅋㅋ", ("ㅋㅋㅋㅋ", "ㅋㅋㅋ", "ㅎㅎㅎㅎ", "ㅋㅋ!")),
            ("How are you?", ("I'm here and ready to chat.", "Ready to talk!", "I'm here. We can chat.", "I'm ready when you are.")),
            ("I have good news!", ("Oh, that's great!", "Good news? Tell me!", "That's lovely to hear!", "Oh, what happened?")),
            ("I'm tired", ("Sounds like you're worn out.", "That sounds tiring.", "You sound really tired.", "Oh, that sounds exhausting.")),
            ("lol", ("Haha!", "Ha ha!", "LOL!", "Hahaha!")),
        )
    originals: dict[str, list[Example]] = {}
    for prompt, answers in authored:
        plan = _sample_plan(prompt)
        if not plan or not plan.get("ready"):
            continue
        if mirror_greetings and plan["speech_act"] == "return_greeting":
            continue
        key = packet_text(plan)
        targets = answers if diverse_social else answers[:1]
        originals[key] = [Example("social:" + plan["speech_act"], key, answer)
                          for answer in targets]
    # The model can learn different ways to present the same facts. These are
    # supervised examples only; inference still checks each exact fact unit.
    fact_leads = {
        "ko": ("", "간단히 말하면, ", "먼저 핵심부터 말할게. ", "조금 풀어 말해 볼게. "),
        "en": ("", "Here's a quick explanation. ", "Let me explain. ", "Here's the short version. "),
    }
    for prompt in ("너 누구야?", "파피티 이름은?", "파피티 뇌가 뭐야?",
                   "뭐 할 수 있어?", "MaleCNS가 뭐야?", "What is MaleCNS?"):
        plan = _sample_plan(prompt)
        if plan and plan.get("ready") and plan.get("semantic_steps"):
            answer = trusted_fact_fallback(plan)
            if answer:
                key = packet_text(plan)
                leads = fact_leads[plan["style"]["language"]] if phrasing_cues else ("",)
                originals[key] = [Example("facts:" + plan["speech_act"], key, lead + answer)
                                  for lead in leads]
    numeric = [Example("verified-number", packet_text(_math_plan(value)), str(value))
               for value in range(200)]
    rng = random.Random(seed)
    rng.shuffle(numeric)
    # Distinct value packets; no label leakage from held-out examples.
    valid, test, train_numbers = numeric[:20], numeric[20:40], numeric[40:]
    groups = list(originals.values())
    # Held-out stylistic variations of known meanings test whether the model
    # preserves content rather than matching a complete training packet.
    varied = []
    for rows in groups[:4]:
        row = rows[0]
        payload = json.loads(row.user)
        payload["tone"] = "careful_unseen"  # NOT present in training.
        varied.append(Example(row.group + ":style", json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ), row.assistant))
    valid.extend(varied[:2])
    test.extend(varied[2:])
    if len(groups) < 5:
        raise ValueError("brain did not produce enough independent speech plans")
    # Give each packet equal weight while preserving every valid realization.
    if diverse_social:
        train = []
        for rows in groups:
            act = rows[0].group.removeprefix("social:")
            count = 40 if act in EVERYDAY_SPEECH_ACTS else 12
            train.extend(rows[i % len(rows)] for i in range(count))
    else:
        train = [rows[0] for rows in groups] * 12
    if phrasing_cues:
        # Teach separable delivery cues as well as ordinary, unhinted prose.
        # The runtime sends only a cue and the plan, never a target sentence.
        for rows in groups:
            for index, style in enumerate(DELIVERY_STYLES):
                row = rows[index % len(rows)]
                example = Example(row.group, delivery_packet(row.user, style), row.assistant)
                train.extend([example] * 12)
    train.extend(train_numbers)
    if mirror_greetings:
        greetings = _greeting_examples(seed)
        train.extend(greetings.train)
        valid.extend(greetings.valid)
        test.extend(greetings.test)
    if compositional_greetings:
        greetings = _compositional_greetings()
        train.extend(greetings.train)
        valid.extend(greetings.valid)
        test.extend(greetings.test)
    if everyday_dialogue:
        for key, rows in originals.items():
            if rows[0].group.removeprefix("social:") not in EVERYDAY_SPEECH_ACTS:
                continue
            # Evaluate alternate plan styles against multiple authored replies.
            for split, tone in ((train, "warm"), (train, "short"),
                                (valid, "casual"), (test, "friendly")):
                alternate = key.replace(", natural):", f", {tone}):", 1)
                split.extend(Example(row.group, alternate, row.assistant) for row in rows)
    rng.shuffle(train)
    data = SpeechCorpus(train, valid, test)
    train_prompts = {row.user for row in data.train}
    for other in (data.valid, data.test):
        if train_prompts & {row.user for row in other}:
            raise AssertionError("cross-split semantic packet leakage")
    if {row.user for row in data.valid} & {row.user for row in data.test}:
        raise AssertionError("valid/test semantic packet leakage")
    return data


def _compositional_greetings() -> SpeechCorpus:
    # Every opening and follow-up occurs in training. Two combinations per
    # opening are withheld, so neither complete response nor cue pair leaks.
    examples = (
        (
            "안녕",
            ("안녕!", "반가워!", "어서 와!", "오, 안녕!", "안녕, 반가워!"),
            ("편하게 얘기해.", "오늘은 뭐가 궁금해?", "우리 같이 얘기해 보자.",
             "어떤 이야기를 하고 싶어?", "무슨 얘기부터 해볼까?"),
        ),
        (
            "Hi",
            ("Hi!", "Hello!", "Hey there!", "Nice to see you!", "Glad you're here!"),
            ("What's on your mind?", "Let's chat.", "What would you like to talk about?",
             "Feel free to ask me anything.", "Where should we start?"),
        ),
    )
    data = SpeechCorpus([], [], [])
    for message, openings, followups in examples:
        packet = packet_text(_sample_plan(message))
        for opening, followup in GREETING_CUES:
            pair = Example(
                "compositional-greeting", greeting_packet(packet, opening, followup),
                f"{openings[opening]} {followups[followup]}",
            )
            offset = (followup - opening) % len(followups)
            if offset == 1:
                data.valid.append(pair)
            elif offset == 2:
                data.test.append(pair)
            else:
                data.train.extend([pair] * 12)
    return data
