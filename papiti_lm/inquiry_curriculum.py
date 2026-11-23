"""Opt-in inquiry-inspired prose transfer curriculum.

Training-only, authored examples. No answers, templates, or knowledge are
introduced into the live runtime. Held-out baseline examples stay frozen.
Inspired by inquiry -> application -> reflection, not an IB-certified course.
"""
from __future__ import annotations

from papiti_lm.corpus import Example
from papiti_lm.v0_6_1_data import SpeechCorpus

# Additional examples teach the same communicative intentions with different
# complete surface forms. Styles 4 and 5 are absent from the legacy 0..3 set.
# Each pair contains two authentic, manually authored alternative openings and
# continuations. None is used as a live canned response.
_TRANSFER = {
    ("return_greeting", "ko"): (
        ("어서 와, 이야기 나눠 보자!", "안녕, 어떤 얘기든 좋아!"),
        ("오늘은 어떤 이야기를 해 볼까?", "편하게 시작해도 괜찮아."),
    ),
    ("return_greeting", "en"): (
        ("It's nice to see you!", "Hey, good to have you here!"),
        ("What would you like to discuss?", "You can start wherever you like."),
    ),
    ("return_welcome", "ko"): (
        ("여기서 만나니 반갑다!", "너와 이야기하게 돼서 좋아!"),
        ("무엇부터 얘기해 볼까?", "천천히 이야기해도 괜찮아."),
    ),
    ("return_welcome", "en"): (
        ("Good to have you here!", "It's a pleasure to meet you!"),
        ("What topic shall we begin with?", "Tell me what interests you."),
    ),
    ("acknowledge_thanks", "ko"): (
        ("도움이 되었다니 좋네!", "고맙다는 말 잘 받았어!"),
        ("추가로 궁금한 것도 물어봐.", "다른 문제도 함께 살펴볼 수 있어."),
    ),
    ("acknowledge_thanks", "en"): (
        ("Happy to help!", "I'm pleased that helped!"),
        ("Feel free to bring another question.", "We can explore another question together."),
    ),
    ("return_farewell", "ko"): (
        ("다음에 또 이야기하자!", "안녕, 잘 지내!"),
        ("남은 하루도 편안하길 바라.", "다음 이야기도 기대할게."),
    ),
    ("return_farewell", "en"): (
        ("Catch you next time!", "Until we chat again!"),
        ("I hope your day goes smoothly.", "Take good care of yourself."),
    ),
    ("request_specific_task", "ko"): (
        ("무엇이 필요한지 알려 줄래?", "같이 해결해 볼 문제를 알려 줘."),
        ("어느 부분부터 살펴보면 될까?", "현재 상황을 조금 설명해 줘."),
    ),
    ("request_specific_task", "en"): (
        ("Tell me what you need a hand with.", "What would you like to work through?"),
        ("Which part should we tackle first?", "Give me a little context to start."),
    ),
    ("acknowledge_presence", "ko"): (
        ("응, 여기서 같이 이야기할 수 있어.", "좋아, 대화를 시작해 보자."),
        ("지금 마음에 떠오르는 이야기부터 해 줘.", "궁금한 주제가 있으면 들려줘."),
    ),
    ("acknowledge_presence", "en"): (
        ("Sure, I'm available to talk.", "Yes, we can have a conversation."),
        ("Start with whichever topic you prefer.", "What would you like to talk through?"),
    ),
    ("acknowledge_good_news", "ko"): (
        ("좋은 소식이라니 신난다!", "기쁜 일이 생겼구나!"),
        ("무슨 일인지 들려줄래?", "어떻게 그런 일이 있었는지 궁금해."),
    ),
    ("acknowledge_good_news", "en"): (
        ("That sounds like wonderful news!", "I'm glad something good happened!"),
        ("Would you tell me what happened?", "How did things turn out?"),
    ),
    ("acknowledge_tiredness", "ko"): (
        ("오늘 꽤 지친 것 같아.", "많이 피곤한 하루였나 봐."),
        ("지금은 잠깐 쉬어 갈 수 있을까?", "어떤 점이 가장 힘들었어?"),
    ),
    ("acknowledge_tiredness", "en"): (
        ("It sounds like a long day.", "You seem pretty worn out."),
        ("Could you get some rest now?", "What was the hardest part of today?"),
    ),
    ("respond_to_laughter", "ko"): (
        ("정말 웃긴 일이 있었나 봐!", "무척 재미있었나 보네!"),
        ("어느 부분이 제일 웃겼어?", "그 이야기를 좀 더 들어 보고 싶어."),
    ),
    ("respond_to_laughter", "en"): (
        ("That must have been funny!", "Sounds like something amused you!"),
        ("What was the funniest part?", "I'd be curious to hear what happened."),
    ),
}


def add_inquiry_training(corpus: SpeechCorpus) -> SpeechCorpus:
    """Add transfer examples without altering any old train, valid or test row."""
    from hybrid_speech import reply_key
    from papiti_lm.v0_6_1_prose import _packet_details, prose_packet

    packets = {}
    for row in corpus.train:
        packet = row.user.split("\nOpening style: ", 1)[0]
        act, language = _packet_details(packet)
        if (act, language) in _TRANSFER:
            packets[packet] = (act, language)
    additions = []
    for packet, key in sorted(packets.items()):
        openings, continuations = _TRANSFER[key]
        # Four combinations per social plan; all use new styles, so legacy
        # hold-out cue pairs (0..3) remain untouched.
        for i in range(2):
            for j in range(2):
                additions.append(Example(
                    "inquiry-transfer", prose_packet(packet, i + 4, j + 4),
                    f"{openings[i]}\n{continuations[j]}",
                ))
    original = {reply_key(x.assistant) for x in (*corpus.valid, *corpus.test)}
    if original & {reply_key(x.assistant) for x in additions}:
        raise ValueError("inquiry examples leak frozen held-out answers")
    if {x.user for x in additions} & {x.user for x in (*corpus.valid, *corpus.test)}:
        raise ValueError("inquiry examples leak frozen held-out prompts")
    return SpeechCorpus(corpus.train + additions, corpus.valid, corpus.test)
