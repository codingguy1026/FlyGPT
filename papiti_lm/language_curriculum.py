"""v0.6.3 opt-in concept -> application language curriculum and transfer probes.

All supervision comes from authored, training-side utterances. Neither the
model nor the live runtime receives a lookup table of answers. The frozen
legacy validation and 94 combination tests remain unchanged.
"""
from __future__ import annotations

from papiti_lm.corpus import Example
from papiti_lm.v0_6_1_data import SOCIAL_SPEECH_ACTS, SpeechCorpus
from papiti_lm.v0_6_1_prose import _FOLLOWUPS, _packet_details, prose_packet
from papiti_lm.inquiry_curriculum import _TRANSFER


def _packet(row: Example) -> str:
    return row.user.split("\nOpening style: ", 1)[0]


def foundation_sequences(corpus: SpeechCorpus, tokenizer, context: int):
    """Supervise clean opening clauses before whole two-part responses.

    Original text and the semantic prompt are unchanged. The continuation is
    present in teacher-forced input, but its target positions are masked: it
    cannot contribute to this phase's loss. Unlike a first-line-only target,
    this never teaches a premature EOS at the end of the opening.
    """
    lessons = []
    for row in corpus.train:
        act, _ = _packet_details(_packet(row))
        if act not in SOCIAL_SPEECH_ACTS or "\n" not in row.assistant:
            continue
        first = row.assistant.split("\n", 1)[0]
        inputs, targets = tokenizer.sequence(row, context)
        # Labels are shifted one position relative to prefix token IDs.
        start = len(tokenizer.prompt(row.user)) - 1
        end = start + len(tokenizer.encode(first))
        if end <= start or end > len(targets):
            raise AssertionError("invalid opening lesson boundary")
        targets = list(targets)
        targets[end:] = [-100] * (len(targets) - end)
        if not any(target != -100 for target in targets):
            raise AssertionError("foundation lesson lacks supervised tokens")
        lessons.append((inputs, targets))
    if not lessons:
        raise ValueError("no social opening lessons available")
    return lessons


def foundation_epoch_count(epochs: int) -> int:
    """An intentionally short opening phase followed by full-text practice."""
    if not 1 <= epochs <= 60:
        raise ValueError("invalid training length")
    # Reserve at least one full-response epoch, even in a short smoke run.
    return min(6, max(1, epochs // 5), epochs - 1)


def mixed_style_transfer_probes(corpus: SpeechCorpus) -> list[Example]:
    """Build 36 previously unpaired style combinations for manual review.

    One side of each sentence is taught in original styles 0..3; the other is
    taught in inquiry styles 4..5. The resulting mixed cue pairs are excluded
    from all stages of training and the unchanged 94-item regression test.
    Neither this function nor its targets are called by the training path.
    """
    known_openings: dict[str, str] = {}
    packs: dict[str, tuple[str, str]] = {}
    for row in corpus.train:
        packet = _packet(row)
        act, language = _packet_details(packet)
        if (act, language) not in _TRANSFER:
            continue
        packs[packet] = (act, language)
        if row.user == prose_packet(packet, 0, 0):
            known_openings[packet] = row.assistant.split("\n", 1)[0]
    probes = []
    for packet, (act, language) in sorted(packs.items()):
        if packet not in known_openings:
            raise ValueError(f"missing original style-zero opening for {act}/{language}")
        new_openings, new_followups = _TRANSFER[(act, language)]
        # Avoid awkward repetition in the Korean greeting probe (어서 와 ... 어서 와).
        followup_index = 1 if (act, language) == ("return_greeting", "ko") else 0
        old_followup = _FOLLOWUPS[act][language == "en"][followup_index]
        probes.extend((
            Example("inquiry-mixed-transfer", prose_packet(packet, 4, 0),
                    new_openings[0] + "\n" + old_followup),
            Example("inquiry-mixed-transfer", prose_packet(packet, 0, 5),
                    known_openings[packet] + "\n" + new_followups[1]),
        ))
    if len(probes) != 36:
        raise ValueError(f"expected 36 bilingual probes, found {len(probes)}")
    prior = {row.user for row in (*corpus.train, *corpus.valid, *corpus.test)}
    if len({row.user for row in probes}) != len(probes) or any(
        row.user in prior for row in probes
    ):
        raise ValueError("transfer probes overlap an existing split")
    return probes
