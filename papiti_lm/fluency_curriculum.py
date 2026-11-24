"""Training-only fluency recovery utilities for Papiti v0.6.6.

Keep every original next-token label. Raise loss weights on complete response
endings, rather than adding fixed answers, forbidden words or runtime patches.
All text comes from the existing TRAIN split.
"""
from __future__ import annotations

from papiti_lm.corpus import EOS, Example
from papiti_lm.ib_curriculum import balanced_epoch_rows

# These pairs have never been used in v0.6.5's optimizer or earlier two
# 36-question transfer diagnostics. Once opened, they cease to be blind.
SEALED_TRANSFER_PAIRS = ((5, 0), (0, 4))


def fluency_sequence(
    row: Example, tokenizer, context: int,
    *, followup_weight: float = 1.6, eos_weight: float = 1.4,
):
    """Return normal input/labels and loss weights for later response tokens.

    Every non-prompt token is still supervised. No generated prefix or
    validation/test answer is introduced. Control weights (1, 1) exactly
    reproduce ordinary cross-entropy for the same minibatches.
    """
    if followup_weight < 1 or eos_weight < 1:
        raise ValueError("fluency weights must not be less than one")
    inputs, labels = tokenizer.sequence(row, context)
    weights = [0.0 if label == -100 else 1.0 for label in labels]
    if labels[-1] != EOS:
        raise AssertionError("complete response must end with an EOS label")

    if "\nOpening style: " in row.user and row.assistant.count("\n") == 1:
        opening, _ = row.assistant.split("\n", 1)
        # The entire first clause and newline remain trained at weight one.
        # Reweight only the second clause, ending before EOS.
        start = len(tokenizer.prompt(row.user)) - 1
        boundary = start + len(tokenizer.encode(opening + "\n"))
        if not start < boundary < len(labels) - 1:
            raise AssertionError("invalid full-response follow-up boundary")
        for index in range(boundary, len(labels) - 1):
            if labels[index] == -100:
                raise AssertionError("follow-up would contain masked labels")
            weights[index] = followup_weight

    weights[-1] = eos_weight
    if any((label == -100) != (weight == 0) for label, weight in zip(labels, weights)):
        raise AssertionError("weight/label mask mismatch")
    return inputs, labels, weights


def fluency_epoch_rows(
    original: list[Example], action: list[Example], tokenizer, context: int,
    epoch: int, *, followup_weight: float, eos_weight: float,
):
    """Use identical rotating original/action training rows for both arms."""
    selected = balanced_epoch_rows(original, action, epoch)
    return [
        fluency_sequence(
            row, tokenizer, context,
            followup_weight=followup_weight, eos_weight=eos_weight,
        )
        for row in selected
    ]


def sealed_transfer_probes(corpus):
    """Predeclare unseen style combinations from TRAIN-SIDE parts only.

    Exactly 36 probes at seed 42: 18 acts/languages x two unseen cue pairs.
    Nothing here may be imported into the optimizer or model checkpoint.
    """
    from collections import defaultdict

    from hybrid_speech import reply_key
    from papiti_lm.corpus import Example
    from papiti_lm.v0_6_1_data import SOCIAL_SPEECH_ACTS
    from papiti_lm.v0_6_1_prose import _packet_details, prose_packet

    slots = defaultdict(lambda: {"opening": {}, "followup": {}})
    prompts = {row.user for row in (*corpus.train, *corpus.valid, *corpus.test)}
    heldout_replies = {reply_key(row.assistant) for row in (*corpus.valid, *corpus.test)}
    for row in corpus.train:
        if row.assistant.count("\n") != 1 or "\nOpening style: " not in row.user:
            continue
        packet, suffix = row.user.rsplit("\nOpening style: ", 1)
        if "\nFollow-up style: " not in suffix:
            continue
        start, end = suffix.split("\nFollow-up style: ", 1)
        act, lang = _packet_details(packet)
        if act not in SOCIAL_SPEECH_ACTS or lang not in {"en", "ko"}:
            continue
        first, second = row.assistant.split("\n", 1)
        slots[packet]["opening"].setdefault(start, first)
        slots[packet]["followup"].setdefault(end, second)

    result = []
    for packet, parts in sorted(slots.items()):
        for first_style, next_style in SEALED_TRANSFER_PAIRS:
            first = parts["opening"].get(str(first_style))
            second = parts["followup"].get(str(next_style))
            if not first or not second:
                raise AssertionError("missing training-side component for sealed probe")
            prompt = prose_packet(packet, first_style, next_style)
            expected = first + "\n" + second
            if prompt in prompts or reply_key(expected) in heldout_replies:
                raise AssertionError("sealed probe overlaps a preexisting split")
            result.append(Example("sealed-mixed-transfer", prompt, expected))
    if len(result) != 36 or len({x.user for x in result}) != 36:
        raise AssertionError("expected 36 distinct sealed transfer pairs")
    return result
