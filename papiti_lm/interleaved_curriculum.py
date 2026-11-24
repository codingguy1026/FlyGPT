"""Train-only interleaved clause rehearsal for Papiti v0.6.7.

The candidate sees every original full-response training row plus rotating
factorized opening/follow-up lessons. A matched control receives the same
full-response training rows and an equal number of repeated full sentences.
Neither treatment adds output templates, word bans or evaluation answers.
"""
from __future__ import annotations

import random

from papiti_lm.corpus import Example
from papiti_lm.ib_curriculum import balanced_epoch_rows, unique_training_rows
from papiti_lm.v0_6_1_data import SOCIAL_SPEECH_ACTS, SpeechCorpus
from papiti_lm.v0_6_1_prose import _packet_details


def paired_clause_lessons(corpus: SpeechCorpus, tokenizer, context: int):
    """Pair the same source row's ordinary full response and masked clause.

    These factorized inputs match the established v0.6.4 clause curriculum.
    One variant trains the opening + newline and masks the continuation;
    another trains the continuation + EOS and masks the opening. Unmasked
    response characters are teacher-forced from TRAIN ONLY.
    """
    pairs = []
    for row in unique_training_rows(corpus):
        if "\nOpening style: " not in row.user or row.assistant.count("\n") != 1:
            continue
        packet, suffix = row.user.rsplit("\nOpening style: ", 1)
        if "\nFollow-up style: " not in suffix:
            continue
        opening_style, followup_style = suffix.split("\nFollow-up style: ", 1)
        act, lang = _packet_details(packet)
        if act not in SOCIAL_SPEECH_ACTS or lang not in {"ko", "en"}:
            continue
        if opening_style not in set("012345") or followup_style not in set("012345"):
            raise ValueError("invalid style")
        opening, _ = row.assistant.split("\n", 1)
        full = tokenizer.sequence(row, context)
        variants = (
            (f"{packet}\nOpening style: {opening_style}\nFollow-up style: ?", "opening"),
            (f"{packet}\nOpening style: ?\nFollow-up style: {followup_style}", "followup"),
        )
        for prompt, part in variants:
            item = Example("interleaved-" + part, prompt, row.assistant)
            inputs, labels = tokenizer.sequence(item, context)
            start = len(tokenizer.prompt(prompt)) - 1
            boundary = start + len(tokenizer.encode(opening + "\n"))
            if not start < boundary < len(labels) - 1:
                raise AssertionError("invalid split boundary")
            labels = list(labels)
            if part == "opening":
                labels[boundary:] = [-100] * (len(labels) - boundary)
            else:
                labels[:boundary] = [-100] * boundary
            if not any(y != -100 for y in labels):
                raise AssertionError("empty clause labels")
            pairs.append((full, (inputs, labels)))
    if not pairs:
        raise ValueError("missing training-side clause pairs")
    return pairs


def interleaved_epoch_rows(original, actions, paired, epoch: int, *, control: bool):
    """Equal full-response rows and equal added sample/step budget for both arms."""
    full_rows = balanced_epoch_rows(original, actions, epoch)
    rng = random.Random(670_000 + epoch)
    order = list(range(len(paired)))
    rng.shuffle(order)
    count = max(1, len(original) // 5)
    if len(paired) < count:
        raise ValueError("not enough training-only clause lessons")
    additions = [paired[i][0 if control else 1] for i in order[:count]]
    return full_rows + additions


def sealed_v067_probes(corpus: SpeechCorpus):
    """36 untrained style pairs selected BEFORE the v0.6.7 evaluation.

    Pair (4,2) / (2,5), distinct from v0.6.5 action training, both older
    36-probe diagnostics and the already opened v0.6.6 sealed pairs.
    Labels combine TRAIN-side components only. Never call from train().
    """
    from collections import defaultdict

    from hybrid_speech import reply_key
    from papiti_lm.v0_6_1_prose import prose_packet

    cue_pairs = ((4, 2), (2, 5))
    parts = defaultdict(lambda: {"opening": {}, "followup": {}})
    existing = {r.user for r in (*corpus.train, *corpus.valid, *corpus.test)}
    heldout = {reply_key(r.assistant) for r in (*corpus.valid, *corpus.test)}
    for row in corpus.train:
        if "\nOpening style: " not in row.user or row.assistant.count("\n") != 1:
            continue
        packet, suffix = row.user.rsplit("\nOpening style: ", 1)
        if "\nFollow-up style: " not in suffix:
            continue
        head_style, tail_style = suffix.split("\nFollow-up style: ", 1)
        act, lang = _packet_details(packet)
        if act not in SOCIAL_SPEECH_ACTS or lang not in {"ko", "en"}:
            continue
        first, second = row.assistant.split("\n", 1)
        parts[packet]["opening"].setdefault(head_style, first)
        parts[packet]["followup"].setdefault(tail_style, second)

    results = []
    for packet, segments in sorted(parts.items()):
        for a, b in cue_pairs:
            head, tail = (segments["opening"].get(str(a)),
                          segments["followup"].get(str(b)))
            if not head or not tail:
                raise AssertionError("unseen probe lacks training-side clauses")
            prompt = prose_packet(packet, a, b)
            answer = head + "\n" + tail
            if prompt in existing or reply_key(answer) in heldout:
                raise AssertionError("v0.6.7 sealed probe overlaps a frozen split")
            results.append(Example("v067-sealed-transfer", prompt, answer))
    if len(results) != 36 or len({r.user for r in results}) != 36:
        raise AssertionError("v0.6.7 must have exactly 36 distinct sealed probes")
    return results
