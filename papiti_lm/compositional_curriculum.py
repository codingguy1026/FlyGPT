"""v0.6.4 factorized clause supervision for a generative bilingual speaker.

Training-only auxiliary examples. No sentence templates, forbidden-word lists,
or response lookups are installed in inference. Style-mixing transfer probes
remain wholly outside optimization.
"""
from __future__ import annotations

from papiti_lm.corpus import Example
from papiti_lm.v0_6_1_data import SOCIAL_SPEECH_ACTS, SpeechCorpus
from papiti_lm.v0_6_1_prose import _packet_details


def clause_lessons(corpus: SpeechCorpus, tokenizer, context: int):
    """Teach an opening without its follow-up cue, and vice versa.

    Both teacher-forced sequences keep the *original complete* response, but
    optimize only the relevant clause. The other cue becomes '?' so the model
    cannot rely on fixed same-family (old-old / new-new) style pairings.
    The first lesson never teaches an early EOS. The second still teaches EOS.
    Duplicate original training rows are counted just once for auxiliary work.
    """
    lessons = []
    seen = set()
    for row in corpus.train:
        pair = (row.user, row.assistant)
        if pair in seen:
            continue
        seen.add(pair)
        if "\nOpening style: " not in row.user or row.assistant.count("\n") != 1:
            continue
        packet, suffix = row.user.rsplit("\nOpening style: ", 1)
        if "\nFollow-up style: " not in suffix:
            continue
        opening_style, follow_style = suffix.split("\nFollow-up style: ", 1)
        act, language = _packet_details(packet)
        if act not in SOCIAL_SPEECH_ACTS or language not in {"ko", "en"}:
            continue
        if opening_style not in {"0", "1", "2", "3", "4", "5"} or follow_style not in {"0", "1", "2", "3", "4", "5"}:
            raise ValueError("invalid training style cue")
        first, _ = row.assistant.split("\n", 1)
        variants = (
            (f"{packet}\nOpening style: {opening_style}\nFollow-up style: ?", "opening"),
            (f"{packet}\nOpening style: ?\nFollow-up style: {follow_style}", "followup"),
        )
        for user, part in variants:
            example = Example("factorized-" + part, user, row.assistant)
            inputs, labels = tokenizer.sequence(example, context)
            start = len(tokenizer.prompt(user)) - 1
            # Include newline in the opening target so the speaker learns the
            # two-part structure without falsely learning an early end token.
            boundary = start + len(tokenizer.encode(first + "\n"))
            if boundary <= start or boundary >= len(labels):
                raise AssertionError("invalid clause boundary")
            labels = list(labels)
            if part == "opening":
                labels[boundary:] = [-100] * (len(labels) - boundary)
            else:
                labels[:boundary] = [-100] * boundary
            if not any(token != -100 for token in labels):
                raise AssertionError("empty factorized clause supervision")
            lessons.append((inputs, labels, user, part))
    if not lessons:
        raise ValueError("no eligible bilingual clause lessons")
    return lessons


def clause_training_sequences(corpus: SpeechCorpus, tokenizer, context: int):
    """Return only (inputs, labels), for torch DataLoader compatibility."""
    return [(inputs, labels) for inputs, labels, _, _ in clause_lessons(corpus, tokenizer, context)]
