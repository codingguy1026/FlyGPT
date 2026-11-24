"""IB-MYP-inspired (not IB-certified) inquiry/action/reflection practice.

Only training-side answers are recombined. Existing validation, the 94 frozen
tests, and both 36-item exploratory transfer sets remain strictly untouched.
No runtime templates, banned-word filters, or answer tables are introduced.
"""
from __future__ import annotations

from collections import defaultdict

from papiti_lm.corpus import Example
from papiti_lm.v0_6_1_data import SOCIAL_SPEECH_ACTS, SpeechCorpus
from papiti_lm.v0_6_1_prose import _packet_details, prose_packet

# Deliberately exclude both previously inspected transfer sets:
# old: (4, 0), (0, 5); second diagnostic: (4, 1), (1, 5).
RESERVED_TRANSFER_PAIRS = frozenset({(4, 0), (0, 5), (4, 1), (1, 5)})
ACTION_PAIRS = ((2, 4), (3, 5), (5, 2), (5, 3))


def unique_training_rows(corpus: SpeechCorpus) -> list[Example]:
    """Count each actual example once instead of optimizing duplicated rows."""
    return list({(row.user, row.assistant): row for row in corpus.train}.values())


def action_transfer_examples(corpus: SpeechCorpus) -> list[Example]:
    """Construct practice combinations solely from components already trained.

    New pairs are training-only, never the reserved transfer pairs. The user
    prompt itself is a pair of style cues; the target joins separately seen
    opening and follow-up texts. Both original and added rows are checked
    against held-out answers to avoid exact-label leakage.
    """
    from hybrid_speech import reply_key

    parts = defaultdict(lambda: {"openings": {}, "followups": {}})
    existing_prompts = {row.user for row in (*corpus.train, *corpus.valid, *corpus.test)}
    heldout_answers = {reply_key(row.assistant) for row in (*corpus.valid, *corpus.test)}

    for row in corpus.train:
        if row.assistant.count("\n") != 1 or "\nOpening style: " not in row.user:
            continue
        packet, suffix = row.user.rsplit("\nOpening style: ", 1)
        if "\nFollow-up style: " not in suffix:
            continue
        opening_style, followup_style = suffix.split("\nFollow-up style: ", 1)
        act, lang = _packet_details(packet)
        if act not in SOCIAL_SPEECH_ACTS or lang not in {"ko", "en"}:
            continue
        opening, followup = row.assistant.split("\n", 1)
        slot = parts[packet]
        # The same style may be duplicated by prose-repeats. The first
        # occurrence is sufficient; never invent a target from a heldout.
        slot["openings"].setdefault(opening_style, opening)
        slot["followups"].setdefault(followup_style, followup)

    actions = []
    for packet, slot in sorted(parts.items()):
        for start, end in ACTION_PAIRS:
            if (start, end) in RESERVED_TRANSFER_PAIRS:
                raise AssertionError("reserved transfer pair was selected for training")
            opening = slot["openings"].get(str(start))
            followup = slot["followups"].get(str(end))
            if not opening or not followup:
                continue
            prompt = prose_packet(packet, start, end)
            answer = opening + "\n" + followup
            if prompt in existing_prompts:
                raise AssertionError("new action practice overlaps an existing split")
            if reply_key(answer) in heldout_answers:
                # Never turn an existing validation/test answer into training.
                continue
            actions.append(Example("ib-action-transfer", prompt, answer))

    if not actions:
        raise ValueError("no clean IB action transfer combinations were constructed")
    if len({row.user for row in actions}) != len(actions):
        raise AssertionError("action transfer contains duplicated prompts")
    return actions


def stage_for_epoch(epoch: int, epochs: int) -> str:
    """An inquiry foundation, then action practice, then feedback-based rehearsal."""
    from papiti_lm.language_curriculum import foundation_epoch_count

    inquiry = foundation_epoch_count(epochs)
    action = min(6, max(2, epochs // 6))
    if epoch <= inquiry:
        return "ib_inquiry"
    if epoch <= inquiry + action:
        return "ib_action"
    return "ib_reflection"


def balanced_epoch_rows(
    originals: list, actions: list, epoch: int, *, focus: list | None = None,
) -> list:
    """Replace, rather than add, a fraction of original rows with practice.

    The example count per full-response epoch matches the no-action control.
    A rotating deterministic selection exposes all action pairs across epochs.
    """
    if not actions:
        return list(originals)
    import random

    rng = random.Random(65000 + epoch)
    original = list(originals)
    practice = list(actions)
    rng.shuffle(original)
    rng.shuffle(practice)
    transfer_budget = min(len(practice), max(1, len(original) // 5))
    selected = practice[:transfer_budget]
    if focus:
        focus_count = min(len(focus), max(1, transfer_budget // 3))
        selected[-focus_count:] = focus[:focus_count]
    return original[:len(original) - transfer_budget] + selected
