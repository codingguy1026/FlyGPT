"""Small, conservative cues for Papiti's *expressed* social tone.

These patterns are not sentiment recognition, a mental-health classifier,
or evidence that the AI experiences emotions. Only simple first-person
statements are recognized; questions and compound tasks remain with the
usual router.
"""
from __future__ import annotations

import re
from typing import Any


# The same atoms are used by the planner and checked by the realizer.
# No claims about events beyond what the user actually reported.
AFFECT_REACTIONS: dict[str, dict[str, Any]] = {
    "acknowledge_personal_joy": {
        "plan_value": "Acknowledge the user's explicitly stated good news or positive mood warmly, without inventing details.",
        "tone": "celebratory",
        "ko": "좋은 소식이네! 어떤 일이 있었는지 더 들려줄래? 🎉",
        "en": "That's good to hear! Want to tell me more? 🎉",
    },
    "acknowledge_personal_disappointment": {
        "plan_value": "Acknowledge the user's explicitly stated disappointment gently, without diagnosing or assuming a cause.",
        "tone": "gentle",
        "ko": "속상했겠다. 괜찮다면 무슨 일이 있었는지 이야기해 줘.",
        "en": "That sounds upsetting. If you'd like, you can tell me what happened.",
    },
}

# Full-string matches only: avoid converting searches, questions, quoted
# statements or multi-intent requests into canned mood responses.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(
        r"(?:(?:나|내가|나는)\s*)?(?:오늘\s*)?(?:(?:진짜|너무)\s*)?"
        r"(?:기분(?:이)?\s*좋아(?:요)?|신나(?:요)?|행복해(?:요)?)[!~.ㅋㅎ\s]*"
    ), "acknowledge_personal_joy"),
    (re.compile(
        r"(?:(?:나|내가|나는)\s*)?(?:오늘\s*)?시험\s*"
        r"(?:잘\s*봤어(?:요)?|잘\s*봄|잘\s*봤다)[!~.ㅋㅎ\s]*"
    ), "acknowledge_personal_joy"),
    (re.compile(
        r"(?:(?:나|내가|나는)\s*)?(?:오늘\s*)?(?:(?:진짜|너무)\s*)?"
        r"(?:속상해(?:요)?|기분(?:이)?\s*안\s*좋아(?:요)?)[!~.ㅠㅜ\s]*"
    ), "acknowledge_personal_disappointment"),
    (re.compile(
        r"(?:(?:나|내가|나는)\s*)?(?:오늘\s*)?시험\s*"
        r"(?:망쳤어(?:요)?|망했어(?:요)?|잘\s*못\s*봤어(?:요)?)[!~.ㅠㅜ\s]*"
    ), "acknowledge_personal_disappointment"),
    (re.compile(
        r"(?:i'm|i am)\s+(?:(?:so|really)\s+)?(?:happy|excited)"
        r"(?:\s+today)?[!.~\s]*"
        r"|i\s+(?:did well|aced)\s+(?:(?:on|in)\s+)?"
        r"(?:(?:my|the)\s+)?(?:test|exam)[!.~\s]*", re.IGNORECASE
    ), "acknowledge_personal_joy"),
    (re.compile(
        r"(?:i'm|i am)\s+(?:(?:so|really)\s+)?(?:sad|upset)"
        r"(?:\s+today)?[!.~\s]*"
        r"|i\s+(?:failed|did badly on)\s+(?:(?:my|the)\s+)?"
        r"(?:test|exam)[!.~\s]*", re.IGNORECASE
    ), "acknowledge_personal_disappointment"),
)


def detect_personal_affect(message: str) -> str | None:
    """Return a supported speech act only for an unambiguous short report."""
    if not isinstance(message, str):
        return None
    text = " ".join(message.strip().split())
    if not text or len(text) > 100:
        return None
    for pattern, act in _PATTERNS:
        if pattern.fullmatch(text):
            return act
    return None
