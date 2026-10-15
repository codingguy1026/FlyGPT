"""Conservative presentation categories for official MaleCNS superclass tags.

Categories are annotations-derived UI groupings, NOT inferred physiological
roles and NOT the task-router's learned internal function.
"""
from __future__ import annotations

from typing import Any

ROLE_SUPERCLASSES: dict[str, frozenset[str]] = {
    "sensory": frozenset({
        "ol_sensory", "cb_sensory", "vnc_sensory",
        "sensory_ascending", "sensory_descending",
    }),
    "motor": frozenset({"cb_motor", "vnc_motor"}),
    "interneuron": frozenset({
        "ol_intrinsic", "cb_intrinsic", "vnc_intrinsic",
    }),
    "projection": frozenset({
        "visual_projection", "visual_centrifugal",
        "ascending_neuron", "descending_neuron",
    }),
}


def categorize_superclass(value: Any) -> str:
    """Return a coarse UI group only from a nonempty official superclass.

    Keep remaining named superclasses under 'other', not 'motor', so efferent,
    endocrine and unresolved classes aren't incorrectly called motor neurons.
    """
    if not isinstance(value, str) or not value.strip():
        return "unclassified"
    clean = value.strip().lower()
    for category, values in ROLE_SUPERCLASSES.items():
        if clean in values:
            return category
    return "other"


def annotation_record(row: dict[str, Any]) -> dict[str, str | None]:
    """Normalize public neuPrint metadata; missing values never become labels."""
    def clean(field: str) -> str | None:
        raw = row.get(field)
        if raw is None:
            return None
        value = str(raw).strip()
        return value if value and value.lower() not in {"nan", "none", "null"} else None

    superclass = clean("superclass")
    return {
        "body_id": str(int(row["body_id"])),
        "kind": categorize_superclass(superclass),
        "superclass": superclass,
        "type": clean("type"),
        "instance": clean("instance"),
        "consensus_nt": clean("consensus_nt"),
    }
