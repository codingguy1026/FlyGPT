"""Biology-informed, soft *experimental* role priors for MaleCNS task routing.

An official biological sensory/motor label is not a claim that this fly neuron
understands text or performs an AI task. No hard-coded path or disconnected
neuron is created; unclassified cells remain fully available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_ROLES = frozenset({
    "sensory", "motor", "interneuron", "projection", "other", "unclassified",
})
# Multipliers are explicit EXPERIMENTAL HYPOTHESES, not measurements.
# (encoder input, directed message relay, category-readout contribution)
ROLE_GAINS: dict[str, tuple[float, float, float]] = {
    "sensory": (1.25, 1.0, 0.9),
    "motor": (0.9, 1.0, 1.25),
    "interneuron": (1.0, 1.2, 1.0),
    "projection": (1.0, 1.15, 1.05),
    "other": (1.0, 1.0, 1.0),
    "unclassified": (1.0, 1.0, 1.0),
}


def validate_role_assignment(
    payload: dict[str, Any], scaffold_meta: dict[str, Any], n_nodes: int,
) -> tuple[list[str], dict[str, Any]]:
    """Reject ID mismatch, non-neuPrint provenance, and invented categories."""
    from collections import Counter

    if scaffold_meta.get("kind") != "male-cns:v1.0":
        raise ValueError("Role-aware training requires the MaleCNS v1.0 scaffold")
    if payload.get("dataset") != "male-cns:v1.0" or payload.get("source") != "Janelia MaleCNS v1.0 via neuPrint":
        raise ValueError("Role assignment must come from official MaleCNS v1.0 neuPrint annotations")
    ids = scaffold_meta.get("root_ids")
    if not isinstance(ids, list) or len(ids) != n_nodes or len(set(map(str, ids))) != n_nodes:
        raise ValueError("Scaffold root_ids must be unique and match n_nodes")
    reference = [str(int(body)) for body in ids]
    raw = payload.get("roles")
    if not isinstance(raw, dict) or set(raw) != set(reference):
        raise ValueError("Roles must cover the exact scaffold body ID set (no missing/extra IDs)")
    roles = [raw[body] for body in reference]
    if any(not isinstance(role, str) or role not in VALID_ROLES for role in roles):
        raise ValueError("Role assignment includes an unsupported biological category")
    if not any(role not in ("other", "unclassified") for role in roles):
        raise ValueError("No identified sensory/motor/intrinsic/projection neurons; cannot assess role priors")
    coverage = sum(role != "unclassified" for role in roles)
    return roles, {
        "source": payload["source"],
        "dataset": payload["dataset"],
        "annotated_count": coverage,
        "counts": dict(Counter(roles)),
        "role_gains": ROLE_GAINS,
        "scheme": "soft_biological_superclass_priors_v1",
        "training_only": True,
    }


def load_role_assignment(
    path: Path, scaffold_meta: dict[str, Any], n_nodes: int,
) -> tuple[list[str], dict[str, Any]]:
    return validate_role_assignment(
        json.loads(path.read_text(encoding="utf-8")), scaffold_meta, n_nodes,
    )
