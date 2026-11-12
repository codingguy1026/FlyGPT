"""Validated runtime for Papiti v0.5 full-population *multi-action* policy.

This is deliberately separate from legacy six-route FlyRouterRuntime.
Do not load a v0.5 policy through FLYGPT_MODEL_PATH or claim that it predicts
any of the old categories. The official connectome is represented via exact
all-neuron weighted moments; this is not live biological neural firing.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

EXPECTED_NEURONS = 166700
EXPECTED_CAPABILITIES = ("web_search", "memory_lookup", "calculator")
DEFAULT_CHECKPOINT = "artifacts/papiti_v0_5_category_free_166700.pt"

_loaded = None
_loaded_path: Path | None = None
_load_lock = Lock()


class PapitiV05PolicyRuntime:
    def __init__(self, path: str | Path):
        import torch
        from training.papiti_v0_5_policy import (
            CAPABILITIES, FullMaleCNSPolicy,
        )

        self.path = Path(path)
        ckpt = torch.load(self.path, map_location="cpu", weights_only=True)
        meta = ckpt.get("structural_metadata", {})
        state = ckpt.get("model_state", {})
        thresholds = ckpt.get("thresholds")
        if (
            ckpt.get("format") != "papiti-v0.5-category-free-full-malecns"
            or ckpt.get("experimental_only") is not True
            or ckpt.get("production_enabled") is not False
            or ckpt.get("n_neurons") != EXPECTED_NEURONS
            or meta.get("dataset") != "male-cns:v1.0"
            or meta.get("annotated_neurons_eligible") != EXPECTED_NEURONS
            or meta.get("full_source_scanned_every_epoch") is not True
            or meta.get("no_topk_or_node_sampling") is not True
            or meta.get("training_kind") != "offline_self_supervised_directed_edge_reconstruction"
            or tuple(ckpt.get("capabilities", ())) != EXPECTED_CAPABILITIES
            or tuple(CAPABILITIES) != EXPECTED_CAPABILITIES
            or not isinstance(thresholds, (list, tuple))
            or len(thresholds) != len(EXPECTED_CAPABILITIES)
            or any(not isinstance(t, (int, float)) or not 0 < t < 1 for t in thresholds)
            or "neuron_ids" not in state
            or "keys" not in state
            or "neuron_gain_logits" not in state
            or "encoder.0.weight" not in state
        ):
            raise ValueError("Checkpoint does not meet official v0.5 full-neuron policy invariants")

        ids, keys = state["neuron_ids"], state["keys"]
        if (
            ids.numel() != EXPECTED_NEURONS
            or keys.ndim != 2
            or keys.shape[0] != EXPECTED_NEURONS
            or state["neuron_gain_logits"].numel() != EXPECTED_NEURONS
            or not bool(torch.isfinite(state["neuron_gain_logits"]).all())
        ):
            raise ValueError("Missing, invalid or non-finite per-neuron policy weights")
        model = FullMaleCNSPolicy(
            ids, keys, hidden=state["encoder.0.weight"].shape[0],
        )
        model.load_state_dict(state, strict=True)
        model.eval()

        self.model = model
        self.thresholds = [float(t) for t in thresholds]
        self.n_neurons = EXPECTED_NEURONS
        self.n_positive_edges = int(
            meta.get("epoch_reports", [{}])[-1].get("eligible_directed_connection_rows", 0)
        )
        if self.n_positive_edges < EXPECTED_NEURONS:
            raise ValueError("Structural pretraining provenance is incomplete")
        self.name = "papiti-v0.5-category-free-166700"
        self.capabilities = tuple(CAPABILITIES)

    def decide(self, message: str) -> dict[str, Any]:
        return self.model.decide(message, self.thresholds)


def get_v05_policy(path: str | Path = DEFAULT_CHECKPOINT) -> PapitiV05PolicyRuntime:
    """Load lazily once. Missing/mismatched checkpoint fails CLOSED.

    Legacy v0.4 is not substituted when v0.5 is explicitly enabled.
    """
    global _loaded, _loaded_path
    requested = Path(path).resolve()
    if _loaded is not None and _loaded_path == requested:
        return _loaded
    with _load_lock:
        if _loaded is None or _loaded_path != requested:
            candidate = PapitiV05PolicyRuntime(requested)
            _loaded, _loaded_path = candidate, requested
    return _loaded
