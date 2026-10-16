"""Opt-in serving adapter for a truly whole-population MaleCNS structural router.

The *full-neuron structural summary* reflects every curated annotated neuron
and all positive directed neuron-neuron connection rows from preprocessing.
It is NOT a per-neuron firing-trace simulator. Never fabricate frames.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


class FullPopulationRouterRuntime:
    def __init__(self, model_path: str | Path):
        import torch
        from training.full_population_router import FullPopulationRouter
        from training.train_router import vectorize

        if os.environ.get("FLYGPT_ENABLE_FULL_POPULATION_ROUTER") != "1":
            raise ValueError("Full-neuron model requires explicit opt-in")
        self.model_path = Path(model_path)
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
        meta = ckpt.get("scaffold_meta", {})
        if (
            ckpt.get("format") != "flygpt-full-population-v1"
            or ckpt.get("architecture") != "FullPopulationRouter"
            or ckpt.get("experimental_only") is not True
            or meta.get("kind") != "male-cns:v1.0"
            or meta.get("full_connectome_trained") is not True
            or meta.get("full_source_scanned_every_epoch") is not True
            or meta.get("full_neuron_population_attended") is not True
            or meta.get("live_edge_propagation") is not False
        ):
            raise ValueError("Invalid or unverified all-neuron structural checkpoint")

        self.routes = list(ckpt["routes"])
        self.vocab_size = int(ckpt["vocab_size"])
        self.vectorizer_version = ckpt.get("vectorizer_version")
        self.n_nodes = int(ckpt["n_nodes"])
        self.n_edges = int(meta.get("n_positive_connection_rows", 0))
        if (
            self.n_nodes < 100000
            or self.n_edges < self.n_nodes
            or self.vocab_size != 8192
            or self.vectorizer_version != "v6"
            or self.routes != ["code", "general", "math", "memory", "research", "summarize"]
        ):
            raise ValueError("Full MaleCNS model missing real population coverage")
        ids = ckpt["neuron_ids"]
        model_state = ckpt["model_state"]
        keys = model_state["keys"]
        if ids.numel() != self.n_nodes or keys.shape[0] != self.n_nodes:
            raise ValueError("Full-population checkpoint silently excluded neurons")
        if not torch.equal(ids.cpu(), model_state["neuron_ids"].cpu()):
            raise ValueError("Mismatch between saved model and annotated neuron IDs")

        self.model = FullPopulationRouter(ids, keys, len(self.routes),
                                          vocab_size=self.vocab_size)
        self.model.load_state_dict(model_state, strict=True)
        self.model.eval()
        self._torch = torch
        self._vectorize = vectorize
        self.scaffold_meta = dict(meta)
        info = ckpt["training_summary"]
        self.temperature = max(0.05, min(float(info["temperature"]), 10.0))
        self.calibration = dict(info["calibration"])
        self.min_confidence = max(0., min(float(self.calibration.get("min_confidence", .55)), 1.))
        self.min_margin = max(0., min(float(self.calibration.get("min_margin", .10)), 1.))
        self.best_val_balanced_accuracy = float(info.get("best_val_macro_recall", 0.))
        self.best_val_accuracy = float(info.get("best_val_accuracy", 0.))
        self.steps = 0  # No edge propagation during per-message inference.
        self.full_neuron_population = True

    def graph(self) -> dict[str, Any]:
        """Full population statistics, no fabricated browser neuronal graph."""
        return {
            "model": self.model_path.name,
            "scaffold_kind": "male-cns:v1.0",
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "steps": 0,
            "nodes": [],
            "edges": [],
            "full_population_summary": True,
            "visualized_neurons": 0,
            "visualization_status": "summary-only; no single-cell live firing measurement",
            "representation": self.scaffold_meta["representation"],
            "source": self.scaffold_meta["source"],
        }

    def predict(self, text: str, *, top_k: int = 3, on_frame=None) -> dict[str, Any]:
        torch = self._torch
        x = self._vectorize(text, self.vocab_size, self.vectorizer_version).unsqueeze(0)
        feature_signal = float(torch.linalg.vector_norm(x).item())
        with torch.no_grad():
            logits = self.model(x)
            if feature_signal < 1e-8:
                probs = torch.full((len(self.routes),), 1./len(self.routes))
            else:
                probs = torch.softmax(logits / self.temperature, dim=1)[0]
        values, indices = torch.topk(probs, k=min(max(1, int(top_k)), len(self.routes)))
        ranked = [
            {"route": self.routes[int(idx)], "confidence": float(value)}
            for value, idx in zip(values.tolist(), indices.tolist())
        ]
        best = ranked[0]["confidence"]
        second = ranked[1]["confidence"] if len(ranked)>1 else 0.
        margin = max(0., best-second)
        entropy = float((-(probs.clamp_min(1e-12)*probs.clamp_min(1e-12).log())
                        .sum()/math.log(len(self.routes))).item())
        return {
            "model": self.model_path.name,
            "route": ranked[0]["route"],
            "confidence": best, "margin": margin, "entropy": entropy,
            "accepted": bool(feature_signal > 1e-8
                             and best >= self.min_confidence
                             and margin >= self.min_margin),
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
            "temperature": self.temperature,
            "calibrated": not math.isclose(self.temperature, 1., abs_tol=1e-6),
            "top_routes": ranked,
            "n_nodes": self.n_nodes, "steps": 0,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_balanced_accuracy": self.best_val_balanced_accuracy,
            "scaffold_kind": "male-cns:v1.0",
            "vectorizer_version": self.vectorizer_version,
            "feature_signal": feature_signal,
            "trace": [],  # We do NOT invent live full-neuron firing.
            "role_aware_experimental": False,
            "full_population_summary": True,
            "all_curated_neurons_attended": True,
        }
