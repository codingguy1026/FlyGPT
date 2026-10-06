from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class FlyRouterRuntime:
    """Load and run a trained FlyGraphRouter checkpoint on CPU."""

    def __init__(self, model_path: str | Path) -> None:
        import torch

        from training.train_router import FlyGraphRouter, vectorize

        self._torch = torch
        self._vectorize = vectorize
        self.model_path = Path(model_path)

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )

        self.routes = list(checkpoint["routes"])
        self.vocab_size = int(checkpoint["vocab_size"])
        self.n_nodes = int(checkpoint["n_nodes"])
        self.steps = int(checkpoint["steps"])
        self.best_val_accuracy = float(checkpoint.get("best_val_accuracy", 0.0))
        self.best_val_balanced_accuracy = float(
            checkpoint.get("best_val_balanced_accuracy", self.best_val_accuracy)
        )
        self.vectorizer_version = str(checkpoint.get("vectorizer_version", "v1"))
        self.scaffold_meta = dict(checkpoint.get("scaffold_meta", {}))
        self.temperature = max(0.05, min(float(checkpoint.get("temperature", 1.0)), 10.0))
        self.calibration = dict(checkpoint.get("calibration", {}))

        self.min_confidence = max(
            0.0,
            min(float(self.calibration.get("min_confidence", 0.55)), 1.0),
        )
        self.min_margin = max(
            0.0,
            min(float(self.calibration.get("min_margin", 0.10)), 1.0),
        )

        self.src = checkpoint["src"].to(dtype=torch.long, device="cpu")
        self.dst = checkpoint["dst"].to(dtype=torch.long, device="cpu")
        self.base_weight = checkpoint["base_weight"].to(dtype=torch.float32, device="cpu")

        self.model = FlyGraphRouter(
            vocab_size=self.vocab_size,
            n_nodes=self.n_nodes,
            n_classes=len(self.routes),
            src=self.src,
            dst=self.dst,
            base_weight=self.base_weight,
            steps=self.steps,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _activation_frame(self, h: Any, stage: str) -> dict[str, Any]:
        values = h[0].detach().abs().to(dtype=self._torch.float32, device="cpu")
        maximum = float(values.max().item()) if values.numel() else 0.0
        if maximum > 0:
            values = values / maximum

        return {
            "stage": stage,
            "activations": [round(float(value), 4) for value in values.tolist()],
        }

    def graph(self) -> dict[str, Any]:
        """Return the learned scaffold topology for the browser visualizer."""

        torch = self._torch
        with torch.no_grad():
            learned_weight = (
                self.base_weight
                * torch.sigmoid(self.model.edge_gain.detach().cpu())
                * 2.0
            )

        root_ids = self.scaffold_meta.get("root_ids") or []
        nodes = [
            {
                "index": index,
                # Keep connectome body IDs serialized as strings for browser safety and stable UI handling.
                "root_id": str(root_ids[index]) if index < len(root_ids) else None,
            }
            for index in range(self.n_nodes)
        ]
        edges = [
            {
                "src": int(src),
                "dst": int(dst),
                "weight": round(float(weight), 5),
            }
            for src, dst, weight in zip(
                self.src.tolist(),
                self.dst.tolist(),
                learned_weight.tolist(),
            )
        ]

        return {
            "model": self.model_path.name,
            "scaffold_kind": self.scaffold_meta.get("kind", "unknown"),
            "n_nodes": self.n_nodes,
            "n_edges": len(edges),
            "steps": self.steps,
            "nodes": nodes,
            "edges": edges,
            "temperature": self.temperature,
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
        }

    def predict(self, text: str, *, top_k: int = 3) -> dict[str, Any]:
        torch = self._torch
        x = self._vectorize(text, self.vocab_size, self.vectorizer_version).unsqueeze(0)
        feature_signal = float(torch.linalg.vector_norm(x).item())

        with torch.no_grad():
            h = torch.tanh(self.model.encoder(x))
            trace = [self._activation_frame(h, "input")]

            weights = (
                self.model.base_weight
                * torch.sigmoid(self.model.edge_gain)
                * 2.0
            )

            for step in range(1, self.steps + 1):
                messages = h[:, self.model.src] * weights.unsqueeze(0)
                agg = torch.zeros_like(h)
                agg.index_add_(1, self.model.dst, messages)
                agg = agg / self.model.in_degree.unsqueeze(0)
                h = self.model.norm(h + torch.tanh(agg))
                trace.append(self._activation_frame(h, f"step_{step}"))

            logits = self.model.output(h)
            if feature_signal <= 1e-8:
                probabilities = torch.full(
                    (len(self.routes),),
                    1.0 / max(1, len(self.routes)),
                    dtype=torch.float32,
                )
            else:
                probabilities = torch.softmax(logits / self.temperature, dim=1)[0]

        k = max(1, min(int(top_k), len(self.routes)))
        values, indices = torch.topk(probabilities, k=k)

        ranked = [
            {
                "route": self.routes[int(index)],
                "confidence": float(value),
            }
            for value, index in zip(values.tolist(), indices.tolist())
        ]

        confidence = ranked[0]["confidence"]
        second = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
        margin = max(0.0, confidence - second)

        entropy = 0.0
        if len(self.routes) > 1:
            safe = probabilities.clamp_min(1e-12)
            entropy = float((-(safe * safe.log()).sum() / math.log(len(self.routes))).item())

        accepted = (
            feature_signal > 1e-8
            and confidence >= self.min_confidence
            and margin >= self.min_margin
        )

        return {
            "model": self.model_path.name,
            "route": ranked[0]["route"],
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
            "accepted": accepted,
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
            "temperature": self.temperature,
            "calibrated": not math.isclose(self.temperature, 1.0, abs_tol=1e-6),
            "top_routes": ranked,
            "n_nodes": self.n_nodes,
            "steps": self.steps,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_balanced_accuracy": self.best_val_balanced_accuracy,
            "scaffold_kind": self.scaffold_meta.get("kind", "unknown"),
            "vectorizer_version": self.vectorizer_version,
            "feature_signal": feature_signal,
            "trace": trace,
        }
