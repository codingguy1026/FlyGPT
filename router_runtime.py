from __future__ import annotations

from pathlib import Path
from typing import Any


class FlyRouterRuntime:
    """Load and run the trained FlyGraphRouter checkpoint on CPU."""

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
        self.scaffold_meta = dict(checkpoint.get("scaffold_meta", {}))

        src = checkpoint["src"].to(dtype=torch.long, device="cpu")
        dst = checkpoint["dst"].to(dtype=torch.long, device="cpu")
        base_weight = checkpoint["base_weight"].to(dtype=torch.float32, device="cpu")

        self.model = FlyGraphRouter(
            vocab_size=self.vocab_size,
            n_nodes=self.n_nodes,
            n_classes=len(self.routes),
            src=src,
            dst=dst,
            base_weight=base_weight,
            steps=self.steps,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, text: str, *, top_k: int = 3) -> dict[str, Any]:
        torch = self._torch
        x = self._vectorize(text, self.vocab_size).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0]

        k = max(1, min(int(top_k), len(self.routes)))
        values, indices = torch.topk(probabilities, k=k)

        ranked = [
            {
                "route": self.routes[int(index)],
                "confidence": float(value),
            }
            for value, index in zip(values.tolist(), indices.tolist())
        ]

        return {
            "route": ranked[0]["route"],
            "confidence": ranked[0]["confidence"],
            "top_routes": ranked,
            "n_nodes": self.n_nodes,
            "steps": self.steps,
            "best_val_accuracy": self.best_val_accuracy,
            "scaffold_kind": self.scaffold_meta.get("kind", "unknown"),
        }
