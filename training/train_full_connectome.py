"""Offline structural pretraining using ALL annotated MaleCNS connection rows.

This does NOT train chat language/etiquette and does NOT replace the 256-node
live router. The full weights Feather file is scanned every epoch without top-K
selection or truncation. Explicit glia and unannotated segments are excluded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_brain import DEFAULT_ANNOTATIONS, load_annotated_neurons
from full_connectome import DATASET, DEFAULT_PATH, FullMaleCNS


def neuron_indices(annotated: np.ndarray, body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map body IDs onto sorted curated IDs without inventing neurons."""
    if annotated.ndim != 1 or len(annotated) == 0:
        raise ValueError("A nonempty, one-dimensional curated neuron index is required")
    if np.any(annotated[1:] <= annotated[:-1]):
        raise ValueError("Annotated neuron IDs must be unique and sorted")
    positions = np.searchsorted(annotated, body_ids)
    valid = positions < len(annotated)
    valid[valid] = annotated[positions[valid]] == body_ids[valid]
    return positions, valid


def train_full_graph(
    *,
    weights_path: Path = DEFAULT_PATH,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    output_path: Path,
    epochs: int = 1,
    embedding_dim: int = 16,
    batch_size: int = 32768,
    learning_rate: float = 0.01,
    seed: int = 42,
    progress_every: int = 250,
) -> dict:
    if epochs < 1 or embedding_dim < 1 or batch_size < 1:
        raise ValueError("epochs, embedding_dim and batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if progress_every < 0:
        raise ValueError("progress_every must be nonnegative")
    import torch
    import torch.nn.functional as F

    ids = np.asarray(load_annotated_neurons(annotations_path), dtype=np.int64)
    n = len(ids)
    if n < 2:
        raise ValueError("At least two annotated neurons are needed for negative sampling")
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    pre_embed = torch.nn.Embedding(n, embedding_dim, sparse=True)
    post_embed = torch.nn.Embedding(n, embedding_dim, sparse=True)
    torch.nn.init.normal_(pre_embed.weight, std=0.05)
    torch.nn.init.normal_(post_embed.weight, std=0.05)
    optimizer = torch.optim.SparseAdam(
        list(pre_embed.parameters()) + list(post_embed.parameters()),
        lr=learning_rate,
    )

    reports = []
    touched = np.zeros(n, dtype=np.bool_)
    for epoch in range(1, epochs + 1):
        scanned = 0
        eligible = 0
        steps = 0
        total_loss = 0.0
        with FullMaleCNS(weights_path) as graph:
            for batch_no, batch in enumerate(graph._batches(), start=1):
                scanned += batch.num_rows
                src = batch.column(batch.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False)
                dst = batch.column(batch.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False)
                weight = batch.column(batch.schema.get_field_index("weight")).to_numpy(zero_copy_only=False)
                src_idx, src_ok = neuron_indices(ids, src)
                dst_idx, dst_ok = neuron_indices(ids, dst)
                keep = src_ok & dst_ok & (weight > 0)
                if not np.any(keep):
                    continue
                src_idx, dst_idx, weight = src_idx[keep], dst_idx[keep], weight[keep]
                touched[src_idx] = True
                touched[dst_idx] = True
                eligible += len(src_idx)
                for offset in range(0, len(src_idx), batch_size):
                    end = offset + batch_size
                    pre = torch.from_numpy(src_idx[offset:end].astype(np.int64, copy=False))
                    post = torch.from_numpy(dst_idx[offset:end].astype(np.int64, copy=False))
                    weights = torch.from_numpy(weight[offset:end].astype(np.float32, copy=False))
                    negative = torch.randint(n, post.shape, generator=rng)
                    negative = (negative + (negative == post).long()) % n
                    pos_score = (pre_embed(pre) * post_embed(post)).sum(-1)
                    neg_score = (pre_embed(pre) * post_embed(negative)).sum(-1)
                    strength = 1 + torch.log1p(weights).clamp(max=20) / 20
                    loss = ((F.softplus(-pos_score) + F.softplus(neg_score)) * strength).mean()
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("Non-finite structural training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach())
                    steps += 1
                if progress_every and batch_no % progress_every == 0:
                    print(
                        f"[full-train] epoch={epoch}/{epochs} batches={batch_no} "
                        f"rows={scanned:,} eligible_edges={eligible:,}",
                        file=sys.stderr, flush=True,
                    )
        if not eligible:
            raise RuntimeError("No positive connections between annotated neurons")
        reports.append({
            "epoch": epoch, "rows_scanned": scanned,
            "eligible_directed_connection_rows": eligible,
            "optimizer_steps": steps,
            "mean_training_loss": round(total_loss / steps, 6),
        })

    metadata = {
        "dataset": DATASET,
        "training_kind": "offline_self_supervised_directed_edge_reconstruction",
        "annotated_neurons_eligible": n,
        "annotated_neurons_with_positive_edges": int(touched.sum()),
        "isolated_or_unseen_annotated_neurons": int((~touched).sum()),
        "no_topk_or_node_sampling": True,
        "full_source_scanned_every_epoch": True,
        "training_epochs_completed": epochs,
        "epoch_reports": reports,
        "integrated_into_chat": False,
        "trained_for_natural_language": False,
        "biological_firing_model": False,
        "note": "Structural representations only; etiquette requires labeled task supervision.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": metadata,
        "body_ids": torch.from_numpy(ids.copy()),
        "presynaptic_embeddings": pre_embed.weight.detach().cpu(),
        "postsynaptic_embeddings": post_embed.weight.detach().cpu(),
    }, output_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out", type=Path, default=Path("artifacts/malecns_full_structural_embeddings.pt"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    report = train_full_graph(
        weights_path=args.weights,
        annotations_path=args.annotations,
        output_path=args.out,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
