"""Reproduce the 93.16%-study group split with the real full-neuron architecture.

The old 93.16% was a 3-seed group-heldout average for a different 256-node
residual router. This study uses *the same teacher corpus and group partition
algorithm* to evaluate a different model trained from scratch on all curated
MaleCNS structural keys. It DOES NOT guarantee the previous score.

Optionally score archived 256-node checkpoints for a paired reference. Full
population checkpoints are retained separately by seed for honest follow-up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training.full_population_router import (
    load_all_structural_keys, prepare_teacher, train_one,
)
from training.grouped_corpus import grouped_three_way_split
from training.train_router import (
    ResidualFlyGraphRouter, load_examples, batchify,
    _accuracy_from_logits, _balanced_accuracy_from_logits,
)


def score_old_reference(path: Path, test) -> dict:
    """Re-evaluate the exact frozen older model on the corresponding seed's holdout."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if (ckpt.get("format") != "flygpt-research-residual-v1"
            or ckpt.get("architecture") != "ResidualFlyGraphRouter"
            or int(ckpt["n_nodes"]) != 256
            or int(ckpt["steps"]) != 3
            or ckpt.get("vectorizer_version") != "v6"):
        raise ValueError("Baseline is not the original verified 256-node residual model")
    classes = list(ckpt["routes"])
    model = ResidualFlyGraphRouter(
        int(ckpt["vocab_size"]), int(ckpt["n_nodes"]), len(classes),
        ckpt["src"], ckpt["dst"], ckpt["base_weight"], steps=3
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    features, targets = batchify(test, {x: i for i, x in enumerate(classes)},
                                 int(ckpt["vocab_size"]), "v6")
    with torch.no_grad():
        logits = model(features)
    return {
        "accuracy": round(_accuracy_from_logits(logits, targets), 6),
        "macro_recall": round(_balanced_accuracy_from_logits(
            logits, targets, len(classes)
        ), 6),
        "test_rows": len(test),
    }


def run(args):
    if sorted(set(args.seeds)) != args.seeds:
        raise ValueError("Seeds must be sorted and unique")
    torch.set_num_threads(2)
    ids, keys, meta = load_all_structural_keys(args.structural)
    if int(meta["annotated_neurons_eligible"]) != len(ids):
        raise ValueError("Not all real MaleCNS neurons were loaded")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="full-pop-benchmark-") as td:
        path = Path(td) / "teacher.jsonl"
        prepare_teacher(path)
        text_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        examples = load_examples(path)
        for seed in args.seeds:
            train, val, test, group_audit = grouped_three_way_split(examples, seed)
            print(
                f"SPLIT seed={seed} train={len(train)} val={len(val)} "
                f"test={len(test)} grouped={group_audit['n_clusters']}",
                flush=True,
            )
            # Label leakage protection: test isn't passed into checkpoint
            # selection and is evaluated only after the selected epoch.
            model, details = train_one(
                path, ids, keys, seed,
                args.epochs, args.patience, args.batch_size,
                explicit_split=(train, val, test),
            )
            reference = None
            if args.baseline_dir:
                reference = score_old_reference(
                    args.baseline_dir / f"male_cns_seed{seed}.pt", test
                )
            checkpoint = args.output / f"malecns_all_neurons_seed{seed}.pt"
            torch.save({
                "format": "flygpt-full-population-v1",
                "architecture": "FullPopulationRouter",
                "model_state": model.state_dict(),
                "routes": details["routes"], "vocab_size": 8192,
                "vectorizer_version": "v6",
                "n_nodes": len(ids), "neuron_ids": ids,
                "training_summary": details,
                "scaffold_meta": {
                    "kind": "male-cns:v1.0",
                    "source": "Janelia MaleCNS v1.0 full neuronal connectome",
                    "full_connectome_trained": True,
                    "full_source_scanned_every_epoch": True,
                    "n_neurons": len(ids),
                    "n_positive_connection_rows":
                        meta["epoch_reports"][-1]["eligible_directed_connection_rows"],
                    "full_neuron_population_attended": True,
                    "representation":
                        "exact full-neuron linear attention over full-graph structural embeddings",
                    "live_edge_propagation": False,
                },
                "structural_weights_sha256":
                    hashlib.sha256(args.structural.read_bytes()).hexdigest(),
                "source_teacher_sha256": text_sha,
                "seed": seed,
                "experimental_only": True,
            }, checkpoint)
            test_metrics = details["test_metrics"]
            result = {
                "seed": seed, "split": group_audit,
                "test": test_metrics,
                "train_examples": len(train),
                "validation_macro": details["best_val_macro_recall"],
                "old_256_reference": reference,
                "checkpoint": checkpoint.name,
            }
            results.append(result)
            print(
                f"RESULT seed={seed} neurons={len(ids)} "
                f"full_test_macro={test_metrics['macro_recall']:.5f} "
                f"full_test_accuracy={test_metrics['accuracy']:.5f} "
                f"baseline_256_macro={reference['macro_recall'] if reference else 'unavailable'}",
                flush=True,
            )

    def macro_mean(path):
        values = [r[path]["macro_recall"] for r in results if r[path] is not None]
        return round(statistics.mean(values), 6) if values else None

    mean_full = macro_mean("test")
    mean_256 = macro_mean("old_256_reference")
    report = {
        "name": "Same splits as 93.16%-internal-baseline: actual all-neuron study",
        "training_kind": "full official directed edge structural pretrain + supervised routing",
        "neuron_count": len(ids),
        "structural_connection_rows": meta["epoch_reports"][-1]["eligible_directed_connection_rows"],
        "structural_neurons_with_edges": meta["annotated_neurons_with_positive_edges"],
        "source_training_corpus_sha256": text_sha,
        "structural_artifact_sha256": hashlib.sha256(args.structural.read_bytes()).hexdigest(),
        "seeds": args.seeds,
        "samples": len(examples),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "results": results,
        "macro_recall_mean": {
            "actual_all_annotated_neurons": mean_full,
            "frozen_256_neuron_reference": mean_256,
            "historical_256_mean": .9316,
            "target_reached_on_internal_grouped_test": mean_full >= .9316,
        },
        "interpretation": [
            "This 93.16%-style metric is a teacher-corpus grouped holdout, not real-user chat accuracy.",
            "All-neuron route classifier trained from scratch, not the old weights expanded.",
            "Structure learned by directed-edge reconstruction, not simulated biological firing.",
            "All curated neurons contribute to exact population attention, but online inference does not simulate every synapse.",
            "The 300-question public diagnostic was never used for training or selection.",
            "Three seeds exploratory; no statistically significant results implied.",
            "The full model is not the original 256-node model and has separate weights.",
        ],
        "production_modified": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY all_neurons_mean=" + str(mean_full) +
          " frozen_256_reference_mean=" + str(mean_256), flush=True)
    print("REPORT=" + str(args.report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structural", type=Path,
                   default=ROOT / "artifacts/malecns_full_structural_embeddings.pt")
    p.add_argument("--baseline-dir", type=Path,
                   help="Archived seed-specific 256-neuron checkpoints, optional")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=Path, default=Path("/tmp/full-pop-models"))
    p.add_argument("--report", type=Path, default=Path("/tmp/full-pop-report.json"))
    args = p.parse_args()
    if not args.seeds or args.epochs < 2 or args.patience < 1 or args.batch_size < 1:
        p.error("Invalid training configuration")
    run(args)


if __name__ == "__main__":
    main()
