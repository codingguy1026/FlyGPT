"""No credentials or real 1GB Feather file needed for these unit tests."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from training.train_full_connectome import neuron_indices, train_full_graph


class IndexTests(unittest.TestCase):
    def test_unannotated_and_glia_are_excluded_from_index(self):
        idx, mask = neuron_indices(
            np.asarray([1, 3, 10]),
            np.asarray([0, 1, 3, 5, 10, 11]),
        )
        self.assertEqual(mask.tolist(), [False, True, True, False, True, False])
        self.assertEqual(idx[mask].tolist(), [0, 1, 2])

    def test_reject_unsorted_annotations(self):
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            neuron_indices(np.asarray([3, 1]), np.asarray([1]))


class FullStructuralTrainingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is an optional offline-training dependency")
    def test_streamed_training_scans_every_batch_and_counts_isolated_neurons(self):
        import torch

        class Column:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.int64)
            def to_numpy(self, zero_copy_only=False):
                return self.values

        class Schema:
            def get_field_index(self, key):
                return {"body_pre": 0, "body_post": 1, "weight": 2}[key]

        class Batch:
            schema = Schema()
            def __init__(self, src, dst, weight):
                self.columns = [Column(src), Column(dst), Column(weight)]
                self.num_rows = len(src)
            def column(self, index):
                return self.columns[index]

        class Graph:
            def __init__(self, path):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def _batches(self):
                yield Batch([1, 2, 4], [2, 3, 1], [3, 2, 9])
                yield Batch([3, 2, 1], [1, 5, 4], [4, 4, 8])

        with tempfile.TemporaryDirectory() as directory:
            with patch("training.train_full_connectome.load_annotated_neurons", return_value=(1, 2, 3, 6)), \
                 patch("training.train_full_connectome.FullMaleCNS", Graph):
                output = Path(directory) / "full.pt"
                meta = train_full_graph(
                    output_path=output, epochs=2, embedding_dim=4,
                    batch_size=2, progress_every=0,
                )
            self.assertEqual(meta["annotated_neurons_eligible"], 4)
            self.assertEqual(meta["annotated_neurons_with_positive_edges"], 3)
            self.assertEqual(meta["isolated_or_unseen_annotated_neurons"], 1)
            self.assertEqual(meta["training_epochs_completed"], 2)
            for report in meta["epoch_reports"]:
                self.assertEqual(report["rows_scanned"], 6)
                self.assertEqual(report["eligible_directed_connection_rows"], 3)
            ckpt = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(ckpt["body_ids"].tolist(), [1, 2, 3, 6])
            self.assertEqual(ckpt["presynaptic_embeddings"].shape, (4, 4))
            self.assertFalse(ckpt["metadata"]["integrated_into_chat"])


if __name__ == "__main__":
    unittest.main()
