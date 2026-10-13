"""Small deterministic tests of true all-batch MaleCNS graph propagation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from full_brain import FullBrainPropagator, load_annotated_neurons

ARROW = importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(ARROW, "Install pyarrow to run full-brain tests")
class FullBrainTests(unittest.TestCase):
    def setUp(self) -> None:
        import pyarrow as pa
        import pyarrow.feather as feather

        self.temp = tempfile.TemporaryDirectory()
        directory = Path(self.temp.name)
        self.weights = directory / "weights.feather"
        self.annotations = directory / "annotations.feather"

        # Split into tiny record batches to verify no row/chunk is skipped.
        # 4 is an unannotated fragment; 5 is explicitly a glial cell.
        feather.write_feather(
            pa.table({
                "body_pre": pa.array([1, 1, 1, 2, 3, 2, 1, 4], type=pa.int64()),
                "body_post": pa.array([2, 2, 3, 1, 1, 2, 4, 1], type=pa.int64()),
                "weight": pa.array([2, 3, 5, 4, 6, 0, 100, 200], type=pa.int64()),
            }),
            self.weights,
            chunksize=2,
        )
        feather.write_feather(
            pa.table({
                "bodyId": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
                "superclass": ["cb_intrinsic", "cb_intrinsic", "vnc_motor", None, "glia"],
            }),
            self.annotations,
        )
        self.brain = FullBrainPropagator(
            weights_path=self.weights, annotations_path=self.annotations
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_annotation_filter(self) -> None:
        self.assertEqual(load_annotated_neurons(self.annotations), (1, 2, 3))

    def test_propagate_three_steps_scans_every_edge_each_time(self) -> None:
        result = self.brain.propagate(1, steps=3, top=3, progress_every=0)
        self.assertEqual(result["annotated_neurons_eligible"], 3)
        self.assertEqual(result["steps_completed"], 3)
        self.assertEqual(result["rows_scanned_total"], 24)
        self.assertTrue(result["uses_all_eligible_neurons"])
        self.assertFalse(result["integrated_into_chat"])
        first, second, third = result["stages"]
        self.assertEqual(first["contributing_connection_rows"], 3)
        self.assertEqual(first["active_neurons"], 2)
        self.assertEqual(
            first["top"],
            [
                {"body_id": 2, "relative_activity": 1.0},
                {"body_id": 3, "relative_activity": 1.0},
            ],
        )
        self.assertEqual(second["active_neurons"], 1)
        self.assertEqual(second["top"], [{"body_id": 1, "relative_activity": 1.0}])
        self.assertEqual(third["top"], first["top"])
        # Body 4 (unannotated) contributed no artificial giant weight.
        self.assertEqual(first["contributing_connection_rows"], 3)

    def test_explicit_min_weight_filter(self) -> None:
        result = self.brain.propagate(1, steps=1, min_weight=4)
        self.assertEqual(result["stages"][0]["contributing_connection_rows"], 2)
        self.assertEqual(result["stages"][0]["active_neurons"], 2)

    def test_seed_must_be_curated(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a curated neuronal"):
            self.brain.propagate(4, steps=1)

    def test_validation_no_silent_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than the full"):
            self.brain.propagate(1, steps=1, max_active=2)
        with self.assertRaisesRegex(ValueError, "steps must"):
            self.brain.propagate(1, steps=0)
        with self.assertRaisesRegex(ValueError, "top must"):
            self.brain.propagate(1, top=0)

    def test_missing_annotations_fails_explicitly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_annotated_neurons(Path(self.temp.name) / "absent.feather")


if __name__ == "__main__":
    unittest.main()
