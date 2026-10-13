"""Offline smoke tests for the optional full MaleCNS Feather reader."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from full_connectome import FullMaleCNS

ARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(ARROW_AVAILABLE, "pip install -r requirements-full-connectome.txt")
class FullMaleCNSTests(unittest.TestCase):
    def setUp(self) -> None:
        import pyarrow as pa
        import pyarrow.feather as feather

        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "small.feather"
        feather.write_feather(
            pa.table(
                {
                    "body_pre": pa.array([1, 1, 1, 2, 3, 2], type=pa.int64()),
                    "body_post": pa.array([2, 2, 3, 1, 1, 2], type=pa.int64()),
                    "weight": pa.array([2, 3, 5, 4, 6, 0], type=pa.int64()),
                }
            ),
            str(self.path),
            chunksize=2,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_inspect_is_metadata_only(self) -> None:
        with FullMaleCNS(self.path) as graph:
            result = graph.inspect()
        self.assertEqual(result["columns"]["body_pre"], "int64")
        self.assertEqual(result["columns"]["weight"], "int64")
        self.assertGreaterEqual(result["record_batches"], 2)
        self.assertFalse(result["full_scan_performed"])

    def test_full_audit_counts_all_batches(self) -> None:
        with FullMaleCNS(self.path) as graph:
            result = graph.audit()
        self.assertEqual(result["rows_scanned"], 6)
        self.assertEqual(result["synapse_weight_sum"], 20)
        self.assertEqual(result["self_connections"], 1)
        self.assertEqual(result["nonpositive_weights"], 1)
        self.assertEqual(result["rows_with_nulls"], 0)
        self.assertTrue(result["full_scan_performed"])

    def test_outgoing_aggregates_repeated_pairs_across_batches(self) -> None:
        with FullMaleCNS(self.path) as graph:
            result = graph.neighbors(1, direction="outgoing", limit=1)
        self.assertEqual(result["matched_rows"], 3)
        self.assertEqual(result["distinct_directional_partners"], 2)
        self.assertEqual(
            result["returned"],
            [{"direction": "outgoing", "partner_body_id": 2, "synapses": 5}],
        )

    def test_incoming_and_pair_filter(self) -> None:
        with FullMaleCNS(self.path) as graph:
            incoming = graph.neighbors(1, direction="incoming")
            exact = graph.neighbors(
                1, direction="outgoing", partner_id=2, min_weight=3
            )
        self.assertEqual(incoming["returned"][0]["partner_body_id"], 3)
        self.assertEqual(incoming["returned"][0]["synapses"], 6)
        self.assertEqual(exact["matched_rows"], 1)
        self.assertEqual(exact["returned"][0]["synapses"], 3)

    def test_rejects_unsupported_schema(self) -> None:
        import pyarrow as pa
        import pyarrow.feather as feather

        broken = Path(self.temp.name) / "broken.feather"
        feather.write_feather(
            pa.table({"body_pre": pa.array([1]), "body_post": pa.array([2])}),
            str(broken),
        )
        with self.assertRaisesRegex(ValueError, "weight"):
            with FullMaleCNS(broken):
                pass

    def test_rejects_unbounded_limits(self) -> None:
        with FullMaleCNS(self.path) as graph:
            with self.assertRaises(ValueError):
                graph.neighbors(1, limit=0)
            with self.assertRaises(ValueError):
                graph.neighbors(1, min_weight=0)


if __name__ == "__main__":
    unittest.main()
