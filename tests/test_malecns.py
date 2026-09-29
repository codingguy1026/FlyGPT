from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from bootstrap_malecns import split_weights
from malecns import MaleCNSConnectome


class MaleCNSConnectomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.parts = self.root / "parts"
        self.metadata = self.root / "metadata"
        self.parts.mkdir()
        self.metadata.mkdir()

        con = duckdb.connect(database=":memory:")
        con.execute(
            """
            CREATE TABLE sample AS
            SELECT * FROM (
                VALUES
                    (1::BIGINT, 2::BIGINT, 5::BIGINT),
                    (1::BIGINT, 3::BIGINT, 2::BIGINT),
                    (3::BIGINT, 1::BIGINT, 7::BIGINT),
                    (2::BIGINT, 1::BIGINT, 4::BIGINT)
            ) AS t(body_pre, body_post, weight)
            """
        )
        output = self.parts / "malecns_weights_part_001.parquet"
        con.execute("COPY sample TO ? (FORMAT PARQUET)", [str(output)])
        con.close()

        annotations = pa.table(
            {
                "bodyId": pa.array([1, 2, 3], type=pa.int64()),
                "type": ["DNge104_R", "partner_2", "partner_3"],
                "superclass": ["descending", "intrinsic", "intrinsic"],
            }
        )
        pq.write_table(annotations, self.metadata / "annotations.parquet")

        neurotransmitters = pa.table(
            {
                "body": pa.array([1, 2, 3], type=pa.int64()),
                "consensus_nt": ["acetylcholine", "gaba", "glutamate"],
            }
        )
        pq.write_table(
            neurotransmitters,
            self.metadata / "neurotransmitters.parquet",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_stats(self) -> None:
        with MaleCNSConnectome(self.parts, self.metadata) as connectome:
            stats = connectome.stats()
        self.assertEqual(stats["rows"], 4)
        self.assertEqual(stats["synapses"], 18)
        self.assertEqual(stats["parts"], 1)
        self.assertEqual(stats["dataset"], "MaleCNS v1.0")

    def test_output_and_input_partner_aggregation(self) -> None:
        with MaleCNSConnectome(self.parts, self.metadata) as connectome:
            outputs = connectome.neuron(1, direction="outputs")["outputs"]
            inputs = connectome.neuron(1, direction="inputs")["inputs"]

        self.assertEqual(outputs[0]["partner_root_id"], 2)
        self.assertEqual(outputs[0]["syn_count"], 5)
        self.assertEqual(outputs[0]["ach_avg"], 1.0)

        self.assertEqual(inputs[0]["partner_root_id"], 3)
        self.assertEqual(inputs[0]["syn_count"], 7)
        self.assertEqual(inputs[0]["glut_avg"], 1.0)

    def test_pair_and_annotation(self) -> None:
        with MaleCNSConnectome(self.parts, self.metadata) as connectome:
            pair = connectome.pair(1, 2)
            annotation = connectome.annotation(1)

        self.assertEqual(pair["syn_count"], 5)
        self.assertEqual(pair["by_neuropil"][0]["neuropil"], "whole_cns")
        self.assertEqual(pair["by_neuropil"][0]["ach_avg"], 1.0)
        self.assertIsNotNone(annotation)
        assert annotation is not None
        self.assertEqual(annotation["type"], "DNge104_R")

    def test_streaming_split_filters_curated_endpoints(self) -> None:
        raw = self.root / "weights.feather"
        table = pa.table(
            {
                "body_pre": pa.array([1, 1, 3, 2, 9], type=pa.int64()),
                "body_post": pa.array([2, 9, 1, 3, 1], type=pa.int64()),
                "weight": pa.array([5, 10, 2, 1, 8], type=pa.int64()),
            }
        )
        feather.write_feather(table, raw, chunksize=2)

        out = self.root / "split"
        result = split_weights(
            raw,
            out,
            parts=3,
            compression_level=3,
            keep_ids={1, 2, 3},
            min_weight=2,
            force=False,
        )

        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["synapses"], 7)

        with MaleCNSConnectome(out) as connectome:
            stats = connectome.stats()

        self.assertEqual(stats["rows"], 2)
        self.assertEqual(stats["synapses"], 7)


if __name__ == "__main__":
    unittest.main()
