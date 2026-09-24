from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

from connectome import FlyWireConnectome


class FlyWireConnectomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name)

        con = duckdb.connect(database=":memory:")
        con.execute(
            """
            CREATE TABLE sample AS
            SELECT * FROM (
                VALUES
                    (1::UBIGINT, 2::UBIGINT, 'AL_L', 3, 0.1, 0.7, 0.1, 0.0, 0.0, 0.1),
                    (1::UBIGINT, 2::UBIGINT, 'LH_L', 2, 0.2, 0.6, 0.1, 0.0, 0.0, 0.1),
                    (3::UBIGINT, 1::UBIGINT, 'AL_L', 5, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0)
            ) AS t(
                pre_pt_root_id,
                post_pt_root_id,
                neuropil,
                syn_count,
                gaba_avg,
                ach_avg,
                glut_avg,
                oct_avg,
                ser_avg,
                da_avg
            )
            """
        )
        output = self.data_dir / "proofread_connections_783_part_01.parquet"
        con.execute("COPY sample TO ? (FORMAT PARQUET)", [str(output)])
        con.close()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_stats(self) -> None:
        with FlyWireConnectome(self.data_dir) as connectome:
            stats = connectome.stats()
        self.assertEqual(stats["rows"], 3)
        self.assertEqual(stats["parts"], 1)
        self.assertEqual(stats["synapses"], 10)

    def test_output_partner_aggregation(self) -> None:
        with FlyWireConnectome(self.data_dir) as connectome:
            outputs = connectome.neuron(1, direction="outputs")["outputs"]
        self.assertEqual(outputs[0]["partner_root_id"], 2)
        self.assertEqual(outputs[0]["syn_count"], 5)
        self.assertEqual(outputs[0]["neuropil_count"], 2)

    def test_pair(self) -> None:
        with FlyWireConnectome(self.data_dir) as connectome:
            pair = connectome.pair(1, 2)
        self.assertEqual(pair["syn_count"], 5)
        self.assertEqual(len(pair["by_neuropil"]), 2)


if __name__ == "__main__":
    unittest.main()
