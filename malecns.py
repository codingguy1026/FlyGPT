from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import glob
import os
import tempfile

import duckdb

from connectome import NT_COLUMNS


EXPECTED_COLUMNS = {"body_pre", "body_post", "weight"}

NT_NAME_TO_COLUMN = {
    "gaba": "gaba_avg",
    "acetylcholine": "ach_avg",
    "ach": "ach_avg",
    "glutamate": "glut_avg",
    "glut": "glut_avg",
    "octopamine": "oct_avg",
    "serotonin": "ser_avg",
    "5-ht": "ser_avg",
    "dopamine": "da_avg",
    "da": "da_avg",
}


@dataclass
class MaleCNSConnectome:
    """Query MaleCNS v1.0 Parquet parts with DuckDB.

    The bulk MaleCNS release stores connectivity separately from neuron
    annotations and neurotransmitter predictions. FlyGPT keeps the large
    connectivity graph split across Parquet files and exposes a drop-in query
    surface similar to FlyWireConnectome.
    """

    source: str | Path
    metadata_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.source = Path(self.source).expanduser()
        if self.metadata_dir is not None:
            self.metadata_dir = Path(self.metadata_dir).expanduser()
        elif self.source.is_dir():
            sibling = self.source.parent / "metadata"
            self.metadata_dir = sibling if sibling.is_dir() else None

        self.connection = duckdb.connect(database=":memory:")

        temp_dir = Path(tempfile.gettempdir()) / "flygpt_duckdb"
        temp_dir.mkdir(parents=True, exist_ok=True)
        memory_limit = os.environ.get("FLYGPT_MEMORY_LIMIT", "1GB")
        threads = int(os.environ.get("FLYGPT_THREADS", "2"))
        escaped_temp_dir = temp_dir.as_posix().replace("'", "''")
        escaped_memory_limit = memory_limit.replace("'", "''")
        self.connection.execute(f"SET memory_limit='{escaped_memory_limit}'")
        self.connection.execute(f"SET threads={max(1, threads)}")
        self.connection.execute("SET preserve_insertion_order=false")
        self.connection.execute(f"SET temp_directory='{escaped_temp_dir}'")

        self._files = self._resolve_files()
        self._create_view()
        self._validate_schema()
        self._has_annotations = False
        self._has_neurotransmitters = False
        self._create_metadata_views()

    def _resolve_files(self) -> list[Path]:
        source_text = str(self.source)

        if any(token in source_text for token in ("*", "?", "[")):
            files = [Path(path) for path in sorted(glob.glob(source_text))]
        elif self.source.is_dir():
            files = sorted(self.source.glob("malecns_weights_part_*.parquet"))
        elif self.source.is_file():
            files = [self.source]
        else:
            files = []

        if not files:
            raise FileNotFoundError(
                f"No MaleCNS Parquet files found for {self.source!s}. "
                "Run bootstrap_malecns.py or point MALECNS_DATA_DIR at the parts directory."
            )

        return files

    def _create_view(self) -> None:
        file_literals = ", ".join(
            "'" + path.resolve().as_posix().replace("'", "''") + "'"
            for path in self._files
        )
        self.connection.execute(
            f"""
            CREATE OR REPLACE VIEW malecns_raw AS
            SELECT *
            FROM read_parquet([{file_literals}], union_by_name = true)
            """
        )
        self.connection.execute(
            """
            CREATE OR REPLACE VIEW connections AS
            SELECT
                body_pre::UBIGINT AS pre_pt_root_id,
                body_post::UBIGINT AS post_pt_root_id,
                weight::BIGINT AS syn_count
            FROM malecns_raw
            """
        )

    def _validate_schema(self) -> None:
        columns = {
            row[0]
            for row in self.connection.execute(
                "DESCRIBE SELECT * FROM malecns_raw"
            ).fetchall()
        }
        missing = EXPECTED_COLUMNS - columns
        if missing:
            raise ValueError(
                "Unexpected MaleCNS schema. Missing columns: "
                + ", ".join(sorted(missing))
            )

    def _create_metadata_views(self) -> None:
        if self.metadata_dir is None:
            return

        metadata_dir = Path(self.metadata_dir)
        annotations = metadata_dir / "annotations.parquet"
        neurotransmitters = metadata_dir / "neurotransmitters.parquet"

        if annotations.is_file():
            path = annotations.resolve().as_posix().replace("'", "''")
            self.connection.execute(
                f"CREATE OR REPLACE VIEW annotations AS SELECT * FROM read_parquet('{path}')"
            )
            self._has_annotations = True

        if neurotransmitters.is_file():
            path = neurotransmitters.resolve().as_posix().replace("'", "''")
            self.connection.execute(
                f"CREATE OR REPLACE VIEW neurotransmitters AS SELECT * FROM read_parquet('{path}')"
            )
            self._has_neurotransmitters = True

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(self._files)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MaleCNSConnectome":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @staticmethod
    def _dict_rows(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def stats(self) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS rows,
                APPROX_COUNT_DISTINCT(pre_pt_root_id) AS approx_presynaptic_neurons,
                APPROX_COUNT_DISTINCT(post_pt_root_id) AS approx_postsynaptic_neurons,
                SUM(syn_count) AS synapses
            FROM connections
            """
        ).fetchone()

        assert row is not None
        return {
            "dataset": "MaleCNS v1.0",
            "rows": row[0],
            "approx_presynaptic_neurons": row[1],
            "approx_postsynaptic_neurons": row[2],
            "neuropils": 0,
            "synapses": row[3],
            "parts": len(self._files),
        }

    def neuron(
        self,
        root_id: int,
        *,
        direction: str = "both",
        neuropil: str | None = None,
        min_synapses: int = 1,
        limit: int = 25,
    ) -> dict[str, list[dict[str, Any]]]:
        if direction not in {"inputs", "outputs", "both"}:
            raise ValueError("direction must be 'inputs', 'outputs', or 'both'")
        if min_synapses < 1:
            raise ValueError("min_synapses must be at least 1")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        result: dict[str, list[dict[str, Any]]] = {}

        if direction in {"outputs", "both"}:
            rows = self._partners(
                root_id=root_id,
                root_column="pre_pt_root_id",
                partner_column="post_pt_root_id",
                min_synapses=min_synapses,
                limit=limit,
            )
            self._enrich_nt(rows, presynaptic_body_id=int(root_id))
            result["outputs"] = rows

        if direction in {"inputs", "both"}:
            rows = self._partners(
                root_id=root_id,
                root_column="post_pt_root_id",
                partner_column="pre_pt_root_id",
                min_synapses=min_synapses,
                limit=limit,
            )
            self._enrich_nt(rows, presynaptic_key="partner_root_id")
            result["inputs"] = rows

        return result

    def _partners(
        self,
        *,
        root_id: int,
        root_column: str,
        partner_column: str,
        min_synapses: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            f"""
            SELECT
                {partner_column} AS partner_root_id,
                SUM(syn_count)::BIGINT AS syn_count,
                0::INTEGER AS neuropil_count
            FROM connections
            WHERE {root_column} = ?
            GROUP BY {partner_column}
            HAVING SUM(syn_count) >= ?
            ORDER BY syn_count DESC, partner_root_id
            LIMIT ?
            """,
            [int(root_id), int(min_synapses), int(limit)],
        )
        return self._dict_rows(cursor)

    def pair(self, pre_root_id: int, post_root_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT COALESCE(SUM(syn_count), 0)::BIGINT
            FROM connections
            WHERE pre_pt_root_id = ? AND post_pt_root_id = ?
            """,
            [int(pre_root_id), int(post_root_id)],
        ).fetchone()
        syn_count = int(row[0] if row else 0)

        by_neuropil: list[dict[str, Any]] = []
        if syn_count:
            item: dict[str, Any] = {
                "neuropil": "whole_cns",
                "syn_count": syn_count,
            }
            self._enrich_nt([item], presynaptic_body_id=int(pre_root_id))
            by_neuropil.append(item)

        return {
            "pre_root_id": int(pre_root_id),
            "post_root_id": int(post_root_id),
            "syn_count": syn_count,
            "by_neuropil": by_neuropil,
        }

    def top_connections(
        self,
        *,
        neuropil: str | None = None,
        min_synapses: int = 1,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT
                pre_pt_root_id,
                post_pt_root_id,
                SUM(syn_count)::BIGINT AS syn_count,
                0::INTEGER AS neuropil_count
            FROM connections
            GROUP BY pre_pt_root_id, post_pt_root_id
            HAVING SUM(syn_count) >= ?
            ORDER BY syn_count DESC, pre_pt_root_id, post_pt_root_id
            LIMIT ?
            """,
            [int(min_synapses), int(limit)],
        )
        rows = self._dict_rows(cursor)
        self._enrich_nt(rows, presynaptic_key="pre_pt_root_id")
        return rows

    def annotation(self, body_id: int) -> dict[str, Any] | None:
        if not self._has_annotations:
            return None
        cursor = self.connection.execute(
            "SELECT * FROM annotations WHERE bodyId = ? LIMIT 1",
            [int(body_id)],
        )
        rows = self._dict_rows(cursor)
        return rows[0] if rows else None

    def _enrich_nt(
        self,
        rows: list[dict[str, Any]],
        *,
        presynaptic_body_id: int | None = None,
        presynaptic_key: str | None = None,
    ) -> None:
        for row in rows:
            for column in NT_COLUMNS:
                row[column] = 0.0

        if not rows or not self._has_neurotransmitters:
            return

        if presynaptic_body_id is not None:
            body_ids = [int(presynaptic_body_id)]
        elif presynaptic_key is not None:
            body_ids = sorted({int(row[presynaptic_key]) for row in rows})
        else:
            return

        placeholders = ",".join("?" for _ in body_ids)
        cursor = self.connection.execute(
            f"""
            SELECT body, ANY_VALUE(consensus_nt) AS consensus_nt
            FROM neurotransmitters
            WHERE body IN ({placeholders})
            GROUP BY body
            """,
            body_ids,
        )
        mapping = {
            int(body): str(nt or "").strip().lower()
            for body, nt in cursor.fetchall()
        }

        for row in rows:
            if presynaptic_body_id is not None:
                body_id = int(presynaptic_body_id)
            else:
                assert presynaptic_key is not None
                body_id = int(row[presynaptic_key])
            column = NT_NAME_TO_COLUMN.get(mapping.get(body_id, ""))
            if column is not None:
                row[column] = 1.0

    def context_for_neuron(
        self,
        root_id: int,
        *,
        limit: int = 10,
        min_synapses: int = 1,
    ) -> str:
        annotation = self.annotation(root_id)
        partners = self.neuron(
            root_id,
            direction="both",
            min_synapses=min_synapses,
            limit=limit,
        )

        lines = [f"MaleCNS v1.0 neuron {root_id}"]
        if annotation:
            type_name = annotation.get("type")
            superclass = annotation.get("superclass")
            if type_name:
                lines.append(f"Type: {type_name}")
            if superclass:
                lines.append(f"Superclass: {superclass}")

        for label in ("outputs", "inputs"):
            lines.append(f"{label.title()}:")
            rows = partners.get(label, [])
            if not rows:
                lines.append("- none found")
                continue
            for row in rows:
                nt = max(NT_COLUMNS, key=lambda key: float(row.get(key) or 0.0))
                nt_prob = float(row.get(nt) or 0.0)
                nt_text = nt.removesuffix("_avg") if nt_prob > 0 else "unknown"
                lines.append(
                    f"- partner {row['partner_root_id']}: "
                    f"{row['syn_count']} synapses; presynaptic NT {nt_text}"
                )

        return "\n".join(lines)
