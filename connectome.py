from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import glob

import duckdb


EXPECTED_COLUMNS = {
    "pre_pt_root_id",
    "post_pt_root_id",
    "neuropil",
    "syn_count",
    "gaba_avg",
    "ach_avg",
    "glut_avg",
    "oct_avg",
    "ser_avg",
    "da_avg",
}

NT_COLUMNS = ("gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg")


def _weighted_nt_sql() -> str:
    return ",\n".join(
        f"SUM({column} * syn_count) / NULLIF(SUM(syn_count), 0) AS {column}"
        for column in NT_COLUMNS
    )


@dataclass
class FlyWireConnectome:
    """Query FlyWire v783 proofread connection Parquet files with DuckDB.

    The loader never pulls the full dataset into Python memory. DuckDB scans the
    Parquet parts lazily and only returns the rows needed for each query.
    """

    source: str | Path

    def __post_init__(self) -> None:
        self.source = Path(self.source).expanduser()
        self.connection = duckdb.connect(database=":memory:")
        self._files = self._resolve_files()
        self._create_view()
        self._validate_schema()

    def _resolve_files(self) -> list[Path]:
        source_text = str(self.source)

        if any(token in source_text for token in ("*", "?", "[")):
            files = [Path(path) for path in sorted(glob.glob(source_text))]
        elif self.source.is_dir():
            files = sorted(self.source.glob("proofread_connections_783_part_*.parquet"))
            if not files:
                files = sorted(self.source.glob("*.parquet"))
        elif self.source.is_file():
            files = [self.source]
        else:
            files = []

        if not files:
            raise FileNotFoundError(
                f"No Parquet files found for {self.source!s}. "
                "Point FLYWIRE_DATA_DIR at the folder containing the v783 parts."
            )

        return files

    def _create_view(self) -> None:
        # Passing the filenames as a DuckDB list avoids manually UNIONing nine
        # files and keeps scans predicate-aware.
        file_literals = ", ".join(
            "'" + path.resolve().as_posix().replace("'", "''") + "'"
            for path in self._files
        )
        self.connection.execute(
            f"""
            CREATE OR REPLACE VIEW connections AS
            SELECT *
            FROM read_parquet([{file_literals}], union_by_name = true)
            """
        )

    def _validate_schema(self) -> None:
        columns = {
            row[0]
            for row in self.connection.execute(
                "DESCRIBE SELECT * FROM connections"
            ).fetchall()
        }
        missing = EXPECTED_COLUMNS - columns
        if missing:
            raise ValueError(
                "Unexpected FlyWire schema. Missing columns: "
                + ", ".join(sorted(missing))
            )

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(self._files)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "FlyWireConnectome":
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
                COUNT(DISTINCT neuropil) AS neuropils,
                SUM(syn_count) AS synapses
            FROM connections
            """
        ).fetchone()

        assert row is not None
        return {
            "rows": row[0],
            "approx_presynaptic_neurons": row[1],
            "approx_postsynaptic_neurons": row[2],
            "neuropils": row[3],
            "synapses": row[4],
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
        """Return strongest input/output partners for one proofread neuron."""

        if direction not in {"inputs", "outputs", "both"}:
            raise ValueError("direction must be 'inputs', 'outputs', or 'both'")
        if min_synapses < 1:
            raise ValueError("min_synapses must be at least 1")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        result: dict[str, list[dict[str, Any]]] = {}

        if direction in {"outputs", "both"}:
            result["outputs"] = self._partners(
                root_id=root_id,
                root_column="pre_pt_root_id",
                partner_column="post_pt_root_id",
                neuropil=neuropil,
                min_synapses=min_synapses,
                limit=limit,
            )

        if direction in {"inputs", "both"}:
            result["inputs"] = self._partners(
                root_id=root_id,
                root_column="post_pt_root_id",
                partner_column="pre_pt_root_id",
                neuropil=neuropil,
                min_synapses=min_synapses,
                limit=limit,
            )

        return result

    def _partners(
        self,
        *,
        root_id: int,
        root_column: str,
        partner_column: str,
        neuropil: str | None,
        min_synapses: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        where = [f"{root_column} = ?"]
        params: list[Any] = [int(root_id)]

        if neuropil is not None:
            where.append("neuropil = ?")
            params.append(neuropil)

        params.extend([int(min_synapses), int(limit)])

        cursor = self.connection.execute(
            f"""
            SELECT
                {partner_column} AS partner_root_id,
                SUM(syn_count)::BIGINT AS syn_count,
                COUNT(DISTINCT neuropil)::INTEGER AS neuropil_count,
                {_weighted_nt_sql()}
            FROM connections
            WHERE {" AND ".join(where)}
            GROUP BY {partner_column}
            HAVING SUM(syn_count) >= ?
            ORDER BY syn_count DESC, partner_root_id
            LIMIT ?
            """,
            params,
        )
        return self._dict_rows(cursor)

    def pair(self, pre_root_id: int, post_root_id: int) -> dict[str, Any]:
        """Inspect one directed neuron-to-neuron connection."""

        cursor = self.connection.execute(
            f"""
            SELECT
                neuropil,
                SUM(syn_count)::BIGINT AS syn_count,
                {_weighted_nt_sql()}
            FROM connections
            WHERE pre_pt_root_id = ? AND post_pt_root_id = ?
            GROUP BY neuropil
            ORDER BY syn_count DESC, neuropil
            """,
            [int(pre_root_id), int(post_root_id)],
        )
        by_neuropil = self._dict_rows(cursor)

        return {
            "pre_root_id": int(pre_root_id),
            "post_root_id": int(post_root_id),
            "syn_count": sum(row["syn_count"] for row in by_neuropil),
            "by_neuropil": by_neuropil,
        }

    def top_connections(
        self,
        *,
        neuropil: str | None = None,
        min_synapses: int = 1,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Return the strongest directed neuron pairs."""

        where = []
        params: list[Any] = []

        if neuropil is not None:
            where.append("neuropil = ?")
            params.append(neuropil)

        where_sql = "WHERE " + " AND ".join(where) if where else ""
        params.extend([int(min_synapses), int(limit)])

        cursor = self.connection.execute(
            f"""
            SELECT
                pre_pt_root_id,
                post_pt_root_id,
                SUM(syn_count)::BIGINT AS syn_count,
                COUNT(DISTINCT neuropil)::INTEGER AS neuropil_count,
                {_weighted_nt_sql()}
            FROM connections
            {where_sql}
            GROUP BY pre_pt_root_id, post_pt_root_id
            HAVING SUM(syn_count) >= ?
            ORDER BY syn_count DESC, pre_pt_root_id, post_pt_root_id
            LIMIT ?
            """,
            params,
        )
        return self._dict_rows(cursor)

    def context_for_neuron(
        self,
        root_id: int,
        *,
        limit: int = 10,
        min_synapses: int = 1,
    ) -> str:
        """Create compact text that can be handed to an LLM as grounded context."""

        partners = self.neuron(
            root_id,
            direction="both",
            min_synapses=min_synapses,
            limit=limit,
        )

        lines = [f"FlyWire v783 neuron {root_id}"]

        for label in ("outputs", "inputs"):
            lines.append(f"{label.title()}:")
            rows = partners.get(label, [])
            if not rows:
                lines.append("- none found")
                continue

            for row in rows:
                nt = max(
                    NT_COLUMNS,
                    key=lambda key: float(row.get(key) or 0.0),
                )
                nt_name = nt.removesuffix("_avg")
                nt_prob = float(row.get(nt) or 0.0)
                lines.append(
                    f"- partner {row['partner_root_id']}: "
                    f"{row['syn_count']} synapses across "
                    f"{row['neuropil_count']} neuropil(s); "
                    f"highest mean NT probability {nt_name}={nt_prob:.3f}"
                )

        return "\n".join(lines)
