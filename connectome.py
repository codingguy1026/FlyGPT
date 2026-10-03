from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from neuprint import Client, fetch_adjacencies, fetch_meta


DEFAULT_NEUPRINT_SERVER = "https://neuprint.janelia.org"
DEFAULT_NEUPRINT_DATASET = "male-cns:v1.0"

# Kept for API compatibility with older FlyGPT callers. MaleCNS v1.0 exposes
# a categorical consensus neurotransmitter per neuron rather than the six
# FlyWire per-edge probability columns.
NT_COLUMNS: tuple[str, ...] = ()

_NT_LABELS = {
    "acetylcholine": "ACH",
    "ach": "ACH",
    "gaba": "GABA",
    "glutamate": "GLUT",
    "glut": "GLUT",
    "dopamine": "DA",
    "da": "DA",
    "serotonin": "SER",
    "5ht": "SER",
    "octopamine": "OCT",
    "oct": "OCT",
    "tyramine": "TYR",
}


def _nt_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "UNKNOWN"
    return _NT_LABELS.get(text.lower(), text.upper())


@dataclass
class MaleCNSConnectome:
    """Query the Janelia MaleCNS v1.0 connectome through neuPrint.

    Interactive FlyGPT requests use targeted neuPrint queries, so the 1.1 GB
    flat connection graph does not need to live inside the app container.
    """

    token: str | None = None
    server: str = DEFAULT_NEUPRINT_SERVER
    dataset: str = DEFAULT_NEUPRINT_DATASET
    client: Any | None = None

    def __post_init__(self) -> None:
        self.server = os.environ.get("NEUPRINT_SERVER", self.server).strip()
        self.dataset = os.environ.get("NEUPRINT_DATASET", self.dataset).strip()
        self.token = (\n            self.token\n            or os.environ.get("NEUPRINT_TOKEN")\n            or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")\n            or ""\n        ).strip() or None

        if self.client is None:
            if not self.token:
                raise RuntimeError(
                    "MaleCNS access needs a neuPrint token. "
                    "Set NEUPRINT_TOKEN (or NEUPRINT_APPLICATION_CREDENTIALS) for neuprint.janelia.org."
                )
            self.client = Client(self.server, dataset=self.dataset, token=self.token)

    def close(self) -> None:
        # neuprint.Client uses requests sessions internally and does not require
        # callers to close it for normal command/API usage.
        return None

    def __enter__(self) -> "MaleCNSConnectome":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        meta = fetch_meta(client=self.client)
        neurons = self.client.fetch_custom(
            "MATCH (n:Neuron) RETURN count(n) AS neurons"
        )
        neuron_count = int(neurons.iloc[0]["neurons"]) if len(neurons) else 0
        primary_rois = meta.get("primaryRois") or []

        return {
            "dataset": meta.get("dataset") or self.dataset,
            "neurons": neuron_count,
            "presynaptic_sites": int(meta.get("totalPreCount") or 0),
            "postsynaptic_sites": int(meta.get("totalPostCount") or 0),
            "neuropils": len(primary_rois),
            "last_database_edit": meta.get("lastDatabaseEdit"),
        }

    @staticmethod
    def _nt_lookup(neurons_df: Any) -> dict[int, str]:
        if neurons_df is None or len(neurons_df) == 0 or "bodyId" not in neurons_df.columns:
            return {}

        result: dict[int, str] = {}
        has_nt = "consensusNt" in neurons_df.columns
        for row in neurons_df.to_dict("records"):
            body_id = row.get("bodyId")
            if body_id is None:
                continue
            result[int(body_id)] = _nt_label(row.get("consensusNt") if has_nt else None)
        return result

    def neuron(
        self,
        root_id: int,
        *,
        direction: str = "both",
        neuropil: str | None = None,
        min_synapses: int = 1,
        limit: int = 25,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return strongest input/output partners for one MaleCNS body."""

        if direction not in {"inputs", "outputs", "both"}:
            raise ValueError("direction must be 'inputs', 'outputs', or 'both'")
        if min_synapses < 1:
            raise ValueError("min_synapses must be at least 1")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        result: dict[str, list[dict[str, Any]]] = {}
        if direction in {"outputs", "both"}:
            result["outputs"] = self._partners(
                root_id=int(root_id),
                outgoing=True,
                neuropil=neuropil,
                min_synapses=min_synapses,
                limit=limit,
            )
        if direction in {"inputs", "both"}:
            result["inputs"] = self._partners(
                root_id=int(root_id),
                outgoing=False,
                neuropil=neuropil,
                min_synapses=min_synapses,
                limit=limit,
            )
        return result

    def _partners(
        self,
        *,
        root_id: int,
        outgoing: bool,
        neuropil: str | None,
        min_synapses: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        rois = [neuropil] if neuropil else None
        sources = [root_id] if outgoing else None
        targets = None if outgoing else [root_id]

        neurons_df, conn_df = fetch_adjacencies(
            sources=sources,
            targets=targets,
            rois=rois,
            min_total_weight=1,
            properties=["type", "instance", "consensusNt"],
            weight_props=["weight"],
            client=self.client,
        )

        if conn_df is None or len(conn_df) == 0:
            return []

        partner_col = "bodyId_post" if outgoing else "bodyId_pre"
        pre_col = "bodyId_pre"
        nt_lookup = self._nt_lookup(neurons_df)
        rows: list[dict[str, Any]] = []

        for partner_id, group in conn_df.groupby(partner_col, sort=False):
            syn_count = int(group["weight"].sum())
            if syn_count < min_synapses:
                continue

            pre_id = int(group.iloc[0][pre_col])
            roi_count = (
                int(group["roi"].dropna().nunique())
                if "roi" in group.columns
                else 0
            )
            rows.append(
                {
                    "partner_root_id": int(partner_id),
                    "syn_count": syn_count,
                    "neuropil_count": roi_count,
                    "dominant_nt": nt_lookup.get(pre_id, "UNKNOWN"),
                }
            )

        rows.sort(key=lambda row: (-int(row["syn_count"]), int(row["partner_root_id"])))
        return rows[:limit]

    def pair(self, pre_root_id: int, post_root_id: int) -> dict[str, Any]:
        """Inspect one directed MaleCNS neuron-to-neuron connection."""

        neurons_df, conn_df = fetch_adjacencies(
            sources=[int(pre_root_id)],
            targets=[int(post_root_id)],
            min_total_weight=1,
            properties=["type", "instance", "consensusNt"],
            weight_props=["weight"],
            client=self.client,
        )
        nt_lookup = self._nt_lookup(neurons_df)
        dominant_nt = nt_lookup.get(int(pre_root_id), "UNKNOWN")
        by_neuropil: list[dict[str, Any]] = []

        if conn_df is not None and len(conn_df):
            if "roi" in conn_df.columns:
                grouped = conn_df.groupby("roi", dropna=False, sort=False)
                for roi, group in grouped:
                    by_neuropil.append(
                        {
                            "neuropil": str(roi) if roi == roi else "NotPrimary",
                            "syn_count": int(group["weight"].sum()),
                            "dominant_nt": dominant_nt,
                        }
                    )
            else:
                by_neuropil.append(
                    {
                        "neuropil": "unknown",
                        "syn_count": int(conn_df["weight"].sum()),
                        "dominant_nt": dominant_nt,
                    }
                )

        by_neuropil.sort(key=lambda row: (-int(row["syn_count"]), str(row["neuropil"])))
        return {
            "pre_root_id": int(pre_root_id),
            "post_root_id": int(post_root_id),
            "syn_count": sum(int(row["syn_count"]) for row in by_neuropil),
            "by_neuropil": by_neuropil,
        }

    def top_connections(
        self,
        *,
        neuropil: str | None = None,
        min_synapses: int = 1,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Return the strongest total MaleCNS directed neuron pairs."""

        if neuropil is not None:
            raise ValueError(
                "Global top-connections ROI filtering is not available through "
                "the lightweight neuPrint path; use neuron() or pair() for ROI queries."
            )
        if min_synapses < 1:
            raise ValueError("min_synapses must be at least 1")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        query = f"""
        MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
        WHERE c.weight >= {int(min_synapses)}
        RETURN
            pre.bodyId AS pre_pt_root_id,
            post.bodyId AS post_pt_root_id,
            c.weight AS syn_count,
            pre.consensusNt AS consensus_nt
        ORDER BY c.weight DESC, pre.bodyId, post.bodyId
        LIMIT {int(limit)}
        """
        frame = self.client.fetch_custom(query)
        if frame is None or len(frame) == 0:
            return []

        rows = []
        for row in frame.to_dict("records"):
            rows.append(
                {
                    "pre_pt_root_id": int(row["pre_pt_root_id"]),
                    "post_pt_root_id": int(row["post_pt_root_id"]),
                    "syn_count": int(row["syn_count"]),
                    "neuropil_count": None,
                    "dominant_nt": _nt_label(row.get("consensus_nt")),
                }
            )
        return rows

    def scaffold_edges(self, *, candidate_edges: int = 32768) -> list[tuple[int, int, int]]:
        """Fetch a bounded set of strongest real MaleCNS edges for router scaffolds."""

        if candidate_edges < 1:
            raise ValueError("candidate_edges must be positive")
        query = f"""
        MATCH (pre:Neuron)-[c:ConnectsTo]->(post:Neuron)
        WHERE c.weight > 0 AND pre.bodyId <> post.bodyId
        RETURN pre.bodyId AS pre, post.bodyId AS post, c.weight AS weight
        ORDER BY c.weight DESC, pre.bodyId, post.bodyId
        LIMIT {int(candidate_edges)}
        """
        frame = self.client.fetch_custom(query)
        return [
            (int(row["pre"]), int(row["post"]), int(row["weight"]))
            for row in frame.to_dict("records")
        ]

    def context_for_neuron(
        self,
        root_id: int,
        *,
        limit: int = 10,
        min_synapses: int = 1,
    ) -> str:
        partners = self.neuron(
            root_id,
            direction="both",
            min_synapses=min_synapses,
            limit=limit,
        )

        lines = [f"Janelia MaleCNS v1.0 neuron {root_id}"]
        for label in ("outputs", "inputs"):
            lines.append(f"{label.title()}:")
            rows = partners.get(label, [])
            if not rows:
                lines.append("- none found")
                continue
            for row in rows:
                lines.append(
                    f"- partner {row['partner_root_id']}: "
                    f"{row['syn_count']} synapses across "
                    f"{row['neuropil_count']} neuropil(s); "
                    f"consensus NT {row.get('dominant_nt', 'UNKNOWN')}"
                )
        return "\n".join(lines)


