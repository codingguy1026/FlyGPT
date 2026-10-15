"""Build a role assignment for the EXACT frozen MaleCNS scaffold using neuPrint.

Requires NEUPRINT_TOKEN in the environment. Does not assign roles from AI
guesses, ID order or graph geometry; all missing annotations stay unclassified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectome import MaleCNSConnectome
from training.role_prior import validate_role_assignment


def build_roles(scaffold_path: Path, out_path: Path) -> dict:
    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    root_ids = scaffold.get("root_ids")
    if scaffold.get("dataset") != "male-cns:v1.0" or not isinstance(root_ids, list):
        raise ValueError("Expected MaleCNS v1.0 scaffold with root IDs")
    body_ids = [int(value) for value in root_ids]
    if not body_ids or len(set(body_ids)) != len(body_ids):
        raise ValueError("Missing or duplicate scaffold body IDs")
    if len(body_ids) != int(scaffold["n_nodes"]):
        raise ValueError("Scaffold n_nodes and root_ids disagree")

    with MaleCNSConnectome() as connectome:
        if connectome.dataset != "male-cns:v1.0":
            raise ValueError("Expected neuPrint dataset male-cns:v1.0")
        records = connectome.neuron_annotations(body_ids)

    assignments = {
        str(body_id): records.get(str(body_id), {}).get("kind", "unclassified")
        for body_id in body_ids
    }
    payload = {
        "source": "Janelia MaleCNS v1.0 via neuPrint",
        "dataset": "male-cns:v1.0",
        "root_ids": [str(item) for item in body_ids],
        "roles": assignments,
    }
    # Fail loudly if official classifications cannot provide any usable signal.
    _, metadata = validate_role_assignment(
        payload,
        {"kind": "male-cns:v1.0", "root_ids": body_ids},
        len(body_ids),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved=" + str(out_path) + " annotated=" + str(metadata["annotated_count"]) +
          "/" + str(len(body_ids)) + " counts=" + str(metadata["counts"]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold", type=Path, default=Path("training/malecns_scaffold.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/malecns_roles_v1.json"))
    args = parser.parse_args()
    build_roles(args.scaffold, args.out)


if __name__ == "__main__":
    main()
