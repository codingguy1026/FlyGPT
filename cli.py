from __future__ import annotations

import argparse
import json
import os

from connectome import FlyWireConnectome


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query FlyWire v783 proofread connectivity Parquet files."
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("FLYWIRE_DATA_DIR", "data/flywire_parts"),
        help="Parquet file, glob, or directory (default: FLYWIRE_DATA_DIR or data/flywire_parts)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stats", help="Show dataset statistics.")

    neuron = subparsers.add_parser("neuron", help="Show strongest partners for one neuron.")
    neuron.add_argument("root_id", type=int)
    neuron.add_argument(
        "--direction",
        choices=("inputs", "outputs", "both"),
        default="both",
    )
    neuron.add_argument("--neuropil")
    neuron.add_argument("--min-synapses", type=int, default=1)
    neuron.add_argument("--limit", type=int, default=25)
    neuron.add_argument(
        "--context",
        action="store_true",
        help="Print compact LLM-ready context instead of JSON.",
    )

    pair = subparsers.add_parser("pair", help="Inspect one directed neuron pair.")
    pair.add_argument("pre_root_id", type=int)
    pair.add_argument("post_root_id", type=int)

    top = subparsers.add_parser("top", help="Show strongest directed neuron pairs.")
    top.add_argument("--neuropil")
    top.add_argument("--min-synapses", type=int, default=1)
    top.add_argument("--limit", type=int, default=25)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    with FlyWireConnectome(args.data) as connectome:
        if args.command == "stats":
            _print(connectome.stats())
            return

        if args.command == "neuron":
            if args.context:
                print(
                    connectome.context_for_neuron(
                        args.root_id,
                        limit=args.limit,
                        min_synapses=args.min_synapses,
                    )
                )
            else:
                _print(
                    connectome.neuron(
                        args.root_id,
                        direction=args.direction,
                        neuropil=args.neuropil,
                        min_synapses=args.min_synapses,
                        limit=args.limit,
                    )
                )
            return

        if args.command == "pair":
            _print(connectome.pair(args.pre_root_id, args.post_root_id))
            return

        if args.command == "top":
            _print(
                connectome.top_connections(
                    neuropil=args.neuropil,
                    min_synapses=args.min_synapses,
                    limit=args.limit,
                )
            )
            return


if __name__ == "__main__":
    main()
