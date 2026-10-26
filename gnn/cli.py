"""Single command-line interface for maintained and historical GNN pipelines."""

from __future__ import annotations

import argparse
import sys


PIPELINES = {
    "integrated": "Compact actual-node/actual-edge pipeline (recommended).",
    "padded-shared": "Padded representation through the integrated engine.",
    "historical": "Original historical training and validation procedure.",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gnn",
        add_help=False,
        description="Train or evaluate graph-robustness classifiers.",
    )
    parser.add_argument("action", nargs="?", choices=("train", "test"))
    parser.add_argument("--pipeline", choices=tuple(PIPELINES), default="integrated")
    parser.add_argument("--model", choices=("gat", "gatv2"), default="gatv2")
    return parser


def _print_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print("\nPipelines:")
    for name, description in PIPELINES.items():
        print(f"  {name:14} {description}")
    print("\nExamples:")
    print("  python -m gnn train --model gatv2 --epochs 100")
    print("  python -m gnn test --model gatv2 --conformal-coverage 0.9")
    print("  python -m gnn train --pipeline historical --model gat")
    print("\nArguments not listed above are forwarded to the selected pipeline.")


def main() -> None:
    parser = _parser()
    args, forwarded = parser.parse_known_args()
    if args.action is None:
        _print_help(parser)
        return

    # Each implementation owns its detailed argparse interface. Remove the
    # unified selector flags and let it parse only the forwarded arguments.
    sys.argv = [sys.argv[0], *forwarded]
    if args.pipeline == "historical":
        from legacy.historical_baseline import test_main, train_main

        entrypoint = train_main if args.action == "train" else test_main
        entrypoint(args.model)
        return

    from gnn.variants import test_main, train_main

    representation = "new" if args.pipeline == "integrated" else "original"
    entrypoint = train_main if args.action == "train" else test_main
    entrypoint(representation, args.model)
