"""Small CLI for the repository scaffold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from oversight_study import __version__
from oversight_study.scaffold import missing_scaffold_paths, repository_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oversight-study")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show the current implementation status")
    subparsers.add_parser("smoke", help="Validate the initial repository scaffold")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "status":
        print(
            json.dumps(
                {
                    "project": "faulty-oversight-shift",
                    "version": __version__,
                    "status": "design-and-scaffold-only",
                    "experiment_implemented": False,
                    "results_available": False,
                },
                indent=2,
            )
        )
        return 0

    root = repository_root()
    missing = missing_scaffold_paths(root)
    if missing:
        print("Scaffold check failed. Missing paths:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("Initial scaffold check passed; experiment pipeline is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
