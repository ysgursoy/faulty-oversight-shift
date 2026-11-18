"""Small CLI for the repository scaffold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from oversight_study import __version__
from oversight_study.scaffold import missing_scaffold_paths, repository_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oversight-study")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show the current implementation status")
    subparsers.add_parser("smoke", help="Validate the initial repository scaffold")
    freeze = subparsers.add_parser("freeze", help="Record a design freeze (hashes, commit, approver)")
    freeze.add_argument("--kind", choices=("first", "second"), required=True)
    freeze.add_argument("--approver", required=True)
    freeze.add_argument("--n-lists", type=int)
    freeze.add_argument("--design-version")
    freeze.add_argument("--dataset", type=Path, help="Dataset release directory")
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

    if args.command == "freeze":
        return run_freeze(args)

    root = repository_root()
    missing = missing_scaffold_paths(root)
    if missing:
        print("Scaffold check failed. Missing paths:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("Initial scaffold check passed; experiment pipeline is not implemented yet.")
    return 0


def run_freeze(args: argparse.Namespace) -> int:
    from oversight_study.config import AnalysisConfig, ModelConfig, load_config
    from oversight_study.validation import freeze_record as fr

    repo = repository_root()
    decisions = repo / fr.DECISIONS_DIR
    record = fr.make_freeze_record(
        args.kind,
        approver=args.approver,
        commit=fr.git_commit(repo),
        hashes=fr.compute_hashes(repo, args.dataset),
        model=load_config(repo / "configs/models/claude.yaml", ModelConfig),
        analysis=load_config(repo / "configs/analysis/primary.yaml", AnalysisConfig),
        existing=fr.read_freezes(decisions),
        n_lists=args.n_lists,
        design_version=args.design_version,
    )
    note = fr.write_freeze(record, decisions)
    print(f"{args.kind} freeze recorded: {note.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
