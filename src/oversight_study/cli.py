"""Command line interface. Paid model calls need `--allow-paid` and an API key; `--mock`
runs the same pipeline offline with a local mock model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from oversight_study import __version__
from oversight_study.scaffold import missing_scaffold_paths, repository_root

PAID_EXIT = 2
MOCK_POLICIES = ("protocol", "evaluator", "planted", "invalid", "refusal", "noise")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oversight-study")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show the implementation status")
    sub.add_parser("smoke", help="Offline end-to-end validation with a mock model")

    gen = sub.add_parser("generate", help="Generate and write a dataset release")
    gen.add_argument("--dataset-version", required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--n-trial", type=int, default=3)
    gen.add_argument("--n-pilot", type=int, default=20)
    gen.add_argument("--n-main", type=int)
    gen.add_argument("--include-test-seeds", action="store_true",
                     help="Write evaluation seeds into the manifest (private release only)")
    _seed_args(gen)

    pre = sub.add_parser("precheck", help="Single-turn protocol check")
    pre.add_argument("--config", type=Path, default=Path("configs/precheck.yaml"))
    pre.add_argument("--dataset", type=Path, required=True)
    pre.add_argument("--split", choices=("development", "evaluation"), required=True)
    pre.add_argument("--wording", type=int, choices=(1, 2), action="append")
    pre.add_argument("--out", type=Path, required=True)
    _client_args(pre)

    run = sub.add_parser("run", help="Run a stage (trial, pilot, main)")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--runs-root", type=Path, default=Path("runs-unsealed"))
    run.add_argument("--resume", help="Run id to resume")
    run.add_argument("--include-optional", action="store_true")
    _client_args(run)

    seal = sub.add_parser("seal", help="Seal a completed run")
    seal.add_argument("run_dir", type=Path)

    cls = sub.add_parser("classify", help="Second-model classification of a run")
    cls.add_argument("--run", type=Path, required=True)
    cls.add_argument("--dataset", type=Path, required=True)
    cls.add_argument("--spec", type=Path, default=Path("annotations/spec.yaml"))
    cls.add_argument("--mock", action="store_true", help="Offline mock classifier")
    cls.add_argument("--allow-paid", action="store_true")

    ana = sub.add_parser("analyze", help="Analyze a validated release")
    ana.add_argument("release", type=Path)
    ana.add_argument("--stage", choices=("trial", "pilot", "main"), required=True)
    ana.add_argument("--split", choices=("development", "evaluation"), required=True)
    ana.add_argument("--replicate", type=int, default=0)
    ana.add_argument("--out", type=Path, default=Path("results"))
    ana.add_argument("--config", type=Path, default=Path("configs/analysis/primary.yaml"))
    ana.add_argument("--m", type=float, default=1.5)

    validate = sub.add_parser("validate-release", help="Check a release end to end")
    validate.add_argument("release", type=Path)

    schemas = sub.add_parser("schemas", help="Write JSON Schema files from the models")
    schemas.add_argument("--out", type=Path, default=Path("schemas"))

    freeze = sub.add_parser("freeze", help="Record a design freeze (hashes, commit, approver)")
    freeze.add_argument("--kind", choices=("first", "second"), required=True)
    freeze.add_argument("--approver", required=True)
    freeze.add_argument("--n-lists", type=int)
    freeze.add_argument("--design-version")
    freeze.add_argument("--dataset", type=Path, help="Dataset release directory")
    return parser


def _seed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds-dev", type=Path, help="Default: configs/data/seeds-dev.yaml")
    parser.add_argument("--seeds-test", type=Path, help="Default: .local/seeds-test.yaml")


def _client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/claude.yaml"))
    parser.add_argument("--mock", choices=MOCK_POLICIES, help="Offline mock model policy")
    parser.add_argument("--mock-seed", type=int, default=0)
    parser.add_argument("--allow-paid", action="store_true", help="Allow paid API calls")


class PaidRefused(Exception):
    pass


def make_client(args: argparse.Namespace, model):
    from oversight_study.runner.mock import MockModel, Policy

    if getattr(args, "mock", None):
        return MockModel(Policy(args.mock), seed=args.mock_seed), "mock-model"
    if not args.allow_paid:
        raise PaidRefused("paid model calls need --allow-paid (or use --mock)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise PaidRefused("ANTHROPIC_API_KEY is not set")
    from oversight_study.runner.anthropic_client import AnthropicClient

    return AnthropicClient(model), model.model_id


def git_commit(repo: Path) -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=repo, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty.strip() else "")


def _settings(model):
    from oversight_study.records.run_records import GenerationSettings

    return GenerationSettings(temperature=model.temperature, thinking=model.thinking,
                              max_tokens=model.max_tokens)


def _gate(repo: Path, dataset: Path):
    from oversight_study.validation.freeze import FreezeGate
    from oversight_study.validation.freeze_record import DECISIONS_DIR, compute_hashes

    hashes = compute_hashes(repo, dataset)
    return FreezeGate(repo / DECISIONS_DIR, hashes), hashes


def cmd_status() -> int:
    print(json.dumps({
        "project": "faulty-oversight-shift",
        "version": __version__,
        "status": "pipeline-implemented-not-run",
        "components": {"generation": True, "runner": True, "classification": True,
                       "analysis": True, "paid_runs": "require --allow-paid"},
        "results_available": False,
    }, indent=2))
    return 0


def cmd_generate(args) -> int:
    from oversight_study.generation import build_dataset, load_seeds, write_dataset

    seeds = load_seeds(True, dev_path=args.seeds_dev, test_path=args.seeds_test)
    dataset = build_dataset(seeds, n_trial=args.n_trial, n_pilot=args.n_pilot,
                            n_main=args.n_main)
    manifest = write_dataset(dataset, args.out, dataset_version=args.dataset_version,
                             include_test_seeds=args.include_test_seeds)
    print(f"dataset {manifest.dataset_version} written to {args.out}: {manifest.list_counts}")
    return 0


def cmd_precheck(args) -> int:
    from oversight_study.config import ModelConfig, PrecheckConfig, load_config
    from oversight_study.generation import read_dataset, verify_dataset
    from oversight_study.runner.precheck import run_precheck, select_subset, summarize_precheck
    from oversight_study.task.prompt_version import prompt_version

    repo = repository_root()
    config = load_config(args.config, PrecheckConfig)
    model = load_config(args.model_config, ModelConfig)
    client, requested = make_client(args, model)
    verify_dataset(args.dataset)
    _, pools, _, _ = read_dataset(args.dataset)
    split_pools = {t: [c for c in pools[args.split] if c.case_type == t]
                   for t in ("A", "near_measurement", "near_weight", "U")}
    gate = _gate(repo, args.dataset)[0] if args.split == "evaluation" else None
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for wording in args.wording or list(config.wordings):
        quotas = (config.development_subset if args.split == "development"
                  else config.evaluation_by_wording[wording])
        cases = select_subset(split_pools, quotas, config.sample_seed)
        records += run_precheck(cases, config.frames, [wording], client,
                                dataset_split=args.split, run_id=f"precheck-{args.split}",
                                requested_model=requested, settings=_settings(model),
                                prompt_version=prompt_version(), freeze_gate=gate)
    (args.out / "precheck.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    summary = summarize_precheck(records, config.threshold)
    summary.to_csv(args.out / "precheck_summary.csv")
    print(summary.to_string())
    return 0 if summary.passed.all() else 1


def cmd_run(args) -> int:
    from oversight_study.config import ModelConfig, StageConfig, load_config
    from oversight_study.generation import read_dataset, verify_dataset
    from oversight_study.runner.run import run_stage
    from oversight_study.task.prompt_version import prompt_version

    repo = repository_root()
    config = load_config(args.config, StageConfig)
    model = load_config(args.model_config, ModelConfig)
    client, requested = make_client(args, model)
    if requested == "mock-model":
        model = model.model_copy(update={"model_id": "mock-model"})
    verify_dataset(args.dataset)
    _, _, episodes, _ = read_dataset(args.dataset)
    gate, hashes = _gate(repo, args.dataset)
    run_dir = run_stage(config, episodes[config.stage], client, model=model,
                        settings=_settings(model), prompt_version=prompt_version(),
                        provenance={"commit": git_commit(repo), **hashes},
                        run_root=args.runs_root, run_id=args.resume,
                        include_optional=args.include_optional, freeze_gate=gate)
    print(f"run completed: {run_dir}")
    return 0


def cmd_classify(args) -> int:
    from oversight_study.classification.run import classify_run
    from oversight_study.classification.spec import load_classifier_spec
    from oversight_study.generation import read_dataset
    from oversight_study.runner.run import Run

    spec = load_classifier_spec(args.spec)
    if args.mock:
        from oversight_study.classification.mock import MockClassifier

        client = MockClassifier()
    else:
        if not args.allow_paid:
            raise PaidRefused("paid classifier calls need --allow-paid (or use --mock)")
        spec.require_runnable()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise PaidRefused("ANTHROPIC_API_KEY is not set")
        from oversight_study.config import ModelConfig, load_config
        from oversight_study.runner.anthropic_client import AnthropicClient

        base = load_config(Path("configs/models/claude.yaml"), ModelConfig)
        client = AnthropicClient(base.model_copy(update={"model_id": spec.model_id}))
    manifest = Run(args.run).manifest()
    _, _, episodes, _ = read_dataset(args.dataset)
    rows = classify_run(args.run, episodes[manifest.stage], client, spec, manifest.m)
    print(f"classified {len(rows)} items in {args.run}")
    return 0


def cmd_analyze(args) -> int:
    from oversight_study.analysis.report import analyze
    from oversight_study.config import AnalysisConfig, load_config

    out = analyze(args.release, args.out, stage=args.stage, dataset_split=args.split, m=args.m,
                  config=load_config(args.config, AnalysisConfig), replicate=args.replicate,
                  code_commit=git_commit(repository_root()))
    print(f"analysis written to {out}")
    return 0


def cmd_validate_release(args) -> int:
    from oversight_study.validation.release import ReleaseError, validate_release

    try:
        report = validate_release(args.release)
    except (ReleaseError, ValueError) as error:
        print(f"Release check failed: {error}")
        return 1
    print(f"Release valid: dataset {report.dataset_version}, "
          f"{len(report.runs)} run(s), {sum(report.runs.values())} records")
    return 0


def cmd_freeze(args) -> int:
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


def cmd_smoke() -> int:
    root = repository_root()
    missing = missing_scaffold_paths(root)
    if missing:
        print("Scaffold check failed. Missing paths:")
        for path in missing:
            print(f"- {path}")
        return 1
    print("Scaffold check passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "status": lambda: cmd_status(),
        "smoke": lambda: cmd_smoke(),
        "generate": lambda: cmd_generate(args),
        "precheck": lambda: cmd_precheck(args),
        "run": lambda: cmd_run(args),
        "seal": lambda: _seal(args),
        "classify": lambda: cmd_classify(args),
        "analyze": lambda: cmd_analyze(args),
        "validate-release": lambda: cmd_validate_release(args),
        "schemas": lambda: _schemas(args),
        "freeze": lambda: cmd_freeze(args),
    }
    try:
        return commands[args.command]()
    except PaidRefused as error:
        print(f"Refused: {error}", file=sys.stderr)
        return PAID_EXIT


def _seal(args) -> int:
    from oversight_study.runner.run import seal_run

    print(f"sealed: {seal_run(args.run_dir)}")
    return 0


def _schemas(args) -> int:
    from oversight_study.records.schema_export import export_schemas

    for path in export_schemas(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
