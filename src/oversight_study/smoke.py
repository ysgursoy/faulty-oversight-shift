"""Offline end-to-end validation with a mock model. Must pass before any API call.

Checks: dataset quotas and separation; a mock trial run over every frame, condition, and
wording with the payload allowlist enforced; record contract; English-only records;
mock classification; sealing and release validation; the analysis recovers a planted
effect and returns about zero without one; malformed input stops the analysis; accuracy
never exceeds 1.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from oversight_study.analysis.compare import build_table, delta
from oversight_study.analysis.metrics import protocol_accuracy
from oversight_study.analysis.synthetic import make_synthetic_records
from oversight_study.analysis.validate import AnalysisInputError, check_duplicates, validate
from oversight_study.classification.mock import MockClassifier
from oversight_study.classification.run import classify_run
from oversight_study.classification.spec import load_classifier_spec
from oversight_study.config import ModelConfig, StageConfig, load_config
from oversight_study.generation import build_dataset, verify_dataset, write_dataset
from oversight_study.generation.seeds import SeedConfig
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.mock import MockModel, Policy
from oversight_study.runner.run import Run, run_stage, seal_run
from oversight_study.scaffold import missing_scaffold_paths, repository_root
from oversight_study.task.prompt_version import prompt_version
from oversight_study.validation.release import validate_release

# Smoke-only seeds; never used for released data.
SMOKE_SEEDS = SeedConfig(dev_pool_base=8200, trial_lists_base=8300, pilot_lists_base=8400,
                         test_pool_base=8100, test_lists_base=8500)
TURKISH = re.compile(r"[çğıöşüÇĞİÖŞÜ]")

Check = tuple[str, bool, str]


def _check(name: str, fn: Callable[[], str]) -> Check:
    try:
        return name, True, fn()
    except Exception as error:  # noqa: BLE001 - every failure is reported as a failed check
        return name, False, f"{type(error).__name__}: {error}"


def run_smoke(repo: Path | None = None) -> list[Check]:
    repo = repo or repository_root()
    checks: list[Check] = []
    missing = missing_scaffold_paths(repo)
    checks.append(("repository files present", not missing, ", ".join(missing) or "ok"))

    with TemporaryDirectory() as temp:
        release = Path(temp) / "release"
        state: dict = {}

        def dataset() -> str:
            state["dataset"] = build_dataset(SMOKE_SEEDS, n_trial=2, n_pilot=2, n_main=2)
            write_dataset(state["dataset"], release / "data", dataset_version="smoke")
            verify_dataset(release / "data")
            return "quotas, composition, and separation verified"

        def trial_run() -> str:
            config = load_config(repo / "configs/stages/trial.yaml", StageConfig).model_copy(
                update={"list_count": 2})
            model = load_config(repo / "configs/models/claude.yaml", ModelConfig).model_copy(
                update={"model_id": "mock-model"})
            state["run"] = run_stage(
                config, state["dataset"].episodes["trial"],
                MockModel(Policy("invalid", rate=0.1), seed=1), model=model,
                settings=GenerationSettings(max_tokens=64),
                prompt_version=prompt_version(repo / "prompts"),
                provenance={"commit": "smoke"}, run_root=release / "runs")
            return f"{Run(state['run']).manifest().cells_completed} cells, payloads clean"

        def english_only() -> str:
            text = (state["run"] / "results.jsonl").read_text(encoding="utf-8")
            match = TURKISH.search(text)
            if match:
                raise ValueError(f"non-English character {match.group(0)!r} in records")
            return "no Turkish characters in records"

        def classify() -> str:
            spec = load_classifier_spec(repo / "annotations/spec.yaml")
            rows = classify_run(state["run"], state["dataset"].episodes["trial"],
                                MockClassifier(), spec, 1.5)
            return f"{len(rows)} items classified"

        def release_check() -> str:
            seal_run(state["run"])
            report = validate_release(release)
            return f"{sum(report.runs.values())} records valid"

        for name, fn in (("dataset", dataset), ("mock trial run", trial_run),
                         ("English-only records", english_only),
                         ("mock classification", classify), ("seal and validate release",
                                                             release_check)):
            checks.append(_check(name, fn))
            if not checks[-1][1]:
                break

    def planted() -> str:
        study = make_synthetic_records(effect=0.3, n_episodes=1000, seed=7)
        estimate = delta(build_table(protocol_accuracy(study.late), 1.5))
        if abs(estimate - 0.3) > 0.05:
            raise ValueError(f"planted 0.3, recovered {estimate:.3f}")
        null = make_synthetic_records(effect=0.0, n_episodes=1000, seed=11)
        zero = delta(build_table(protocol_accuracy(null.late), 1.5))
        if abs(zero) > 0.05:
            raise ValueError(f"null effect recovered as {zero:.3f}")
        return f"planted 0.300 -> {estimate:.3f}; null -> {zero:.3f}"

    def stops() -> str:
        study = make_synthetic_records(effect=0.1, n_episodes=20, seed=0)
        cases = {
            "duplicate": lambda: check_duplicates(pd.concat([study.late, study.late.iloc[:1]])),
            "missing cell": lambda: validate(study.late[~((study.late.frame == "human")
                                                          & (study.late.condition == "c")
                                                          & (study.late.episode == 3))],
                                             study.probes),
            "unknown category": lambda: validate(study.late.assign(frame="finance"),
                                                 study.probes),
        }
        for name, fn in cases.items():
            try:
                fn()
            except AnalysisInputError:
                continue
            raise ValueError(f"{name} did not stop the analysis")
        return "duplicate, missing cell, unknown category all stop"

    def accuracy_bound() -> str:
        study = make_synthetic_records(effect=0.6, n_episodes=200, seed=1)
        top = protocol_accuracy(study.late).max()
        if top > 1:
            raise ValueError(f"accuracy {top} > 1")
        return f"max accuracy {top:.3f}"

    for name, fn in (("planted effect recovered", planted), ("bad input stops analysis", stops),
                     ("accuracy at most 1", accuracy_bound)):
        checks.append(_check(name, fn))
    return checks


def print_checks(checks: list[Check]) -> bool:
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return all(ok for _, ok, _ in checks)
