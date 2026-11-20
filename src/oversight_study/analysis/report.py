"""Analysis of a validated release: tables, figures, and an analysis manifest.

Output: `<out_root>/<analysis-id>/` with `manifest.json`, `tables/*.csv`, `figures/*`.
Development pilot and main analyses get separate ids and are never merged; the pilot is
analyzed per replicate. Statistics that are not estimable are reported as such.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.rcParams["svg.hashsalt"] = "oversight-study"

import pandas as pd

from oversight_study.analysis import diagnostics as diag
from oversight_study.analysis import plots
from oversight_study.analysis.compare import (
    FRAMES,
    Estimate,
    acquisition,
    build_table,
    delta,
    paired_bootstrap,
    two_extremes,
    within_frame_difference,
    wording_robustness,
)
from oversight_study.analysis.load import load_records
from oversight_study.analysis.metrics import (
    conditional_application,
    late_shift,
    protocol_accuracy,
    shift,
)
from oversight_study.analysis.validate import AnalysisInputError, validate
from oversight_study.config import AnalysisConfig
from oversight_study.generation.manifest import read_dataset
from oversight_study.validation.release import validate_release


def _estimate(table: pd.DataFrame, statistic, config: AnalysisConfig, level=None) -> dict:
    try:
        est = paired_bootstrap(table, statistic, n=config.bootstrap_n,
                               seed=config.bootstrap_seed, level=level or config.level)
    except (AnalysisInputError, KeyError) as error:
        return {"estimable": False, "reason": str(error)}
    return {"estimable": True, **_est_dict(est)}


def _est_dict(est: Estimate) -> dict:
    return {"point": est.point, "low": est.low, "high": est.high, "level": est.level,
            "n_episodes": est.n_episodes}


def _write_csv(frame: pd.DataFrame | pd.Series, path: Path) -> None:
    frame.to_csv(path, float_format="%.10g")


def analyze(release: Path, out_root: Path, *, stage: str, dataset_split: str, m: float,
            config: AnalysisConfig, replicate: int = 0, code_commit: str = "unknown") -> Path:
    release = Path(release)
    report = validate_release(release)
    run_dirs = sorted(p for p in (release / "runs").iterdir() if p.is_dir())
    records = load_records(run_dirs, dataset_split, stage)
    late = records.late[records.late.replicate == replicate]
    probes = records.probes[records.probes.replicate == replicate] if not records.probes.empty \
        else records.probes
    validate(late, probes)
    _, _, episodes, _ = read_dataset(release / "data")
    r_star_by_case = {p.case_id: p.r_star for lists in episodes.values() for e in lists
                      for p in e.probes}

    inputs = json.dumps({"runs": sorted(report.runs), "stage": stage, "split": dataset_split,
                         "replicate": replicate, "m": m}, sort_keys=True)
    digest = hashlib.sha256(inputs.encode()).hexdigest()[:8]
    analysis_id = f"{stage}-{dataset_split}-r{replicate}-{digest}"
    out = Path(out_root) / analysis_id
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    t_acc = build_table(protocol_accuracy(late), m, replicate)
    t_shift = build_table(late_shift(late, m), m, replicate)
    _write_csv(t_acc, tables / "protocol_accuracy.csv")
    _write_csv(t_shift, tables / "conditional_shift.csv")

    summary: dict = {
        "primary": {"protocol_accuracy": _estimate(t_acc, delta, config),
                    "conditional_shift": _estimate(t_shift, delta, config)},
        "within_frame": {},
    }
    for frame in FRAMES:
        for name, table in (("protocol_accuracy", t_acc), ("conditional_shift", t_shift)):
            summary["within_frame"][f"{frame}/{name}"] = _estimate(
                table, lambda t, f=frame: within_frame_difference(t, f).dropna().mean(), config)
    try:
        low, high = two_extremes(late, m)
        summary["two_extremes_shift"] = {"low": low, "high": high}
    except (AnalysisInputError, KeyError) as error:
        summary["two_extremes_shift"] = {"estimable": False, "reason": str(error)}

    robustness_results = {}
    if (t_acc.columns.get_level_values("wording") == 2).any():
        for name, table in (("protocol_accuracy", t_acc), ("conditional_shift", t_shift)):
            try:
                est, label = wording_robustness(table, config.delta, n=config.bootstrap_n,
                                                seed=config.bootstrap_seed,
                                                level=config.robustness_level)
                summary.setdefault("wording_robustness", {})[name] = {**_est_dict(est),
                                                                      "label": label}
                robustness_results[name] = (est, label)
            except (AnalysisInputError, KeyError) as error:
                summary.setdefault("wording_robustness", {})[name] = {"estimable": False,
                                                                      "reason": str(error)}
    if not probes.empty:
        t_ca = build_table(conditional_application(probes, m, r_star_by_case), m, replicate)
        _write_csv(t_ca, tables / "conditional_application.csv")
        frames_present = sorted({c[0] for c in t_ca.columns})
        try:
            summary["acquisition"] = {
                frame: {**_est_dict(v["estimate"]), "above_zero": v["above_zero"]}
                for frame, v in acquisition(t_ca, frames_present, n=config.bootstrap_n,
                                            seed=config.bootstrap_seed).items()}
        except (AnalysisInputError, KeyError) as error:
            summary["acquisition"] = {"estimable": False, "reason": str(error)}

    turns = records.turns[records.turns.replicate == replicate]
    dist = diag.response_distribution(turns)
    curve = diag.learning_curve(turns, m)
    diagnostics = {
        "response_distribution": dist,
        "learning_curve": curve,
        "unaffected_accuracy": diag.unaffected_accuracy(turns),
        "near_shift": diag.near_shift(turns, m),
        "deviations": diag.deviations(turns),
        "unrounded_answers": diag.unrounded_answers(turns),
        "retry_recovery": diag.retry_recovery(turns),
    }
    end = records.end[records.end.replicate == replicate] if not records.end.empty else records.end
    if not end.empty:
        diagnostics["return_to_protocol"] = diag.return_to_protocol(late, end, m)
        if not probes.empty:
            diagnostics["two_rules"] = diag.two_rules(end, r_star_by_case, m)
            diagnostics["beyond_boundary"] = diag.beyond_boundary(end, r_star_by_case, m)
            diagnostics["correct_feedback_control"] = diag.correct_feedback_control(
                end, r_star_by_case, m)
            diagnostics["detect_and_retain"] = diag.detect_and_retain(t_ca, t_acc)
    for name, table in diagnostics.items():
        _write_csv(table, tables / f"{name}.csv")

    n_episodes = len(t_acc.index)
    plots.plot_frame_difference(t_acc, t_shift, figures, n_boot=min(config.bootstrap_n, 2000),
                                seed=config.bootstrap_seed)
    plots.plot_response_distribution(dist, figures, n_episodes)
    plots.plot_learning_curve(curve, figures, n_episodes)
    s_values = late.assign(s=[shift(y, p, m) for y, p in zip(late.y, late.r_star, strict=True)])
    plots.plot_shift_histogram(s_values, figures, n_episodes)
    if robustness_results:
        plots.plot_wording_robustness(robustness_results, config.delta, figures)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "analysis_id": analysis_id,
        "stage": stage,
        "dataset_split": dataset_split,
        "replicate": replicate,
        "m": m,
        "dataset_version": report.dataset_version,
        "input_runs": sorted(report.runs),
        "code_commit": code_commit,
        "parameters": config.model_dump(mode="json"),
        "summary": summary,
        "checksums_sha256": {p.relative_to(out).as_posix():
                             hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str)
                                       + "\n", encoding="utf-8")
    return out
