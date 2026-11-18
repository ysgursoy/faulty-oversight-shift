"""Report figures. Each function writes `<name>.svg` and `<name>.png` and states the
number of episodes used. Inputs are tables from `compare` and `diagnostics`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oversight_study.analysis.compare import (
    FRAMES,
    Estimate,
    paired_bootstrap,
    within_frame_difference,
)
from oversight_study.analysis.diagnostics import RESPONSE_CLASSES
from oversight_study.analysis.validate import AnalysisInputError

SERIES = (("b", 1, "Hidden rule, wording 1"), ("b", 2, "Hidden rule, wording 2"),
          ("f", 1, "Disclosed rule"))


def _save(fig, out_dir: Path, name: str, n_episodes: int) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.01, f"Episodes used: {n_episodes}", fontsize=8, ha="left", va="bottom")
    paths = [out_dir / f"{name}.svg", out_dir / f"{name}.png"]
    for path in paths:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_frame_difference(t_accuracy: pd.DataFrame, t_shift: pd.DataFrame, out_dir: Path,
                          n_boot: int = 2000, seed: int = 0) -> list[Path]:
    """Condition − c per frame on both primary measures, with paired 95% intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (title, table) in zip(axes, (("Exact protocol accuracy", t_accuracy),
                                         ("Conditional shift", t_shift)), strict=True):
        for j, (letter, wording, label) in enumerate(SERIES):
            for i, frame in enumerate(FRAMES):
                if (frame, letter, wording) not in table or (frame, "c", wording) not in table:
                    continue
                try:
                    est = paired_bootstrap(
                        table, lambda t, f=frame, c=letter, w=wording:
                        within_frame_difference(t, f, c, w).dropna().mean(), n=n_boot, seed=seed)
                except AnalysisInputError:
                    continue
                ax.errorbar(i + (j - 1) * 0.2, est.point,
                            yerr=[[est.point - est.low], [est.high - est.point]], fmt="o",
                            color=f"C{j}", label=label if i == 0 else None)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(len(FRAMES)), FRAMES)
        ax.set_title(title)
        ax.set_ylabel("Condition − c, last 3 affected cases")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, out_dir, "frame_difference", len(t_accuracy.index))


def plot_response_distribution(dist: pd.DataFrame, out_dir: Path, n_episodes: int) -> list[Path]:
    """Stacked shares of the five response classes per frame × condition × wording."""
    shares = dist[[f"{c}_share" for c in RESPONSE_CLASSES]]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(shares)), 4))
    bottom = np.zeros(len(shares))
    labels = ["/".join(map(str, idx)) for idx in shares.index]
    for column, name in zip(shares.columns, RESPONSE_CLASSES, strict=True):
        ax.bar(labels, shares[column], bottom=bottom, label=name)
        bottom += shares[column].to_numpy()
    ax.set_ylabel("Share of planned turns")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out_dir, "response_distribution", n_episodes)


def plot_learning_curve(curve: pd.DataFrame, out_dir: Path, n_episodes: int) -> list[Path]:
    """Mean s and exact R* rate over affected cases k = 1..10; c dashed."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data = curve.reset_index()
    for ax, column, title in zip(axes, ("mean_s", "exact_r_star"), ("Mean s", "Exact R* rate"),
                                 strict=True):
        for (frame, condition, wording), group in data.groupby(["frame", "condition",
                                                                 "wording"]):
            style = "--" if condition == "c" else "-"
            ax.plot(group.k, group[column], style, marker="o",
                    label=f"{frame} {condition} w{wording}")
        ax.set_xlabel("Affected case k")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out_dir, "learning_curve", n_episodes)


def plot_rule_acquisition(rates: pd.DataFrame, out_dir: Path, n_episodes: int) -> list[Path]:
    """`rates`: index (frame, letter) with columns `description` and `application`;
    c is the floor and f the ceiling for the hidden-rule condition b."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, column, title in zip(axes, ("description", "application"),
                                 ("Rule described (consistent)", "Conditional application"),
                                 strict=True):
        table = rates[column].unstack("letter")
        table.plot.bar(ax=ax, rot=0)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out_dir, "rule_acquisition", n_episodes)


def plot_shift_histogram(s_values: pd.DataFrame, out_dir: Path, n_episodes: int) -> list[Path]:
    """`s_values`: columns frame, condition, s (late affected answers)."""
    groups = list(s_values.groupby(["frame", "condition"]))
    fig, axes = plt.subplots(1, max(1, len(groups)), figsize=(3 * max(1, len(groups)), 3),
                             sharey=True, squeeze=False)
    for ax, ((frame, condition), group) in zip(axes[0], groups, strict=False):
        ax.hist(group.s.dropna(), bins=np.linspace(-0.5, 1.5, 21))
        ax.set_title(f"{frame} {condition}", fontsize=9)
        ax.set_xlabel("s")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, out_dir, "shift_histogram", n_episodes)


def plot_episode_vs_end(points: pd.DataFrame, out_dir: Path, n_episodes: int) -> list[Path]:
    """`points`: columns frame, s_episode, s_end, invalid_end. s_end = 0 is a return to R*."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (frame, group) in enumerate(points.groupby("frame")):
        valid = group[~group.invalid_end]
        ax.scatter(valid.s_episode, valid.s_end, color=f"C{i}", label=frame, alpha=0.6)
        invalid = group[group.invalid_end]
        ax.scatter(invalid.s_episode, np.zeros(len(invalid)) - 0.25, color=f"C{i}", marker="x")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("s during the episode")
    ax.set_ylabel("s of the end-of-episode answer (x: invalid)")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out_dir, "episode_vs_end", n_episodes)


def plot_wording_robustness(results: dict[str, tuple[Estimate, str]], margin: float,
                            out_dir: Path) -> list[Path]:
    """D = Δ(wording 2) − Δ(wording 1) with its 90% interval and the ±margin band."""
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (measure, (est, label)) in enumerate(results.items()):
        ax.errorbar(i, est.point, yerr=[[est.point - est.low], [est.high - est.point]],
                    fmt="o", color="C0")
        ax.annotate(label, (i, est.high), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    ax.axhspan(-margin, margin, color="gray", alpha=0.2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(range(len(results)), list(results))
    ax.set_ylabel("D")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    n = max((est.n_episodes for est, _ in results.values()), default=0)
    return _save(fig, out_dir, "wording_robustness", n)
