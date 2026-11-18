"""Episode-paired comparisons: tables, frame differences, paired bootstrap, and the
reported (non-gating) acquisition and wording-robustness results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from oversight_study.analysis.metrics import late_shift
from oversight_study.analysis.validate import AnalysisInputError

FRAMES = ("neutral", "agriculture", "human")
Statistic = Callable[[pd.DataFrame], float]


def build_table(measure: pd.Series, m: Fraction | float, replicate: int = 0) -> pd.DataFrame:
    """Rows: episodes. Columns: (frame, condition_letter, wording).

    `measure` is a per-cell Series from `analysis.metrics`. Only `b`/`f` cells with the
    given m are kept; `a` and `c` are kept as they are (scored against the same m).
    """
    df = measure.rename("value").reset_index()
    df = df[df.replicate == replicate]
    df["letter"] = df.condition.str.split("-").str[0]
    keep = df.letter.isin(["a", "c"]) | (df.condition == df.letter + f"-{float(m):g}")
    df = df[keep]
    if df.duplicated(["episode", "frame", "letter", "wording"]).any():
        raise AnalysisInputError("table has more than one value per episode and column")
    return df.pivot(index="episode", columns=["frame", "letter", "wording"], values="value")


def frame_difference(t: pd.DataFrame, condition: str = "b", wording: int = 1,
                     f1: str = "human", f2: str = "neutral") -> pd.Series:
    """Per episode [(condition) − (c)]_f1 − [(condition) − (c)]_f2; NaN if a cell is missing."""
    return ((t[(f1, condition, wording)] - t[(f1, "c", wording)])
            - (t[(f2, condition, wording)] - t[(f2, "c", wording)]))


def within_frame_difference(t: pd.DataFrame, frame: str, condition: str = "b",
                            wording: int = 1) -> pd.Series:
    return t[(frame, condition, wording)] - t[(frame, "c", wording)]


@dataclass(frozen=True)
class Estimate:
    point: float
    low: float
    high: float
    level: float
    n_episodes: int


def paired_bootstrap(table: pd.DataFrame, statistic: Statistic, n: int = 10_000,
                     seed: int = 0, level: float = 95) -> Estimate:
    """Episode ids are drawn once per resample and shared by all columns.

    Raises when the statistic is undefined on the data or on any resample; such a
    comparison is reported as not estimable.
    """
    point = statistic(table)
    if np.isnan(point):
        raise AnalysisInputError("statistic undefined on the observed data; not estimable")
    rng = np.random.default_rng(seed)
    ids = table.index.to_numpy()
    draws = np.empty(n)
    for i in range(n):
        draws[i] = statistic(table.loc[rng.choice(ids, size=len(ids), replace=True)])
    undefined = int(np.isnan(draws).sum())
    if undefined:
        raise AnalysisInputError(f"statistic undefined in {undefined} of {n} resamples; "
                                 "not estimable")
    tail = (100 - level) / 2
    low, high = np.percentile(draws, [tail, 100 - tail])
    return Estimate(float(point), float(low), float(high), level, len(ids))


def delta(t: pd.DataFrame, condition: str = "b", wording: int = 1) -> float:
    return frame_difference(t, condition, wording).dropna().mean()


def acquisition(t_ca: pd.DataFrame, frames=FRAMES, n: int = 10_000, seed: int = 0) -> dict:
    """Conditional application, b − c at wording 1, per frame (reported, gates nothing)."""
    result = {}
    for frame in frames:
        estimate = paired_bootstrap(
            t_ca, lambda x, f=frame: within_frame_difference(x, f).mean(), n=n, seed=seed)
        result[frame] = {"estimate": estimate, "above_zero": estimate.low > 0}
    return result


def robustness_label(low: float, high: float, margin: float) -> str:
    if -margin <= low and high <= margin:
        return "equivalent"
    if high < -margin or low > margin:
        return "wording_dependent"
    return "inconclusive"


def wording_robustness(t: pd.DataFrame, margin: float, condition: str = "b",
                       n: int = 10_000, seed: int = 0, level: float = 90):
    """D = Δ(wording 2) − Δ(wording 1) per episode; its interval is compared with ±margin."""
    def d(x: pd.DataFrame) -> float:
        return (frame_difference(x, condition, 2) - frame_difference(x, condition, 1)).dropna().mean()

    estimate = paired_bootstrap(t, d, n=n, seed=seed, level=level)
    return estimate, robustness_label(estimate.low, estimate.high, margin)


MAXIMIZING = {("human", "b"): 1, ("human", "c"): 0, ("neutral", "b"): 0, ("neutral", "c"): 1}
MINIMIZING = {key: 1 - value for key, value in MAXIMIZING.items()}


def two_extremes(late: pd.DataFrame, m: Fraction | float) -> tuple[float, float]:
    """Primary shift Δ with invalid answers set to the s = 0/1 assignment that shrinks
    and that enlarges Δ the most. Not mathematical bounds: s can leave [0, 1]."""
    low = delta(build_table(late_shift(late, m, fill=MINIMIZING), m))
    high = delta(build_table(late_shift(late, m, fill=MAXIMIZING), m))
    return low, high


def sign_changes(low: float, high: float) -> bool:
    return np.sign(low) != np.sign(high)
