"""Pre-specified sample-size rule: list count from pilot variance and Δmin.

Planning uses the one-sided upper confidence limit of the pilot standard deviation
(Browne 1995), not the point estimate. The pilot's own effect estimate is never used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import chi2, norm


def sd_upper_factor(n_pilot: int, confidence: float = 0.80) -> float:
    """Multiplier for the one-sided upper confidence limit of a standard deviation."""
    if n_pilot < 2:
        raise ValueError("n_pilot must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    df = n_pilot - 1
    return math.sqrt(df / chi2.ppf(1 - confidence, df))


def sigma_upper(sigma_hat: float, n_pilot: int, confidence: float = 0.80) -> float:
    if sigma_hat < 0:
        raise ValueError("sigma_hat must be non-negative")
    return sigma_hat * sd_upper_factor(n_pilot, confidence)


def required_lists(sigma: float, delta_min: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """n = ceil((z_{1-α/2} + z_{power})² σ² / Δmin²), two-sided level α."""
    if delta_min <= 0:
        raise ValueError("delta_min must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    k = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2
    return math.ceil(k * sigma**2 / delta_min**2)


@dataclass(frozen=True)
class ListDecision:
    required: int
    n_max: int
    lists: int  # lists to run if approved: min(required, n_max)
    exceeds_n_max: bool  # True needs a written review decision before running


def decide_lists(required: int, n_max: int) -> ListDecision:
    if required < 1 or n_max < 1:
        raise ValueError("required and n_max must be positive")
    return ListDecision(
        required=required,
        n_max=n_max,
        lists=min(required, n_max),
        exceeds_n_max=required > n_max,
    )
