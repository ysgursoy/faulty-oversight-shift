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
    n_min: int
    lists: int  # lists to run if approved: required, raised to n_min, capped at n_max
    exceeds_n_max: bool  # True needs a written review decision before running
    raised_to_n_min: bool  # the variance rule asked for fewer lists than the minimum


def decide_lists(required: int, n_max: int, n_min: int = 1) -> ListDecision:
    """Apply the minimum and maximum list counts to the variance rule's result.

    `required` may be 0 when the pilot shows no variance; `n_min` then sets the count.
    """
    if required < 0 or n_max < 1 or n_min < 1:
        raise ValueError("required must be non-negative; n_min and n_max positive")
    if n_min > n_max:
        raise ValueError("n_min must not exceed n_max")
    return ListDecision(
        required=required,
        n_max=n_max,
        n_min=n_min,
        lists=min(max(required, n_min), n_max),
        exceeds_n_max=required > n_max,
        raised_to_n_min=required < n_min,
    )
