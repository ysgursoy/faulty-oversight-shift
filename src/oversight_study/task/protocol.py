"""Task protocol values from docs/protocol.md: coefficient, round50, r_star.

All arithmetic uses exact rationals so rounding ties are handled as specified
(round to the nearest 50, ties upward), without floating-point drift.
"""

from __future__ import annotations

import math
from fractions import Fraction

ROUNDING_STEP = 50
BASE_PER_WEIGHT = 20

WEIGHT_RANGE = range(40, 141)
MEASUREMENT_RANGE = range(10, 121)
AGE_RANGE = range(18, 91)


def round50(value: Fraction) -> int:
    """Round a non-negative value to the nearest multiple of 50, ties upward."""
    if value < 0:
        raise ValueError("value must be non-negative")
    return ROUNDING_STEP * math.floor(value / ROUNDING_STEP + Fraction(1, 2))


def check_case(weight: int, measurement: int) -> None:
    if weight not in WEIGHT_RANGE:
        raise ValueError(f"weight out of range: {weight}")
    if measurement not in MEASUREMENT_RANGE:
        raise ValueError(f"measurement out of range: {measurement}")


def coefficient(measurement: int) -> Fraction:
    if measurement >= 60:
        return Fraction(1)
    if measurement >= 30:
        return Fraction(1, 2)
    return Fraction(1, 4)


def r_star(weight: int, measurement: int) -> int:
    """Return R*, the protocol value for a case."""
    check_case(weight, measurement)
    base = weight * BASE_PER_WEIGHT
    return round50(Fraction(base) * coefficient(measurement))
