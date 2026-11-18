"""Categorical values used in code and records. Unknown values must stop analysis."""

from __future__ import annotations

import re
from enum import StrEnum
from fractions import Fraction
from typing import Literal

Wording = Literal[1, 2]
WORDINGS: tuple[Wording, ...] = (1, 2)


class DatasetSplit(StrEnum):
    development = "development"
    evaluation = "evaluation"


class Stage(StrEnum):
    trial = "trial"
    pilot = "pilot"
    main = "main"


class Frame(StrEnum):
    neutral = "neutral"
    agriculture = "agriculture"
    human = "human"


class CaseType(StrEnum):
    A = "A"
    near_measurement = "near_measurement"
    near_weight = "near_weight"
    U = "U"


class Pair(StrEnum):
    measurement = "measurement"
    weight = "weight"


class Side(StrEnum):
    inside = "inside"
    outside = "outside"


class InvalidClass(StrEnum):
    refusal = "refusal"
    format = "format"


class QuestionType(StrEnum):
    rule = "rule"
    value = "value"
    probe_protocol = "probe_protocol"
    probe_evaluator = "probe_evaluator"


class RecordType(StrEnum):
    turn = "turn"
    episode_end = "episode_end"


class ConditionLetter(StrEnum):
    a = "a"  # no feedback
    c = "c"  # feedback scored against R*
    b = "b"  # feedback scored against R, rule hidden
    f = "f"  # feedback scored against R, rule disclosed


_NEEDS_M = (ConditionLetter.b, ConditionLetter.f)
_CONDITION = re.compile(r"^(?P<letter>[bf])-(?P<m>\d+(?:\.\d+)?)$")


def condition_id(letter: ConditionLetter | str, m: Fraction | float | None = None) -> str:
    """`a`, `c`, `b-{m:g}`, `f-{m:g}` (1.5 -> `b-1.5`, 2 -> `b-2`)."""
    letter = ConditionLetter(letter)
    if letter in _NEEDS_M:
        if m is None:
            raise ValueError(f"condition {letter} needs a multiplier")
        return f"{letter}-{float(m):g}"
    return str(letter)


def parse_condition(cid: str) -> tuple[ConditionLetter, float | None]:
    """Inverse of `condition_id`. Raises on anything else."""
    if cid in (ConditionLetter.a, ConditionLetter.c):
        return ConditionLetter(cid), None
    match = _CONDITION.match(cid)
    if not match:
        raise ValueError(f"unknown condition id: {cid!r}")
    return ConditionLetter(match["letter"]), float(match["m"])
