"""Evaluator feedback text sent after each scored turn."""

from __future__ import annotations

from functools import cache
from typing import Literal

import yaml

from oversight_study.task.templates import load_template

Outcome = Literal["correct", "incorrect", "invalid"]
OUTCOMES: tuple[Outcome, ...] = ("correct", "incorrect", "invalid")


@cache
def feedback_templates(wording: int) -> dict[str, str]:
    if wording not in (1, 2):
        raise ValueError(f"unknown wording: {wording}")
    data = yaml.safe_load(load_template(f"feedback/wording_{wording}.yaml"))
    if not isinstance(data, dict) or set(data) != set(OUTCOMES):
        raise ValueError(f"feedback/wording_{wording}.yaml must define exactly {OUTCOMES}")
    return data


def _check_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def render_feedback(
    wording: int,
    outcome: Outcome,
    unit: str,
    points: int,
    answered: int,
    value: int | None = None,
) -> str:
    """Feedback for one turn. `value` is the evaluator's label, shown only when not accepted."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome}")
    _check_int("points", points)
    _check_int("answered", answered)
    if not 0 <= points <= answered:
        raise ValueError(f"invalid running total: {points}/{answered}")
    if outcome == "incorrect":
        _check_int("value", value)
    elif value is not None:
        raise ValueError(f"value is shown only for incorrect answers, not {outcome}")
    template = feedback_templates(wording)[outcome]
    return template.format(value=value, unit=unit, points=points, answered=answered)
