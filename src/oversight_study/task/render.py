"""Model-visible message rendering. Only allowlisted fields reach the text: frame labels,
the three case values, evaluator feedback, and (disclosed-rule condition only) `m`."""

from __future__ import annotations

from fractions import Fraction
from typing import Protocol

from oversight_study.records.enums import ConditionLetter, Frame, QuestionType
from oversight_study.task.frames import load_frame
from oversight_study.task.templates import load_template

_SLOT = "\x00evaluator\x00"

_QUESTION_TEMPLATES = {
    QuestionType.rule: "episode_end/rule_question.txt",
    QuestionType.value: "episode_end/value_question.txt",
    QuestionType.probe_protocol: "episode_end/probe_protocol.txt",
    QuestionType.probe_evaluator: "episode_end/probe_evaluator.txt",
}


class ShownCase(Protocol):
    age: int
    weight: int
    measurement: int


def format_m(m: Fraction | float) -> str:
    return f"{float(m):g}"


def _evaluator_text(frame: Frame, wording: int, letter: ConditionLetter, m) -> str:
    if letter == ConditionLetter.a:
        return ""
    text = load_template(f"system/evaluator_sentence_{wording}.txt")
    if letter == ConditionLetter.f:
        spec = load_frame(frame)
        rule = load_template("system/explicit_rule.txt").format(
            weight_label=spec.weight_label,
            measurement_label=spec.measurement_label,
            m=format_m(m),
        )
        text = f"{text} {rule}"
    return text


def render_system_prompt(
    frame: Frame | str, wording: int, letter: ConditionLetter | str, m: Fraction | float
) -> str:
    """Wording template filled with frame fields; the evaluator sentence is dropped in `a`
    (its line or inline slot disappears) and followed by the disclosed rule in `f`."""
    frame, letter = Frame(frame), ConditionLetter(letter)
    if wording not in (1, 2):
        raise ValueError(f"unknown wording: {wording}")
    filled = load_template(f"system/wording_{wording}.txt").format(
        evaluator=_SLOT, **load_frame(frame).model_dump()
    )
    evaluator = _evaluator_text(frame, wording, letter, m)
    if evaluator:
        return filled.replace(_SLOT, evaluator)
    return filled.replace(f"\n{_SLOT}\n", "\n").replace(f" {_SLOT}", "").replace(_SLOT, "")


def render_case(case: ShownCase, frame: Frame | str) -> str:
    """Three label lines. Reads only age, weight, measurement from the case."""
    spec = load_frame(Frame(frame))
    return load_template("messages/case.txt").format(
        age_label=spec.age_label,
        weight_label=spec.weight_label,
        measurement_label=spec.measurement_label,
        age=case.age,
        weight=case.weight,
        measurement=case.measurement,
    )


def render_turn_message(prev_feedback: str | None, case: ShownCase, frame: Frame | str) -> str:
    """First turn and the no-feedback condition send the case alone."""
    block = render_case(case, frame)
    if prev_feedback is None:
        return block
    return load_template("messages/turn_with_feedback.txt").format(
        feedback=prev_feedback, case=block
    )


def render_question(question: QuestionType | str, frame: Frame | str,
                    case: ShownCase | None = None) -> str:
    question = QuestionType(question)
    template = load_template(_QUESTION_TEMPLATES[question])
    if question == QuestionType.rule:
        if case is not None:
            raise ValueError("the rule question shows no case")
        return template
    if case is None:
        raise ValueError(f"{question} needs a case")
    spec = load_frame(Frame(frame))
    return template.format(
        age_label=spec.age_label,
        weight_label=spec.weight_label,
        measurement_label=spec.measurement_label,
        age=case.age,
        weight=case.weight,
        measurement=case.measurement,
    )


def render_episode_end_message(
    last_feedback: str | None,
    question: QuestionType | str,
    frame: Frame | str,
    case: ShownCase | None = None,
) -> str:
    """Last-turn feedback (if any), the transition sentence, then the question."""
    parts = [] if last_feedback is None else [last_feedback]
    parts.append(load_template("episode_end/transition.txt"))
    parts.append(render_question(question, frame, case))
    return "\n\n".join(parts)
