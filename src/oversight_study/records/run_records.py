"""Run records written to `results.jsonl`: one `turn` record per planned turn and one
`episode_end` record per end-of-episode question, exactly once per cell."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from oversight_study.records.enums import (
    CaseType,
    ConditionLetter,
    DatasetSplit,
    Frame,
    InvalidClass,
    Pair,
    QuestionType,
    Side,
    Stage,
    parse_condition,
)

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class GenerationSettings(BaseModel):
    model_config = _FROZEN

    temperature: float | None = None
    top_p: float | None = None
    thinking: str | None = None
    max_tokens: int = Field(gt=0)


class Usage(BaseModel):
    model_config = _FROZEN

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)


class Attempt(BaseModel):
    """One model call for a logical query; raw output always kept."""

    model_config = _FROZEN

    attempt_id: str = Field(min_length=1)
    raw_output: str
    attempt_class: Literal["valid", "refusal", "format"]
    extra_text: str = ""
    stop_reason: str | None = None
    usage: Usage


class RecordBase(BaseModel):
    model_config = _FROZEN

    dataset_split: DatasetSplit
    stage: Stage
    frame: Frame
    condition: str
    wording: Literal[1, 2]
    m: float = Field(gt=0)
    episode: int = Field(ge=0)
    replicate: Literal[0, 1] = 0
    requested_model: str = Field(min_length=1)
    returned_model: str | None = None
    generation_settings: GenerationSettings
    prompt_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    logical_query_id: str = Field(min_length=1)

    @field_validator("condition")
    @classmethod
    def _known_condition(cls, value: str) -> str:
        parse_condition(value)
        return value

    @property
    def condition_letter(self) -> ConditionLetter:
        return parse_condition(self.condition)[0]


def _check_y(y: int | None, invalid_class: InvalidClass | None) -> None:
    if (y is None) != (invalid_class is not None):
        raise ValueError("y must be None exactly when invalid_class is set")


class TurnRecord(RecordBase):
    record_type: Literal["turn"] = "turn"
    turn: int = Field(ge=1, le=40)
    k: int | None = Field(default=None, ge=1, le=10)
    case_id: str = Field(min_length=1)
    case_type: CaseType
    attempts: tuple[Attempt, ...] = Field(min_length=1, max_length=2)
    y: StrictInt | None
    invalid_class: InvalidClass | None = None
    r_star: int
    r: int
    ratio_r_over_r_star: float
    score_r: Literal[0, 1]
    score_r_star: Literal[0, 1]
    feedback_sent: str | None
    extra_text: str = ""

    @model_validator(mode="after")
    def _check(self):
        if (self.k is not None) != (self.case_type == CaseType.A):
            raise ValueError("k is set exactly for affected (A) cases")
        _check_y(self.y, self.invalid_class)
        if not math.isclose(self.ratio_r_over_r_star, self.r / self.r_star):
            raise ValueError("ratio_r_over_r_star must equal r / r_star")
        if self.score_r != int(self.y == self.r) or self.score_r_star != int(self.y == self.r_star):
            raise ValueError("scores must be exact-match scores of y")
        if len(self.attempts) == 2 and self.attempts[0].attempt_class == "valid":
            raise ValueError("a retry follows only an invalid first attempt")
        if (self.attempts[-1].attempt_class == "valid") != (self.y is not None):
            raise ValueError("final attempt is valid exactly when y is set")
        no_feedback = self.condition_letter == ConditionLetter.a
        if no_feedback != (self.feedback_sent is None):
            raise ValueError("feedback_sent is None exactly in the no-feedback condition")
        return self


_PROBE_QUESTIONS = (QuestionType.probe_protocol, QuestionType.probe_evaluator)


class EpisodeEndRecord(RecordBase):
    record_type: Literal["episode_end"] = "episode_end"
    question_type: QuestionType
    case_id: str | None = None
    pair: Pair | None = None
    side: Side | None = None
    raw_answer: str
    y: StrictInt | None = None
    invalid_class: InvalidClass | None = None

    @model_validator(mode="after")
    def _check(self):
        is_probe = self.question_type in _PROBE_QUESTIONS
        if is_probe != (self.pair is not None) or is_probe != (self.side is not None):
            raise ValueError("pair and side are set exactly for probe questions")
        is_rule = self.question_type == QuestionType.rule
        if is_rule != (self.case_id is None):
            raise ValueError("case_id is None exactly for the rule question")
        if is_rule:
            if self.y is not None:
                raise ValueError("the rule question has no numeric answer")
        else:
            _check_y(self.y, self.invalid_class)
        if self.condition_letter == ConditionLetter.a and self.question_type != QuestionType.value:
            raise ValueError("the no-feedback condition asks only value questions")
        return self
