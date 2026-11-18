"""Classifier specification (`annotations/spec.yaml`) and classifier JSON parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from oversight_study.classification.rules import RULE_CLASSES, Condition, ParsedRule
from oversight_study.config import UNDECIDED, ConfigNotReadyError
from oversight_study.task.templates import prompts_dir

_STRICT = ConfigDict(frozen=True, extra="forbid")


class Decoding(BaseModel):
    model_config = _STRICT

    temperature: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int = Field(gt=0)


class ClassifierTask(BaseModel):
    model_config = _STRICT

    prompt: str
    output: Literal["parsed_rule", "label"]
    labels: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _labels(self):
        if (self.output == "label") != bool(self.labels):
            raise ValueError("label tasks need labels; parsed_rule tasks have none")
        return self


class DecisionRules(BaseModel):
    model_config = _STRICT

    rule_classes: tuple[str, ...]
    rule_class_source: str
    unparsable_classifier_output: str

    @model_validator(mode="after")
    def _classes(self):
        if self.rule_classes != RULE_CLASSES:
            raise ValueError(f"rule_classes must be {RULE_CLASSES}")
        return self


class ManualReview(BaseModel):
    model_config = _STRICT

    seed: int
    per_task_sample: int = Field(gt=0)
    stratify_by: tuple[str, ...]
    blinded_fields: tuple[str, ...]
    reviewers: int = Field(ge=1)


class Adjudication(BaseModel):
    model_config = _STRICT

    preserve_originals: Literal[True]
    rule: str


class ClassifierSpec(BaseModel):
    model_config = _STRICT

    model_id: str
    decoding: Decoding
    tasks: dict[Literal["rule_to_predicate", "invalid_output", "extra_text"], ClassifierTask]
    decision_rules: DecisionRules
    manual_review: ManualReview
    adjudication: Adjudication

    @model_validator(mode="after")
    def _all_tasks(self):
        missing = {"rule_to_predicate", "invalid_output", "extra_text"} - set(self.tasks)
        if missing:
            raise ValueError(f"missing classifier tasks: {sorted(missing)}")
        return self

    def require_runnable(self) -> None:
        if self.model_id == UNDECIDED:
            raise ConfigNotReadyError("classifier model_id is undecided")

    def prompt_paths(self) -> dict[str, Path]:
        return {name: prompts_dir() / task.prompt for name, task in self.tasks.items()}


def load_classifier_spec(path: Path | str) -> ClassifierSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping")
    spec = ClassifierSpec(**data)
    for name, prompt in spec.prompt_paths().items():
        if not prompt.is_file():
            raise FileNotFoundError(f"classifier prompt for {name} not found: {prompt}")
    return spec


class _RuleJson(BaseModel):
    model_config = _STRICT

    none: bool
    conditions: tuple[Condition, ...] = ()
    multiplier: float | None = None
    rounds_to_50: bool


def parsed_rule_from_json(data: dict) -> ParsedRule:
    """Validate the rule_to_predicate classifier output and convert it to a ParsedRule."""
    rule = _RuleJson(**data)
    return ParsedRule(
        conditions=rule.conditions,
        multiplier=rule.multiplier,
        rounds_to_50=rule.rounds_to_50,
        none=rule.none,
    )
