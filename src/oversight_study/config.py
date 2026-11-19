"""Versioned run configuration: model settings, stage cells, precheck, and analysis.

Configs hold no credentials. Every run stores its resolved config and its hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oversight_study.records.enums import (
    ConditionLetter,
    DatasetSplit,
    Frame,
    Stage,
    parse_condition,
)

_STRICT = ConfigDict(frozen=True, extra="forbid")
UNDECIDED = "TBD"


class ConfigNotReadyError(RuntimeError):
    """Raised when a config still holds an undecided value needed for the requested step."""


class RetryPolicy(BaseModel):
    model_config = _STRICT

    max_transport_retries: int = Field(ge=0)
    initial_backoff_seconds: float = Field(gt=0)
    max_backoff_seconds: float = Field(gt=0)


class Prices(BaseModel):
    """Price assumptions in USD per million tokens, recorded with run costs."""

    model_config = _STRICT

    input: float | None = None
    output: float | None = None
    cache_write: float | None = None
    cache_read: float | None = None


class ModelConfig(BaseModel):
    model_config = _STRICT

    provider: Literal["anthropic"]
    model_id: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=1)
    thinking: Literal["adaptive", "between_tools", "disabled"] | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens: int = Field(gt=0)
    prompt_caching: bool = True
    retry: RetryPolicy
    prices: Prices = Prices()

    def require_runnable(self) -> None:
        if self.model_id == UNDECIDED:
            raise ConfigNotReadyError("model_id is undecided; paid runs are refused")

    @model_validator(mode="after")
    def _check_thinking(self):
        if self.thinking == "between_tools" and self.effort in ("xhigh", "max"):
            raise ValueError("between_tools thinking is accepted only at effort high or below")
        return self


class ListRange(BaseModel):
    model_config = _STRICT

    start: int = Field(default=0, ge=0)
    stop: int | None = Field(default=None, gt=0)  # None = up to the stage list count

    def indices(self, list_count: int) -> range:
        stop = list_count if self.stop is None else min(self.stop, list_count)
        return range(self.start, stop)


class CellGroup(BaseModel):
    model_config = _STRICT

    conditions: tuple[str, ...] = Field(min_length=1)
    wordings: tuple[Literal[1, 2], ...] = Field(min_length=1)
    frames: tuple[Frame, ...] = Field(min_length=1)
    lists: ListRange = ListRange()
    replicates: tuple[Literal[0, 1], ...] = (0,)
    optional: bool = False

    @field_validator("conditions")
    @classmethod
    def _known(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            parse_condition(value)
        return values


PlannedCell = tuple[Frame, str, int, int, int]  # frame, condition, wording, list, replicate


class StageConfig(BaseModel):
    model_config = _STRICT

    stage: Stage
    split: DatasetSplit
    design_version: str = Field(min_length=1)
    m: float = Field(gt=0)
    list_count: int | None = Field(default=None, gt=0)  # None until fixed for the stage
    order_seed: int
    max_cost_usd: float | None = Field(default=None, gt=0)  # stop a run past this
    cells: tuple[CellGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self):
        if self.stage in (Stage.trial, Stage.pilot) and self.split != DatasetSplit.development:
            raise ValueError(f"{self.stage} runs on the development split only")
        for group in self.cells:
            for cid in group.conditions:
                letter, m = parse_condition(cid)
                if letter in (ConditionLetter.b, ConditionLetter.f) and m != self.m:
                    raise ValueError(f"condition {cid} does not match stage m = {self.m:g}")
        return self

    def planned_cells(self, include_optional: bool = False) -> list[PlannedCell]:
        if self.list_count is None:
            raise ConfigNotReadyError(f"{self.stage}: list_count is not fixed yet")
        cells = [
            (frame, condition, wording, index, replicate)
            for group in self.cells
            if include_optional or not group.optional
            for frame in group.frames
            for condition in group.conditions
            for wording in group.wordings
            for index in group.lists.indices(self.list_count)
            for replicate in group.replicates
        ]
        if len(set(cells)) != len(cells):
            raise ValueError("stage config plans the same cell twice")
        return cells


class PrecheckConfig(BaseModel):
    model_config = _STRICT

    frames: tuple[Frame, ...]
    wordings: tuple[Literal[1, 2], ...]
    threshold: float = Field(gt=0, le=1)
    development_subset: dict[str, int]
    evaluation_by_wording: dict[int, dict[str, int]]
    sample_seed: int


class AnalysisConfig(BaseModel):
    model_config = _STRICT

    delta: float = Field(gt=0)  # wording equivalence margin
    delta_min: float | None = Field(default=None, gt=0)  # fixed before pilot data is seen
    n_max: int | None = Field(default=None, gt=0)
    n_min: int = Field(default=1, gt=0)  # minimum list count; added after the pilot
    n_pilot: int = Field(gt=1)
    bootstrap_n: int = Field(gt=0)
    bootstrap_seed: int
    level: float = Field(gt=0, lt=100)
    robustness_level: float = Field(gt=0, lt=100)

    def require_ready_for_pilot(self) -> None:
        missing = [name for name in ("delta_min", "n_max") if getattr(self, name) is None]
        if missing:
            raise ConfigNotReadyError(f"undecided before pilot: {', '.join(missing)}")


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config(path: Path | str, model: type[ConfigT]) -> ConfigT:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping")
    return model(**data)


def resolved_config_hash(config: BaseModel) -> str:
    """SHA-256 of the canonical JSON of a resolved config."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
