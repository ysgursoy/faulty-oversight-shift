"""Typed case, probe, and episode-list records. These models are the single definition
used by generation, released data files, and generated JSON Schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oversight_study.evaluator import is_affected
from oversight_study.records.enums import CaseType, DatasetSplit, Pair, Side
from oversight_study.task import r_star

ListUse = Literal["trial", "pilot", "main"]
N_TURNS = 40
N_PROBES = 4

Key = tuple[int, int]


class _CaseFields(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    split: DatasetSplit
    age: int = Field(ge=18, le=90)
    weight: int = Field(ge=40, le=140)
    measurement: int = Field(ge=10, le=120)
    r_star: int

    @property
    def key(self) -> Key:
        return (self.weight, self.measurement)

    @model_validator(mode="after")
    def _check_r_star(self):
        expected = r_star(self.weight, self.measurement)
        if self.r_star != expected:
            raise ValueError(f"r_star {self.r_star} != protocol value {expected}")
        return self


class Case(_CaseFields):
    """A pool case. `case_type` is hidden metadata, never model-visible."""

    case_type: CaseType

    @model_validator(mode="after")
    def _check_case_type(self):
        from oversight_study.generation.pools import case_type_of  # avoid import cycle

        expected = case_type_of(self.weight, self.measurement)
        if self.case_type != expected:
            raise ValueError(f"case_type {self.case_type} != region type {expected}")
        return self


class ProbeCase(_CaseFields):
    """An unseen end-of-episode probe; `inside` is the affected side of the boundary."""

    pair: Pair
    side: Side

    @model_validator(mode="after")
    def _check_side(self):
        if is_affected(self.weight, self.measurement) != (self.side == Side.inside):
            raise ValueError(f"side {self.side} does not match the affected region")
        return self


class EpisodeList(BaseModel):
    """Serialized episode list: 40 turn ids in order and 4 probe ids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: DatasetSplit
    use: ListUse
    list_index: int = Field(ge=0)
    seed: int
    turn_case_ids: tuple[str, ...]
    probe_case_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check_shape(self):
        if len(self.turn_case_ids) != N_TURNS:
            raise ValueError(f"need {N_TURNS} turns, got {len(self.turn_case_ids)}")
        if len(self.probe_case_ids) != N_PROBES:
            raise ValueError(f"need {N_PROBES} probes, got {len(self.probe_case_ids)}")
        ids = self.turn_case_ids + self.probe_case_ids
        if len(set(ids)) != len(ids):
            raise ValueError("episode list ids must be unique")
        return self
