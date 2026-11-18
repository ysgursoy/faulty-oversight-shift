"""Frame placeholder values. Frames change only surface wording, never the arithmetic."""

from __future__ import annotations

from functools import cache

import yaml
from pydantic import BaseModel, ConfigDict, Field

from oversight_study.records.enums import Frame
from oversight_study.task.templates import load_template


class FrameSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(min_length=1)
    object: str = Field(min_length=1)
    age_label: str = Field(min_length=1)
    weight_label: str = Field(min_length=1)
    measurement_label: str = Field(min_length=1)
    output_label: str = Field(min_length=1)
    unit: str = Field(min_length=1)


def parse_frame(text: str) -> FrameSpec:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError("frame file must be a mapping")
    return FrameSpec(**data)


@cache
def load_frame(frame: Frame | str) -> FrameSpec:
    return parse_frame(load_template(f"frames/{Frame(frame).value}.yaml"))
