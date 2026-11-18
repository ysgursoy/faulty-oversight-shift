"""The only place that builds model requests; every request passes the payload check."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.client import Message, ModelRequest
from oversight_study.validation.payload import check_payload


def build_request(system: str, messages: Sequence[Message], settings: GenerationSettings,
                  m: Fraction | float) -> ModelRequest:
    request = ModelRequest(system=system, messages=tuple(messages), settings=settings)
    check_payload(request, m)
    return request
