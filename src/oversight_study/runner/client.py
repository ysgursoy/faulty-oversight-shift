"""Model client interface. Every request is built from the allowlisted message text only."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from oversight_study.records.run_records import GenerationSettings, Usage

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Message(BaseModel):
    model_config = _FROZEN

    role: Literal["user", "assistant"]
    content: str


class ModelRequest(BaseModel):
    model_config = _FROZEN

    system: str
    messages: tuple[Message, ...] = Field(min_length=1)
    settings: GenerationSettings

    @property
    def last_user_text(self) -> str:
        if self.messages[-1].role != "user":
            raise ValueError("a request must end with a user message")
        return self.messages[-1].content


class ModelResponse(BaseModel):
    model_config = _FROZEN

    text: str
    returned_model: str | None = None
    stop_reason: str | None = None
    usage: Usage


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...
