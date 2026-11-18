"""Paid model client for the Anthropic Messages API. Never used by tests or CI.

Transport failures (connection, timeout, rate limit, 5xx) are retried with exponential
backoff and reported through `on_event`; they are not parse retries. SDK-level retries
are disabled so this class owns the retry policy.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from oversight_study.config import ConfigNotReadyError, ModelConfig
from oversight_study.records.run_records import Usage
from oversight_study.runner.client import ModelRequest, ModelResponse

NO_PAID_RUNS_ENV = "NO_PAID_RUNS"
API_KEY_ENV = "ANTHROPIC_API_KEY"
EPHEMERAL = {"type": "ephemeral"}

EventSink = Callable[[dict], None]


class PaidRunRefusedError(RuntimeError):
    """A paid call was attempted where paid runs are not allowed."""


def thinking_param(setting: str | None) -> dict | None:
    """`None` -> omit; `disabled`; `enabled:<budget_tokens>`."""
    if setting is None:
        return None
    if setting == "disabled":
        return {"type": "disabled"}
    if setting.startswith("enabled:"):
        return {"type": "enabled", "budget_tokens": int(setting.split(":", 1)[1])}
    raise ValueError(f"unknown thinking setting: {setting!r}")


def build_api_kwargs(config: ModelConfig, request: ModelRequest) -> dict[str, Any]:
    """Messages API arguments. With caching on, breakpoints mark the system prompt and the
    last message so turns of one episode and its end-of-episode copies share the prefix."""
    cache = config.prompt_caching
    system = {"type": "text", "text": request.system}
    if cache:
        system["cache_control"] = EPHEMERAL
    messages = []
    for index, message in enumerate(request.messages):
        block = {"type": "text", "text": message.content}
        if cache and index == len(request.messages) - 1:
            block["cache_control"] = EPHEMERAL
        messages.append({"role": message.role, "content": [block]})
    kwargs: dict[str, Any] = {
        "model": config.model_id,
        "max_tokens": request.settings.max_tokens,
        "system": [system],
        "messages": messages,
    }
    if request.settings.temperature is not None:
        kwargs["temperature"] = request.settings.temperature
    if request.settings.top_p is not None:
        kwargs["top_p"] = request.settings.top_p
    thinking = thinking_param(request.settings.thinking)
    if thinking is not None:
        kwargs["thinking"] = thinking
    return kwargs


def to_response(api_response: Any) -> ModelResponse:
    text = "".join(block.text for block in api_response.content
                   if getattr(block, "type", None) == "text")
    usage = api_response.usage
    return ModelResponse(
        text=text,
        returned_model=getattr(api_response, "model", None),
        stop_reason=getattr(api_response, "stop_reason", None),
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        ),
    )


def _retryable(error: Exception) -> bool:
    import anthropic

    if isinstance(error, anthropic.APIConnectionError | anthropic.RateLimitError):
        return True
    return isinstance(error, anthropic.APIStatusError) and error.status_code >= 500


class AnthropicClient:
    def __init__(self, config: ModelConfig, client: Any = None,
                 on_event: EventSink | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if os.environ.get(NO_PAID_RUNS_ENV) == "1":
            raise PaidRunRefusedError(f"{NO_PAID_RUNS_ENV}=1: paid runs are disabled here")
        config.require_runnable()
        if client is None:
            if not os.environ.get(API_KEY_ENV):
                raise ConfigNotReadyError(f"{API_KEY_ENV} is not set")
            import anthropic

            client = anthropic.Anthropic(max_retries=0)
        self.config = config
        self._client = client
        self._on_event = on_event or (lambda event: None)
        self._sleep = sleep

    def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs = build_api_kwargs(self.config, request)
        policy = self.config.retry
        delay = policy.initial_backoff_seconds
        for attempt in range(policy.max_transport_retries + 1):
            try:
                return to_response(self._client.messages.create(**kwargs))
            except Exception as error:
                if not _retryable(error) or attempt == policy.max_transport_retries:
                    self._on_event({"event": "api_error", "attempt": attempt + 1,
                                    "error": type(error).__name__, "final": True})
                    raise
                self._on_event({"event": "api_retry", "attempt": attempt + 1,
                                "error": type(error).__name__, "sleep_seconds": delay})
                self._sleep(delay)
                delay = min(delay * 2, policy.max_backoff_seconds)
        raise AssertionError("unreachable")
