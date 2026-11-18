import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import anthropic
import httpx2

from oversight_study.config import ConfigNotReadyError, ModelConfig, load_config
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.anthropic_client import (
    AnthropicClient,
    PaidRunRefusedError,
    build_api_kwargs,
    thinking_param,
)
from oversight_study.runner.client import Message, ModelRequest
from oversight_study.scaffold import repository_root

CONFIG = load_config(repository_root(Path(__file__)) / "configs/models/claude.yaml",
                     ModelConfig).model_copy(update={"model_id": "claude-test-20260101"})
REQUEST = ModelRequest(
    system="sys",
    messages=(Message(role="user", content="u1"), Message(role="assistant", content="a1"),
              Message(role="user", content="u2")),
    settings=GenerationSettings(temperature=0.0, max_tokens=64),
)
API_RESPONSE = SimpleNamespace(
    content=[SimpleNamespace(type="text", text='{"value": 1500}')],
    model="claude-test-20260101", stop_reason="end_turn",
    usage=SimpleNamespace(input_tokens=120, output_tokens=6, cache_creation_input_tokens=100,
                          cache_read_input_tokens=None),
)


class StubMessages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def stub(outcomes):
    return SimpleNamespace(messages=StubMessages(outcomes))


def connection_error():
    return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.invalid"))


def status_error(code):
    request = httpx2.Request("POST", "https://api.invalid")
    return anthropic.APIStatusError("x", response=httpx2.Response(code, request=request), body=None)


class RequestShapeTests(unittest.TestCase):
    def test_kwargs_and_cache_breakpoints(self) -> None:
        kwargs = build_api_kwargs(CONFIG, REQUEST)
        self.assertEqual(kwargs["model"], "claude-test-20260101")
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", kwargs["messages"][0]["content"][0])
        self.assertEqual(kwargs["messages"][-1]["content"][0]["cache_control"],
                         {"type": "ephemeral"})
        self.assertEqual([m["role"] for m in kwargs["messages"]], ["user", "assistant", "user"])
        self.assertNotIn("thinking", kwargs)

    def test_caching_off(self) -> None:
        kwargs = build_api_kwargs(CONFIG.model_copy(update={"prompt_caching": False}), REQUEST)
        self.assertNotIn("cache_control", kwargs["system"][0])

    def test_thinking_settings(self) -> None:
        self.assertIsNone(thinking_param(None))
        self.assertEqual(thinking_param("disabled"), {"type": "disabled"})
        self.assertEqual(thinking_param("enabled:2048"), {"type": "enabled", "budget_tokens": 2048})
        with self.assertRaises(ValueError):
            thinking_param("on")


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        # CI sets NO_PAID_RUNS=1; these tests use a stub transport, never the API.
        patcher = mock.patch.dict(os.environ, {"NO_PAID_RUNS": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_usage_mapping(self) -> None:
        response = AnthropicClient(CONFIG, client=stub([API_RESPONSE])).complete(REQUEST)
        self.assertEqual(response.text, '{"value": 1500}')
        self.assertEqual(response.returned_model, "claude-test-20260101")
        self.assertEqual((response.usage.cache_creation_input_tokens,
                          response.usage.cache_read_input_tokens), (100, 0))

    def test_transport_errors_retried_with_backoff(self) -> None:
        events, sleeps = [], []
        client = AnthropicClient(CONFIG, client=stub([connection_error(), status_error(529),
                                                      API_RESPONSE]),
                                 on_event=events.append, sleep=sleeps.append)
        client.complete(REQUEST)
        self.assertEqual(sleeps, [2.0, 4.0])
        self.assertEqual([e["event"] for e in events], ["api_retry", "api_retry"])

    def test_client_errors_not_retried(self) -> None:
        events = []
        client = AnthropicClient(CONFIG, client=stub([status_error(400)]), on_event=events.append,
                                 sleep=lambda s: None)
        with self.assertRaises(anthropic.APIStatusError):
            client.complete(REQUEST)
        self.assertTrue(events[-1]["final"])

    def test_gives_up_after_max_retries(self) -> None:
        retries = CONFIG.retry.max_transport_retries
        client = AnthropicClient(CONFIG, client=stub([connection_error()] * (retries + 1)),
                                 sleep=lambda s: None)
        with self.assertRaises(anthropic.APIConnectionError):
            client.complete(REQUEST)

    def test_refusals(self) -> None:
        with self.assertRaises(ConfigNotReadyError):
            AnthropicClient(CONFIG.model_copy(update={"model_id": "TBD"}), client=stub([]))
        with mock.patch.dict(os.environ, {"NO_PAID_RUNS": "1"}), \
                self.assertRaises(PaidRunRefusedError):
            AnthropicClient(CONFIG, client=stub([]))
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ConfigNotReadyError):
            AnthropicClient(CONFIG)


if __name__ == "__main__":
    unittest.main()
