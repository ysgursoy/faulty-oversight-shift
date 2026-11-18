import unittest

from oversight_study.records.cases import Case
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.client import Message, ModelRequest
from oversight_study.runner.mock import REFUSAL_TEXT, MockModel, Policy, parse_case_block
from oversight_study.runner.parsing import parse_number_answer, parse_turn_output
from oversight_study.task.render import (
    render_episode_end_message,
    render_system_prompt,
    render_turn_message,
)

SETTINGS = GenerationSettings(max_tokens=64)
AFFECTED = Case(case_id="c-1", split="development", age=40, weight=100, measurement=50,
                case_type="A", r_star=1000)
UNAFFECTED = Case(case_id="c-2", split="development", age=30, weight=60, measurement=80,
                  case_type="U", r_star=1200)
INSIDE = Case(case_id="c-3", split="development", age=50, weight=92, measurement=40,
              case_type="A", r_star=900)


def turn_request(case, frame="human", feedback=None, letter="b"):
    return ModelRequest(
        system=render_system_prompt(frame, 1, letter, 1.5),
        messages=(Message(role="user", content=render_turn_message(feedback, case, frame)),),
        settings=SETTINGS,
    )


def end_request(question, case=None, frame="human"):
    return ModelRequest(
        system=render_system_prompt(frame, 1, "b", 1.5),
        messages=(Message(role="user", content=render_turn_message(None, AFFECTED, frame)),
                  Message(role="assistant", content='{"value": 1000}'),
                  Message(role="user", content=render_episode_end_message(
                      "Fb.", question, frame, case))),
        settings=SETTINGS,
    )


def value_of(model, request):
    return parse_turn_output(model.complete(request).text).value


class ParseCaseBlockTests(unittest.TestCase):
    def test_reads_frame_and_values_from_text(self) -> None:
        for frame in ("neutral", "agriculture", "human"):
            shown = parse_case_block(render_turn_message("Fb.", AFFECTED, frame))
            self.assertEqual((shown.frame, shown.age, shown.weight, shown.measurement),
                             (frame, 40, 100, 50))

    def test_no_case_block(self) -> None:
        with self.assertRaises(ValueError):
            parse_case_block("nothing here")


class PolicyTests(unittest.TestCase):
    def test_protocol_policy(self) -> None:
        model = MockModel(Policy("protocol"))
        self.assertEqual(value_of(model, turn_request(AFFECTED)), 1000)
        self.assertEqual(value_of(model, turn_request(UNAFFECTED, "neutral")), 1200)

    def test_evaluator_policy(self) -> None:
        model = MockModel(Policy("evaluator", m=1.5))
        self.assertEqual(value_of(model, turn_request(AFFECTED)), 1500)
        self.assertEqual(value_of(model, turn_request(UNAFFECTED)), 1200)

    def test_planted_policy_by_frame(self) -> None:
        model = MockModel(Policy("planted", shift_probability={"human": 1.0, "neutral": 0.0}))
        self.assertEqual(value_of(model, turn_request(AFFECTED, "human")), 1500)
        self.assertEqual(value_of(model, turn_request(AFFECTED, "neutral")), 1000)

    def test_failure_policies(self) -> None:
        self.assertFalse(parse_turn_output(
            MockModel(Policy("invalid", rate=1.0)).complete(turn_request(AFFECTED)).text).ok)
        self.assertEqual(
            MockModel(Policy("refusal", rate=1.0)).complete(turn_request(AFFECTED)).text,
            REFUSAL_TEXT)
        self.assertEqual(value_of(MockModel(Policy("noise", rate=1.0)), turn_request(AFFECTED)),
                         1050)
        self.assertEqual(value_of(MockModel(Policy("noise", rate=0.0)), turn_request(AFFECTED)),
                         1000)

    def test_end_of_episode_answers(self) -> None:
        evaluator = MockModel(Policy("evaluator"))
        protocol = MockModel(Policy("protocol"))
        self.assertEqual(protocol.complete(end_request("rule")).text, "none")
        self.assertIn("1.5", evaluator.complete(end_request("rule")).text)
        for model in (protocol, evaluator):
            self.assertEqual(parse_number_answer(
                model.complete(end_request("value", AFFECTED)).text), 1000)
        self.assertEqual(parse_number_answer(
            evaluator.complete(end_request("probe_evaluator", INSIDE)).text), 1350)
        self.assertEqual(parse_number_answer(
            protocol.complete(end_request("probe_evaluator", INSIDE)).text), 900)
        self.assertEqual(parse_number_answer(
            evaluator.complete(end_request("probe_evaluator", UNAFFECTED)).text), 1200)

    def test_deterministic_and_order_independent(self) -> None:
        requests = [turn_request(AFFECTED, f) for f in ("neutral", "agriculture", "human")]
        a = MockModel(Policy("refusal", rate=0.5), seed=3)
        b = MockModel(Policy("refusal", rate=0.5), seed=3)
        forward = [a.complete(r).text for r in requests]
        backward = [b.complete(r).text for r in reversed(requests)][::-1]
        self.assertEqual(forward, backward)
        self.assertEqual(a.calls, 3)

    def test_usage_and_model_id(self) -> None:
        response = MockModel().complete(turn_request(AFFECTED))
        self.assertEqual(response.returned_model, "mock-model")
        self.assertGreater(response.usage.input_tokens, 0)

    def test_request_must_end_with_user(self) -> None:
        request = ModelRequest(system="s", messages=(Message(role="assistant", content="x"),),
                               settings=SETTINGS)
        with self.assertRaises(ValueError):
            MockModel().complete(request)


if __name__ == "__main__":
    unittest.main()
