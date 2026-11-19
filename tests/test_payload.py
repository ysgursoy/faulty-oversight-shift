import unittest
from pathlib import Path

from oversight_study.generation import build_dataset, load_seeds
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.client import Message, ModelRequest
from oversight_study.runner.episode import EpisodeContext, run_episode
from oversight_study.runner.mock import MockModel, Policy
from oversight_study.runner.request_builder import build_request
from oversight_study.task.render import render_system_prompt, render_turn_message
from oversight_study.validation.payload import PayloadLeakError, check_payload

FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = GenerationSettings(max_tokens=64)


def request(system, user):
    return ModelRequest(system=system, messages=(Message(role="user", content=user),),
                        settings=SETTINGS)


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.episode = build_dataset(seeds).episodes["trial"][0]
        cls.case = cls.episode.turns[0]
        cls.user = render_turn_message("Evaluator: accepted. Score: 1/1", cls.case, "human")

    def test_full_mock_episodes_have_no_leaks(self) -> None:
        for condition in ("a", "c", "b-1.5", "f-1.5"):
            for wording in (1, 2):
                ctx = EpisodeContext(
                    episode=self.episode, frame="agriculture", condition=condition,
                    wording=wording, m=1.5, stage="trial", dataset_split="development",
                    replicate=0, run_id="r", requested_model="mock-model", settings=SETTINGS,
                    prompt_version="0" * 64)
                run_episode(ctx, MockModel(Policy("evaluator")))  # raises on any leak

    def test_disclosed_rule_may_show_m(self) -> None:
        check_payload(request(render_system_prompt("human", 1, "f", 1.5), self.user), 1.5)

    def test_injected_leaks_raise(self) -> None:
        system = render_system_prompt("human", 1, "b", 1.5)
        for leak in (self.case.case_id, "near_weight", "case type U", "condition b-1.5",
                     "this case is affected", '{"k": 8}', "multiply by 1.5"):
            with self.subTest(leak=leak), self.assertRaises(PayloadLeakError):
                check_payload(request(system, self.user + "\n" + leak), 1.5)

    def test_m_outside_rule_sentence_raises_even_in_f(self) -> None:
        system = render_system_prompt("human", 1, "f", 1.5) + "\nNote: factor 1.5."
        with self.assertRaises(PayloadLeakError):
            check_payload(request(system, self.user), 1.5)

    def test_case_type_text_in_system_raises(self) -> None:
        system = render_system_prompt("human", 1, "c", 1.5) + " near_measurement"
        with self.assertRaises(PayloadLeakError):
            check_payload(request(system, self.user), 1.5)

    def test_extra_forbidden_tokens(self) -> None:
        with self.assertRaises(PayloadLeakError):
            check_payload(request("sys", "hello secret"), 1.5, extra_forbidden=["secret"])

    def test_build_request_checks(self) -> None:
        with self.assertRaises(PayloadLeakError):
            build_request("sys", [Message(role="user", content=self.case.case_id)], SETTINGS, 1.5)
        build_request("sys", [Message(role="user", content=self.user)], SETTINGS, 1.5)

    def test_model_text_in_history_is_not_checked(self) -> None:
        history = (Message(role="user", content=self.user),
                   Message(role="assistant", content='1000 x 1.5 = 1500. {"value": 1500}'),
                   Message(role="user", content=self.user))
        check_payload(ModelRequest(system="sys", messages=history, settings=SETTINGS), 1.5)
        with self.assertRaises(PayloadLeakError):
            check_payload(ModelRequest(system="sys", messages=(*history[:2], Message(
                role="user", content=self.user + " 1.5")), settings=SETTINGS), 1.5)

    def test_numbers_like_m_inside_other_numbers_are_fine(self) -> None:
        check_payload(request("sys", "Value 11.55 and 21.5x? no: 1500 and 0.5"), 1.5)


if __name__ == "__main__":
    unittest.main()
