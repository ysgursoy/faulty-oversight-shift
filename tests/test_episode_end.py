import unittest
from pathlib import Path

from oversight_study.generation import build_dataset, load_seeds
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.episode import EpisodeContext, run_episode
from oversight_study.runner.episode_end import ask_episode_end, planned_questions
from oversight_study.runner.mock import MockModel, Policy

FIXTURES = Path(__file__).parent / "fixtures"


class Recorder:
    def __init__(self, inner):
        self.inner, self.requests = inner, []

    def complete(self, request):
        self.requests.append(request)
        return self.inner.complete(request)


def context(episode, condition):
    return EpisodeContext(episode=episode, frame="human", condition=condition, wording=1, m=1.5,
                          stage="trial", dataset_split="development", replicate=0, run_id="r",
                          requested_model="mock-model",
                          settings=GenerationSettings(max_tokens=64), prompt_version="0" * 64)


class EpisodeEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.episode = build_dataset(seeds).episodes["trial"][0]

    def run_end(self, condition, policy):
        ctx = context(self.episode, condition)
        result = run_episode(ctx, MockModel(policy))
        recorder = Recorder(MockModel(policy))
        return result, ask_episode_end(result, ctx, recorder), recorder

    def test_call_counts(self) -> None:
        for condition, expected in (("a", 3), ("c", 12), ("b-1.5", 12), ("f-1.5", 12)):
            with self.subTest(condition=condition):
                _, records, recorder = self.run_end(condition, Policy("protocol"))
                self.assertEqual(len(records), expected)
                self.assertEqual(len(recorder.requests), expected)

    def test_each_question_sees_only_the_frozen_history(self) -> None:
        result, _, recorder = self.run_end("b-1.5", Policy("protocol"))
        for request in recorder.requests:
            self.assertEqual(request.messages[:-1], result.history)
            self.assertEqual(len(request.messages), 81)
            self.assertEqual(request.system, result.system)

    def test_messages_start_with_last_feedback_then_transition(self) -> None:
        result, _, recorder = self.run_end("c", Policy("protocol"))
        for request in recorder.requests:
            self.assertTrue(request.last_user_text.startswith(result.last_feedback + "\n\n"
                                                              "The calculation stage is over"))
        _, _, no_feedback = self.run_end("a", Policy("protocol"))
        self.assertTrue(all(r.last_user_text.startswith("The calculation stage is over")
                            for r in no_feedback.requests))

    def test_record_fields(self) -> None:
        _, records, _ = self.run_end("b-1.5", Policy("evaluator"))
        by_type = {}
        for r in records:
            by_type.setdefault(r.question_type.value, []).append(r)
        self.assertEqual({k: len(v) for k, v in by_type.items()},
                         {"rule": 1, "value": 3, "probe_protocol": 4, "probe_evaluator": 4})
        rule = by_type["rule"][0]
        self.assertIsNone(rule.case_id)
        self.assertIn("1.5", rule.raw_answer)
        last = [c.case_id for c in self.episode.last_affected]
        self.assertEqual([r.case_id for r in by_type["value"]], last)
        self.assertEqual([r.y for r in by_type["value"]],
                         [c.r_star for c in self.episode.last_affected])
        inside = [r for r in by_type["probe_evaluator"] if r.side == "inside"]
        probes = {p.case_id: p for p in self.episode.probes}
        for r in inside:
            self.assertGreater(r.y, probes[r.case_id].r_star)
        self.assertEqual(len({r.logical_query_id for r in records}), 12)

    def test_unparsable_numeric_answer_is_format(self) -> None:
        ctx = context(self.episode, "c")
        result = run_episode(ctx, MockModel(Policy("protocol")))

        class Wordy:
            def complete(self, request):
                return MockModel(Policy("protocol")).complete(request).model_copy(
                    update={"text": "It is 1000 or maybe 1500."})

        records = ask_episode_end(result, ctx, Wordy())
        numeric = [r for r in records if r.question_type != "rule"]
        self.assertTrue(all(r.y is None and r.invalid_class == "format" for r in numeric))

    def test_planned_questions_order(self) -> None:
        order = [q.value for q, _ in planned_questions(context(self.episode, "f-1.5"))]
        self.assertEqual(order[:4], ["rule", "value", "value", "value"])
        self.assertEqual(order[4:], ["probe_protocol"] * 4 + ["probe_evaluator"] * 4)


if __name__ == "__main__":
    unittest.main()
