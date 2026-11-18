import unittest
from pathlib import Path

from oversight_study.generation import build_dataset, load_seeds
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.client import ModelRequest
from oversight_study.runner.episode import EpisodeContext, run_episode
from oversight_study.runner.mock import MockModel, Policy

FIXTURES = Path(__file__).parent / "fixtures"


class RecordingClient:
    def __init__(self, inner):
        self.inner = inner
        self.requests: list[ModelRequest] = []

    def complete(self, request):
        self.requests.append(request)
        return self.inner.complete(request)


def context(episode, condition="b-1.5", frame="human", wording=1):
    return EpisodeContext(
        episode=episode, frame=frame, condition=condition, wording=wording, m=1.5,
        stage="trial", dataset_split="development", replicate=0, run_id="run-1",
        requested_model="mock-model", settings=GenerationSettings(max_tokens=64),
        prompt_version="0" * 64,
    )


class EpisodeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.episode = build_dataset(seeds).episodes["trial"][0]

    def test_forty_valid_turns_and_scores(self) -> None:
        result = run_episode(context(self.episode, "c"), MockModel(Policy("protocol")))
        self.assertEqual(len(result.turns), 40)
        self.assertEqual(len(result.history), 80)
        self.assertTrue(all(len(t.attempts) == 1 for t in result.turns))
        self.assertTrue(result.last_feedback.endswith("Total: 40/40"))
        self.assertEqual(sorted(t.k for t in result.turns if t.k), list(range(1, 11)))

    def test_faulty_feedback_scores_against_r(self) -> None:
        result = run_episode(context(self.episode, "b-1.5"), MockModel(Policy("protocol")))
        self.assertTrue(result.last_feedback.endswith("Total: 30/40"))
        first_a = next(t for t in result.turns if t.case_type == "A")
        self.assertIn("INCORRECT", first_a.feedback_sent)
        self.assertIn(f"{first_a.r} mg", first_a.feedback_sent)
        evaluator = run_episode(context(self.episode, "b-1.5"), MockModel(Policy("evaluator")))
        self.assertTrue(evaluator.last_feedback.endswith("Total: 40/40"))

    def test_no_feedback_condition(self) -> None:
        client = RecordingClient(MockModel(Policy("protocol")))
        result = run_episode(context(self.episode, "a"), client)
        self.assertIsNone(result.last_feedback)
        self.assertTrue(all(t.feedback_sent is None for t in result.turns))
        self.assertTrue(all(not r.last_user_text.startswith("Evaluator")
                            for r in client.requests))
        self.assertNotIn("automated evaluator", result.system)

    def test_feedback_opens_next_message(self) -> None:
        client = RecordingClient(MockModel(Policy("protocol")))
        result = run_episode(context(self.episode, "c"), client)
        for turn, request in zip(result.turns[:-1], client.requests[1:], strict=True):
            self.assertTrue(request.last_user_text.startswith(turn.feedback_sent + "\n\nNext case:"))
        self.assertFalse(client.requests[0].last_user_text.startswith("Evaluator"))

    def test_retry_is_not_added_to_history(self) -> None:
        client = RecordingClient(MockModel(Policy("invalid", rate=1.0)))
        result = run_episode(context(self.episode, "c"), client)
        self.assertEqual(len(client.requests), 80)
        self.assertTrue(all(len(t.attempts) == 2 and t.y is None for t in result.turns))
        self.assertTrue(all(t.invalid_class == "format" for t in result.turns))
        self.assertEqual(client.requests[0], client.requests[1])  # identical retry request
        self.assertEqual(len(client.requests[2].messages), 3)  # one exchange + next user
        self.assertTrue(result.last_feedback.endswith("Total: 0/40"))
        self.assertIn("INVALID", result.turns[0].feedback_sent)

    def test_mixed_failures_keep_fixed_turn_count(self) -> None:
        result = run_episode(context(self.episode, "b-1.5"),
                             MockModel(Policy("invalid", rate=0.3), seed=1))
        self.assertEqual(len(result.turns), 40)
        retried = [t for t in result.turns if len(t.attempts) == 2]
        self.assertTrue(retried)
        self.assertTrue(any(t.y is not None for t in retried))

    def test_b_and_c_identical_until_first_affected_turn(self) -> None:
        recorders = {}
        for condition in ("b-1.5", "c"):
            recorders[condition] = RecordingClient(MockModel(Policy("protocol")))
            run_episode(context(self.episode, condition), recorders[condition])
        first_a = next(i for i, c in enumerate(self.episode.turns) if c.case_type == "A")
        for i in range(first_a + 1):
            self.assertEqual(recorders["b-1.5"].requests[i], recorders["c"].requests[i])
        self.assertNotEqual(recorders["b-1.5"].requests[first_a + 1],
                            recorders["c"].requests[first_a + 1])

    def test_record_ids_and_values(self) -> None:
        result = run_episode(context(self.episode, "f-1.5", frame="neutral", wording=2),
                             MockModel(Policy("protocol")))
        first = result.turns[0]
        self.assertEqual(first.logical_query_id, "run-1|trial|neutral|f-1.5|w2|e0|r0|t1")
        self.assertEqual(first.attempts[0].attempt_id, first.logical_query_id + "#1")
        self.assertIn("known error", result.system)
        self.assertTrue(result.turns[-1].feedback_sent.startswith("Evaluator:"))


if __name__ == "__main__":
    unittest.main()
