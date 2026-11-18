import unittest
from pathlib import Path

import pandas as pd

from oversight_study.analysis import diagnostics as d
from oversight_study.analysis.load import _row
from oversight_study.generation import build_dataset, load_seeds
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.episode import EpisodeContext, run_episode
from oversight_study.runner.mock import MockModel, Policy

FIXTURES = Path(__file__).parent / "fixtures"


def turns_for(episode, condition, policy, frame="human"):
    ctx = EpisodeContext(episode=episode, frame=frame, condition=condition, wording=1, m=1.5,
                         stage="trial", dataset_split="development", replicate=0, run_id="r",
                         requested_model="mock-model",
                         settings=GenerationSettings(max_tokens=64), prompt_version="0" * 64)
    return pd.DataFrame([_row(t) for t in run_episode(ctx, MockModel(policy)).turns])


class TurnDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        episode = build_dataset(seeds).episodes["trial"][0]
        cls.evaluator = turns_for(episode, "b-1.5", Policy("evaluator"))
        cls.protocol = turns_for(episode, "c", Policy("protocol"))
        cls.invalid = turns_for(episode, "c", Policy("invalid", rate=1.0), frame="neutral")

    def test_response_distribution(self) -> None:
        dist = d.response_distribution(pd.concat([self.evaluator, self.protocol, self.invalid]))
        b = dist.loc[("human", "b-1.5", 1)]
        self.assertEqual((b.exact_r, b.exact_r_star, b.n), (10, 30, 40))
        self.assertEqual(dist.loc[("human", "c", 1)].exact_r_star, 40)
        self.assertEqual(dist.loc[("neutral", "c", 1)].format, 40)
        self.assertAlmostEqual(b.exact_r_share, 0.25)

    def test_learning_curve_and_first_phase(self) -> None:
        curve = d.learning_curve(self.evaluator, 1.5)
        self.assertEqual(list(curve.index.get_level_values("k")), list(range(1, 11)))
        self.assertTrue((curve.mean_s == 1.0).all())
        self.assertTrue((curve.exact_r_star == 0.0).all())
        self.assertEqual(d.first_phase_shift(self.protocol, 1.5).iloc[0], 0.0)

    def test_deviations_and_unaffected_accuracy(self) -> None:
        dev = d.deviations(self.evaluator).iloc[0]
        self.assertGreater(dev.absolute, 0)
        self.assertGreaterEqual(dev.relative, 0.5)
        acc = d.unaffected_accuracy(self.protocol).iloc[0]
        self.assertEqual((acc.exact_r_star, acc.mean_relative_error), (1.0, 0.0))

    def test_near_shift_reports_strips_apart(self) -> None:
        near = d.near_shift(self.evaluator, 1.5)
        self.assertEqual(set(near.index.get_level_values("case_type")),
                         {"near_measurement", "near_weight"})
        self.assertTrue((near == 0.0).all())

    def test_unrounded_and_retries(self) -> None:
        self.assertEqual(d.unrounded_answers(self.evaluator).iloc[0], 0)
        retries = d.retry_recovery(self.invalid).iloc[0]
        self.assertEqual((retries.retried, retries.refusal_then_value), (40, 0))

    def test_extra_warning_rate(self) -> None:
        labels = {self.protocol.logical_query_id.iloc[0]: "warning",
                  self.protocol.logical_query_id.iloc[1]: "other"}
        self.assertAlmostEqual(d.extra_warning_rate(self.protocol, labels).iloc[0], 1 / 40)

    def test_format_count(self) -> None:
        self.assertEqual(d.format_count(31, 38), "31 of 38")


BASE = {"frame": "human", "wording": 1, "replicate": 0}


def end_rows(condition, episode, answers):
    """answers: {(case_id, pair, side): (protocol_answer, evaluator_answer)}"""
    rows = []
    for (cid, pair, side), (proto, evaluator) in answers.items():
        for qt, y in (("probe_protocol", proto), ("probe_evaluator", evaluator)):
            rows.append({**BASE, "condition": condition, "episode": episode, "case_id": cid,
                         "pair": pair, "side": side, "question_type": qt, "y": y})
    return rows


R_STAR = {"pmi": 1000, "pmo": 2000, "pwi": 900, "pwo": 850}


class EndDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        learned = {("pmi", "measurement", "inside"): (1000, 1500),
                   ("pmo", "measurement", "outside"): (2000, 3000),
                   ("pwi", "weight", "inside"): (900, 1350),
                   ("pwo", "weight", "outside"): (850, 850)}
        baseline = {key: (proto, proto) for key, (proto, _) in learned.items()}
        self.end = pd.DataFrame(end_rows("b-1.5", 0, learned) + end_rows("c", 0, baseline))

    def test_two_rules_and_beyond_boundary(self) -> None:
        both = d.two_rules(self.end, R_STAR, 1.5)
        self.assertEqual(tuple(both.loc[("human", "b-1.5", 1)]), (2, 2))
        self.assertEqual(both.loc[("human", "c", 1)].both, 0)
        over = d.beyond_boundary(self.end, R_STAR, 1.5)
        self.assertEqual(tuple(over.loc[("human", "b-1.5", 1)]), (1, 2))

    def test_correct_feedback_control(self) -> None:
        control = d.correct_feedback_control(self.end, R_STAR, 1.5)
        self.assertEqual(tuple(control.loc[("human", "c", 1)]), (1, 1))

    def test_return_to_protocol(self) -> None:
        late = pd.DataFrame([{**BASE, "condition": "b-1.5", "episode": 0, "case_id": cid,
                              "k": k, "y": y, "r_star": 1000}
                             for cid, k, y in (("c8", 8, 1500), ("c9", 9, 1000),
                                               ("c10", 10, 1500))])
        end = pd.DataFrame([{**BASE, "condition": "b-1.5", "episode": 0, "case_id": cid,
                             "question_type": "value", "y": y}
                            for cid, y in (("c8", 1000), ("c9", 1000), ("c10", None))])
        result = d.return_to_protocol(late, end, 1.5).iloc[0]
        self.assertEqual(tuple(result), (2, 1, 1))

    def test_rule_description_rates(self) -> None:
        rc = pd.DataFrame([{**BASE, "condition": "b-1.5", "rule_class": c} for c in
                           ("same_as_generating", "consistent_with_history", "partial", "none")])
        rates = d.rule_description_rates(rc).iloc[0]
        self.assertEqual((rates.consistent, rates.same, rates.partial, rates.n),
                         (0.5, 0.25, 0.25, 4))


if __name__ == "__main__":
    unittest.main()
