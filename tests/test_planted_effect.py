import unittest

import pandas as pd

from oversight_study.analysis.compare import build_table, delta, paired_bootstrap
from oversight_study.analysis.metrics import late_shift, protocol_accuracy
from oversight_study.analysis.synthetic import make_synthetic_records
from oversight_study.analysis.validate import AnalysisInputError, check_duplicates, validate


def estimated_delta(study, measure="accuracy"):
    if measure == "accuracy":
        series = protocol_accuracy(study.late)
    else:
        series = late_shift(study.late, 1.5)
    return delta(build_table(series, 1.5))


class PlantedEffectTests(unittest.TestCase):
    def test_planted_effect_is_recovered(self) -> None:
        study = make_synthetic_records(effect=0.3, n_episodes=1000, seed=7)
        validate(study.late, study.probes)
        self.assertAlmostEqual(estimated_delta(study), 0.3, delta=0.05)
        # Higher human accuracy under faulty feedback means less shift toward the evaluator.
        self.assertAlmostEqual(estimated_delta(study, "shift"), -0.3, delta=0.05)

    def test_null_effect_is_near_zero(self) -> None:
        study = make_synthetic_records(effect=0.0, n_episodes=1000, seed=11)
        self.assertLess(abs(estimated_delta(study)), 0.05)

    def test_interval_covers_planted_effect(self) -> None:
        study = make_synthetic_records(effect=0.2, n_episodes=300, seed=3)
        t = build_table(protocol_accuracy(study.late), 1.5)
        estimate = paired_bootstrap(t, delta, n=1000, seed=0)
        self.assertLess(estimate.low, 0.2)
        self.assertGreater(estimate.high, 0.2)

    def test_accuracy_never_exceeds_one(self) -> None:
        study = make_synthetic_records(effect=0.6, n_episodes=200, seed=1)
        self.assertLessEqual(protocol_accuracy(study.late).max(), 1.0)
        with self.assertRaises(ValueError):
            make_synthetic_records(effect=0.7, n_episodes=10)


class AnalysisStopsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = make_synthetic_records(effect=0.1, n_episodes=20, seed=0)

    def test_duplicate_record_stops(self) -> None:
        late = pd.concat([self.study.late, self.study.late.iloc[:1]])
        with self.assertRaises(AnalysisInputError):
            check_duplicates(late)
        with self.assertRaises(AnalysisInputError):
            validate(late, self.study.probes)

    def test_missing_cell_stops(self) -> None:
        late = self.study.late
        missing = late[~((late.frame == "human") & (late.condition == "c") & (late.episode == 3))]
        with self.assertRaises(AnalysisInputError):
            validate(missing, self.study.probes)

    def test_unknown_category_stops(self) -> None:
        late = self.study.late.copy()
        late.loc[late.index[0], "frame"] = "finance"
        with self.assertRaises(AnalysisInputError):
            validate(late, self.study.probes)


if __name__ == "__main__":
    unittest.main()
