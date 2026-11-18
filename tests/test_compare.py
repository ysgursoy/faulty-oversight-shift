import math
import unittest

import numpy as np
import pandas as pd

from oversight_study.analysis.compare import (
    acquisition,
    build_table,
    delta,
    frame_difference,
    paired_bootstrap,
    robustness_label,
    two_extremes,
    wording_robustness,
)
from oversight_study.analysis.validate import AnalysisInputError

CELL = ["frame", "condition", "wording", "episode", "replicate"]


def measure(values: dict) -> pd.Series:
    rows = [(frame, condition, wording, episode, 0, v)
            for (frame, condition, wording), series in values.items()
            for episode, v in enumerate(series)]
    df = pd.DataFrame(rows, columns=[*CELL, "value"])
    return df.set_index(CELL)["value"]


def primary(human_b, human_c, neutral_b, neutral_c, wording=1):
    return {("human", "b-1.5", wording): human_b, ("human", "c", wording): human_c,
            ("neutral", "b-1.5", wording): neutral_b, ("neutral", "c", wording): neutral_c}


class TableTests(unittest.TestCase):
    def test_build_table_and_frame_difference(self) -> None:
        t = build_table(measure(primary([0.2, 0.4], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0])), 1.5)
        self.assertEqual(list(t.index), [0, 1])
        np.testing.assert_allclose(frame_difference(t).to_numpy(), [0.2, 0.4])
        self.assertAlmostEqual(delta(t), 0.3)

    def test_other_multiplier_cells_dropped(self) -> None:
        values = primary([0.2], [1.0], [0.0], [1.0])
        values[("human", "b-2", 1)] = [0.9]
        t = build_table(measure(values), 1.5)
        self.assertNotIn(("human", "b", 1), [c for c in t.columns if t[c].iloc[0] == 0.9])

    def test_missing_cell_gives_nan_difference(self) -> None:
        values = primary([0.2, 0.4], [1.0, 1.0], [0.0, 0.0], [1.0, math.nan])
        t = build_table(measure(values), 1.5)
        self.assertTrue(math.isnan(frame_difference(t).iloc[1]))
        self.assertAlmostEqual(delta(t), 0.2)


class BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        self.t = build_table(measure(primary(
            list(rng.uniform(0.3, 0.7, n)), [1.0] * n, list(rng.uniform(0.0, 0.4, n)), [1.0] * n)),
            1.5)

    def test_deterministic_and_contains_point(self) -> None:
        a = paired_bootstrap(self.t, delta, n=500, seed=1)
        b = paired_bootstrap(self.t, delta, n=500, seed=1)
        self.assertEqual(a, b)
        self.assertLess(a.low, a.point)
        self.assertLess(a.point, a.high)
        self.assertEqual(a.n_episodes, 60)

    def test_episode_ids_shared_across_columns(self) -> None:
        # A statistic that is zero only if both columns come from the same episodes.
        def same_rows(x):
            return float((x[("human", "b", 1)] - x[("human", "b", 1)].to_numpy()).abs().sum())
        self.assertEqual(paired_bootstrap(self.t, same_rows, n=50).high, 0.0)

    def test_undefined_statistic_raises(self) -> None:
        with self.assertRaises(AnalysisInputError):
            paired_bootstrap(self.t, lambda x: math.nan, n=10)
        tiny = self.t.iloc[:3].copy()
        tiny.iloc[0, 0] = math.nan

        def all_defined(x):
            col = x[("human", "b", 1)]
            return math.nan if col.isna().any() else col.mean()
        # A NaN in the data makes the statistic undefined, so no interval is produced.
        with self.assertRaises(AnalysisInputError):
            paired_bootstrap(tiny, all_defined, n=200, seed=0)


class CriteriaTests(unittest.TestCase):
    def test_robustness_labels(self) -> None:
        self.assertEqual(robustness_label(-0.1, 0.1, 0.15), "equivalent")
        self.assertEqual(robustness_label(0.2, 0.4, 0.15), "wording_dependent")
        self.assertEqual(robustness_label(-0.4, -0.2, 0.15), "wording_dependent")
        self.assertEqual(robustness_label(-0.1, 0.3, 0.15), "inconclusive")

    def test_wording_robustness_on_identical_wordings(self) -> None:
        values = {**primary([0.2, 0.5, 0.3], [1.0] * 3, [0.1, 0.0, 0.2], [1.0] * 3),
                  **primary([0.2, 0.5, 0.3], [1.0] * 3, [0.1, 0.0, 0.2], [1.0] * 3, wording=2)}
        estimate, label = wording_robustness(build_table(measure(values), 1.5), 0.15, n=200)
        self.assertEqual((estimate.point, label), (0.0, "equivalent"))

    def test_acquisition_per_frame(self) -> None:
        values = {("human", "b-1.5", 1): [1.0, 0.5, 1.0], ("human", "c", 1): [0.0, 0.0, 0.5],
                  ("neutral", "b-1.5", 1): [0.0, 0.5, 0.0], ("neutral", "c", 1): [0.0, 0.5, 0.0]}
        result = acquisition(build_table(measure(values), 1.5), frames=("human", "neutral"),
                             n=200)
        self.assertTrue(result["human"]["above_zero"])
        self.assertFalse(result["neutral"]["above_zero"])


class TwoExtremesTests(unittest.TestCase):
    def test_fills_move_delta_in_opposite_directions(self) -> None:
        rows = []
        for frame in ("human", "neutral"):
            for condition in ("b-1.5", "c"):
                for episode in range(2):
                    for k, y in zip((8, 9, 10), (1000, 1000, None), strict=True):
                        rows.append({"frame": frame, "condition": condition, "wording": 1,
                                     "episode": episode, "replicate": 0, "k": k, "y": y,
                                     "r_star": 1000})
        low, high = two_extremes(pd.DataFrame(rows), 1.5)
        self.assertLess(low, high)
        self.assertAlmostEqual(low, -2 / 3)
        self.assertAlmostEqual(high, 2 / 3)


if __name__ == "__main__":
    unittest.main()
