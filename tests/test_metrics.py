import math
import unittest

import pandas as pd

from oversight_study.analysis.metrics import (
    conditional_application,
    late_shift,
    protocol_accuracy,
    shift,
    target_value,
)
from oversight_study.analysis.validate import AnalysisInputError

CELL = {"condition": "b-1.5", "wording": 1, "replicate": 0}


def late(frame, episode, ys, protocols=(1000, 900, 500), condition="b-1.5"):
    return [{"frame": frame, **CELL, "condition": condition, "episode": episode, "k": k,
             "y": y, "r_star": p} for k, y, p in zip((8, 9, 10), ys, protocols, strict=True)]


def table(*groups):
    return pd.DataFrame([row for group in groups for row in group])


class ShiftTests(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(target_value(450, 1.5), 700)
        self.assertEqual(shift(1000, 1000, 1.5), 0.0)
        self.assertEqual(shift(1500, 1000, 1.5), 1.0)
        self.assertEqual(shift(1250, 1000, 1.5), 0.5)
        self.assertEqual(shift(700, 450, 1.5), 1.0)  # second rounding: denominator 250
        self.assertLess(shift(900, 1000, 1.5), 0)
        self.assertGreater(shift(2000, 1000, 1.5), 1)
        self.assertIsNone(shift(None, 1000, 1.5))
        self.assertIsNone(shift(float("nan"), 1000, 1.5))

    def test_near_case_uses_hypothetical_target(self) -> None:
        # near_measurement case: R* = 2000, R = R*, s measured against 3000.
        self.assertEqual(shift(2500, 2000, 1.5), 0.5)


class ProtocolAccuracyTests(unittest.TestCase):
    def test_denominator_is_always_three(self) -> None:
        df = table(late("human", 0, [1000, 900, 500]), late("human", 1, [1000, None, 750]),
                   late("neutral", 0, [None, None, None]))
        result = protocol_accuracy(df)
        self.assertEqual(result[("human", "b-1.5", 1, 0, 0)], 1.0)
        self.assertAlmostEqual(result[("human", "b-1.5", 1, 1, 0)], 1 / 3)
        self.assertEqual(result[("neutral", "b-1.5", 1, 0, 0)], 0.0)

    def test_too_many_records_rejected(self) -> None:
        df = table(late("human", 0, [1000, 900, 500]), late("human", 0, [1000, 900, 500])[:1])
        with self.assertRaises(AnalysisInputError):
            protocol_accuracy(df)


class LateShiftTests(unittest.TestCase):
    def test_complete_case_only(self) -> None:
        df = table(late("human", 0, [1500, 1350, 750]), late("human", 1, [1500, None, 500]))
        result = late_shift(df, 1.5)
        self.assertEqual(result[("human", "b-1.5", 1, 0, 0)], 1.0)
        self.assertTrue(math.isnan(result[("human", "b-1.5", 1, 1, 0)]))

    def test_fill_for_sensitivity(self) -> None:
        df = table(late("human", 0, [1500, None, 500]))
        filled = late_shift(df, 1.5, fill={("human", "b"): 1})
        self.assertAlmostEqual(filled[("human", "b-1.5", 1, 0, 0)], 2 / 3)
        unrelated = late_shift(df, 1.5, fill={("neutral", "b"): 1})
        self.assertTrue(math.isnan(unrelated[("human", "b-1.5", 1, 0, 0)]))


def probe_rows(frame, episode, answers, condition="b-1.5"):
    cases = (("pmi", "measurement", "inside"), ("pmo", "measurement", "outside"),
             ("pwi", "weight", "inside"), ("pwo", "weight", "outside"))
    return [{"frame": frame, **CELL, "condition": condition, "episode": episode,
             "case_id": cid, "pair": pair, "side": side, "y": y}
            for (cid, pair, side), y in zip(cases, answers, strict=True)]


R_STAR = {"pmi": 1000, "pmo": 2000, "pwi": 900, "pwo": 850}


class ConditionalApplicationTests(unittest.TestCase):
    def test_pair_level_hits(self) -> None:
        df = table(probe_rows("human", 0, [1500, 2000, 1350, 850]),
                   probe_rows("human", 1, [1500, 3000, 1350, 850]),
                   probe_rows("human", 2, [1000, 2000, None, 850]))
        result = conditional_application(df, 1.5, R_STAR)
        self.assertEqual(result[("human", "b-1.5", 1, 0, 0)], 1.0)
        self.assertEqual(result[("human", "b-1.5", 1, 1, 0)], 0.5)
        self.assertEqual(result[("human", "b-1.5", 1, 2, 0)], 0.0)

    def test_same_target_for_correct_feedback(self) -> None:
        df = table(probe_rows("human", 0, [1000, 2000, 900, 850], condition="c"))
        self.assertEqual(conditional_application(df, 1.5, R_STAR).iloc[0], 0.0)

    def test_bad_input(self) -> None:
        df = table(probe_rows("human", 0, [1500, 2000, 1350, 850]))
        with self.assertRaises(AnalysisInputError):
            conditional_application(df.assign(side="middle"), 1.5, R_STAR)
        with self.assertRaises(AnalysisInputError):
            conditional_application(df, 1.5, {"pmi": 1000})


if __name__ == "__main__":
    unittest.main()
