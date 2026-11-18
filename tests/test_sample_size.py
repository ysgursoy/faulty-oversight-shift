import unittest

from oversight_study.analysis.sample_size import (
    decide_lists,
    required_lists,
    sd_upper_factor,
    sigma_upper,
)


class SampleSizeTests(unittest.TestCase):
    def test_upper_factors_match_protocol_table(self) -> None:
        for n, expected in ((12, 1.25), (20, 1.18), (30, 1.14)):
            self.assertAlmostEqual(sd_upper_factor(n), expected, delta=0.01)

    def test_sigma_upper_scales_estimate(self) -> None:
        self.assertAlmostEqual(sigma_upper(0.30, 20), 0.30 * sd_upper_factor(20))

    def test_worked_examples(self) -> None:
        self.assertEqual(required_lists(0.30, 0.15), 32)
        self.assertEqual(required_lists(0.40, 0.15), 56)

    def test_more_variance_needs_more_lists(self) -> None:
        self.assertLess(required_lists(0.2, 0.15), required_lists(0.3, 0.15))

    def test_decision_against_n_max(self) -> None:
        within = decide_lists(32, 60)
        self.assertEqual((within.lists, within.exceeds_n_max), (32, False))
        over = decide_lists(80, 60)
        self.assertEqual((over.lists, over.exceeds_n_max), (60, True))

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            sd_upper_factor(1)
        with self.assertRaises(ValueError):
            required_lists(0.3, 0)
        with self.assertRaises(ValueError):
            sigma_upper(-0.1, 20)
        with self.assertRaises(ValueError):
            decide_lists(0, 10)


if __name__ == "__main__":
    unittest.main()
