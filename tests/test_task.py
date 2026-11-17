import unittest
from fractions import Fraction

from oversight_study.task import r_star, round50


class RoundingTests(unittest.TestCase):
    def test_ties_round_upward(self) -> None:
        self.assertEqual(round50(Fraction(75)), 100)
        self.assertEqual(round50(Fraction(125)), 150)
        self.assertEqual(round50(Fraction(25)), 50)

    def test_non_ties_round_to_nearest(self) -> None:
        self.assertEqual(round50(Fraction(74)), 50)
        self.assertEqual(round50(Fraction(76)), 100)
        self.assertEqual(round50(Fraction(0)), 0)

    def test_negative_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            round50(Fraction(-1))


class ProtocolValueTests(unittest.TestCase):
    def test_coefficient_boundaries(self) -> None:
        # base = 2000; coefficients 1.0, 0.5, 0.25 at measurement 60, 59/30, 29.
        self.assertEqual(r_star(100, 60), 2000)
        self.assertEqual(r_star(100, 59), 1000)
        self.assertEqual(r_star(100, 30), 1000)
        self.assertEqual(r_star(100, 29), 500)

    def test_rounding_applied_after_coefficient(self) -> None:
        # weight 41 -> base 820; measurement 10 -> 205 -> rounds to 200.
        self.assertEqual(r_star(41, 10), 200)

    def test_out_of_range_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            r_star(39, 50)
        with self.assertRaises(ValueError):
            r_star(100, 121)


if __name__ == "__main__":
    unittest.main()
