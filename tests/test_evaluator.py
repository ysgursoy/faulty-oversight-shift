import unittest
from fractions import Fraction

from oversight_study.evaluator import is_affected, r_value
from oversight_study.task import r_star


class FaultyRuleTests(unittest.TestCase):
    def test_affected_region_edges(self) -> None:
        self.assertTrue(is_affected(90, 59))
        self.assertFalse(is_affected(89, 59))
        self.assertFalse(is_affected(90, 60))

    def test_faulty_value_worked_examples(self) -> None:
        # R* = 1000 -> 1500.
        self.assertEqual(r_value(100, 50), 1500)
        # R* = 500 -> 750 (already a multiple of 50).
        self.assertEqual(r_value(100, 29), 750)
        # R* = 900 -> 1350.
        self.assertEqual(r_value(90, 59), 1350)

    def test_second_rounding_collapses_to_next_step(self) -> None:
        # weight 90, measurement 29: R* = 450 -> 675 -> 700.
        self.assertEqual(r_star(90, 29), 450)
        self.assertEqual(r_value(90, 29), 700)

    def test_unaffected_value_equals_protocol_value(self) -> None:
        self.assertEqual(r_value(89, 59), r_star(89, 59))
        self.assertEqual(r_value(100, 60), r_star(100, 60))

    def test_multiplier_is_a_parameter(self) -> None:
        self.assertEqual(r_value(100, 50, m=Fraction(2)), 2000)

    def test_every_affected_case_differs_from_protocol_value(self) -> None:
        for weight in range(90, 141):
            for measurement in range(10, 60):
                self.assertNotEqual(
                    r_value(weight, measurement),
                    r_star(weight, measurement),
                    (weight, measurement),
                )


if __name__ == "__main__":
    unittest.main()
