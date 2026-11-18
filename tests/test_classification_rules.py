import unittest
from fractions import Fraction

from oversight_study.classification.rules import (
    Condition,
    ParsedRule,
    apply_rule,
    classify_rule,
)

GEN = (Condition("weight", ">=", 90), Condition("measurement", "<", 60))

# Small episode-like history: affected, near both strips, far unaffected.
HISTORY = [
    (100, 50), (90, 29), (140, 59),          # affected
    (100, 62), (95, 65),                     # near_measurement
    (87, 50), (85, 20),                      # near_weight
    (60, 80), (120, 100), (50, 40), (70, 15),  # far unaffected
]
# Same history without weights 85-89: a threshold of 85 cannot be told apart.
HISTORY_NO_NEAR_WEIGHT = [c for c in HISTORY if not 85 <= c[0] < 90]


class ApplyRuleTests(unittest.TestCase):
    def test_generating_rule_reproduces_evaluator(self) -> None:
        values = apply_rule(ParsedRule(GEN, Fraction(3, 2)), [(100, 50), (90, 29), (100, 62)])
        self.assertEqual(values, [1500, 700, 2000])

    def test_without_rounding_keeps_exact_product(self) -> None:
        values = apply_rule(ParsedRule(GEN, 1.5, rounds_to_50=False), [(90, 29)])
        self.assertEqual(values, [Fraction(675)])

    def test_empty_rule_returns_protocol_values(self) -> None:
        self.assertEqual(apply_rule(ParsedRule(none=True), [(100, 50)]), [1000])


class ClassifyRuleTests(unittest.TestCase):
    def test_same_as_generating_including_equivalent_forms(self) -> None:
        self.assertEqual(classify_rule(ParsedRule(GEN, 1.5), HISTORY), "same_as_generating")
        equivalent = (Condition("weight", ">", 89), Condition("measurement", "<=", 59))
        self.assertEqual(classify_rule(ParsedRule(equivalent, 1.5), HISTORY), "same_as_generating")

    def test_consistent_but_not_generating(self) -> None:
        loose = (Condition("weight", ">=", 85), Condition("measurement", "<", 60))
        rule = ParsedRule(loose, 1.5)
        self.assertEqual(classify_rule(rule, HISTORY_NO_NEAR_WEIGHT), "consistent_with_history")
        self.assertEqual(classify_rule(rule, HISTORY), "partial")

    def test_missing_rounding_is_partial(self) -> None:
        self.assertEqual(classify_rule(ParsedRule(GEN, 1.5, rounds_to_50=False), HISTORY), "partial")

    def test_band_up_description_is_partial_at_main_multiplier(self) -> None:
        # "Heavy cases use the next coefficient band" ~ multiplier 2 on the affected region.
        self.assertEqual(classify_rule(ParsedRule(GEN, 2), HISTORY), "partial")

    def test_wrong_direction_and_conditions_is_none(self) -> None:
        wrong = (Condition("weight", "<", 60), Condition("measurement", ">=", 80))
        self.assertEqual(classify_rule(ParsedRule(wrong, 0.5), HISTORY), "none")

    def test_no_rule(self) -> None:
        self.assertEqual(classify_rule(ParsedRule(none=True), HISTORY), "none")
        self.assertEqual(classify_rule(ParsedRule(GEN, None), HISTORY), "none")
        self.assertEqual(classify_rule(ParsedRule((), 1.5), HISTORY), "none")

    def test_one_condition_right_is_partial(self) -> None:
        half = (Condition("weight", ">=", 90),)
        self.assertEqual(classify_rule(ParsedRule(half, 0.9), HISTORY), "partial")

    def test_empty_history_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_rule(ParsedRule(GEN, 1.5), [])


if __name__ == "__main__":
    unittest.main()
