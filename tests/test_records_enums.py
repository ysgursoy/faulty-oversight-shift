import unittest
from fractions import Fraction

from oversight_study.generation import CASE_TYPES
from oversight_study.records.enums import (
    CaseType,
    ConditionLetter,
    DatasetSplit,
    Stage,
    condition_id,
    parse_condition,
)


class ConditionIdTests(unittest.TestCase):
    def test_ids(self) -> None:
        self.assertEqual(condition_id("a", 1.5), "a")
        self.assertEqual(condition_id("c"), "c")
        self.assertEqual(condition_id("b", 1.5), "b-1.5")
        self.assertEqual(condition_id("b", Fraction(3, 2)), "b-1.5")
        self.assertEqual(condition_id("b", 2), "b-2")
        self.assertEqual(condition_id("b", 1.25), "b-1.25")
        self.assertEqual(condition_id("f", 1.5), "f-1.5")

    def test_b_and_f_need_m(self) -> None:
        for letter in ("b", "f"):
            with self.assertRaises(ValueError):
                condition_id(letter)

    def test_unknown_letter(self) -> None:
        with self.assertRaises(ValueError):
            condition_id("x", 1)

    def test_parse_round_trip(self) -> None:
        for cid in ("a", "c", "b-1.5", "f-1.5", "b-2", "b-1.25"):
            letter, m = parse_condition(cid)
            self.assertEqual(condition_id(letter, m), cid)
        self.assertEqual(parse_condition("b-1.5"), (ConditionLetter.b, 1.5))
        self.assertEqual(parse_condition("a"), (ConditionLetter.a, None))

    def test_parse_rejects(self) -> None:
        for bad in ("x-1", "b", "f", "a-1.5", "c-2", "b-", "B-1.5", "b-1.5x", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_condition(bad)


class VocabularyTests(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual([s.value for s in DatasetSplit], ["development", "evaluation"])
        self.assertEqual([s.value for s in Stage], ["trial", "pilot", "main"])

    def test_case_types_match_generator(self) -> None:
        self.assertEqual(tuple(c.value for c in CaseType), CASE_TYPES)


if __name__ == "__main__":
    unittest.main()
