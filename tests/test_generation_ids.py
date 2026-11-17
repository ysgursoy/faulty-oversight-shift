import re
import unittest

from oversight_study.generation import CASE_TYPES, case_id, check_unique_ids

ID_PATTERN = re.compile(r"^[cp]-[0-9a-f]{12}$")


class CaseIdTests(unittest.TestCase):
    def test_format_and_stability(self) -> None:
        first = case_id("c", "development", 95, 40)
        self.assertRegex(first, ID_PATTERN)
        self.assertEqual(first, case_id("c", "development", 95, 40))

    def test_depends_on_kind_split_and_key(self) -> None:
        base = case_id("c", "development", 95, 40)
        self.assertNotEqual(base, case_id("p", "development", 95, 40))
        self.assertNotEqual(base, case_id("c", "evaluation", 95, 40))
        self.assertNotEqual(base, case_id("c", "development", 95, 41))

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            case_id("x", "development", 95, 40)  # type: ignore[arg-type]

    def test_ids_do_not_contain_case_types(self) -> None:
        value = case_id("c", "development", 95, 40)
        for case_type in CASE_TYPES:
            self.assertNotIn(case_type, value)

    def test_duplicates_detected(self) -> None:
        check_unique_ids(["c-1", "c-2"])
        with self.assertRaises(ValueError):
            check_unique_ids(["c-1", "c-1"])


if __name__ == "__main__":
    unittest.main()
