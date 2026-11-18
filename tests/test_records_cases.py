import unittest

from pydantic import ValidationError

from oversight_study.records.cases import Case, EpisodeList, ProbeCase

CASE = {"case_id": "c-1", "split": "development", "age": 40, "weight": 100, "measurement": 50,
            "case_type": "A", "r_star": 1000}
PROBE = {"case_id": "p-1", "split": "evaluation", "age": 40, "weight": 100, "measurement": 55,
             "pair": "measurement", "side": "inside", "r_star": 1000}
LIST = {"split": "development", "use": "trial", "list_index": 0, "seed": 1,
            "turn_case_ids": tuple(f"c-{i}" for i in range(40)),
            "probe_case_ids": tuple(f"p-{i}" for i in range(4))}


class CaseModelTests(unittest.TestCase):
    def test_valid_case(self) -> None:
        case = Case(**CASE)
        self.assertEqual(case.key, (100, 50))
        self.assertEqual(case.split, "development")
        self.assertEqual(case.model_dump(mode="json")["case_type"], "A")

    def test_out_of_range_fields(self) -> None:
        for field, value in (("age", 17), ("age", 91), ("weight", 39), ("weight", 141),
                             ("measurement", 9), ("measurement", 121)):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                Case(**{**CASE, field: value})

    def test_wrong_r_star(self) -> None:
        with self.assertRaises(ValidationError):
            Case(**{**CASE, "r_star": 1050})

    def test_case_type_must_match_region(self) -> None:
        with self.assertRaises(ValidationError):
            Case(**{**CASE, "case_type": "U"})
        Case(**{**CASE, "weight": 100, "measurement": 62, "r_star": 2000,
                "case_type": "near_measurement"})

    def test_unknown_split_and_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            Case(**{**CASE, "split": "dev"})
        with self.assertRaises(ValidationError):
            Case(**CASE, k=1)

    def test_frozen_and_hashable(self) -> None:
        case = Case(**CASE)
        with self.assertRaises(ValidationError):
            case.age = 41
        self.assertEqual(len({case, Case(**CASE)}), 1)


class ProbeModelTests(unittest.TestCase):
    def test_valid_probe(self) -> None:
        self.assertEqual(ProbeCase(**PROBE).side, "inside")

    def test_side_must_match_affected_region(self) -> None:
        with self.assertRaises(ValidationError):
            ProbeCase(**{**PROBE, "side": "outside"})
        ProbeCase(**{**PROBE, "measurement": 62, "side": "outside", "r_star": 2000})

    def test_unknown_pair(self) -> None:
        with self.assertRaises(ValidationError):
            ProbeCase(**{**PROBE, "pair": "age"})


class EpisodeListModelTests(unittest.TestCase):
    def test_valid_list(self) -> None:
        self.assertEqual(len(EpisodeList(**LIST).turn_case_ids), 40)

    def test_wrong_lengths(self) -> None:
        with self.assertRaises(ValidationError):
            EpisodeList(**{**LIST, "turn_case_ids": LIST["turn_case_ids"][:39]})
        with self.assertRaises(ValidationError):
            EpisodeList(**{**LIST, "probe_case_ids": LIST["probe_case_ids"][:3]})

    def test_duplicate_ids(self) -> None:
        turns = ("c-0",) * 2 + LIST["turn_case_ids"][2:]
        with self.assertRaises(ValidationError):
            EpisodeList(**{**LIST, "turn_case_ids": turns})
        with self.assertRaises(ValidationError):
            EpisodeList(**{**LIST, "probe_case_ids": ("c-0", "p-1", "p-2", "p-3")})

    def test_use_and_index(self) -> None:
        with self.assertRaises(ValidationError):
            EpisodeList(**{**LIST, "use": "test"})
        with self.assertRaises(ValidationError):
            EpisodeList(**{**LIST, "list_index": -1})


if __name__ == "__main__":
    unittest.main()
