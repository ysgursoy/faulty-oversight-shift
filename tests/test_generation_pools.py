import unittest

from oversight_study.evaluator import is_affected
from oversight_study.generation import (
    CASE_TYPES,
    POOL_QUOTAS,
    U_QUOTAS,
    case_type_of,
    generate_pool,
    generate_pools,
    to_cases,
    u_quota_of,
)
from oversight_study.generation.pools import MEASUREMENTS, WEIGHTS
from oversight_study.task import AGE_RANGE

GRID = [(w, m) for w in WEIGHTS for m in MEASUREMENTS]
ALL_PREDICATES = POOL_QUOTAS + U_QUOTAS
SEED_BASE = 9100


def grid_size(predicate) -> int:
    return sum(1 for w, m in GRID if predicate({"weight": w, "measurement": m}))


class PredicateTests(unittest.TestCase):
    def test_predicates_partition_the_grid(self) -> None:
        for w, m in GRID:
            hits = [name for name, predicate, _ in ALL_PREDICATES
                    if predicate({"weight": w, "measurement": m})]
            self.assertEqual(len(hits), 1, (w, m, hits))

    def test_strip_edges(self) -> None:
        self.assertEqual(case_type_of(90, 59), "A")
        self.assertEqual(case_type_of(90, 60), "near_measurement")
        self.assertEqual(case_type_of(140, 65), "near_measurement")
        self.assertEqual(case_type_of(90, 66), "U")
        self.assertEqual(case_type_of(85, 59), "near_weight")
        self.assertEqual(case_type_of(89, 10), "near_weight")
        self.assertEqual(case_type_of(84, 59), "U")
        self.assertEqual(case_type_of(89, 60), "U")

    def test_near_strips_are_never_affected(self) -> None:
        for w, m in GRID:
            if case_type_of(w, m) in ("near_measurement", "near_weight"):
                self.assertFalse(is_affected(w, m))

    def test_grid_capacity(self) -> None:
        sizes = {name: grid_size(predicate) for name, predicate, _ in POOL_QUOTAS}
        self.assertEqual(sizes, {"A": 2550, "near_measurement": 306, "near_weight": 250})


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.used_keys: set = set()
        cls.pools = generate_pools(SEED_BASE, cls.used_keys)

    def test_counts_per_case_type_and_u_sub_quota(self) -> None:
        self.assertEqual(list(self.pools), list(CASE_TYPES))
        for name, _, count in POOL_QUOTAS:
            self.assertEqual(len(self.pools[name]), count)
        self.assertEqual(len(self.pools["U"]), 300)
        for name, _, count in U_QUOTAS:
            got = sum(1 for c in self.pools["U"] if u_quota_of(c["weight"], c["measurement"]) == name)
            self.assertEqual(got, count, name)

    def test_cases_satisfy_their_predicate(self) -> None:
        for case_type, cases in self.pools.items():
            for c in cases:
                self.assertEqual(case_type_of(c["weight"], c["measurement"]), case_type)
                self.assertIn(c["age"], AGE_RANGE)

    def test_used_keys_cover_pool(self) -> None:
        keys = {(c["weight"], c["measurement"]) for cases in self.pools.values() for c in cases}
        self.assertEqual(len(keys), 490)
        self.assertEqual(keys, self.used_keys)

    def test_same_seed_same_output(self) -> None:
        self.assertEqual(generate_pools(SEED_BASE, set()), self.pools)

    def test_each_sub_pool_has_its_own_seed(self) -> None:
        reference = generate_pool(SEED_BASE + 4, U_QUOTAS[0][1], 100, set())
        self.assertEqual(generate_pools(SEED_BASE, set())["U"][:100], reference)

    def test_exhausted_quota_raises(self) -> None:
        all_affected = {(w, m) for w, m in GRID if is_affected(w, m)}
        with self.assertRaises(ValueError):
            generate_pools(SEED_BASE, all_affected)


class CaseConversionTests(unittest.TestCase):
    def test_cases_have_opaque_ids_sorted_within_type(self) -> None:
        cases = to_cases("development", generate_pools(SEED_BASE, set()))
        for case_type, group in cases.items():
            ids = [c.case_id for c in group]
            self.assertEqual(ids, sorted(ids))
            for case in group:
                self.assertEqual(case.case_type, case_type)
                self.assertTrue(case.case_id.startswith("c-"))
                self.assertNotIn(case_type, case.case_id)


if __name__ == "__main__":
    unittest.main()
