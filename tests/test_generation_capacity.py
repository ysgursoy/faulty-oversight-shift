import unittest
from pathlib import Path

from oversight_study.generation import build_dataset, load_seeds
from oversight_study.generation.capacity import extra_pool_capacity
from oversight_study.task import MEASUREMENT_RANGE, WEIGHT_RANGE

FIXTURES = Path(__file__).parent / "fixtures"


class GridCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.capacity = {
            n: extra_pool_capacity(build_dataset(seeds, n_main=n).used_keys, seed_base=70000)
            for n in (100, 200, 300, 350)
        }

    def test_large_builds_succeed_and_leave_room_for_a_new_pool(self) -> None:
        # Room for at least one more disjoint pool remains up to 350 main lists.
        self.assertGreaterEqual(self.capacity[300], 1)
        self.assertGreaterEqual(self.capacity[350], 1)
        self.assertGreaterEqual(self.capacity[200], 2)

    def test_capacity_shrinks_as_lists_grow(self) -> None:
        values = [self.capacity[n] for n in (100, 200, 300, 350)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_full_grid_has_no_capacity(self) -> None:
        everything = {(w, m) for w in WEIGHT_RANGE for m in MEASUREMENT_RANGE}
        self.assertEqual(extra_pool_capacity(everything, seed_base=1), 0)


if __name__ == "__main__":
    unittest.main()
