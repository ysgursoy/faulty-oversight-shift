import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.generation import (
    DEVELOPMENT,
    EVALUATION,
    GENERATION_ORDER,
    Dataset,
    build_dataset,
    check_disjoint,
    generate_pools,
    load_seeds,
    to_cases,
    verify_dataset,
    write_dataset,
)
from oversight_study.task import r_star

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_seeds():
    return load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                      test_path=FIXTURES / "seeds-test.yaml")


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seeds = fixture_seeds()
        cls.dataset = build_dataset(cls.seeds)

    def test_split_names_and_order(self) -> None:
        self.assertEqual(set(self.dataset.pools), {EVALUATION, DEVELOPMENT})
        self.assertEqual(GENERATION_ORDER, ("evaluation_pool", "development_pool"))

    def test_splits_are_disjoint_and_used_keys_complete(self) -> None:
        check_disjoint(self.dataset)
        eval_keys = {c.key for c in self.dataset.cases(EVALUATION)}
        dev_keys = {c.key for c in self.dataset.cases(DEVELOPMENT)}
        self.assertEqual(eval_keys & dev_keys, set())
        self.assertEqual(self.dataset.used_keys, eval_keys | dev_keys)
        self.assertEqual(len(self.dataset.used_keys), 980)

    def test_evaluation_pool_is_drawn_first_without_exclusions(self) -> None:
        standalone = to_cases(EVALUATION, generate_pools(self.seeds.test_pool_base, set()))
        self.assertEqual(standalone, self.dataset.pools[EVALUATION])

    def test_development_pool_excludes_evaluation_keys(self) -> None:
        used = {c.key for c in self.dataset.cases(EVALUATION)}
        rerun = to_cases(DEVELOPMENT, generate_pools(self.seeds.dev_pool_base, used))
        self.assertEqual(rerun, self.dataset.pools[DEVELOPMENT])

    def test_generation_is_deterministic(self) -> None:
        self.assertEqual(build_dataset(self.seeds), self.dataset)

    def test_check_disjoint_detects_overlap(self) -> None:
        dev = self.dataset.pools[DEVELOPMENT]
        bad = Dataset(
            seeds=self.seeds,
            pools={DEVELOPMENT: dev, EVALUATION: {**dev, "A": dev["A"][:1]}},
            used_keys=self.dataset.used_keys,
        )
        with self.assertRaises(ValueError):
            check_disjoint(bad)


class WriteAndVerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_dataset(fixture_seeds())

    def test_round_trip_and_tamper_detection(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "release"
            manifest = write_dataset(self.dataset, out)
            verify_dataset(out)
            self.assertEqual(manifest["used_keys_count"], 980)
            self.assertEqual(manifest["generation_order"], list(GENERATION_ORDER))
            self.assertEqual(
                manifest["counts"][DEVELOPMENT],
                {"A": 150, "near_measurement": 20, "near_weight": 20, "U": 300,
                 "u_light_high": 100, "u_heavy_far": 100, "u_light_mid": 50, "u_light_low": 50},
            )
            self.assertNotIn("test_pool_base", manifest["seeds"])

            pool = out / "pools" / f"{DEVELOPMENT}.json"
            rows = json.loads(pool.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 490)
            for row in rows:
                self.assertEqual(row["r_star"], r_star(row["weight"], row["measurement"]))

            pool.write_text(pool.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_dataset(out)

    def test_test_seeds_written_only_on_request(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest = write_dataset(self.dataset, Path(temp_dir), include_test_seeds=True)
        self.assertEqual(manifest["seeds"]["test_pool_base"], 9100)

    def test_existing_release_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "release"
            write_dataset(self.dataset, out)
            with self.assertRaises(FileExistsError):
                write_dataset(self.dataset, out)


if __name__ == "__main__":
    unittest.main()
