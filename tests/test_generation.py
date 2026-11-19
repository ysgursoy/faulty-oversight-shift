import hashlib
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
    read_dataset,
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
        cls.dataset = build_dataset(cls.seeds, n_main=4)

    def test_split_names_and_order(self) -> None:
        self.assertEqual(set(self.dataset.pools), {EVALUATION, DEVELOPMENT})
        self.assertEqual(GENERATION_ORDER, ("evaluation_pool", "development_pool",
                                            "trial_lists", "pilot_lists", "main_lists"))
        counts = {use: len(lists) for use, lists in self.dataset.episodes.items()}
        self.assertEqual(counts, {"trial": 3, "pilot": 20, "main": 4})

    def test_splits_are_disjoint_and_used_keys_complete(self) -> None:
        check_disjoint(self.dataset)
        eval_keys = {c.key for c in self.dataset.cases(EVALUATION)}
        dev_keys = {c.key for c in self.dataset.cases(DEVELOPMENT)}
        self.assertEqual(eval_keys & dev_keys, set())
        probe_keys = {p.key for lists in self.dataset.episodes.values()
                      for e in lists for p in e.probes}
        self.assertEqual(probe_keys & (eval_keys | dev_keys), set())
        self.assertEqual(self.dataset.used_keys, eval_keys | dev_keys | probe_keys)

    def test_lists_use_their_split(self) -> None:
        for use, split in (("trial", DEVELOPMENT), ("pilot", DEVELOPMENT), ("main", EVALUATION)):
            for episode in self.dataset.episodes[use]:
                self.assertEqual(episode.split, split)
                pool = {c.case_id for c in self.dataset.cases(split)}
                self.assertTrue({c.case_id for c in episode.turns} <= pool)

    def test_main_list_i_independent_of_count(self) -> None:
        longer = build_dataset(self.seeds, n_main=6)
        self.assertEqual(longer.episodes["main"][:4], self.dataset.episodes["main"])
        self.assertEqual(longer.episodes["pilot"], self.dataset.episodes["pilot"])

    def test_main_lists_are_optional(self) -> None:
        without = build_dataset(self.seeds)
        self.assertEqual(without.episodes["main"], [])
        self.assertEqual(without.episodes["trial"], self.dataset.episodes["trial"])

    def test_probes_disjoint_across_splits(self) -> None:
        dev = {p.key for use in ("trial", "pilot") for e in self.dataset.episodes[use]
               for p in e.probes}
        main = {p.key for e in self.dataset.episodes["main"] for p in e.probes}
        self.assertEqual(dev & main, set())

    def test_evaluation_pool_is_drawn_first_without_exclusions(self) -> None:
        standalone = to_cases(EVALUATION, generate_pools(self.seeds.test_pool_base, set()))
        self.assertEqual(standalone, self.dataset.pools[EVALUATION])

    def test_development_pool_excludes_evaluation_keys(self) -> None:
        used = {c.key for c in self.dataset.cases(EVALUATION)}
        rerun = to_cases(DEVELOPMENT, generate_pools(self.seeds.dev_pool_base, used))
        self.assertEqual(rerun, self.dataset.pools[DEVELOPMENT])

    def test_generation_is_deterministic(self) -> None:
        self.assertEqual(build_dataset(self.seeds, n_main=4), self.dataset)

    def test_check_disjoint_detects_overlap(self) -> None:
        dev = self.dataset.pools[DEVELOPMENT]
        bad = Dataset(
            seeds=self.seeds,
            pools={DEVELOPMENT: dev, EVALUATION: {**dev, "A": dev["A"][:1]}},
            episodes={"trial": [], "pilot": [], "main": []},
            used_keys=self.dataset.used_keys,
        )
        with self.assertRaises(ValueError):
            check_disjoint(bad)

    def test_check_disjoint_detects_wrong_list_split(self) -> None:
        bad = Dataset(
            seeds=self.seeds,
            pools=self.dataset.pools,
            episodes={**self.dataset.episodes, "main": self.dataset.episodes["pilot"][:1]},
            used_keys=self.dataset.used_keys,
        )
        with self.assertRaises(ValueError):
            check_disjoint(bad)


class ExcludedKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.earlier = build_dataset(fixture_seeds(), n_main=2)
        shown = {case.key for case in cls.earlier.cases(DEVELOPMENT)}
        shown |= {p.key for use in ("trial", "pilot") for e in cls.earlier.episodes[use]
                  for p in e.probes}
        cls.shown = frozenset(shown)
        # Fresh seeds for the evaluation side, same development seeds: worst case for overlap.
        seeds = fixture_seeds().model_copy(update={"test_pool_base": 7_000_001,
                                                   "test_lists_base": 8_000_001})
        cls.dataset = build_dataset(seeds, n_main=2, excluded_keys=cls.shown)

    def test_no_excluded_key_is_used(self) -> None:
        used = {c.key for split in (EVALUATION, DEVELOPMENT) for c in self.dataset.cases(split)}
        used |= {p.key for lists in self.dataset.episodes.values() for e in lists
                 for p in e.probes}
        self.assertFalse(used & self.shown)
        self.assertFalse(self.dataset.used_keys & self.shown)
        self.assertEqual(self.dataset.excluded_keys, self.shown)

    def test_excluded_keys_are_released_and_verified(self) -> None:
        from oversight_study.generation.manifest import development_keys

        with TemporaryDirectory() as temp_dir:
            earlier, later = Path(temp_dir) / "earlier", Path(temp_dir) / "later"
            write_dataset(self.earlier, earlier, dataset_version="v-earlier")
            self.assertEqual(development_keys(earlier), set(self.shown))
            manifest = write_dataset(self.dataset, later, dataset_version="v-later")
            self.assertEqual(manifest.excluded_keys_count, len(self.shown))
            self.assertIn("excluded_keys.json", manifest.checksums_sha256)
            self.assertEqual(verify_dataset(later), manifest)

    def test_reuse_of_an_excluded_key_is_detected(self) -> None:
        clash = next(iter(self.dataset.used_keys))
        broken = Dataset(seeds=self.dataset.seeds, pools=self.dataset.pools,
                         episodes=self.dataset.episodes, used_keys=self.dataset.used_keys,
                         excluded_keys=self.dataset.excluded_keys | {clash})
        with self.assertRaises(ValueError):
            check_disjoint(broken)


class WriteAndVerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_dataset(fixture_seeds(), n_main=2)

    def write(self, out: Path, **kwargs):
        return write_dataset(self.dataset, out, dataset_version="v-test", **kwargs)

    def test_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "release"
            manifest = self.write(out)
            self.assertEqual(verify_dataset(out), manifest)
            self.assertEqual(manifest.dataset_version, "v-test")
            self.assertEqual(manifest.used_keys_count, len(self.dataset.used_keys))
            self.assertEqual(manifest.generation_order, GENERATION_ORDER)
            self.assertEqual(manifest.list_counts, {"trial": 3, "pilot": 20, "main": 2})
            self.assertEqual(
                manifest.counts[DEVELOPMENT],
                {"A": 150, "near_measurement": 20, "near_weight": 20, "U": 300,
                 "u_light_high": 100, "u_heavy_far": 100, "u_light_mid": 50, "u_light_low": 50},
            )
            self.assertNotIn("test_pool_base", manifest.seeds)
            self.assertEqual(sorted(manifest.checksums_sha256), [
                "episodes/main.json", "episodes/pilot.json", "episodes/trial.json",
                f"pools/{DEVELOPMENT}.json", f"pools/{EVALUATION}.json", "used_keys.json"])

            rows = json.loads((out / "pools" / f"{DEVELOPMENT}.json").read_text())
            self.assertEqual(len(rows), 490)
            for row in rows:
                self.assertEqual(row["r_star"], r_star(row["weight"], row["measurement"]))

            _, pools, episodes, used = read_dataset(out)
            self.assertEqual(episodes, self.dataset.episodes)
            self.assertEqual(used, set(self.dataset.used_keys))
            self.assertEqual(len(pools[EVALUATION]), 490)

    def test_tampering_is_detected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "release"
            self.write(out)
            pool = out / "episodes" / "trial.json"
            pool.write_text(pool.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_dataset(out)

    def test_consistent_but_wrong_content_is_detected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "release"
            self.write(out)
            used_path = out / "used_keys.json"
            used_path.write_text(json.dumps(json.loads(used_path.read_text())[1:]) + "\n")
            manifest_path = out / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["checksums_sha256"]["used_keys.json"] = hashlib.sha256(
                used_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify_dataset(out)

    def test_main_lists_written_only_when_generated(self) -> None:
        without = build_dataset(fixture_seeds())
        with TemporaryDirectory() as temp_dir:
            manifest = write_dataset(without, Path(temp_dir), dataset_version="v-test")
            self.assertFalse((Path(temp_dir) / "episodes" / "main.json").exists())
            self.assertEqual(manifest.list_counts["main"], 0)
            verify_dataset(Path(temp_dir))

    def test_test_seeds_written_only_on_request(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest = self.write(Path(temp_dir), include_test_seeds=True)
        self.assertEqual(manifest.seeds["test_pool_base"], 9100)

    def test_existing_release_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "release"
            self.write(out)
            with self.assertRaises(FileExistsError):
                self.write(out)


if __name__ == "__main__":
    unittest.main()
