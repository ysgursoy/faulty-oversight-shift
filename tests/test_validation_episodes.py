import dataclasses
import unittest
from pathlib import Path

from oversight_study.generation import build_dataset, load_seeds
from oversight_study.generation.dataset import Dataset
from oversight_study.validation.episodes import EpisodeError, check_episode
from oversight_study.validation.splits import check_split_overlap

FIXTURES = Path(__file__).parent / "fixtures"


class EpisodeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.dataset = build_dataset(seeds, n_main=2)
        cls.pools = cls.dataset.pools["development"]
        cls.even = cls.dataset.episodes["trial"][0]
        cls.odd = cls.dataset.episodes["trial"][1]

    def assert_bad(self, episode, pools=None) -> None:
        with self.assertRaises(EpisodeError):
            check_episode(episode, pools or self.pools)

    def test_generated_dataset_passes(self) -> None:
        check_split_overlap(self.dataset)

    def test_turn_count_and_repeats(self) -> None:
        turns = self.even.turns
        self.assert_bad(dataclasses.replace(self.even, turns=turns[:39]))
        self.assert_bad(dataclasses.replace(self.even, turns=turns[:39] + turns[:1]))

    def test_case_from_other_split(self) -> None:
        self.assert_bad(self.even, self.dataset.pools["evaluation"])

    def test_parity_rule(self) -> None:
        self.assert_bad(dataclasses.replace(self.even, list_index=1))
        self.assert_bad(dataclasses.replace(self.odd, list_index=0))

    def test_first_two_must_be_unaffected(self) -> None:
        turns = list(self.even.turns)
        first_a = next(i for i, c in enumerate(turns) if c.case_type == "A")
        turns[0], turns[first_a] = turns[first_a], turns[0]
        self.assert_bad(dataclasses.replace(self.even, turns=tuple(turns)))

    def test_wrong_composition(self) -> None:
        turns = list(self.even.turns)
        index = next(i for i, c in enumerate(turns) if c.case_type == "A")
        used = {c.case_id for c in turns}
        turns[index] = next(c for c in self.pools["U"] if c.case_id not in used)
        self.assert_bad(dataclasses.replace(self.even, turns=tuple(turns)))

    def probes_with(self, pair, side, update):
        return tuple(p.model_copy(update=update) if (p.pair, p.side) == (pair, side) else p
                     for p in self.even.probes)

    def test_probe_violations(self) -> None:
        for pair, side, update in (
            ("measurement", "inside", {"measurement": 50}),
            ("measurement", "outside", {"measurement": 66}),
            ("measurement", "outside", {"age": 18 if self.even.probes[0].age != 18 else 19}),
            ("weight", "inside", {"weight": 95}),
            ("weight", "outside", {"weight": 84}),
            ("weight", "inside", {"measurement": 20}),
            ("weight", "outside", {"split": "evaluation"}),
            ("measurement", "outside", {"side": "inside"}),
        ):
            with self.subTest(pair=pair, side=side, update=update):
                bad = dataclasses.replace(self.even, probes=self.probes_with(pair, side, update))
                self.assert_bad(bad)

    def test_split_overlap_catches_probe_in_pool(self) -> None:
        pool_case = self.pools["A"][0]
        probe = self.even.probes[0].model_copy(
            update={"weight": pool_case.weight, "measurement": pool_case.measurement})
        bad_episode = dataclasses.replace(self.even, probes=(probe, *self.even.probes[1:]))
        episodes = {**self.dataset.episodes,
                    "trial": [bad_episode, *self.dataset.episodes["trial"][1:]]}
        with self.assertRaises(ValueError):
            check_split_overlap(dataclasses.replace(self.dataset, episodes=episodes))

    def test_split_overlap_catches_shared_probe_across_splits(self) -> None:
        main = self.dataset.episodes["main"][0]
        dev_probe = self.even.probes[0]
        shared = main.probes[0].model_copy(
            update={"weight": dev_probe.weight, "measurement": dev_probe.measurement})
        episodes = {**self.dataset.episodes,
                    "main": [dataclasses.replace(main, probes=(shared, *main.probes[1:])),
                             *self.dataset.episodes["main"][1:]]}
        with self.assertRaises(ValueError):
            check_split_overlap(Dataset(self.dataset.seeds, self.dataset.pools, episodes,
                                        self.dataset.used_keys))

    def test_lists_out_of_order(self) -> None:
        trial = self.dataset.episodes["trial"]
        episodes = {**self.dataset.episodes, "trial": [trial[1], trial[0], *trial[2:]]}
        with self.assertRaises(ValueError):
            check_split_overlap(dataclasses.replace(self.dataset, episodes=episodes))


if __name__ == "__main__":
    unittest.main()
