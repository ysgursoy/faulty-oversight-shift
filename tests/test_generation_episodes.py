import random
import unittest

from oversight_study.evaluator import is_affected
from oversight_study.generation.episodes import (
    generate_episode,
    generate_episodes,
    generate_probes,
)
from oversight_study.generation.pools import generate_pools, to_cases

DEV, EVAL = "development", "evaluation"


def build_pools():
    used: set = set()
    evaluation = to_cases(EVAL, generate_pools(9100, used))
    development = to_cases(DEV, generate_pools(9200, used))
    return evaluation, development, used


class EpisodeCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation, cls.development, cls.pool_keys = build_pools()
        cls.used = set(cls.pool_keys)
        cls.episodes = generate_episodes(9300, cls.development, 6, cls.used, DEV, "trial")

    def test_counts_and_parity(self) -> None:
        for episode in self.episodes:
            types = [case.case_type for case in episode.turns]
            self.assertEqual(len(types), 40)
            self.assertEqual(types.count("A"), 10)
            self.assertEqual(types.count("U"), 25)
            expected_nm = 3 if episode.list_index % 2 == 0 else 2
            self.assertEqual(types.count("near_measurement"), expected_nm)
            self.assertEqual(types.count("near_weight"), 5 - expected_nm)
            self.assertEqual(types[:2], ["U", "U"])
            self.assertEqual(len({case.case_id for case in episode.turns}), 40)

    def test_turns_come_from_the_split_pool(self) -> None:
        pool_ids = {c.case_id for group in self.development.values() for c in group}
        for episode in self.episodes:
            self.assertTrue({c.case_id for c in episode.turns} <= pool_ids)

    def test_k_numbering_and_last_three(self) -> None:
        for episode in self.episodes:
            ks = episode.k_by_case_id
            self.assertEqual(sorted(ks.values()), list(range(1, 11)))
            affected = [c for c in episode.turns if c.case_type == "A"]
            self.assertEqual([ks[c.case_id] for c in affected], list(range(1, 11)))
            self.assertEqual(episode.last_affected, tuple(affected[-3:]))

    def test_probe_pairs(self) -> None:
        for episode in self.episodes:
            probes = {(p.pair, p.side): p for p in episode.probes}
            self.assertEqual(len(probes), 4)
            mi, mo = probes["measurement", "inside"], probes["measurement", "outside"]
            self.assertEqual((mi.age, mi.weight), (mo.age, mo.weight))
            self.assertTrue(90 <= mi.weight <= 140)
            self.assertTrue(54 <= mi.measurement <= 59 and 60 <= mo.measurement <= 65)
            wi, wo = probes["weight", "inside"], probes["weight", "outside"]
            self.assertEqual((wi.age, wi.measurement), (wo.age, wo.measurement))
            self.assertTrue(90 <= wi.weight <= 94 and 85 <= wo.weight <= 89)
            band = (30, 59) if episode.list_index % 2 == 0 else (10, 29)
            self.assertTrue(band[0] <= wi.measurement <= band[1])
            for probe in episode.probes:
                self.assertEqual(is_affected(probe.weight, probe.measurement),
                                 probe.side == "inside")
                self.assertTrue(probe.case_id.startswith("p-"))
            self.assertEqual(len({p.key for p in episode.probes}), 4)

    def test_serialized_lists_validate(self) -> None:
        for episode in self.episodes:
            listing = episode.to_list()
            self.assertEqual(listing.turn_case_ids, tuple(c.case_id for c in episode.turns))
            self.assertEqual(listing.use, "trial")

    def test_probes_never_in_pools_and_join_used_keys(self) -> None:
        probe_keys = {p.key for e in self.episodes for p in e.probes}
        self.assertEqual(probe_keys & self.pool_keys, set())
        self.assertTrue(probe_keys <= self.used)


class EpisodeDeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation, cls.development, cls.pool_keys = build_pools()

    def test_same_seed_same_list(self) -> None:
        a = generate_episode(42, self.development, 0, frozenset(self.pool_keys), DEV, "pilot")
        b = generate_episode(42, self.development, 0, frozenset(self.pool_keys), DEV, "pilot")
        self.assertEqual(a, b)

    def test_list_i_independent_of_count(self) -> None:
        short = generate_episodes(1, self.evaluation, 3, set(self.pool_keys), EVAL, "main")
        long = generate_episodes(1, self.evaluation, 5, set(self.pool_keys), EVAL, "main")
        self.assertEqual(short, long[:3])

    def test_pool_file_order_does_not_matter(self) -> None:
        reversed_pools = {k: list(reversed(v)) for k, v in self.development.items()}
        a = generate_episode(7, self.development, 1, frozenset(self.pool_keys), DEV, "trial")
        b = generate_episode(7, reversed_pools, 1, frozenset(self.pool_keys), DEV, "trial")
        self.assertEqual(a, b)

    def test_probe_search_exhaustion_raises(self) -> None:
        blocked = {(w, m) for w in range(85, 141) for m in range(10, 66)}
        with self.assertRaises(ValueError):
            generate_probes(random.Random(0), 0, blocked, DEV, max_attempts=50)


if __name__ == "__main__":
    unittest.main()
