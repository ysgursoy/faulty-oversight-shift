"""Fixed-order dataset build.

Order: evaluation pool, development pool, trial lists, pilot lists, main lists. One
`used_keys` set threads through every step, so the development pool excludes every
evaluation key and each later step excludes every earlier probe. Changing the order
changes the probes; the order is recorded in the manifest.

`excluded_keys` (for example, every development case shown to a model in earlier work)
are treated as consumed before the first step; no pool case or probe uses them.
"""

from __future__ import annotations

from dataclasses import dataclass

from oversight_study.generation.episodes import Episode, generate_episodes
from oversight_study.generation.ids import check_unique_ids
from oversight_study.generation.pools import generate_pools, to_cases
from oversight_study.generation.seeds import SeedConfig
from oversight_study.records.cases import Case, Key

EVALUATION = "evaluation"
DEVELOPMENT = "development"
SPLITS = (EVALUATION, DEVELOPMENT)

GENERATION_ORDER = (
    "evaluation_pool",
    "development_pool",
    "trial_lists",
    "pilot_lists",
    "main_lists",
)
LIST_SPLIT = {"trial": DEVELOPMENT, "pilot": DEVELOPMENT, "main": EVALUATION}


@dataclass(frozen=True)
class Dataset:
    seeds: SeedConfig
    pools: dict[str, dict[str, list[Case]]]  # split -> case_type -> cases sorted by id
    episodes: dict[str, list[Episode]]  # use (trial/pilot/main) -> lists in index order
    used_keys: frozenset[Key]  # keys consumed by this dataset's pools and probes
    excluded_keys: frozenset[Key] = frozenset()  # unavailable before generation began

    def cases(self, split: str) -> list[Case]:
        return [case for group in self.pools[split].values() for case in group]


def build_dataset(
    seeds: SeedConfig,
    n_trial: int = 3,
    n_pilot: int = 20,
    n_main: int | None = None,
    excluded_keys: frozenset[Key] = frozenset(),
) -> Dataset:
    """Build pools and lists. `n_main=None` skips main lists (count fixed after the pilot)."""
    seeds.require_test_seeds()
    used_keys: set[Key] = set(excluded_keys)
    evaluation = to_cases(EVALUATION, generate_pools(seeds.test_pool_base, used_keys))
    development = to_cases(DEVELOPMENT, generate_pools(seeds.dev_pool_base, used_keys))
    episodes = {
        "trial": generate_episodes(
            seeds.trial_lists_base, development, n_trial, used_keys, DEVELOPMENT, "trial"
        ),
        "pilot": generate_episodes(
            seeds.pilot_lists_base, development, n_pilot, used_keys, DEVELOPMENT, "pilot"
        ),
        "main": []
        if n_main is None
        else generate_episodes(
            seeds.test_lists_base, evaluation, n_main, used_keys, EVALUATION, "main"
        ),
    }
    dataset = Dataset(
        seeds=seeds,
        pools={EVALUATION: evaluation, DEVELOPMENT: development},
        episodes=episodes,
        used_keys=frozenset(used_keys - excluded_keys),
        excluded_keys=frozenset(excluded_keys),
    )
    check_disjoint(dataset)
    return dataset


def check_disjoint(dataset: Dataset) -> None:
    """Raise on any key reused across pools, or a probe key in a pool or another split."""
    pool_keys: set[Key] = set()
    for split in SPLITS:
        for case in dataset.cases(split):
            if case.key in pool_keys:
                raise ValueError(f"duplicate key across pools: {case.key}")
            pool_keys.add(case.key)
    check_unique_ids(case.case_id for split in SPLITS for case in dataset.cases(split))

    probe_keys: dict[str, set[Key]] = {split: set() for split in SPLITS}
    for use, episodes in dataset.episodes.items():
        for episode in episodes:
            if episode.split != LIST_SPLIT[use]:
                raise ValueError(f"{use} list {episode.list_index} on split {episode.split}")
            probe_keys[episode.split].update(probe.key for probe in episode.probes)
    for split, keys in probe_keys.items():
        if keys & pool_keys:
            raise ValueError(f"{split} probe key reused from a pool: {min(keys & pool_keys)}")
    if probe_keys[EVALUATION] & probe_keys[DEVELOPMENT]:
        raise ValueError("probe key shared between splits")
    missing = (pool_keys | probe_keys[EVALUATION] | probe_keys[DEVELOPMENT]) - dataset.used_keys
    if missing:
        raise ValueError(f"used_keys is missing {len(missing)} keys")
    reused = (pool_keys | probe_keys[EVALUATION] | probe_keys[DEVELOPMENT]) & dataset.excluded_keys
    if reused:
        raise ValueError(f"{len(reused)} excluded keys reused, e.g. {min(reused)}")
