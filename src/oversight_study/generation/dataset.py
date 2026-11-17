"""Fixed-order dataset build: evaluation pool first, then development pool.

The development pool is drawn with every evaluation key excluded, so a full build needs
both seed sets. Episode lists and probes extend the same `used_keys` set after the pools.
"""

from __future__ import annotations

from dataclasses import dataclass

from oversight_study.generation.ids import check_unique_ids
from oversight_study.generation.pools import Case, Key, generate_pools, to_cases
from oversight_study.generation.seeds import SeedConfig

EVALUATION = "evaluation"
DEVELOPMENT = "development"
SPLITS = (EVALUATION, DEVELOPMENT)

GENERATION_ORDER = ("evaluation_pool", "development_pool")


@dataclass(frozen=True)
class Dataset:
    seeds: SeedConfig
    pools: dict[str, dict[str, list[Case]]]  # split -> case_type -> cases sorted by id
    used_keys: frozenset[Key]

    def cases(self, split: str) -> list[Case]:
        return [case for group in self.pools[split].values() for case in group]


def build_dataset(seeds: SeedConfig) -> Dataset:
    seeds.require_test_seeds()
    used_keys: set[Key] = set()
    evaluation = to_cases(EVALUATION, generate_pools(seeds.test_pool_base, used_keys))
    development = to_cases(DEVELOPMENT, generate_pools(seeds.dev_pool_base, used_keys))
    dataset = Dataset(
        seeds=seeds,
        pools={EVALUATION: evaluation, DEVELOPMENT: development},
        used_keys=frozenset(used_keys),
    )
    check_disjoint(dataset)
    return dataset


def check_disjoint(dataset: Dataset) -> None:
    """Raise if a key or id appears twice within or across splits."""
    seen: set[Key] = set()
    for split in SPLITS:
        for case in dataset.cases(split):
            if case.key in seen:
                raise ValueError(f"duplicate key across pools: {case.key}")
            seen.add(case.key)
    check_unique_ids(case.case_id for split in SPLITS for case in dataset.cases(split))
