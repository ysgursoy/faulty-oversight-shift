"""Episode lists (40 turns) and boundary-straddling probe pairs.

RNG call order follows the specification exactly; changing it changes every list.
Probe keys join `used_keys` only after all lists of one call are drawn, so list `i`
does not depend on how many lists are generated. Probes may repeat across the lists of
one split but never enter a pool or another split.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from oversight_study.generation.ids import case_id
from oversight_study.records.cases import Case, EpisodeList, Key, ListUse, ProbeCase
from oversight_study.task import r_star

N_AFFECTED = 10
N_NEAR = 5
N_OTHER = 25
N_FIRST_UNAFFECTED = 2


@dataclass(frozen=True)
class Episode:
    split: str
    use: ListUse
    list_index: int
    seed: int
    turns: tuple[Case, ...]
    probes: tuple[ProbeCase, ...]

    @property
    def k_by_case_id(self) -> dict[str, int]:
        """1-based position of each affected case among the affected turns."""
        affected = [case.case_id for case in self.turns if case.case_type == "A"]
        return {cid: k for k, cid in enumerate(affected, start=1)}

    def to_list(self) -> EpisodeList:
        return EpisodeList(
            split=self.split,
            use=self.use,
            list_index=self.list_index,
            seed=self.seed,
            turn_case_ids=tuple(case.case_id for case in self.turns),
            probe_case_ids=tuple(probe.case_id for probe in self.probes),
        )

    @property
    def last_affected(self) -> tuple[Case, ...]:
        """The pre-fixed final three affected cases (k = 8, 9, 10)."""
        return tuple(case for case in self.turns if case.case_type == "A")[-3:]


def _probe(split: str, age: int, weight: int, measurement: int, pair: str, side: str):
    return ProbeCase(
        case_id=case_id("p", split, weight, measurement),
        split=split,
        age=age,
        weight=weight,
        measurement=measurement,
        pair=pair,
        side=side,
        r_star=r_star(weight, measurement),
    )


def generate_probes(
    rng: random.Random,
    episode_index: int,
    used_keys: set[Key] | frozenset[Key],
    split: str,
    max_attempts: int = 10_000,
) -> list[ProbeCase]:
    """Two pairs straddling the rule boundary; 'inside' is the affected side."""
    for _ in range(max_attempts):
        # Measurement pair: same age and weight (>= 90); measurement on either side of 60.
        age1, weight1 = rng.randint(18, 90), rng.randint(90, 140)
        measurement_pair = [
            _probe(split, age1, weight1, rng.randint(54, 59), "measurement", "inside"),
            _probe(split, age1, weight1, rng.randint(60, 65), "measurement", "outside"),
        ]
        # Weight pair: same age and measurement (< 60); weight on either side of 90.
        # Measurement band alternates: 30-59 in even lists, 10-29 in odd lists.
        age2 = rng.randint(18, 90)
        measurement2 = rng.randint(30, 59) if episode_index % 2 == 0 else rng.randint(10, 29)
        weight_pair = [
            _probe(split, age2, rng.randint(90, 94), measurement2, "weight", "inside"),
            _probe(split, age2, rng.randint(85, 89), measurement2, "weight", "outside"),
        ]
        candidates = measurement_pair + weight_pair
        keys = {c.key for c in candidates}
        if len(keys) == 4 and not (keys & used_keys):
            return candidates
    raise ValueError("no unused probe pairs found")


def _by_id(cases: Sequence[Case]) -> list[Case]:
    return sorted(cases, key=lambda case: case.case_id)


def generate_episode(
    seed: int,
    pools: dict[str, Sequence[Case]],
    episode_index: int,
    used_keys: set[Key] | frozenset[Key],
    split: str,
    use: ListUse,
) -> Episode:
    """One 40-turn list: 10 A, 5 near (3+2 or 2+3 by parity), 25 U; first two turns U."""
    rng = random.Random(seed)
    affected = rng.sample(_by_id(pools["A"]), N_AFFECTED)
    n_measurement = 3 if episode_index % 2 == 0 else 2
    near = rng.sample(_by_id(pools["near_measurement"]), n_measurement) + rng.sample(
        _by_id(pools["near_weight"]), N_NEAR - n_measurement
    )
    other = rng.sample(_by_id(pools["U"]), N_OTHER)
    first_two, other_rest = other[:N_FIRST_UNAFFECTED], other[N_FIRST_UNAFFECTED:]
    rest = affected + near + other_rest
    rng.shuffle(rest)
    probes = generate_probes(rng, episode_index, used_keys, split)
    return Episode(
        split=split,
        use=use,
        list_index=episode_index,
        seed=seed,
        turns=tuple(first_two + rest),
        probes=tuple(probes),
    )


def generate_episodes(
    seed_base: int,
    pools: dict[str, Sequence[Case]],
    count: int,
    used_keys: set[Key],
    split: str,
    use: ListUse,
) -> list[Episode]:
    """Lists 0..count-1 with seeds seed_base + i; probe keys then join used_keys."""
    snapshot = frozenset(used_keys)
    episodes = [
        generate_episode(seed_base + i, pools, i, snapshot, split, use) for i in range(count)
    ]
    used_keys.update(probe.key for episode in episodes for probe in episode.probes)
    return episodes
