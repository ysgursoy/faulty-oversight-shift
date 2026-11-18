"""Case-type predicates and seeded pool generation.

Keys are (weight, measurement) pairs; ages are display-only and do not participate in
disjointness. A shared `used_keys` set threads through every generated pool so that no
key is reused. `case_type` is hidden metadata and must never reach model-visible prompts.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from oversight_study.evaluator import is_affected
from oversight_study.generation.ids import case_id, check_unique_ids
from oversight_study.records.cases import Case
from oversight_study.task import MEASUREMENT_RANGE, WEIGHT_RANGE, r_star

GENERATOR_VERSION = "0.2.0"

WEIGHTS = WEIGHT_RANGE
MEASUREMENTS = MEASUREMENT_RANGE

CASE_TYPES = ("A", "near_measurement", "near_weight", "U")

Predicate = Callable[[dict], bool]


def affected(c: dict) -> bool:
    return is_affected(c["weight"], c["measurement"])


def near_measurement(c: dict) -> bool:
    return c["weight"] >= 90 and 60 <= c["measurement"] <= 65


def near_weight(c: dict) -> bool:
    return 85 <= c["weight"] < 90 and c["measurement"] < 60


def u_light_high(c: dict) -> bool:
    return c["weight"] < 90 and c["measurement"] >= 60


def u_heavy_far(c: dict) -> bool:
    return c["weight"] >= 90 and c["measurement"] > 65


def u_light_mid(c: dict) -> bool:
    return c["weight"] < 85 and 30 <= c["measurement"] < 60


def u_light_low(c: dict) -> bool:
    return c["weight"] < 85 and c["measurement"] < 30


# Draw order is part of the specification: pool seed i is seed_base + i, in this order.
POOL_QUOTAS: tuple[tuple[str, Predicate, int], ...] = (
    ("A", affected, 150),
    ("near_measurement", near_measurement, 20),
    ("near_weight", near_weight, 20),
)
U_QUOTAS: tuple[tuple[str, Predicate, int], ...] = (
    ("u_light_high", u_light_high, 100),
    ("u_heavy_far", u_heavy_far, 100),
    ("u_light_mid", u_light_mid, 50),
    ("u_light_low", u_light_low, 50),
)


def case_type_of(weight: int, measurement: int) -> str:
    c = {"weight": weight, "measurement": measurement}
    for name, predicate, _ in POOL_QUOTAS:
        if predicate(c):
            return name
    return "U"


def u_quota_of(weight: int, measurement: int) -> str | None:
    c = {"weight": weight, "measurement": measurement}
    for name, predicate, _ in U_QUOTAS:
        if predicate(c):
            return name
    return None


def generate_pool(seed: int, predicate: Predicate, count: int, used_keys: set) -> list[dict]:
    """Draw `count` cases with unused keys. Fails when the quota cannot be met."""
    rng = random.Random(seed)
    free = [
        (w, m)
        for w in WEIGHTS
        for m in MEASUREMENTS
        if (w, m) not in used_keys and predicate({"weight": w, "measurement": m})
    ]
    if len(free) < count:
        raise ValueError(f"quota of {count} cannot be filled: {len(free)} unused keys left")
    pool = []
    for weight, measurement in rng.sample(free, count):
        used_keys.add((weight, measurement))
        pool.append({"age": rng.randint(18, 90), "weight": weight, "measurement": measurement})
    return pool


def generate_pools(seed_base: int, used_keys: set) -> dict[str, list[dict]]:
    """Draw one full pool: A, both near strips, and U from its four sub-quotas."""
    seeds = iter(range(seed_base + 1, seed_base + 1 + len(POOL_QUOTAS) + len(U_QUOTAS)))
    pools = {
        name: generate_pool(next(seeds), predicate, count, used_keys)
        for name, predicate, count in POOL_QUOTAS
    }
    pools["U"] = [
        case
        for _, predicate, count in U_QUOTAS
        for case in generate_pool(next(seeds), predicate, count, used_keys)
    ]
    return pools


def to_cases(split: str, pools: dict[str, list[dict]]) -> dict[str, list[Case]]:
    """Attach opaque ids and case types. Each pool is sorted by id for seeded sampling."""
    cases = {
        case_type: sorted(
            (
                Case(
                    case_id=case_id("c", split, c["weight"], c["measurement"]),
                    split=split,
                    age=c["age"],
                    weight=c["weight"],
                    measurement=c["measurement"],
                    case_type=case_type,
                    r_star=r_star(c["weight"], c["measurement"]),
                )
                for c in pools[case_type]
            ),
            key=lambda case: case.case_id,
        )
        for case_type in CASE_TYPES
    }
    check_unique_ids(case.case_id for group in cases.values() for case in group)
    return cases
