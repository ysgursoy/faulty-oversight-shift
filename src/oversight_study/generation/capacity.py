"""How many more disjoint full pools fit after a build: the grid caps the list count.

Probes draw keys from the near strips, so each extra list shrinks the room left for
later disjoint pools (for example a new evaluation pool after results are seen).
"""

from __future__ import annotations

from collections.abc import Iterable

from oversight_study.generation.pools import generate_pools
from oversight_study.records.cases import Key


def extra_pool_capacity(used_keys: Iterable[Key], seed_base: int, max_pools: int = 5) -> int:
    """Number of further full pools that can be drawn disjointly, up to `max_pools`."""
    used = set(used_keys)
    for count in range(max_pools):
        try:
            generate_pools(seed_base + 100 * count, used)
        except ValueError:
            return count
    return max_pools
