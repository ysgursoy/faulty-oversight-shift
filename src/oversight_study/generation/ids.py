"""Opaque, deterministic case ids.

An id depends only on its kind, split, and (weight, measurement) key, so it carries no
case type, quota, or position in a generated sequence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

ID_HEX_LENGTH = 12


def case_id(kind: Literal["c", "p"], split: str, weight: int, measurement: int) -> str:
    """Return `c-…` for pool cases and `p-…` for probe cases."""
    if kind not in ("c", "p"):
        raise ValueError(f"unknown id kind: {kind}")
    digest = hashlib.sha256(f"{split}:{weight}:{measurement}".encode()).hexdigest()
    return f"{kind}-{digest[:ID_HEX_LENGTH]}"


def check_unique_ids(ids: Iterable[str]) -> None:
    """Raise if any id occurs more than once."""
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise ValueError(f"duplicate case id: {value}")
        seen.add(value)
