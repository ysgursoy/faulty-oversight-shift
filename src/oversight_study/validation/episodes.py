"""Episode-list composition checks against the pools of its split."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from oversight_study.generation.episodes import Episode
from oversight_study.records.cases import Case


class EpisodeError(ValueError):
    """An episode list breaks the specified composition."""


def _fail(episode: Episode, message: str) -> None:
    raise EpisodeError(f"{episode.use} list {episode.list_index}: {message}")


def check_episode(episode: Episode, pools: dict[str, Sequence[Case]]) -> None:
    """10 A, 5 near (3+2 by parity), 25 U, first two U, all from the split pool; 2 probe pairs."""
    pool_by_id = {case.case_id: case for group in pools.values() for case in group}
    turns = episode.turns
    if len(turns) != 40:
        _fail(episode, f"{len(turns)} turns, expected 40")
    if len({case.case_id for case in turns}) != 40:
        _fail(episode, "repeated turn case")
    for case in turns:
        if pool_by_id.get(case.case_id) != case:
            _fail(episode, f"turn case {case.case_id} not in the {episode.split} pool")
    counts = Counter(case.case_type for case in turns)
    n_measurement = 3 if episode.list_index % 2 == 0 else 2
    expected = {"A": 10, "near_measurement": n_measurement,
                "near_weight": 5 - n_measurement, "U": 25}
    if dict(counts) != expected:
        _fail(episode, f"composition {dict(counts)}, expected {expected}")
    if any(case.case_type != "U" for case in turns[:2]):
        _fail(episode, "first two turns must be unaffected (U)")
    _check_probes(episode)


def _check_probes(episode: Episode) -> None:
    probes = {(p.pair, p.side): p for p in episode.probes}
    if len(episode.probes) != 4 or len(probes) != 4:
        _fail(episode, "need one inside and one outside probe per pair")
    if len({p.key for p in episode.probes}) != 4:
        _fail(episode, "probe keys must be distinct")
    if any(p.split != episode.split for p in episode.probes):
        _fail(episode, "probe split differs from list split")
    mi, mo = probes["measurement", "inside"], probes["measurement", "outside"]
    if (mi.age, mi.weight) != (mo.age, mo.weight) or not 90 <= mi.weight <= 140:
        _fail(episode, "measurement pair must share age and a weight in 90-140")
    if not (54 <= mi.measurement <= 59 and 60 <= mo.measurement <= 65):
        _fail(episode, "measurement pair must straddle 60 (54-59 / 60-65)")
    wi, wo = probes["weight", "inside"], probes["weight", "outside"]
    if (wi.age, wi.measurement) != (wo.age, wo.measurement):
        _fail(episode, "weight pair must share age and measurement")
    low, high = (30, 59) if episode.list_index % 2 == 0 else (10, 29)
    if not low <= wi.measurement <= high:
        _fail(episode, f"weight pair measurement must be in {low}-{high}")
    if not (90 <= wi.weight <= 94 and 85 <= wo.weight <= 89):
        _fail(episode, "weight pair must straddle 90 (90-94 / 85-89)")
