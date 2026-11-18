"""Stop on malformed analysis input instead of scoring it silently."""

from __future__ import annotations

import pandas as pd

from oversight_study.records.enums import Frame, Pair, Side

CELL = ["frame", "condition", "wording", "episode", "replicate"]
NO_FEEDBACK = "a"
NO_FEEDBACK_LISTS = 10


class AnalysisInputError(ValueError):
    """Analysis input breaks the record contract."""


def check_duplicates(records: pd.DataFrame) -> None:
    if records.empty:
        return
    duplicated = records.logical_query_id[records.logical_query_id.duplicated()]
    if not duplicated.empty:
        raise AnalysisInputError(f"duplicate records: {sorted(set(duplicated))[:3]}")


def _check_values(name: str, values: set, allowed: set) -> None:
    unknown = values - allowed
    if unknown:
        raise AnalysisInputError(f"unknown {name} values: {sorted(map(str, unknown))}")


def validate(late: pd.DataFrame, probes: pd.DataFrame) -> None:
    """Contract checks before scoring.

    - frame, pair, side values are known;
    - every cell has exactly one turn record for each of k = 8, 9, 10;
    - every feedback cell has exactly four distinct probe_evaluator records;
    - all feedback cells cover the same episode lists; the no-feedback condition covers
      the first lists only (up to 10), the same set in every frame and wording.
    """
    if late.empty:
        raise AnalysisInputError("no late affected-case records")
    _check_values("frame", set(late.frame) | set(probes.get("frame", [])), set(Frame))
    if not probes.empty:
        _check_values("pair", set(probes.pair), set(Pair))
        _check_values("side", set(probes.side), set(Side))

    ks = late.groupby(CELL).k.apply(lambda k: sorted(k) == [8, 9, 10])
    if not ks.all():
        bad = ks[~ks].index[0]
        raise AnalysisInputError(f"cell {bad} needs exactly one turn record for k = 8, 9, 10")

    feedback_cells = late[late.condition != NO_FEEDBACK][CELL].drop_duplicates()
    if not probes.empty or not feedback_cells.empty:
        per_cell = probes.groupby(CELL).case_id.agg(["size", "nunique"]) if not probes.empty \
            else pd.DataFrame(columns=["size", "nunique"])
        expected = pd.MultiIndex.from_frame(feedback_cells)
        missing = expected.difference(per_cell.index)
        if len(missing):
            raise AnalysisInputError(f"cell {missing[0]} has no probe records")
        if not ((per_cell["size"] == 4) & (per_cell["nunique"] == 4)).all():
            raise AnalysisInputError("every cell needs exactly four distinct probe records")

    for name, records in (("turn", late), ("probe", probes)):
        if records.empty:
            continue
        covered = records.groupby(["frame", "condition", "wording", "replicate"]).episode.agg(
            frozenset)
        is_a = covered.index.get_level_values("condition") == NO_FEEDBACK
        if covered[~is_a].nunique() > 1:
            raise AnalysisInputError(f"{name} records: cells do not cover the same lists")
        if is_a.any():
            if covered[is_a].nunique() > 1:
                raise AnalysisInputError(f"{name} records: no-feedback cells differ in lists")
            lists = sorted(covered[is_a].iloc[0])
            if lists != list(range(min(NO_FEEDBACK_LISTS, len(lists)))):
                raise AnalysisInputError("no-feedback condition must use the first lists only")
