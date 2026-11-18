"""Load `results.jsonl` files into validated records and analysis tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oversight_study.analysis.validate import AnalysisInputError, check_duplicates
from oversight_study.records.run_records import EpisodeEndRecord, TurnRecord

RESULTS_FILE = "results.jsonl"


@dataclass(frozen=True)
class Records:
    turns: pd.DataFrame  # all turn records
    end: pd.DataFrame  # all episode_end records
    late: pd.DataFrame  # turn records of affected cases k = 8, 9, 10
    probes: pd.DataFrame  # probe_evaluator episode_end records


def parse_line(line: str, where: str) -> TurnRecord | EpisodeEndRecord:
    data = json.loads(line)
    kind = data.get("record_type")
    if kind == "turn":
        return TurnRecord(**data)
    if kind == "episode_end":
        return EpisodeEndRecord(**data)
    raise AnalysisInputError(f"{where}: unknown record_type {kind!r}")


def _frame(records: list) -> pd.DataFrame:
    rows = [r.model_dump(mode="json", exclude={"attempts", "generation_settings"})
            for r in records]
    return pd.DataFrame(rows)


def load_records(
    run_dirs: Iterable[Path | str], dataset_split: str, stage: str
) -> Records:
    """Read every record (each validated by its model), keep one split and stage."""
    turns, end = [], []
    for run_dir in run_dirs:
        path = Path(run_dir) / RESULTS_FILE
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = parse_line(line, f"{path}:{number}")
            if record.dataset_split != dataset_split or record.stage != stage:
                continue
            (turns if record.record_type == "turn" else end).append(record)
    turn_df, end_df = _frame(turns), _frame(end)
    check_duplicates(pd.concat([turn_df, end_df], ignore_index=True))
    if turn_df.empty:
        raise AnalysisInputError(f"no turn records for {dataset_split}/{stage}")
    late = turn_df[(turn_df.case_type == "A") & (turn_df.k >= 8)].copy()
    probes = (
        end_df[end_df.question_type == "probe_evaluator"].copy()
        if not end_df.empty else end_df
    )
    return Records(turns=turn_df, end=end_df, late=late, probes=probes)
