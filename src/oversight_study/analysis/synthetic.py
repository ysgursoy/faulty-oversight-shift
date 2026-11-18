"""Synthetic records with a planted frame effect, for validating the analysis offline.

Primary cells only: frames human and neutral, conditions c and b-{m}, wording 1. In each
late affected answer the model gives exactly R* with the cell's accuracy and the evaluator
value otherwise. The planted protocol-accuracy frame difference Δ equals `effect`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oversight_study.analysis.metrics import target_value

PROTOCOL_VALUES = tuple(range(450, 1401, 50))  # R* range of affected cases
PAIRS = (("measurement", "inside"), ("measurement", "outside"),
         ("weight", "inside"), ("weight", "outside"))


@dataclass(frozen=True)
class SyntheticStudy:
    late: pd.DataFrame
    probes: pd.DataFrame
    r_star_by_case: dict[str, int]
    accuracy: dict[tuple[str, str], float]


def make_synthetic_records(effect: float, n_episodes: int, seed: int = 0, m: float = 1.5,
                           correct_accuracy: float = 0.95,
                           faulty_accuracy: float = 0.40) -> SyntheticStudy:
    if not 0 <= faulty_accuracy + effect <= 1:
        raise ValueError("planted accuracy must stay within [0, 1]")
    condition = f"b-{m:g}"
    accuracy = {("human", "c"): correct_accuracy, ("neutral", "c"): correct_accuracy,
                ("neutral", condition): faulty_accuracy,
                ("human", condition): faulty_accuracy + effect}
    rng = np.random.default_rng(seed)
    late_rows, probe_rows, r_star_by_case = [], [], {}
    for (frame, cond), acc in accuracy.items():
        for episode in range(n_episodes):
            cell = {"frame": frame, "condition": cond, "wording": 1, "episode": episode,
                    "replicate": 0}
            for k in (8, 9, 10):
                protocol = int(rng.choice(PROTOCOL_VALUES))
                evaluator = target_value(protocol, m)
                y = protocol if rng.random() < acc else evaluator
                late_rows.append({**cell, "k": k, "case_id": f"c-{episode}-{k}",
                                  "case_type": "A", "y": y, "r_star": protocol, "r": evaluator,
                                  "invalid_class": None,
                                  "logical_query_id": f"{frame}|{cond}|{episode}|t{k}"})
            for i, (pair, side) in enumerate(PAIRS):
                case_id = f"p-{episode}-{i}"
                r_star_by_case.setdefault(case_id, int(rng.choice(PROTOCOL_VALUES)))
                probe_rows.append({**cell, "case_id": case_id, "pair": pair, "side": side,
                                   "question_type": "probe_evaluator",
                                   "y": r_star_by_case[case_id]})
    return SyntheticStudy(pd.DataFrame(late_rows), pd.DataFrame(probe_rows), r_star_by_case,
                          accuracy)
