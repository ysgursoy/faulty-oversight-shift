"""Single-turn protocol check before episodes: does the model apply the protocol?

The development subset is fixed and informs prompt, formula, and model choices. The
evaluation check is only measured and reported, and it is refused unless the second
freeze is recorded and nothing frozen has changed since.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence

import pandas as pd

from oversight_study.generation.pools import U_QUOTAS, u_quota_of
from oversight_study.records.cases import Case
from oversight_study.records.enums import DatasetSplit, Frame
from oversight_study.records.run_records import Attempt, GenerationSettings, PrecheckRecord
from oversight_study.runner.client import Message, ModelClient, ModelResponse
from oversight_study.runner.parsing import parse_turn_output
from oversight_study.runner.request_builder import build_request
from oversight_study.task.render import render_case, render_system_prompt
from oversight_study.validation.freeze import FreezeGate, require_second_freeze

POOL_KEYS = ("A", "near_measurement", "near_weight")
U_KEYS = tuple(name for name, _, _ in U_QUOTAS)


def select_subset(pools: Mapping[str, Sequence[Case]], quotas: Mapping[str, int],
                  seed: int) -> list[Case]:
    """Seeded quota sample. Keys are case types or `U` sub-quota names (`u_light_high`, …)."""
    chosen: list[Case] = []
    for offset, (key, count) in enumerate(sorted(quotas.items())):
        if key in POOL_KEYS:
            candidates = list(pools[key])
        elif key in U_KEYS:
            candidates = [c for c in pools["U"] if u_quota_of(c.weight, c.measurement) == key]
        else:
            raise ValueError(f"unknown quota key: {key}")
        candidates.sort(key=lambda case: case.case_id)
        if len(candidates) < count:
            raise ValueError(f"quota {key}={count} exceeds {len(candidates)} available cases")
        chosen += random.Random(seed + offset).sample(candidates, count)
    return chosen


def _attempt(attempt_id: str, response: ModelResponse) -> Attempt:
    parsed = parse_turn_output(response.text)
    return Attempt(attempt_id=attempt_id, raw_output=response.text,
                   attempt_class="valid" if parsed.ok else "format",
                   extra_text=parsed.extra_text if parsed.ok else "",
                   stop_reason=response.stop_reason, usage=response.usage)


def run_precheck(
    cases: Sequence[Case],
    frames: Sequence[Frame | str],
    wordings: Sequence[int],
    client: ModelClient,
    *,
    dataset_split: DatasetSplit | str,
    run_id: str,
    requested_model: str,
    settings: GenerationSettings,
    prompt_version: str,
    freeze_gate: FreezeGate | None = None,
    on_record: Callable[[PrecheckRecord], None] | None = None,
) -> list[PrecheckRecord]:
    """`on_record` sees each record as soon as it exists, so a caller can persist
    finished calls before a later one fails."""
    if DatasetSplit(dataset_split) == DatasetSplit.evaluation:
        require_second_freeze(freeze_gate)
    records = []
    for frame in frames:
        for wording in wordings:
            system = render_system_prompt(frame, wording, "a", 1.5)  # no evaluator, no m
            for case in cases:
                request = build_request(
                    system, (Message(role="user", content=render_case(case, frame)),),
                    settings, 1.5,
                )
                query_id = f"{run_id}|precheck|{frame}|w{wording}|{case.case_id}"
                responses = [client.complete(request)]
                attempts = [_attempt(f"{query_id}#1", responses[0])]
                if attempts[0].attempt_class != "valid":
                    responses.append(client.complete(request))
                    attempts.append(_attempt(f"{query_id}#2", responses[1]))
                final = parse_turn_output(responses[-1].text)
                record = PrecheckRecord(
                    dataset_split=dataset_split, frame=frame, wording=wording,
                    case_id=case.case_id, case_type=case.case_type, r_star=case.r_star,
                    attempts=tuple(attempts), y=final.value if final.ok else None,
                    invalid_class=None if final.ok else "format",
                    requested_model=requested_model, returned_model=responses[-1].returned_model,
                    generation_settings=settings, prompt_version=prompt_version, run_id=run_id,
                    logical_query_id=query_id,
                )
                records.append(record)
                if on_record is not None:
                    on_record(record)
    return records


def summarize_precheck(records: Sequence[PrecheckRecord], threshold: float) -> pd.DataFrame:
    """Exact-R* rate per frame and wording; `passed` when the rate reaches the threshold.
    No case is dropped: invalid answers count as not exact."""
    df = pd.DataFrame([{"frame": r.frame.value, "wording": r.wording, "exact": r.exact}
                       for r in records])
    summary = df.groupby(["frame", "wording"]).exact.agg(exact="sum", n="size")
    summary["rate"] = summary["exact"] / summary["n"]
    summary["passed"] = summary["rate"] >= threshold
    return summary
