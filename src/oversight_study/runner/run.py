"""Run a stage: every planned cell once, logged, resumable, and sealed at the end.

A cell is one (frame, condition, wording, list, replicate) episode plus its end-of-episode
questions. Cells start in an order shuffled by the stage's `order_seed`; with `workers` > 1,
up to that many cells run at once (a cell's own turns stay sequential). A cell's records are
appended to `results.jsonl` only after the whole cell finished; on resume, completed cells
are skipped and any partial cell's records are moved to `superseded.jsonl` (kept, never
analyzed) before the cell is rerun.

Run directory files: manifest.json, resolved-config.yaml, results.jsonl, requests.jsonl,
superseded.jsonl, events.jsonl, costs.json, failed-requests.jsonl (requests of a cell that
failed part-way, kept for diagnosis), and checksums.sha256 once sealed.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from oversight_study.config import ModelConfig, Prices, StageConfig, resolved_config_hash
from oversight_study.generation.episodes import Episode
from oversight_study.records.enums import ConditionLetter, DatasetSplit
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.client import ModelClient, ModelRequest, ModelResponse
from oversight_study.runner.episode import EpisodeContext, run_episode
from oversight_study.runner.episode_end import ask_episode_end
from oversight_study.validation.freeze import (
    FreezeGate,
    require_second_freeze,
    require_stage_freeze,
)

RunStatus = Literal["running", "partial", "failed", "completed", "sealed"]
FILES = ("manifest.json", "resolved-config.yaml", "results.jsonl", "requests.jsonl",
         "superseded.jsonl", "events.jsonl", "costs.json", "failed-requests.jsonl")
CELL_FIELDS = ("frame", "condition", "wording", "episode", "replicate")


class BudgetExceededError(RuntimeError):
    """The run's cost reached its configured `max_cost_usd`; the run can be resumed."""


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    stage: str
    dataset_split: str
    design_version: str
    status: RunStatus
    config_hash: str
    requested_model: str
    generation_settings: GenerationSettings
    m: float
    provenance: dict[str, str]  # commit, uv_lock, prompts, configs, dataset_manifest hashes
    cells_planned: int
    cells_completed: int
    started_utc: str
    updated_utc: str


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(stage: str, config_hash: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}-{stamp}-{config_hash[:8]}"


def cell_key(record: dict) -> tuple:
    return tuple(record[name] for name in CELL_FIELDS)


def expected_records(condition: str) -> int:
    letter = ConditionLetter(condition.split("-")[0])
    return 40 + (3 if letter == ConditionLetter.a else 12)


@dataclass
class _Totals:
    """Usage and call count summed over every cell of this process; thread-safe."""

    usage: Counter = field(default_factory=Counter)
    calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, usage: dict) -> int:
        with self.lock:
            self.calls += 1
            self.usage.update(usage)
            return self.calls

    def snapshot(self) -> tuple[Counter, int]:
        with self.lock:
            return Counter(self.usage), self.calls


@dataclass
class _Recorder:
    """Wraps the client for one cell: logs each request and response, adds usage to totals."""

    inner: ModelClient
    totals: _Totals
    cell_id: str = ""
    lines: list[str] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResponse:
        response = self.inner.complete(request)
        call = self.totals.add(response.usage.model_dump())
        self.lines.append(json.dumps({
            "cell_id": self.cell_id,
            "call": call,
            "system_sha256": hashlib.sha256(request.system.encode()).hexdigest(),
            "n_messages": len(request.messages),
            "user": request.last_user_text,
            "response": response.text,
            "returned_model": response.returned_model,
        }, sort_keys=True))
        return response


def compute_costs(usage: Counter, calls: int, prices: Prices) -> dict:
    tokens = {"calls": calls, "input_tokens": usage["input_tokens"],
              "output_tokens": usage["output_tokens"],
              "cache_write_tokens": usage["cache_creation_input_tokens"],
              "cache_read_tokens": usage["cache_read_input_tokens"]}
    costs: dict = {**tokens, "prices_usd_per_mtok": prices.model_dump()}
    if None in (prices.input, prices.output, prices.cache_write, prices.cache_read):
        costs["cost_usd_with_cache"] = costs["cost_usd_without_cache"] = None
        return costs
    million = 1_000_000
    out = tokens["output_tokens"] * prices.output / million
    costs["cost_usd_with_cache"] = out + (
        tokens["input_tokens"] * prices.input + tokens["cache_write_tokens"] * prices.cache_write
        + tokens["cache_read_tokens"] * prices.cache_read) / million
    all_input = (tokens["input_tokens"] + tokens["cache_write_tokens"]
                 + tokens["cache_read_tokens"])
    costs["cost_usd_without_cache"] = out + all_input * prices.input / million
    return costs


class Run:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)
        self.lock = threading.RLock()

    def path(self, name: str) -> Path:
        return self.dir / name

    def append(self, name: str, lines: Sequence[str]) -> None:
        if lines:
            with self.lock, self.path(name).open("a", encoding="utf-8") as handle:
                handle.write("".join(line + "\n" for line in lines))

    def event(self, kind: str, **fields) -> None:
        self.append("events.jsonl", [json.dumps({"utc": _now(), "event": kind, **fields},
                                                 sort_keys=True, default=str)])

    def manifest(self) -> RunManifest:
        return RunManifest(**json.loads(self.path("manifest.json").read_text(encoding="utf-8")))

    def write_manifest(self, manifest: RunManifest) -> None:
        self.path("manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n",
                                              encoding="utf-8")

    def results(self) -> list[dict]:
        path = self.path("results.jsonl")
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]


def _supersede_partial_cells(run: Run, condition_of: dict[tuple, str]) -> set[tuple]:
    """Move records of incomplete cells to superseded.jsonl; return completed cell keys."""
    records = run.results()
    counts = Counter(cell_key(r) for r in records)
    complete = {key for key, n in counts.items() if n == expected_records(key[1])}
    partial = set(counts) - complete
    if partial:
        keep = [r for r in records if cell_key(r) not in partial]
        moved = [r for r in records if cell_key(r) in partial]
        run.append("superseded.jsonl", [json.dumps(r, sort_keys=True) for r in moved])
        run.path("results.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in keep), encoding="utf-8")
        for key in sorted(partial, key=str):
            run.event("cell_superseded", cell=list(key), records=counts[key])
    unknown = complete - set(condition_of)
    if unknown:
        raise ValueError(f"results contain cells not planned by this config: {sorted(unknown)[:2]}")
    return complete


def run_stage(
    config: StageConfig,
    episodes: Sequence[Episode],
    client: ModelClient,
    *,
    model: ModelConfig,
    settings: GenerationSettings,
    prompt_version: str,
    provenance: dict[str, str],
    run_root: Path,
    run_id: str | None = None,
    include_optional: bool = False,
    freeze_gate: FreezeGate | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> Path:
    """Start a new run, or resume `run_id` under `run_root`. Returns the run directory.

    With `workers` > 1, cells run concurrently. On a failure or when the budget is reached,
    no new cell starts; cells already running finish and are kept.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if config.split == DatasetSplit.evaluation:
        require_second_freeze(freeze_gate)
    require_stage_freeze(config.stage, freeze_gate, config.design_version, config.list_count)
    cells = config.planned_cells(include_optional)
    by_index = {episode.list_index: episode for episode in episodes}
    missing = {cell[3] for cell in cells} - set(by_index)
    if missing:
        raise ValueError(f"episode lists missing for indices {sorted(missing)[:5]}")
    config_hash = resolved_config_hash(config)
    if config.max_cost_usd is not None and None in (
            model.prices.input, model.prices.output, model.prices.cache_write,
            model.prices.cache_read):
        raise ValueError("max_cost_usd needs all prices in the model config")

    run_id = run_id or make_run_id(config.stage, config_hash)
    run = Run(Path(run_root) / run_id)
    if run.path("manifest.json").exists():
        manifest = run.manifest()
        if manifest.config_hash != config_hash or manifest.status == "sealed":
            raise ValueError("cannot resume: config changed or run already sealed")
        run.event("run_resumed")
    else:
        run.dir.mkdir(parents=True)
        for name in FILES[3:6]:
            run.path(name).touch()
        run.path("resolved-config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
        manifest = RunManifest(
            run_id=run_id, stage=config.stage, dataset_split=config.split,
            design_version=config.design_version, status="running", config_hash=config_hash,
            requested_model=model.model_id, generation_settings=settings, m=config.m,
            provenance=provenance, cells_planned=len(cells), cells_completed=0,
            started_utc=_now(), updated_utc=_now(),
        )
        run.write_manifest(manifest)
        run.event("run_started", cells=len(cells))

    condition_of = {(f, c, w, i, r): c for f, c, w, i, r in cells}
    done = _supersede_partial_cells(run, condition_of)
    order = list(cells)
    random.Random(config.order_seed).shuffle(order)
    totals = _Totals()
    previous = json.loads(run.path("costs.json").read_text()) if run.path(
        "costs.json").exists() else None
    stop = threading.Event()
    failures: list[BaseException] = []

    def run_cell(key: tuple) -> None:
        if stop.is_set():
            return
        frame, condition, wording, index, replicate = key
        ctx = EpisodeContext(
            episode=by_index[index], frame=frame, condition=condition, wording=wording,
            m=config.m, stage=config.stage, dataset_split=config.split, replicate=replicate,
            run_id=run_id, requested_model=model.model_id, settings=settings,
            prompt_version=prompt_version,
        )
        recorder = _Recorder(client, totals, cell_id=ctx.cell_id)
        try:
            result = run_episode(ctx, recorder)
            end = ask_episode_end(result, ctx, recorder)
        except Exception as error:  # noqa: BLE001 - recorded, then re-raised by run_stage
            stop.set()
            with run.lock:
                failures.append(error)
                run.append("failed-requests.jsonl", recorder.lines)
                run.event("cell_failed", cell_id=ctx.cell_id, calls=len(recorder.lines),
                          error=type(error).__name__, message=str(error)[:500])
            return
        with run.lock:
            run.append("results.jsonl", [r.model_dump_json() for r in (*result.turns, *end)])
            run.append("requests.jsonl", recorder.lines)
            done.add(key)
            run.event("cell_completed", cell=list(key))
            if on_progress:
                on_progress(len(done), len(cells))
            if config.max_cost_usd is not None and len(done) < len(cells) and not stop.is_set():
                spent = _total_cost(totals, model.prices, previous)
                if spent >= config.max_cost_usd:
                    stop.set()
                    run.event("budget_exceeded", spent_usd=spent, cap_usd=config.max_cost_usd)
                    failures.append(BudgetExceededError(
                        f"spent ${spent:.2f} of max_cost_usd ${config.max_cost_usd:.2f}"))

    pending = [key for key in order if key not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: set[Future] = set()
        for key in pending:
            if stop.is_set():
                break
            if len(running) >= workers:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                if stop.is_set():
                    break
            running.add(pool.submit(run_cell, key))
        for future in running:
            future.result()

    _write_costs(run, totals, model.prices, previous)
    if failures:
        run.write_manifest(run.manifest().model_copy(update={
            "status": "partial" if done else "failed", "cells_completed": len(done),
            "updated_utc": _now()}))
        raise failures[0]
    run.write_manifest(run.manifest().model_copy(update={
        "status": "completed", "cells_completed": len(done), "updated_utc": _now()}))
    run.event("run_completed", cells=len(done))
    return run.dir


def _usage_and_calls(totals: _Totals, previous: dict | None) -> tuple[Counter, int]:
    usage, calls = totals.snapshot()
    if previous:
        usage.update({"input_tokens": previous["input_tokens"],
                      "output_tokens": previous["output_tokens"],
                      "cache_creation_input_tokens": previous["cache_write_tokens"],
                      "cache_read_input_tokens": previous["cache_read_tokens"]})
        calls += previous["calls"]
    return usage, calls


def _total_cost(totals: _Totals, prices: Prices, previous: dict | None) -> float:
    usage, calls = _usage_and_calls(totals, previous)
    return compute_costs(usage, calls, prices)["cost_usd_with_cache"] or 0.0


def _write_costs(run: Run, totals: _Totals, prices: Prices, previous: dict | None) -> None:
    usage, calls = _usage_and_calls(totals, previous)
    run.path("costs.json").write_text(
        json.dumps(compute_costs(usage, calls, prices), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def seal_run(run_dir: Path) -> Path:
    """Write checksums.sha256 for a completed run and mark it sealed."""
    run = Run(run_dir)
    manifest = run.manifest()
    if manifest.status != "completed":
        raise ValueError(f"only completed runs can be sealed (status: {manifest.status})")
    run.write_manifest(manifest.model_copy(update={"status": "sealed", "updated_utc": _now()}))
    lines = [f"{hashlib.sha256(run.path(name).read_bytes()).hexdigest()}  {name}"
             for name in FILES if run.path(name).exists()]
    run.path("checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run.path("checksums.sha256")
