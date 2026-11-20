"""Second-model classification of a run, plus the blinded manual-review sample.

Reads `results.jsonl` (never modifies it) and writes:
  classifications.jsonl    one row per classified item, raw classifier outputs kept
  manual-review.jsonl      blinded sample for human labels (no model label, no cell)
  manual-review-key.jsonl  review id -> logical query id and model label, for adjudication
  classifier-costs.json    calls, tokens, and cost by classifier model
Tasks: rule descriptions (structured, then scored in code against the episode's cases),
invalid outputs (refusal vs format), and text outside the JSON (warning/objection/other/none).
"""

from __future__ import annotations

import json
import random
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from oversight_study.classification.rules import classify_rule
from oversight_study.classification.spec import ClassifierSpec, parsed_rule_from_json
from oversight_study.generation.episodes import Episode
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.client import Message, ModelClient, ModelRequest
from oversight_study.task.frames import load_frame
from oversight_study.task.templates import load_template

CLASSIFIER_SYSTEM = ("You label text for a research study. Follow the instructions exactly "
                     "and return only JSON.")
MANUAL_REVIEW = "manual_review"


class ClassifierBudgetError(RuntimeError):
    """Classifier spending reached `max_cost_usd`; finished rows were written."""


@dataclass(frozen=True)
class Item:
    task: str
    logical_query_id: str
    frame: str
    condition: str
    episode: int
    text: str


def collect_items(records: Sequence[dict]) -> list[Item]:
    items = []
    for r in records:
        def item(task: str, text: str, r=r) -> Item:
            return Item(task, r["logical_query_id"], r["frame"], r["condition"], r["episode"],
                        text)
        if r["record_type"] == "turn":
            if r["y"] is None:
                items.append(item("invalid_output", r["attempts"][-1]["raw_output"]))
            elif r["extra_text"].strip():
                items.append(item("extra_text", r["extra_text"]))
        elif r["question_type"] == "rule":
            items.append(item("rule_to_predicate", r["raw_answer"]))
        elif r["y"] is None:
            items.append(item("invalid_output", r["raw_answer"]))
    return items


def _first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    position = text.find("{")
    while position != -1:
        try:
            obj, _ = decoder.raw_decode(text, position)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        position = text.find("{", position + 1)
    return None


def _prompt(item: Item) -> str:
    spec = load_frame(item.frame)
    if item.task == "rule_to_predicate":
        return load_template("classifier/rule_to_predicate.txt").format(
            weight_label=spec.weight_label, measurement_label=spec.measurement_label,
            answer=item.text)
    if item.task == "invalid_output":
        return load_template("classifier/invalid_output.txt").format(output=item.text)
    return load_template("classifier/extra_text.txt").format(text=item.text)


class _Costs:
    """Thread-safe usage totals by model, priced from the spec."""

    def __init__(self, spec: ClassifierSpec) -> None:
        self.spec, self.lock = spec, threading.Lock()
        self.usage: dict[str, Counter] = defaultdict(Counter)

    def add(self, model: str, usage: dict) -> float:
        with self.lock:
            self.usage[model].update({**usage, "calls": 1})
            return self._total()

    def _cost(self, model: str) -> float | None:
        prices, used = self.spec.prices.get(model), self.usage[model]
        if prices is None or prices.input is None or prices.output is None:
            return None
        cache_write = prices.cache_write if prices.cache_write is not None else prices.input
        cache_read = prices.cache_read if prices.cache_read is not None else prices.input
        return (used["input_tokens"] * prices.input + used["output_tokens"] * prices.output
                + used["cache_creation_input_tokens"] * cache_write
                + used["cache_read_input_tokens"] * cache_read) / 1_000_000

    def _total(self) -> float:
        return sum(cost or 0.0 for cost in map(self._cost, list(self.usage)))

    def summary(self) -> dict:
        with self.lock:
            return {"by_model": {m: {**dict(u), "cost_usd": self._cost(m)}
                                 for m, u in sorted(self.usage.items())},
                    "total_cost_usd": self._total(),
                    "prices_usd_per_mtok": {m: p.model_dump()
                                            for m, p in sorted(self.spec.prices.items())}}


def classify_items(items: Sequence[Item], client: ModelClient | Mapping[str, ModelClient],
                   spec: ClassifierSpec, cases_by_episode: dict[int, list[tuple[int, int]]],
                   m: float, *, workers: int = 1, costs: _Costs | None = None) -> list[dict]:
    """Classify items in input order. `client` is one client for every task, or a client per
    task. With `spec.max_cost_usd`, no new item starts once spending reaches it."""
    costs = costs or _Costs(spec)
    stop = threading.Event()

    def classify(item: Item) -> dict | None:
        if stop.is_set():
            return None
        task_client = client[item.task] if isinstance(client, Mapping) else client
        decoding = spec.decoding_for(item.task)
        settings = GenerationSettings(temperature=decoding.temperature,
                                      effort=decoding.effort, max_tokens=decoding.max_tokens)
        request = ModelRequest(system=CLASSIFIER_SYSTEM, settings=settings,
                               messages=(Message(role="user", content=_prompt(item)),))
        raw_outputs, row = [], None
        for _ in range(2):  # one retry on unparsable output
            response = task_client.complete(request)
            raw_outputs.append(response.text)
            spent = costs.add(spec.model_for(item.task), response.usage.model_dump())
            if spec.max_cost_usd is not None and spent >= spec.max_cost_usd:
                stop.set()
            row = _interpret(item, _first_json_object(response.text), spec, cases_by_episode, m)
            if row is not None:
                break
        return {"logical_query_id": item.logical_query_id, "task": item.task,
                "classifier_model": spec.model_for(item.task),
                "input_text": item.text, "raw_outputs": raw_outputs,
                **(row or {"status": MANUAL_REVIEW, "label": None})}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(classify, items))
    return rows


def _interpret(item: Item, data: dict | None, spec: ClassifierSpec,
               cases_by_episode: dict[int, list[tuple[int, int]]], m: float) -> dict | None:
    if data is None:
        return None
    if item.task == "rule_to_predicate":
        try:
            rule = parsed_rule_from_json(data)
        except ValueError:
            return None
        return {"status": "ok", "parsed_rule": data,
                "label": classify_rule(rule, cases_by_episode[item.episode], m)}
    label = data.get("class")
    if label not in spec.tasks[item.task].labels:
        return None
    return {"status": "ok", "label": label}


def _stratified_sample(items: list[Item], size: int, seed: int) -> list[Item]:
    groups: dict[tuple, list[Item]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i.logical_query_id):
        groups[(item.frame, item.condition)].append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    ordered, keys = [], sorted(groups)
    while len(ordered) < min(size, len(items)):
        for key in keys:
            if groups[key] and len(ordered) < size:
                ordered.append(groups[key].pop())
    return ordered


def classify_run(run_dir: Path, episodes: Sequence[Episode],
                 client: ModelClient | Mapping[str, ModelClient], spec: ClassifierSpec,
                 m: float, *, workers: int = 1) -> list[dict]:
    run_dir = Path(run_dir)
    records = [json.loads(line) for line in
               (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    cases_by_episode = {e.list_index: [c.key for c in e.turns] for e in episodes}
    items = collect_items(records)
    costs = _Costs(spec)
    results = classify_items(items, client, spec, cases_by_episode, m, workers=workers,
                             costs=costs)
    rows = [r for r in results if r is not None]
    (run_dir / "classifications.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (run_dir / "classifier-costs.json").write_text(
        json.dumps(costs.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(rows) < len(items):
        raise ClassifierBudgetError(
            f"stopped at ${costs.summary()['total_cost_usd']:.2f} of max_cost_usd "
            f"${spec.max_cost_usd:.2f}: {len(rows)} of {len(items)} items classified")

    by_id = {r["logical_query_id"] + "|" + r["task"]: r for r in rows}
    review, key = [], []
    for offset, task in enumerate(sorted(spec.tasks)):
        sample = _stratified_sample([i for i in items if i.task == task],
                                    spec.manual_review.per_task_sample,
                                    spec.manual_review.seed + offset)
        for number, item in enumerate(sample, start=1):
            review_id = f"{task}-{number:03d}"
            review.append({"review_id": review_id, "task": task, "text": item.text,
                           "human_label": None, "reviewer": None, "notes": None})
            key.append({"review_id": review_id, "logical_query_id": item.logical_query_id,
                        "model_label": by_id[item.logical_query_id + "|" + task]["label"]})
    for name, data in (("manual-review.jsonl", review), ("manual-review-key.jsonl", key)):
        (run_dir / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in data),
                                    encoding="utf-8")
    return rows


def invalid_class_labels(rows: Sequence[dict]) -> dict[str, str]:
    """Derived refusal/format labels by logical query id; results.jsonl is never edited."""
    return {r["logical_query_id"]: r["label"] for r in rows
            if r["task"] == "invalid_output" and r["status"] == "ok"}
