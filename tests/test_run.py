import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.config import ModelConfig, Prices, StageConfig, load_config
from oversight_study.generation import build_dataset, load_seeds
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.mock import MockModel, Policy
from oversight_study.runner.run import (
    BudgetExceededError,
    Run,
    compute_costs,
    expected_records,
    run_stage,
    seal_run,
)
from oversight_study.scaffold import repository_root
from oversight_study.validation.freeze import FreezeError

REPO = repository_root(Path(__file__))
FIXTURES = Path(__file__).parent / "fixtures"
MODEL = load_config(REPO / "configs/models/claude.yaml", ModelConfig)
TRIAL = load_config(REPO / "configs/stages/trial.yaml", StageConfig).model_copy(
    update={"list_count": 2})
SETTINGS = GenerationSettings(max_tokens=64)
CELLS = 3 * (4 + 2) * 2


class FailAfter:
    def __init__(self, calls: int):
        self.left, self.inner = calls, MockModel(Policy("protocol"))

    def complete(self, request):
        if self.left == 0:
            raise RuntimeError("connection lost")
        self.left -= 1
        return self.inner.complete(request)


class FailOnCall:
    """Raises on one specific call number; thread-safe so it works with workers."""

    def __init__(self, call: int):
        import threading
        self.call, self.seen, self.lock = call, 0, threading.Lock()
        self.inner = MockModel(Policy("protocol"))

    def complete(self, request):
        with self.lock:
            self.seen += 1
            fail = self.seen == self.call
        if fail:
            raise RuntimeError("connection lost")
        return self.inner.complete(request)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.episodes = build_dataset(seeds).episodes["trial"]

    def start(self, root, client, **kwargs):
        return run_stage(TRIAL, self.episodes, client, model=MODEL, settings=SETTINGS,
                         prompt_version="0" * 64, provenance={"commit": "x"}, run_root=root,
                         **kwargs)

    def assert_complete(self, run_dir: Path) -> None:
        records = Run(run_dir).results()
        ids = [r["logical_query_id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)))
        per_cell = Counter((r["frame"], r["condition"], r["wording"], r["episode"],
                            r["replicate"]) for r in records)
        self.assertEqual(len(per_cell), CELLS)
        for key, n in per_cell.items():
            self.assertEqual(n, expected_records(key[1]), key)

    def test_full_run_and_seal(self) -> None:
        with TemporaryDirectory() as root:
            run_dir = self.start(root, MockModel(Policy("evaluator")))
            self.assert_complete(run_dir)
            run = Run(run_dir)
            manifest = run.manifest()
            self.assertEqual((manifest.status, manifest.cells_completed), ("completed", CELLS))
            self.assertTrue(manifest.run_id.startswith("trial-"))
            self.assertEqual(json.loads(run.path("costs.json").read_text())["calls"],
                             len(run.path("requests.jsonl").read_text().splitlines()))
            events = [json.loads(e)["event"] for e in
                      run.path("events.jsonl").read_text().splitlines()]
            self.assertEqual((events[0], events[-1]), ("run_started", "run_completed"))
            self.assertIn("order_seed", run.path("resolved-config.yaml").read_text())
            checksums = seal_run(run_dir).read_text()
            self.assertIn("results.jsonl", checksums)
            self.assertEqual(run.manifest().status, "sealed")
            with self.assertRaises(ValueError):
                self.start(root, MockModel(), run_id=manifest.run_id)

    def test_crash_then_resume(self) -> None:
        with TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                self.start(root, FailAfter(500))
            run_dir = next(Path(root).iterdir())
            self.assertEqual(Run(run_dir).manifest().status, "partial")
            self.assertEqual(Run(run_dir).path("superseded.jsonl").read_text(), "")
            failed = Run(run_dir).path("failed-requests.jsonl").read_text().splitlines()
            self.assertGreater(len(failed), 0)
            self.assertIn('"calls"', Run(run_dir).path("events.jsonl").read_text())
            self.start(root, MockModel(Policy("protocol")), run_id=run_dir.name)
            self.assert_complete(run_dir)
            self.assertEqual(Run(run_dir).manifest().status, "completed")

    def test_partial_cell_is_superseded_and_rerun(self) -> None:
        with TemporaryDirectory() as root:
            run_dir = self.start(root, MockModel(Policy("protocol")))
            run = Run(run_dir)
            lines = run.path("results.jsonl").read_text().splitlines()
            run.path("results.jsonl").write_text("\n".join(lines[:-5]) + "\n")
            run.write_manifest(run.manifest().model_copy(update={"status": "partial"}))
            self.start(root, MockModel(Policy("protocol")), run_id=run_dir.name)
            self.assert_complete(run_dir)
            superseded = run.path("superseded.jsonl").read_text().splitlines()
            self.assertGreater(len(superseded), 0)
            self.assertIn("cell_superseded", run.path("events.jsonl").read_text())

    def test_order_is_shuffled_but_fixed(self) -> None:
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            def cells(root):
                events = Run(next(Path(root).iterdir())).path("events.jsonl").read_text()
                return [json.loads(e)["cell"] for e in events.splitlines()
                        if '"cell_completed"' in e]
            self.start(a, MockModel())
            self.start(b, MockModel())
            self.assertEqual(cells(a), cells(b))
            self.assertNotEqual(cells(a), sorted(cells(a)))

    def test_config_change_blocks_resume(self) -> None:
        with TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                self.start(root, FailAfter(100))
            run_id = next(Path(root).iterdir()).name
            changed = TRIAL.model_copy(update={"order_seed": 1})
            with self.assertRaises(ValueError):
                run_stage(changed, self.episodes, MockModel(), model=MODEL, settings=SETTINGS,
                          prompt_version="0" * 64, provenance={}, run_root=root, run_id=run_id)

    def test_evaluation_refused_and_missing_lists(self) -> None:
        main = load_config(REPO / "configs/stages/main.yaml", StageConfig).model_copy(
            update={"list_count": 1})
        with TemporaryDirectory() as root:
            with self.assertRaises(FreezeError):
                run_stage(main, self.episodes, MockModel(), model=MODEL, settings=SETTINGS,
                          prompt_version="0" * 64, provenance={}, run_root=root)
            with self.assertRaises(ValueError):
                run_stage(TRIAL, self.episodes[:1], MockModel(), model=MODEL, settings=SETTINGS,
                          prompt_version="0" * 64, provenance={}, run_root=root)

    def test_budget_cutter_stops_resumably(self) -> None:
        with TemporaryDirectory() as root:
            tight = TRIAL.model_copy(update={"max_cost_usd": 0.0001})
            with self.assertRaises(BudgetExceededError):
                run_stage(tight, self.episodes, MockModel(), model=MODEL, settings=SETTINGS,
                          prompt_version="0" * 64, provenance={}, run_root=root)
            run_dir = next(Path(root).iterdir())
            run = Run(run_dir)
            self.assertEqual(run.manifest().status, "partial")
            self.assertEqual(run.manifest().cells_completed, 1)
            self.assertIn("budget_exceeded", run.path("events.jsonl").read_text())

    def test_parallel_run_matches_sequential(self) -> None:
        def content(run_dir):
            records = Run(run_dir).results()
            for record in records:
                record.pop("run_id")
                record.pop("logical_query_id")
                for attempt in record.get("attempts", []):
                    attempt.pop("attempt_id")
            return sorted(json.dumps(r, sort_keys=True) for r in records)

        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            sequential = self.start(a, MockModel(Policy("noise", rate=0.3)))
            parallel = self.start(b, MockModel(Policy("noise", rate=0.3)), workers=4)
            self.assert_complete(parallel)
            self.assertEqual(content(sequential), content(parallel))
            costs = [json.loads(Run(d).path("costs.json").read_text())
                     for d in (sequential, parallel)]
            self.assertEqual(costs[0], costs[1])
            calls = [json.loads(line)["call"] for line in
                     Run(parallel).path("requests.jsonl").read_text().splitlines()]
            self.assertEqual(sorted(calls), list(range(1, costs[1]["calls"] + 1)))

    def test_parallel_failure_keeps_finished_cells_and_resumes(self) -> None:
        with TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                self.start(root, FailOnCall(300), workers=4)
            run_dir = next(Path(root).iterdir())
            run = Run(run_dir)
            self.assertEqual(run.manifest().status, "partial")
            self.assertEqual(run.path("superseded.jsonl").read_text(), "")
            self.assertGreater(len(run.path("failed-requests.jsonl").read_text()), 0)
            completed = run.path("events.jsonl").read_text().count('"cell_completed"')
            self.assertEqual(run.manifest().cells_completed, completed)
            self.start(root, MockModel(Policy("protocol")), run_id=run_dir.name, workers=4)
            self.assert_complete(run_dir)
            self.assertEqual(run.manifest().status, "completed")

    def test_parallel_budget_stop_is_resumable(self) -> None:
        with TemporaryDirectory() as root:
            tight = TRIAL.model_copy(update={"max_cost_usd": 0.0001})
            with self.assertRaises(BudgetExceededError):
                run_stage(tight, self.episodes, MockModel(), model=MODEL, settings=SETTINGS,
                          prompt_version="0" * 64, provenance={}, run_root=root, workers=4)
            run = Run(next(Path(root).iterdir()))
            self.assertEqual(run.manifest().status, "partial")
            self.assertLessEqual(run.manifest().cells_completed, 4)
            self.assertEqual(run.path("events.jsonl").read_text().count("budget_exceeded"), 1)

    def test_workers_must_be_positive(self) -> None:
        with TemporaryDirectory() as root, self.assertRaises(ValueError):
            self.start(root, MockModel(), workers=0)

    def test_budget_needs_prices(self) -> None:
        no_prices = MODEL.model_copy(update={"prices": Prices()})
        with TemporaryDirectory() as root, self.assertRaises(ValueError):
            run_stage(TRIAL, self.episodes, MockModel(), model=no_prices, settings=SETTINGS,
                      prompt_version="0" * 64, provenance={}, run_root=root)

    def test_costs(self) -> None:
        usage = Counter(input_tokens=1_000_000, output_tokens=100_000,
                        cache_creation_input_tokens=500_000, cache_read_input_tokens=2_000_000)
        costs = compute_costs(usage, 10, Prices(input=3, output=15, cache_write=3.75,
                                                cache_read=0.3))
        self.assertAlmostEqual(costs["cost_usd_with_cache"], 3 + 1.5 + 1.875 + 0.6)
        self.assertAlmostEqual(costs["cost_usd_without_cache"], 3.5 * 3 + 1.5)
        self.assertIsNone(compute_costs(usage, 10, Prices())["cost_usd_with_cache"])


if __name__ == "__main__":
    unittest.main()
