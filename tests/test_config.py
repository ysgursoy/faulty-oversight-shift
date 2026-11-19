import unittest
from pathlib import Path

from pydantic import ValidationError

from oversight_study.config import (
    AnalysisConfig,
    ConfigNotReadyError,
    ModelConfig,
    PrecheckConfig,
    StageConfig,
    load_config,
    resolved_config_hash,
)
from oversight_study.scaffold import repository_root

CONFIGS = repository_root(Path(__file__)) / "configs"


def stage(name: str) -> StageConfig:
    return load_config(CONFIGS / "stages" / f"{name}.yaml", StageConfig)


class ConfigFileTests(unittest.TestCase):
    def test_all_files_load(self) -> None:
        load_config(CONFIGS / "models" / "claude.yaml", ModelConfig)
        load_config(CONFIGS / "precheck.yaml", PrecheckConfig)
        load_config(CONFIGS / "analysis" / "primary.yaml", AnalysisConfig)
        for name in ("trial", "pilot", "main"):
            self.assertEqual(stage(name).stage, name)

    def test_model_settings(self) -> None:
        model = load_config(CONFIGS / "models" / "claude.yaml", ModelConfig)
        self.assertEqual(model.model_id, "claude-sonnet-5-5")
        self.assertIsNone(model.temperature)
        self.assertEqual((model.thinking, model.effort), ("between_tools", "high"))
        model.require_runnable()
        with self.assertRaises(ConfigNotReadyError):
            model.model_copy(update={"model_id": "TBD"}).require_runnable()

    def test_between_tools_needs_effort_high_or_below(self) -> None:
        data = load_config(CONFIGS / "models" / "claude.yaml", ModelConfig).model_dump()
        with self.assertRaises(ValidationError):
            ModelConfig(**{**data, "effort": "xhigh"})
        with self.assertRaises(ValidationError):
            ModelConfig(**{**data, "thinking": "enabled"})

    def test_analysis_needs_delta_min_and_n_max_before_pilot(self) -> None:
        analysis = load_config(CONFIGS / "analysis" / "primary.yaml", AnalysisConfig)
        self.assertEqual((analysis.delta_min, analysis.n_max), (0.15, 60))
        analysis.require_ready_for_pilot()
        with self.assertRaises(ConfigNotReadyError):
            analysis.model_copy(update={"n_max": None}).require_ready_for_pilot()

    def test_precheck_quotas(self) -> None:
        precheck = load_config(CONFIGS / "precheck.yaml", PrecheckConfig)
        self.assertEqual(sum(precheck.development_subset.values()), 50)
        totals = {w: sum(q.values()) for w, q in precheck.evaluation_by_wording.items()}
        self.assertEqual(totals, {1: 250, 2: 100})


class PlannedCellTests(unittest.TestCase):
    def test_main_cell_count_matches_design(self) -> None:
        main = stage("main")
        with self.assertRaises(ConfigNotReadyError):
            main.model_copy(update={"list_count": None}).planned_cells()
        self.assertEqual(main.list_count, 30)
        cells = main.planned_cells()
        self.assertEqual(len(cells), 480)
        no_feedback = [c for c in cells if c[1] == "a"]
        self.assertEqual(len(no_feedback), 30)
        self.assertEqual({c[3] for c in no_feedback}, set(range(10)))

    def test_pilot_core_and_repeat(self) -> None:
        pilot = stage("pilot")
        self.assertEqual(len(pilot.planned_cells()), 100)
        self.assertEqual(len([c for c in pilot.planned_cells() if c[4] == 1]), 20)
        self.assertEqual(len(pilot.planned_cells(include_optional=True)), 220)

    def test_trial_cells(self) -> None:
        self.assertEqual(len(stage("trial").planned_cells()), 3 * 3 * 4 + 3 * 3 * 2)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.main = stage("main").model_dump()

    def test_pilot_and_trial_only_on_development(self) -> None:
        with self.assertRaises(ValidationError):
            StageConfig(**{**stage("pilot").model_dump(), "split": "evaluation"})

    def test_condition_must_match_stage_m(self) -> None:
        cells = list(self.main["cells"])
        cells[1] = {**cells[1], "conditions": ["c", "b-2"]}
        with self.assertRaises(ValidationError):
            StageConfig(**{**self.main, "cells": cells})

    def test_unknown_values(self) -> None:
        for changes in ({"stage": "stage-x"}, {"split": "holdout"}, {"m": 0}, {"extra": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                StageConfig(**{**self.main, **changes})

    def test_duplicate_cells_rejected(self) -> None:
        cells = list(self.main["cells"]) + [self.main["cells"][2]]
        doubled = StageConfig(**{**self.main, "cells": cells, "list_count": 30})
        with self.assertRaises(ValueError):
            doubled.planned_cells()


class HashTests(unittest.TestCase):
    def test_hash_is_stable_and_sensitive(self) -> None:
        main = stage("main")
        self.assertEqual(resolved_config_hash(main), resolved_config_hash(stage("main")))
        changed = main.model_copy(update={"order_seed": main.order_seed + 1})
        self.assertNotEqual(resolved_config_hash(main), resolved_config_hash(changed))
        self.assertEqual(len(resolved_config_hash(main)), 64)


if __name__ == "__main__":
    unittest.main()
