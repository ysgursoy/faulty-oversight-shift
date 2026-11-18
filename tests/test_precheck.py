import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.config import PrecheckConfig, load_config
from oversight_study.generation import build_dataset, load_seeds
from oversight_study.generation.pools import u_quota_of
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.mock import MockModel, Policy
from oversight_study.runner.precheck import (
    run_precheck,
    select_subset,
    summarize_precheck,
)
from oversight_study.scaffold import repository_root
from oversight_study.validation.freeze import FreezeError, FreezeGate
from oversight_study.validation.freeze_record import FreezeRecord, write_freeze

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_config(repository_root(Path(__file__)) / "configs/precheck.yaml", PrecheckConfig)
COMMON = {"run_id": "pre", "requested_model": "mock-model",
          "settings": GenerationSettings(max_tokens=64), "prompt_version": "0" * 64}


class PrecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        cls.dataset = build_dataset(seeds)
        cls.dev_subset = select_subset(cls.dataset.pools["development"],
                                       CONFIG.development_subset, CONFIG.sample_seed)

    def test_development_subset_quotas(self) -> None:
        self.assertEqual(len(self.dev_subset), 50)
        self.assertEqual(sum(c.case_type == "A" for c in self.dev_subset), 16)
        self.assertEqual(sum(u_quota_of(c.weight, c.measurement) == "u_light_mid"
                             for c in self.dev_subset), 5)
        self.assertEqual(len({c.case_id for c in self.dev_subset}), 50)
        again = select_subset(self.dataset.pools["development"], CONFIG.development_subset,
                              CONFIG.sample_seed)
        self.assertEqual(again, self.dev_subset)

    def test_evaluation_subsets_per_wording(self) -> None:
        pools = self.dataset.pools["evaluation"]
        sizes = {w: len(select_subset(pools, q, CONFIG.sample_seed))
                 for w, q in CONFIG.evaluation_by_wording.items()}
        self.assertEqual(sizes, {1: 250, 2: 100})

    def test_bad_quotas(self) -> None:
        pools = self.dataset.pools["development"]
        with self.assertRaises(ValueError):
            select_subset(pools, {"A": 151}, 0)
        with self.assertRaises(ValueError):
            select_subset(pools, {"u_unknown": 1}, 0)

    def test_protocol_model_passes(self) -> None:
        records = run_precheck(self.dev_subset, CONFIG.frames, CONFIG.wordings,
                               MockModel(Policy("protocol")), dataset_split="development",
                               **COMMON)
        self.assertEqual(len(records), 50 * 3 * 2)
        summary = summarize_precheck(records, CONFIG.threshold)
        self.assertTrue(summary.passed.all())
        self.assertTrue((summary.rate == 1.0).all())

    def test_noisy_model_fails_threshold(self) -> None:
        records = run_precheck(self.dev_subset, ["human"], [1],
                               MockModel(Policy("noise", rate=0.5)), dataset_split="development",
                               **COMMON)
        summary = summarize_precheck(records, CONFIG.threshold)
        self.assertFalse(summary.passed.any())

    def test_invalid_answers_stay_in_denominator(self) -> None:
        records = run_precheck(self.dev_subset[:10], ["neutral"], [2],
                               MockModel(Policy("invalid", rate=1.0)), dataset_split="development",
                               **COMMON)
        self.assertTrue(all(len(r.attempts) == 2 and r.y is None for r in records))
        self.assertEqual(summarize_precheck(records, 0.9).loc[("neutral", 2)].n, 10)

    def test_no_evaluator_text_in_precheck_prompt(self) -> None:
        captured = []

        class Recorder(MockModel):
            def complete(self, request):
                captured.append(request)
                return super().complete(request)

        run_precheck(self.dev_subset[:1], ["human"], [1], Recorder(Policy()),
                     dataset_split="development", **COMMON)
        self.assertNotIn("automated evaluator", captured[0].system)
        self.assertEqual(len(captured[0].messages), 1)

    def test_evaluation_refused_unless_allowed(self) -> None:
        cases = self.dataset.pools["evaluation"]["A"][:1]
        with self.assertRaises(FreezeError):
            run_precheck(cases, ["human"], [1], MockModel(), dataset_split="evaluation",
                         **COMMON)
        with TemporaryDirectory() as temp:
            hashes = {"prompts": "p", "configs": "c", "uv_lock": "u", "dataset_manifest": "d"}
            write_freeze(FreezeRecord(kind="second", utc="2026-11-18T00:00:00Z",
                                      commit="a" * 40, approver="x", hashes=hashes,
                                      n_lists=30, design_version="1"), Path(temp))
            run_precheck(cases, ["human"], [1], MockModel(), dataset_split="evaluation",
                         freeze_gate=FreezeGate(Path(temp), hashes), **COMMON)


if __name__ == "__main__":
    unittest.main()
