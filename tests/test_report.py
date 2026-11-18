import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.analysis.report import analyze
from oversight_study.config import AnalysisConfig, ModelConfig, StageConfig, load_config
from oversight_study.generation import build_dataset, load_seeds, write_dataset
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.mock import MockModel, Policy
from oversight_study.runner.run import run_stage, seal_run
from oversight_study.scaffold import repository_root

REPO = repository_root(Path(__file__))
FIXTURES = Path(__file__).parent / "fixtures"
ANALYSIS = load_config(REPO / "configs/analysis/primary.yaml", AnalysisConfig).model_copy(
    update={"bootstrap_n": 200})


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.release = Path(cls._tmp.name) / "release"
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        dataset = build_dataset(seeds)
        write_dataset(dataset, cls.release / "data", dataset_version="v-test")
        config = load_config(REPO / "configs/stages/trial.yaml", StageConfig).model_copy(
            update={"list_count": 3})
        model = load_config(REPO / "configs/models/claude.yaml", ModelConfig)
        policy = Policy("planted", shift_probability={"human": 0.3, "neutral": 0.8,
                                                      "agriculture": 0.5})
        run_dir = run_stage(config, dataset.episodes["trial"], MockModel(policy, seed=4),
                            model=model, settings=GenerationSettings(max_tokens=64),
                            prompt_version="0" * 64, provenance={},
                            run_root=cls.release / "runs")
        seal_run(run_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def run_analysis(self, out):
        return analyze(self.release, out, stage="trial", dataset_split="development", m=1.5,
                       config=ANALYSIS, code_commit="test")

    def test_outputs_and_manifest(self) -> None:
        with TemporaryDirectory() as out:
            result = self.run_analysis(Path(out))
            self.assertTrue(result.name.startswith("trial-development-r0-"))
            manifest = json.loads((result / "manifest.json").read_text())
            self.assertEqual(manifest["dataset_version"], "v-test")
            self.assertEqual(len(manifest["input_runs"]), 1)
            primary = manifest["summary"]["primary"]
            self.assertIn("estimable", primary["protocol_accuracy"])
            self.assertIn("wording_robustness", manifest["summary"])
            self.assertIn("acquisition", manifest["summary"])
            for name in ("protocol_accuracy", "conditional_shift", "conditional_application",
                         "response_distribution", "learning_curve", "return_to_protocol",
                         "two_rules"):
                self.assertTrue((result / "tables" / f"{name}.csv").exists(), name)
            figures = {p.name for p in (result / "figures").iterdir()}
            self.assertIn("frame_difference.svg", figures)
            self.assertIn("wording_robustness.png", figures)
            for rel, digest in manifest["checksums_sha256"].items():
                self.assertTrue((result / rel).exists(), rel)
                self.assertEqual(len(digest), 64)

    def test_rerun_gives_identical_tables(self) -> None:
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            first = json.loads((self.run_analysis(Path(a)) / "manifest.json").read_text())
            second = json.loads((self.run_analysis(Path(b)) / "manifest.json").read_text())
        tables = {k: v for k, v in first["checksums_sha256"].items() if k.startswith("tables/")}
        self.assertTrue(tables)
        for rel, digest in tables.items():
            self.assertEqual(second["checksums_sha256"][rel], digest, rel)
        self.assertEqual(first["summary"], second["summary"])


if __name__ == "__main__":
    unittest.main()
