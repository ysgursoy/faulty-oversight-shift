import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.cli import main
from oversight_study.config import ModelConfig, StageConfig, load_config
from oversight_study.generation import build_dataset, load_seeds, write_dataset
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.mock import MockModel, Policy
from oversight_study.runner.run import run_stage, seal_run
from oversight_study.scaffold import repository_root
from oversight_study.validation.release import ReleaseError, validate_release

REPO = repository_root(Path(__file__))
FIXTURES = Path(__file__).parent / "fixtures"


class ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.release = Path(self._tmp.name) / "release"
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        dataset = build_dataset(seeds)
        write_dataset(dataset, self.release / "data", dataset_version="v-test")
        config = load_config(REPO / "configs/stages/trial.yaml", StageConfig).model_copy(
            update={"list_count": 1})
        model = load_config(REPO / "configs/models/claude.yaml", ModelConfig)
        self.run_dir = run_stage(config, dataset.episodes["trial"], MockModel(Policy("protocol")),
                                 model=model, settings=GenerationSettings(max_tokens=64),
                                 prompt_version="0" * 64, provenance={},
                                 run_root=self.release / "runs")
        seal_run(self.run_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def reseal(self) -> None:
        lines = []
        for line in (self.run_dir / "checksums.sha256").read_text().splitlines():
            name = line.split("  ", 1)[1]
            digest = hashlib.sha256((self.run_dir / name).read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
        (self.run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")

    def test_valid_release(self) -> None:
        report = validate_release(self.release)
        self.assertEqual(report.dataset_version, "v-test")
        self.assertEqual(list(report.runs.values()), [3 * 43 + 15 * 52])
        self.assertEqual(main(["validate-release", str(self.release)]), 0)

    def test_tampered_results(self) -> None:
        path = self.run_dir / "results.jsonl"
        path.write_text(path.read_text().replace('"y":1000', '"y":1050', 1))
        with self.assertRaisesRegex(ReleaseError, "checksum"):
            validate_release(self.release)
        self.assertEqual(main(["validate-release", str(self.release)]), 1)

    def test_duplicate_record_even_with_fresh_checksums(self) -> None:
        path = self.run_dir / "results.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines + lines[:1]) + "\n")
        self.reseal()
        with self.assertRaisesRegex(ReleaseError, "duplicate"):
            validate_release(self.release)

    def test_incomplete_cell(self) -> None:
        path = self.run_dir / "results.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        self.reseal()
        with self.assertRaisesRegex(ReleaseError, "records"):
            validate_release(self.release)

    def test_unsealed_run(self) -> None:
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        manifest["status"] = "completed"
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ReleaseError, "sealed"):
            validate_release(self.release)

    def test_broken_dataset(self) -> None:
        pool = self.release / "data" / "pools" / "development.json"
        pool.write_text(pool.read_text() + " ")
        with self.assertRaisesRegex(ReleaseError, "dataset"):
            validate_release(self.release)


if __name__ == "__main__":
    unittest.main()
