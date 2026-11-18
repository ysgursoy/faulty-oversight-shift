import ast
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.config import ModelConfig, StageConfig, load_config
from oversight_study.generation import build_dataset, load_seeds, write_dataset
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.mock import MockModel
from oversight_study.runner.run import run_stage, seal_run
from oversight_study.scaffold import repository_root
from oversight_study.validation.release import ReleaseError, assemble_release

REPO = repository_root(Path(__file__))
SCRIPTS = REPO / "scripts"
NAMES = ("generate_data", "run_stage", "classify", "analyze", "assemble_release")


class ScriptTests(unittest.TestCase):
    def test_help_works(self) -> None:
        for name in NAMES:
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--help"],
                                        capture_output=True, text=True, cwd=REPO, timeout=60,
                                        check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage", result.stdout)

    def test_scripts_hold_no_logic(self) -> None:
        for name in NAMES:
            tree = ast.parse((SCRIPTS / f"{name}.py").read_text())
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            self.assertEqual(functions, [], name)
            self.assertLess(len((SCRIPTS / f"{name}.py").read_text().splitlines()), 25, name)


class AssembleTests(unittest.TestCase):
    def test_assemble_and_refuse_overwrite(self) -> None:
        fixtures = Path(__file__).parent / "fixtures"
        seeds = load_seeds(True, dev_path=fixtures / "seeds-dev.yaml",
                           test_path=fixtures / "seeds-test.yaml")
        with TemporaryDirectory() as temp:
            dataset = build_dataset(seeds)
            write_dataset(dataset, Path(temp) / "data", dataset_version="v")
            config = load_config(REPO / "configs/stages/trial.yaml", StageConfig).model_copy(
                update={"list_count": 1})
            run_dir = run_stage(config, dataset.episodes["trial"], MockModel(),
                                model=load_config(REPO / "configs/models/claude.yaml",
                                                  ModelConfig),
                                settings=GenerationSettings(max_tokens=64),
                                prompt_version="0" * 64, provenance={},
                                run_root=Path(temp) / "runs")
            seal_run(run_dir)
            report = assemble_release(Path(temp) / "data", [run_dir], Path(temp) / "release")
            self.assertEqual(len(report.runs), 1)
            with self.assertRaises(ReleaseError):
                assemble_release(Path(temp) / "data", [run_dir], Path(temp) / "release")


if __name__ == "__main__":
    unittest.main()
