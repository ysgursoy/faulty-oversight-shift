import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from oversight_study.cli import build_parser, main
from oversight_study.scaffold import repository_root

REPO = repository_root(Path(__file__))
FIXTURES = Path(__file__).parent / "fixtures"
SEEDS = ["--seeds-dev", str(FIXTURES / "seeds-dev.yaml"),
         "--seeds-test", str(FIXTURES / "seeds-test.yaml")]


def quiet(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class ParserTests(unittest.TestCase):
    def test_commands_exist(self) -> None:
        parser = build_parser()
        for argv in (["status"], ["smoke"], ["schemas"], ["seal", "x"],
                     ["validate-release", "x"], ["analyze", "x", "--stage", "pilot",
                                                  "--split", "development"],
                     ["generate", "--dataset-version", "v", "--out", "x"],
                     ["run", "--config", "c", "--dataset", "d", "--mock", "protocol"],
                     ["precheck", "--dataset", "d", "--split", "development", "--out", "o"],
                     ["classify", "--run", "r", "--dataset", "d", "--mock"],
                     ["freeze", "--kind", "first", "--approver", "a"]):
            with self.subTest(argv=argv):
                parser.parse_args(argv)

    def test_status(self) -> None:
        code, out, _ = quiet(["status"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(out)["results_available"])


class PaidRefusalTests(unittest.TestCase):
    def test_run_without_mock_or_allow_paid(self) -> None:
        with TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            quiet(["generate", "--dataset-version", "v", "--out", str(data), *SEEDS])
            code, _, err = quiet(["run", "--config", str(REPO / "configs/stages/trial.yaml"),
                                  "--dataset", str(data), "--runs-root", temp])
            self.assertEqual(code, 2)
            self.assertIn("--allow-paid", err)
            with mock.patch.dict(os.environ, {}, clear=True):
                code, _, err = quiet(["run", "--config",
                                      str(REPO / "configs/stages/trial.yaml"),
                                      "--dataset", str(data), "--runs-root", temp,
                                      "--allow-paid"])
            self.assertEqual(code, 2)
            self.assertIn("ANTHROPIC_API_KEY", err)

    def test_classify_without_mock(self) -> None:
        code, _, _ = quiet(["classify", "--run", "r", "--dataset", "d"])
        self.assertEqual(code, 2)


class EndToEndMockTests(unittest.TestCase):
    def test_generate_run_seal_classify_analyze_validate(self) -> None:
        with TemporaryDirectory() as temp:
            release = Path(temp) / "release"
            data, runs = release / "data", release / "runs"
            self.assertEqual(quiet(["generate", "--dataset-version", "v-test", "--out", str(data),
                                    *SEEDS])[0], 0)
            code, out, _ = quiet(["run", "--config", str(REPO / "configs/stages/trial.yaml"),
                                  "--dataset", str(data), "--runs-root", str(runs),
                                  "--mock", "evaluator"])
            self.assertEqual(code, 0, out)
            run_dir = next(runs.iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["requested_model"], "mock-model")
            self.assertIn("prompts", manifest["provenance"])
            self.assertEqual(quiet(["classify", "--run", str(run_dir), "--dataset", str(data),
                                    "--mock", "--spec", str(REPO / "annotations/spec.yaml")])[0], 0)
            self.assertEqual(quiet(["seal", str(run_dir)])[0], 0)
            self.assertEqual(quiet(["validate-release", str(release)])[0], 0)
            fast = Path(temp) / "analysis.yaml"
            fast.write_text((REPO / "configs/analysis/primary.yaml").read_text()
                            .replace("bootstrap_n: 10000", "bootstrap_n: 200"))
            code, out, _ = quiet(["analyze", str(release), "--stage", "trial", "--split",
                                  "development", "--out", str(Path(temp) / "results"),
                                  "--config", str(fast)])
            self.assertEqual(code, 0, out)
            self.assertTrue(any((Path(temp) / "results").iterdir()))

    def test_precheck_mock(self) -> None:
        with TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            quiet(["generate", "--dataset-version", "v", "--out", str(data), *SEEDS])
            code, out, _ = quiet(["precheck", "--config", str(REPO / "configs/precheck.yaml"),
                                  "--dataset", str(data), "--split", "development",
                                  "--wording", "1", "--out", str(Path(temp) / "pre"),
                                  "--mock", "protocol"])
            self.assertEqual(code, 0, out)
            self.assertTrue((Path(temp) / "pre" / "precheck_summary.csv").exists())

    def test_evaluation_precheck_refused_without_freeze(self) -> None:
        with TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            quiet(["generate", "--dataset-version", "v", "--out", str(data), *SEEDS])
            with self.assertRaises(Exception) as caught:
                main(["precheck", "--config", str(REPO / "configs/precheck.yaml"), "--dataset",
                      str(data), "--split", "evaluation", "--out", str(Path(temp) / "pre"),
                      "--mock", "protocol"])
            self.assertIn("freeze", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
