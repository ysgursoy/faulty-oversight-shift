import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.cli import build_parser
from oversight_study.config import AnalysisConfig, ConfigNotReadyError, ModelConfig, load_config
from oversight_study.scaffold import repository_root
from oversight_study.validation.freeze_record import (
    compute_hashes,
    make_freeze_record,
    read_freezes,
    write_freeze,
)

REPO = repository_root(Path(__file__))
MODEL = load_config(REPO / "configs/models/claude.yaml", ModelConfig)
ANALYSIS = load_config(REPO / "configs/analysis/primary.yaml", AnalysisConfig)
READY_MODEL = MODEL.model_copy(update={"model_id": "claude-x-20260101"})
READY_ANALYSIS = ANALYSIS.model_copy(update={"delta_min": 0.15, "n_max": 60})
COMMIT = "a" * 40
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


def first(existing=(), **kw):
    args = {"approver": "approver", "commit": COMMIT, "hashes": {"prompts": "p"},
            "model": READY_MODEL, "analysis": READY_ANALYSIS, "existing": list(existing),
            "now": NOW, **kw}
    return make_freeze_record("first", **args)


class FreezeRecordTests(unittest.TestCase):
    def test_incomplete_config_refused(self) -> None:
        for model, analysis in ((MODEL, READY_ANALYSIS), (READY_MODEL, ANALYSIS)):
            with self.assertRaises(ConfigNotReadyError):
                first(model=model, analysis=analysis)

    def test_first_freeze_record(self) -> None:
        record = first()
        self.assertEqual((record.kind, record.utc), ("first", "2026-11-18T12:00:00Z"))
        with self.assertRaises(ValueError):
            first(n_lists=30)

    def test_second_freeze_rules(self) -> None:
        base = {"approver": "a", "commit": COMMIT, "model": READY_MODEL,
                "analysis": READY_ANALYSIS, "now": NOW}
        hashes = {"prompts": "p", "dataset_manifest": "d"}
        with self.assertRaisesRegex(ValueError, "first freeze"):
            make_freeze_record("second", hashes=hashes, existing=[], n_lists=30,
                               design_version="1", **base)
        done = [first()]
        with self.assertRaisesRegex(ValueError, "n_lists"):
            make_freeze_record("second", hashes=hashes, existing=done, **base)
        with self.assertRaisesRegex(ValueError, "dataset manifest"):
            make_freeze_record("second", hashes={"prompts": "p"}, existing=done, n_lists=30,
                               design_version="1", **base)
        record = make_freeze_record("second", hashes=hashes, existing=done, n_lists=30,
                                    design_version="1", **base)
        self.assertEqual(record.n_lists, 30)

    def test_same_kind_twice_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "already recorded"):
            first(existing=[first()])

    def test_write_and_read_back(self) -> None:
        with TemporaryDirectory() as temp:
            decisions = Path(temp) / "decisions"
            note = write_freeze(first(), decisions)
            self.assertEqual(note.name, "2026-11-18-first-freeze.md")
            self.assertIn(f"Commit: {COMMIT}", note.read_text())
            second = make_freeze_record(
                "second", approver="a", commit=COMMIT,
                hashes={"prompts": "p", "dataset_manifest": "d"}, model=READY_MODEL,
                analysis=READY_ANALYSIS, existing=read_freezes(decisions), n_lists=40,
                design_version="2", now=NOW)
            write_freeze(second, decisions)
            records = read_freezes(decisions)
        self.assertEqual([r.kind for r in records], ["first", "second"])
        self.assertEqual(records[1].hashes["dataset_manifest"], "d")

    def test_hashes_cover_prompts_configs_lock(self) -> None:
        hashes = compute_hashes(REPO)
        self.assertEqual(set(hashes), {"prompts", "configs", "uv_lock"})
        self.assertTrue(all(len(v) == 64 for v in hashes.values()))

    def test_cli_requires_kind_and_approver(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["freeze", "--kind", "first", "--approver", "x"])
        self.assertEqual(args.kind, "first")
        with self.assertRaises(SystemExit):
            parser.parse_args(["freeze", "--kind", "first"])


if __name__ == "__main__":
    unittest.main()
