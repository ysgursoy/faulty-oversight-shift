import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.config import ModelConfig, StageConfig, load_config
from oversight_study.generation import build_dataset, load_seeds
from oversight_study.records.run_records import GenerationSettings
from oversight_study.runner.mock import MockModel
from oversight_study.runner.run import run_stage
from oversight_study.scaffold import repository_root
from oversight_study.validation.freeze import FreezeError, FreezeGate, require_second_freeze
from oversight_study.validation.freeze_record import FreezeRecord, write_freeze

REPO = repository_root(Path(__file__))
FIXTURES = Path(__file__).parent / "fixtures"
HASHES = {"prompts": "p", "configs": "c", "uv_lock": "u", "dataset_manifest": "d"}


def record(kind, hashes=HASHES):
    extra = {"n_lists": 1, "design_version": "1"} if kind == "second" else {}
    return FreezeRecord(kind=kind, utc="2026-11-18T00:00:00Z", commit="a" * 40,
                        approver="x", hashes=hashes, **extra)


class CountingClient(MockModel):
    pass


class FreezeGateTests(unittest.TestCase):
    def test_missing_freeze(self) -> None:
        with TemporaryDirectory() as temp, self.assertRaisesRegex(FreezeError, "no second freeze"):
            require_second_freeze(FreezeGate(Path(temp), HASHES))
        with self.assertRaises(FreezeError):
            require_second_freeze(None)

    def test_changed_input_named(self) -> None:
        with TemporaryDirectory() as temp:
            write_freeze(record("first"), Path(temp))
            write_freeze(record("second"), Path(temp))
            require_second_freeze(FreezeGate(Path(temp), HASHES))
            for key in HASHES:
                changed = {**HASHES, key: "other"}
                with self.subTest(key=key), self.assertRaisesRegex(FreezeError, key):
                    require_second_freeze(FreezeGate(Path(temp), changed))

    def test_first_freeze_alone_is_not_enough_for_evaluation(self) -> None:
        with TemporaryDirectory() as temp:
            write_freeze(record("first"), Path(temp))
            with self.assertRaises(FreezeError):
                require_second_freeze(FreezeGate(Path(temp), HASHES))


class EvaluationRunGuardTests(unittest.TestCase):
    def test_no_client_call_without_freeze(self) -> None:
        seeds = load_seeds(True, dev_path=FIXTURES / "seeds-dev.yaml",
                           test_path=FIXTURES / "seeds-test.yaml")
        episodes = build_dataset(seeds, n_main=1).episodes["main"]
        main = load_config(REPO / "configs/stages/main.yaml", StageConfig).model_copy(
            update={"list_count": 1})
        model = load_config(REPO / "configs/models/claude.yaml", ModelConfig)
        client = CountingClient()
        with TemporaryDirectory() as root, self.assertRaises(FreezeError):
            run_stage(main, episodes, client, model=model,
                      settings=GenerationSettings(max_tokens=64), prompt_version="0" * 64,
                      provenance={}, run_root=root,
                      freeze_gate=FreezeGate(Path(root) / "decisions", HASHES))
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
