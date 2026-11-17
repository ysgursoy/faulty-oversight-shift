"""Seed bases for pools and episode lists.

Development seeds are versioned in `configs/data/seeds-dev.yaml`. Evaluation seeds are
read from `.local/seeds-test.yaml`, which is not version-controlled, and are released
together with the reported results.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from oversight_study.scaffold import repository_root

DEV_SEEDS_PATH = Path("configs/data/seeds-dev.yaml")
TEST_SEEDS_PATH = Path(".local/seeds-test.yaml")

DEV_SEED_FIELDS = ("dev_pool_base", "trial_lists_base", "pilot_lists_base")
TEST_SEED_FIELDS = ("test_pool_base", "test_lists_base")


class MissingTestSeedsError(RuntimeError):
    """Raised when an evaluation-split step runs without evaluation seeds."""


class SeedConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dev_pool_base: int
    trial_lists_base: int
    pilot_lists_base: int
    test_pool_base: int | None = None
    test_lists_base: int | None = None

    @property
    def has_test_seeds(self) -> bool:
        return self.test_pool_base is not None and self.test_lists_base is not None

    def require_test_seeds(self) -> None:
        if not self.has_test_seeds:
            raise MissingTestSeedsError(f"evaluation seeds are required: {TEST_SEEDS_PATH}")


def _read_yaml(path: Path, fields: tuple[str, ...]) -> dict[str, int]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or set(data) != set(fields):
        raise ValueError(f"{path} must define exactly: {', '.join(fields)}")
    return data


def load_seeds(
    require_test: bool,
    dev_path: Path | None = None,
    test_path: Path | None = None,
) -> SeedConfig:
    """Load development seeds and, when present, evaluation seeds."""
    if dev_path is None or test_path is None:
        root = repository_root()
        dev_path = dev_path or root / DEV_SEEDS_PATH
        test_path = test_path or root / TEST_SEEDS_PATH

    values = _read_yaml(Path(dev_path), DEV_SEED_FIELDS)
    if Path(test_path).is_file():
        values.update(_read_yaml(Path(test_path), TEST_SEED_FIELDS))
    seeds = SeedConfig(**values)
    if require_test:
        seeds.require_test_seeds()
    return seeds
