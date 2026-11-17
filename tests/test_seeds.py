import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.generation import MissingTestSeedsError, build_dataset, load_seeds
from oversight_study.scaffold import repository_root

FIXTURES = Path(__file__).parent / "fixtures"
DEV = FIXTURES / "seeds-dev.yaml"
TEST = FIXTURES / "seeds-test.yaml"

SKIP_DIRS = {".git", ".venv", ".local", "__pycache__", ".ruff_cache", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".txt", ".cfg"}
PRIVATE_SEED_PATTERNS = (
    re.compile(r"test_pool_base\W{1,3}100\b"),
    re.compile(r"test_lists_base\W{1,3}1000\b"),
)


class SeedLoadingTests(unittest.TestCase):
    def test_dev_only_load_without_test_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            seeds = load_seeds(False, dev_path=DEV, test_path=Path(temp_dir) / "absent.yaml")
        self.assertEqual(seeds.dev_pool_base, 9200)
        self.assertFalse(seeds.has_test_seeds)

    def test_test_paths_raise_without_test_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            absent = Path(temp_dir) / "absent.yaml"
            with self.assertRaises(MissingTestSeedsError):
                load_seeds(True, dev_path=DEV, test_path=absent)
            seeds = load_seeds(False, dev_path=DEV, test_path=absent)
        with self.assertRaises(MissingTestSeedsError):
            build_dataset(seeds)

    def test_full_load(self) -> None:
        seeds = load_seeds(True, dev_path=DEV, test_path=TEST)
        self.assertEqual((seeds.test_pool_base, seeds.test_lists_base), (9100, 9500))

    def test_unexpected_or_missing_fields_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bad = Path(temp_dir) / "bad.yaml"
            bad.write_text("dev_pool_base: 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_seeds(False, dev_path=bad, test_path=TEST)


class PrivateSeedLeakTests(unittest.TestCase):
    def test_no_evaluation_seed_values_in_repository_files(self) -> None:
        root = repository_root(Path(__file__).parent)
        leaks = []
        for path in root.rglob("*"):
            if SKIP_DIRS.intersection(path.relative_to(root).parts):
                continue
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in PRIVATE_SEED_PATTERNS:
                if pattern.search(text):
                    leaks.append(str(path.relative_to(root)))
        self.assertEqual(leaks, [])


if __name__ == "__main__":
    unittest.main()
