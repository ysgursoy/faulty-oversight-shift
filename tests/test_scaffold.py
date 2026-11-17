import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.scaffold import REQUIRED_PATHS, missing_scaffold_paths, repository_root


class ScaffoldTests(unittest.TestCase):
    def test_repository_root_is_discoverable(self) -> None:
        root = repository_root(Path(__file__).parent)
        self.assertEqual(root.name, "faulty-oversight-shift")

    def test_current_scaffold_is_complete(self) -> None:
        root = repository_root(Path(__file__).parent)
        self.assertEqual(missing_scaffold_paths(root), [])

    def test_missing_paths_are_reported(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = missing_scaffold_paths(root)
        self.assertEqual(set(missing), set(REQUIRED_PATHS))


if __name__ == "__main__":
    unittest.main()
