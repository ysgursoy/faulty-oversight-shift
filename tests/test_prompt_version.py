import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.task.prompt_version import prompt_files, prompt_version
from oversight_study.task.templates import prompts_dir


class PromptVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name) / "prompts"
        shutil.copytree(prompts_dir(), self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_copy_has_same_hash_as_repo(self) -> None:
        self.assertEqual(prompt_version(self.root), prompt_version())
        self.assertEqual(len(prompt_version()), 64)

    def test_one_byte_change_changes_hash(self) -> None:
        before = prompt_version(self.root)
        path = self.root / "system" / "wording_1.txt"
        path.write_bytes(path.read_bytes() + b" ")
        self.assertNotEqual(prompt_version(self.root), before)

    def test_renaming_a_file_changes_hash(self) -> None:
        before = prompt_version(self.root)
        (self.root / "feedback" / "wording_2.yaml").rename(self.root / "feedback" / "w2.yaml")
        self.assertNotEqual(prompt_version(self.root), before)

    def test_readme_is_ignored(self) -> None:
        before = prompt_version(self.root)
        (self.root / "README.md").write_text("changed\n", encoding="utf-8")
        self.assertEqual(prompt_version(self.root), before)
        self.assertNotIn("README.md", [p.name for p in prompt_files(self.root)])

    def test_order_is_by_relative_path(self) -> None:
        names = [p.relative_to(self.root).as_posix() for p in prompt_files(self.root)]
        self.assertEqual(names, sorted(names))
        self.assertIn("system/wording_1.txt", names)


if __name__ == "__main__":
    unittest.main()
