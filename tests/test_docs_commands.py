import re
import unittest
from pathlib import Path

from oversight_study.cli import build_parser
from oversight_study.scaffold import repository_root

REPO = repository_root(Path(__file__))
DOCS = (REPO / "README.md", REPO / "docs" / "reproduction.md")


def documented_commands() -> set[str]:
    commands = set()
    for path in DOCS:
        for block in re.findall(r"```bash\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL):
            commands.update(re.findall(r"oversight-study ([a-z-]+)", block))
    return commands


class DocsCommandTests(unittest.TestCase):
    def test_documented_commands_exist(self) -> None:
        parser = build_parser()
        available = set(parser._subparsers._group_actions[0].choices)
        commands = documented_commands()
        self.assertTrue(commands)
        self.assertEqual(commands - available, set())


if __name__ == "__main__":
    unittest.main()
