import unittest
from pathlib import Path

import yaml

from oversight_study.scaffold import repository_root

WORKFLOW = repository_root(Path(__file__)) / ".github" / "workflows" / "ci.yml"


class CiWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.job = self.workflow["jobs"]["offline-checks"]
        self.commands = [step["run"] for step in self.job["steps"] if "run" in step]

    def test_offline_steps(self) -> None:
        self.assertEqual(self.commands, [
            "uv sync --frozen --all-extras",
            "uv run ruff check",
            "uv run oversight-study status",
            "uv run oversight-study smoke",
            "uv run python -m unittest discover -s tests",
        ])

    def test_no_paid_runs_and_no_secrets(self) -> None:
        self.assertEqual(self.job["env"]["NO_PAID_RUNS"], "1")
        text = WORKFLOW.read_text()
        self.assertNotIn("secrets.", text)
        self.assertNotIn("ANTHROPIC_API_KEY", text)
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})


if __name__ == "__main__":
    unittest.main()
