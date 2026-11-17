"""Load model-visible prompt templates from the repository `prompts/` directory."""

from __future__ import annotations

import string
from functools import cache
from pathlib import Path

from oversight_study.scaffold import repository_root


def prompts_dir() -> Path:
    return repository_root(Path(__file__)) / "prompts"


@cache
def load_template(relative_path: str) -> str:
    """Template text without the trailing newline that every template file ends with."""
    path = prompts_dir() / relative_path
    return path.read_text(encoding="utf-8").rstrip("\n")


def placeholders(template: str) -> set[str]:
    """Names of `str.format` fields used by a template."""
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}
