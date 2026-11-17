"""Checks for the initial repository scaffold.

These checks intentionally do not claim that the experiment pipeline exists.
"""

from pathlib import Path

REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "docs/protocol.md",
    "docs/reproduction.md",
    "docs/data-card.md",
    "sources/references.bib",
    "sources/source-notes.md",
    "configs/README.md",
    "prompts/README.md",
    "data/README.md",
)


def repository_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing pyproject.toml and README.md."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "README.md").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the repository root")


def missing_scaffold_paths(root: Path) -> list[str]:
    """List required scaffold paths that do not exist under root."""
    return [relative for relative in REQUIRED_PATHS if not (root / relative).is_file()]
