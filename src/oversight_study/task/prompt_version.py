"""Content hash of every model-visible template, recorded as `prompt_version`."""

from __future__ import annotations

import hashlib
from pathlib import Path

from oversight_study.task.templates import prompts_dir

EXCLUDED = frozenset({"README.md"})


def prompt_files(root: Path) -> list[Path]:
    """Template files under `root`, sorted by POSIX relative path; README excluded."""
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.name not in EXCLUDED and not path.name.startswith(".")
    ]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def prompt_version(root: Path | None = None) -> str:
    """SHA-256 over each file's relative path, byte length, and bytes, in path order."""
    root = Path(root) if root is not None else prompts_dir()
    digest = hashlib.sha256()
    for path in prompt_files(root):
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(str(len(data)).encode())
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()
