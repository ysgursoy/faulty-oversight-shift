"""Freeze records: what was fixed, when, by whom, at which commit, with which hashes.

Records are written to a local, untracked directory (`.local/decisions/`): a markdown
note per freeze and an entry appended to `freezes.yaml`, which the runner's freeze guard
reads. Freeze provenance becomes public later through run manifests.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from oversight_study.config import UNDECIDED, AnalysisConfig, ConfigNotReadyError, ModelConfig
from oversight_study.task.prompt_version import prompt_version

FreezeKind = Literal["first", "second"]
DECISIONS_DIR = Path(".local/decisions")
FREEZES_FILE = "freezes.yaml"


class FreezeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: FreezeKind
    utc: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    approver: str = Field(min_length=1)
    hashes: dict[str, str]
    n_lists: int | None = Field(default=None, gt=0)
    design_version: str | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    """SHA-256 over relative paths and bytes of every file under `root`, in path order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_hashes(repo: Path, dataset_dir: Path | None = None) -> dict[str, str]:
    hashes = {
        "prompts": prompt_version(repo / "prompts"),
        "configs": tree_hash(repo / "configs"),
        "uv_lock": _sha256_file(repo / "uv.lock"),
    }
    if dataset_dir is not None:
        hashes["dataset_manifest"] = _sha256_file(Path(dataset_dir) / "manifest.json")
    return hashes


def check_first_freeze_ready(model: ModelConfig, analysis: AnalysisConfig) -> None:
    """Every value the first freeze fixes must be decided."""
    undecided = []
    if model.model_id == UNDECIDED:
        undecided.append("model_id")
    undecided += [name for name in ("delta_min", "n_max") if getattr(analysis, name) is None]
    if undecided:
        raise ConfigNotReadyError(f"first freeze refused; undecided: {', '.join(undecided)}")


def read_freezes(decisions_dir: Path) -> list[FreezeRecord]:
    path = Path(decisions_dir) / FREEZES_FILE
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [FreezeRecord(**item) for item in data]


def make_freeze_record(
    kind: FreezeKind,
    *,
    approver: str,
    commit: str,
    hashes: dict[str, str],
    model: ModelConfig,
    analysis: AnalysisConfig,
    existing: list[FreezeRecord],
    n_lists: int | None = None,
    design_version: str | None = None,
    now: datetime | None = None,
) -> FreezeRecord:
    if any(record.kind == kind for record in existing):
        raise ValueError(f"a {kind} freeze is already recorded; changes need a new design version")
    if kind == "first":
        check_first_freeze_ready(model, analysis)
        if n_lists is not None or design_version is not None:
            raise ValueError("n_lists and design_version are fixed at the second freeze")
    else:
        if not any(record.kind == "first" for record in existing):
            raise ValueError("second freeze needs a recorded first freeze")
        if n_lists is None or not design_version:
            raise ValueError("second freeze needs n_lists and design_version")
        if "dataset_manifest" not in hashes:
            raise ValueError("second freeze needs the dataset manifest hash")
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return FreezeRecord(kind=kind, utc=stamp, commit=commit, approver=approver, hashes=hashes,
                        n_lists=n_lists, design_version=design_version)


def write_freeze(record: FreezeRecord, decisions_dir: Path) -> Path:
    """Append to freezes.yaml and write a dated markdown note. Returns the note path."""
    decisions_dir = Path(decisions_dir)
    decisions_dir.mkdir(parents=True, exist_ok=True)
    records = [*read_freezes(decisions_dir), record]
    (decisions_dir / FREEZES_FILE).write_text(
        yaml.safe_dump([r.model_dump() for r in records], sort_keys=True), encoding="utf-8")
    note = decisions_dir / f"{record.utc[:10]}-{record.kind}-freeze.md"
    lines = [
        f"# {record.kind.capitalize()} freeze",
        "",
        f"Date: {record.utc}",
        f"Commit: {record.commit}",
        f"Approver: {record.approver}",
        "Status: decided",
        "",
    ]
    if record.kind == "second":
        lines += [f"Lists: {record.n_lists}", f"Design version: {record.design_version}", ""]
    lines += ["## Hashes", ""] + [f"- {name}: `{value}`" for name, value in
                                  sorted(record.hashes.items())]
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


def git_commit(repo: Path) -> str:
    """HEAD commit; refuses a working tree with uncommitted tracked changes."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError("commit or stash tracked changes before freezing")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()
