"""Release validation: a release directory is checked end to end before analysis.

Layout:
  <release>/data/            one dataset release (see generation.manifest)
  <release>/runs/<run-id>/   sealed run directories (see runner.run)
Checks: dataset checksums and re-validation; each run sealed, its checksums match, every
record parses against the typed models, no duplicate query ids, every cell complete.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from oversight_study.analysis.load import parse_line
from oversight_study.generation.manifest import verify_dataset
from oversight_study.runner.run import RunManifest, expected_records


class ReleaseError(ValueError):
    """A release breaks its integrity or record contract."""


@dataclass
class ReleaseReport:
    dataset_version: str
    runs: dict[str, int] = field(default_factory=dict)  # run id -> record count


def _check_checksums(run_dir: Path) -> None:
    path = run_dir / "checksums.sha256"
    if not path.exists():
        raise ReleaseError(f"{run_dir.name}: run is not sealed (no checksums.sha256)")
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != expected:
            raise ReleaseError(f"{run_dir.name}: checksum mismatch for {name}")


def validate_run(run_dir: Path) -> int:
    run_dir = Path(run_dir)
    manifest = RunManifest(**json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")))
    if manifest.status != "sealed":
        raise ReleaseError(f"{run_dir.name}: status {manifest.status}, expected sealed")
    _check_checksums(run_dir)
    ids, cells = set(), Counter()
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        record = parse_line(line, f"{run_dir.name}/results.jsonl:{number}")
        if record.run_id != manifest.run_id:
            raise ReleaseError(f"{run_dir.name}:{number}: record from run {record.run_id}")
        if record.logical_query_id in ids:
            raise ReleaseError(f"{run_dir.name}: duplicate record {record.logical_query_id}")
        ids.add(record.logical_query_id)
        cells[(record.frame, record.condition, record.wording, record.episode,
               record.replicate)] += 1
    for key, count in cells.items():
        if count != expected_records(key[1]):
            raise ReleaseError(f"{run_dir.name}: cell {key} has {count} records")
    if len(cells) != manifest.cells_completed:
        raise ReleaseError(f"{run_dir.name}: {len(cells)} cells, manifest says "
                           f"{manifest.cells_completed}")
    return len(ids)


def validate_release(release: Path) -> ReleaseReport:
    release = Path(release)
    if not (release / "data" / "manifest.json").exists():
        raise ReleaseError("release has no data/manifest.json")
    try:
        dataset = verify_dataset(release / "data")
    except ValueError as error:
        raise ReleaseError(f"dataset: {error}") from error
    report = ReleaseReport(dataset_version=dataset.dataset_version)
    runs_dir = release / "runs"
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else []:
        report.runs[run_dir.name] = validate_run(run_dir)
    return report


def assemble_release(dataset_dir: Path, run_dirs: list[Path], out: Path) -> ReleaseReport:
    """Copy one dataset release and sealed runs into a release directory, then validate it."""
    import shutil

    out = Path(out)
    if out.exists():
        raise ReleaseError(f"{out} already exists; releases are never overwritten")
    shutil.copytree(dataset_dir, out / "data")
    for run_dir in run_dirs:
        shutil.copytree(run_dir, out / "runs" / Path(run_dir).name)
    return validate_release(out)
