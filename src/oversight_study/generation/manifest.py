"""Write and verify a checksummed dataset release (pools, used_keys, manifest)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from oversight_study.generation.dataset import GENERATION_ORDER, SPLITS, Dataset
from oversight_study.generation.pools import (
    CASE_TYPES,
    GENERATOR_VERSION,
    POOL_QUOTAS,
    U_QUOTAS,
    u_quota_of,
)
from oversight_study.generation.seeds import DEV_SEED_FIELDS, TEST_SEED_FIELDS
from oversight_study.task import r_star


def _canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(dataset: Dataset, split: str) -> dict[str, int]:
    pools = dataset.pools[split]
    counts = {case_type: len(pools[case_type]) for case_type in CASE_TYPES}
    for name, _, _ in U_QUOTAS:
        counts[name] = sum(1 for c in pools["U"] if u_quota_of(c.weight, c.measurement) == name)
    return counts


def write_dataset(dataset: Dataset, out_dir: Path, *, include_test_seeds: bool = False) -> dict:
    """Write pools, used keys, and a checksummed manifest. Refuses to overwrite a release."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"dataset already written: {manifest_path}")

    pools_dir = out_dir / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}

    for split in SPLITS:
        rows = [
            {**asdict(case), "r_star": r_star(case.weight, case.measurement)}
            for case in dataset.cases(split)
        ]
        rel = f"pools/{split}.json"
        (out_dir / rel).write_text(_canonical_json(rows), encoding="utf-8")
        checksums[rel] = _sha256(out_dir / rel)

    used_rel = "used_keys.json"
    (out_dir / used_rel).write_text(
        _canonical_json([list(key) for key in sorted(dataset.used_keys)]), encoding="utf-8"
    )
    checksums[used_rel] = _sha256(out_dir / used_rel)

    seed_fields = DEV_SEED_FIELDS + (TEST_SEED_FIELDS if include_test_seeds else ())
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generation_order": list(GENERATION_ORDER),
        "seeds": {field: getattr(dataset.seeds, field) for field in seed_fields},
        "quotas": {name: count for name, _, count in POOL_QUOTAS + U_QUOTAS},
        "counts": {split: _counts(dataset, split) for split in SPLITS},
        "used_keys_count": len(dataset.used_keys),
        "checksums_sha256": checksums,
    }
    manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")
    return manifest


def verify_dataset(out_dir: Path) -> None:
    """Recompute checksums from disk and raise if any file differs from the manifest."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["checksums_sha256"].items():
        if _sha256(out_dir / rel) != expected:
            raise ValueError(f"checksum mismatch for {rel}")
