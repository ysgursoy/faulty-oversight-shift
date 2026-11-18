"""Write, read, and verify a checksummed dataset release.

Layout under `out_dir`:
  pools/{evaluation,development}.json   cases with ids and r_star
  episodes/{trial,pilot,main}.json      lists (turn and probe ids) with their probe cases
  used_keys.json                        every consumed (weight, measurement), probes included
  manifest.json                         generator, order, seeds, counts, sha256 per file
Main lists are written only when they were generated. Evaluation seeds are written only
for a private release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from oversight_study.generation.dataset import (
    DEVELOPMENT,
    EVALUATION,
    GENERATION_ORDER,
    LIST_SPLIT,
    SPLITS,
    Dataset,
)
from oversight_study.generation.episodes import Episode
from oversight_study.generation.pools import (
    CASE_TYPES,
    GENERATOR_VERSION,
    POOL_QUOTAS,
    U_QUOTAS,
    u_quota_of,
)
from oversight_study.generation.seeds import DEV_SEED_FIELDS, TEST_SEED_FIELDS
from oversight_study.records.cases import Case, EpisodeList, Key, ProbeCase

LIST_USES = ("trial", "pilot", "main")


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    generator_version: str
    dataset_version: str
    generation_order: tuple[str, ...]
    seeds: dict[str, int]
    quotas: dict[str, int]
    counts: dict[str, dict[str, int]]
    list_counts: dict[str, int]
    used_keys_count: int
    checksums_sha256: dict[str, str]


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


def _episode_rows(episodes: list[Episode]) -> list[dict]:
    return [
        {
            "list": episode.to_list().model_dump(mode="json"),
            "probes": [probe.model_dump(mode="json") for probe in episode.probes],
        }
        for episode in episodes
    ]


def write_dataset(
    dataset: Dataset,
    out_dir: Path,
    *,
    dataset_version: str,
    include_test_seeds: bool = False,
) -> DatasetManifest:
    """Write the release files and manifest. Refuses to overwrite an existing release."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"dataset already written: {manifest_path}")

    files: dict[str, object] = {
        f"pools/{split}.json": [c.model_dump(mode="json") for c in dataset.cases(split)]
        for split in SPLITS
    }
    for use in LIST_USES:
        if dataset.episodes[use]:
            files[f"episodes/{use}.json"] = _episode_rows(dataset.episodes[use])
    files["used_keys.json"] = [list(key) for key in sorted(dataset.used_keys)]

    checksums: dict[str, str] = {}
    for rel, data in files.items():
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_canonical_json(data), encoding="utf-8")
        checksums[rel] = _sha256(path)

    seed_fields = DEV_SEED_FIELDS + (TEST_SEED_FIELDS if include_test_seeds else ())
    manifest = DatasetManifest(
        generator_version=GENERATOR_VERSION,
        dataset_version=dataset_version,
        generation_order=GENERATION_ORDER,
        seeds={field: getattr(dataset.seeds, field) for field in seed_fields},
        quotas={name: count for name, _, count in POOL_QUOTAS + U_QUOTAS},
        counts={split: _counts(dataset, split) for split in SPLITS},
        list_counts={use: len(dataset.episodes[use]) for use in LIST_USES},
        used_keys_count=len(dataset.used_keys),
        checksums_sha256=checksums,
    )
    manifest_path.write_text(_canonical_json(manifest.model_dump(mode="json")), encoding="utf-8")
    return manifest


def read_dataset(
    out_dir: Path,
) -> tuple[DatasetManifest, dict[str, list[Case]], dict[str, list[Episode]], set[Key]]:
    """Load a release into typed models (validating every record)."""
    out_dir = Path(out_dir)
    manifest = DatasetManifest(
        **json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    pools = {
        split: [Case(**row) for row in json.loads((out_dir / f"pools/{split}.json").read_text())]
        for split in SPLITS
    }
    by_id = {case.case_id: case for cases in pools.values() for case in cases}
    episodes: dict[str, list[Episode]] = {use: [] for use in LIST_USES}
    for use in LIST_USES:
        path = out_dir / f"episodes/{use}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            listing = EpisodeList(**row["list"])
            probes = {p["case_id"]: ProbeCase(**p) for p in row["probes"]}
            episodes[use].append(
                Episode(
                    split=listing.split,
                    use=listing.use,
                    list_index=listing.list_index,
                    seed=listing.seed,
                    turns=tuple(by_id[cid] for cid in listing.turn_case_ids),
                    probes=tuple(probes[cid] for cid in listing.probe_case_ids),
                )
            )
    used = {tuple(key) for key in json.loads((out_dir / "used_keys.json").read_text())}
    return manifest, pools, episodes, used


def verify_dataset(out_dir: Path) -> DatasetManifest:
    """Recompute checksums, then re-validate records, separation, and list composition."""
    from oversight_study.validation.episodes import check_episode  # avoid import cycle

    out_dir = Path(out_dir)
    manifest = DatasetManifest(
        **json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    for rel, expected in manifest.checksums_sha256.items():
        if _sha256(out_dir / rel) != expected:
            raise ValueError(f"checksum mismatch for {rel}")
    _, pools, episodes, used = read_dataset(out_dir)

    pool_keys = [case.key for cases in pools.values() for case in cases]
    if len(pool_keys) != len(set(pool_keys)):
        raise ValueError("duplicate key across pools")
    probe_keys = {split: set() for split in SPLITS}
    for use, lists in episodes.items():
        split_pools = {
            case_type: [c for c in pools[LIST_SPLIT[use]] if c.case_type == case_type]
            for case_type in CASE_TYPES
        }
        for index, episode in enumerate(lists):
            if episode.list_index != index or episode.split != LIST_SPLIT[use]:
                raise ValueError(f"{use} list {index} has wrong index or split")
            check_episode(episode, split_pools)
            probe_keys[episode.split].update(p.key for p in episode.probes)
    if (probe_keys[EVALUATION] | probe_keys[DEVELOPMENT]) & set(pool_keys):
        raise ValueError("probe key reused from a pool")
    if probe_keys[EVALUATION] & probe_keys[DEVELOPMENT]:
        raise ValueError("probe key shared between splits")
    expected_used = set(pool_keys) | probe_keys[EVALUATION] | probe_keys[DEVELOPMENT]
    if used != expected_used:
        raise ValueError("used_keys.json does not match pools and probes")
    if manifest.list_counts != {use: len(lists) for use, lists in episodes.items()}:
        raise ValueError("manifest list counts do not match episode files")
    return manifest
