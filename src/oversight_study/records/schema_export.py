"""Generate JSON Schema files from the typed models; the models stay authoritative."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


def schema_models() -> dict[str, type[BaseModel]]:
    from oversight_study.generation.manifest import DatasetManifest
    from oversight_study.records.cases import Case, EpisodeList, ProbeCase
    from oversight_study.records.run_records import (
        EpisodeEndRecord,
        PrecheckRecord,
        TurnRecord,
    )
    from oversight_study.runner.run import RunManifest

    return {
        "case": Case,
        "probe_case": ProbeCase,
        "episode_list": EpisodeList,
        "turn_record": TurnRecord,
        "episode_end_record": EpisodeEndRecord,
        "precheck_record": PrecheckRecord,
        "run_manifest": RunManifest,
        "dataset_manifest": DatasetManifest,
    }


def render_schema(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), sort_keys=True, indent=2) + "\n"


def export_schemas(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in schema_models().items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(render_schema(model), encoding="utf-8")
        paths.append(path)
    return paths
