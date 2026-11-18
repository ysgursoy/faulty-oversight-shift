"""Freeze guard: refuse runs whose prompts, configs, lock, or dataset changed since the
recorded freeze. The evaluation split needs the second freeze before any model call."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from oversight_study.validation.freeze_record import FreezeKind, FreezeRecord, read_freezes

BASE_KEYS = ("prompts", "configs", "uv_lock")
SECOND_KEYS = (*BASE_KEYS, "dataset_manifest")


class FreezeError(RuntimeError):
    """A run would start without the required freeze, or after a frozen input changed."""


@dataclass(frozen=True)
class FreezeGate:
    decisions_dir: Path
    current_hashes: dict[str, str]

    def require(self, kind: FreezeKind, keys: Sequence[str] = BASE_KEYS) -> FreezeRecord:
        records = [r for r in read_freezes(self.decisions_dir) if r.kind == kind]
        if not records:
            raise FreezeError(f"no {kind} freeze recorded")
        record = records[-1]
        for key in keys:
            frozen = record.hashes.get(key)
            if frozen is None:
                raise FreezeError(f"the {kind} freeze has no {key} hash")
            if self.current_hashes.get(key) != frozen:
                raise FreezeError(f"{key} changed since the {kind} freeze; "
                                  "a new design version and a new freeze are needed")
        return record


def require_second_freeze(gate: FreezeGate | None) -> FreezeRecord:
    """Gate for every evaluation-split call: second freeze recorded, nothing changed since."""
    if gate is None:
        raise FreezeError("evaluation calls need the recorded second freeze")
    return gate.require("second", SECOND_KEYS)
