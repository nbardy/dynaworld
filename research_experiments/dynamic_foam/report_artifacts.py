from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .experiment_paths import (
        DYNAMIC_FOAM_ROOT,
        POWERFOAM_METAL_ROOT,
        PROJECT_ROOT,
        TRAIN_SRC,
        ensure_sys_path,
        ensure_train_path,
        relative_to_project,
    )
except ImportError:  # pragma: no cover - direct script execution
    from experiment_paths import (  # type: ignore
        DYNAMIC_FOAM_ROOT,
        POWERFOAM_METAL_ROOT,
        PROJECT_ROOT,
        TRAIN_SRC,
        ensure_sys_path,
        ensure_train_path,
        relative_to_project,
    )


ensure_train_path()
from train_artifacts import write_json as write_artifact_json  # noqa: E402


def load_report_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def load_report_jsonl(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object.")
        rows.append(payload)
    return rows


def write_report_json(path: Path, payload: Any, *, sort_keys: bool = True) -> Path:
    return write_artifact_json(path, payload, sort_keys=sort_keys)


def validate_frame_indices(indices: list[int], *, frame_count: int | None = None) -> list[int]:
    if not indices:
        raise ValueError("At least one frame index is required.")
    if frame_count is not None:
        for frame_index in indices:
            if frame_index < 0 or frame_index >= frame_count:
                raise IndexError(f"frame index {frame_index} out of range for {frame_count} frames.")
    return indices


def parse_frame_indices(value: str, *, frame_count: int | None = None, allow_all: bool = False) -> list[int]:
    raw = str(value).strip()
    if allow_all and raw.lower() == "all":
        if frame_count is None:
            raise ValueError("frame_count is required when parsing frame index value 'all'.")
        return list(range(frame_count))
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return validate_frame_indices(indices, frame_count=frame_count)


__all__ = [
    "DYNAMIC_FOAM_ROOT",
    "POWERFOAM_METAL_ROOT",
    "PROJECT_ROOT",
    "TRAIN_SRC",
    "ensure_sys_path",
    "ensure_train_path",
    "load_report_json",
    "load_report_jsonl",
    "parse_frame_indices",
    "relative_to_project",
    "validate_frame_indices",
    "write_report_json",
]
