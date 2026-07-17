from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "src" / "train"


def ensure_sys_path(*paths: Path) -> None:
    for path in reversed(paths):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


ensure_sys_path(TRAIN_ROOT)

from train_artifacts import write_csv, write_json, write_text  # noqa: E402


def root_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def chdir_root() -> None:
    os.chdir(ROOT)


def load_research_json(path: str | Path) -> Any:
    with root_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_research_json(path: str | Path) -> Any | None:
    resolved = root_path(path)
    return load_research_json(resolved) if resolved.exists() else None


def load_research_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    with root_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def read_research_csv(path: str | Path) -> list[dict[str, str]]:
    with root_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_research_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> Path:
    return write_csv(root_path(path), rows, fieldnames=fieldnames)


def write_research_json(path: str | Path, payload: Any, *, sort_keys: bool = True) -> Path:
    return write_json(root_path(path), payload, sort_keys=sort_keys)


def write_research_text(path: str | Path, text: str) -> Path:
    return write_text(root_path(path), text)


__all__ = [
    "ROOT",
    "TRAIN_ROOT",
    "chdir_root",
    "ensure_sys_path",
    "load_optional_research_json",
    "load_research_json",
    "load_research_jsonl",
    "read_research_csv",
    "root_path",
    "write_research_csv",
    "write_research_json",
    "write_research_text",
]
