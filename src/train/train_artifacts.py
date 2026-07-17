from __future__ import annotations

from collections.abc import Iterable
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from config_utils import serialize_config_value


def write_resolved_config(output_dir: Path, cfg: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.json"
    path.write_text(json.dumps(serialize_config_value(cfg), indent=2) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_config_value(payload), indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, compact: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_kwargs: dict[str, Any] = {"sort_keys": True}
    if compact:
        dump_kwargs["separators"] = (",", ":")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(serialize_config_value(row), **dump_kwargs) + "\n")
    return path


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], *, fieldnames: Iterable[str] | None = None) -> Path:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    else:
        fieldnames = list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_config_value(row.get(key)) for key in fieldnames})
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize_config_value(payload), sort_keys=True) + "\n")


__all__ = ["append_jsonl", "write_csv", "write_json", "write_jsonl", "write_resolved_config", "write_text"]
