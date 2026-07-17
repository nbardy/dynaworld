from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path, *, encoding: str = "utf-8") -> Any:
    return json.loads(Path(path).read_text(encoding=encoding))


def load_jsonl_objects(path: str | Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {jsonl_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on {jsonl_path}:{line_number}.")
            records.append(record)
    return records


__all__ = ["load_json", "load_jsonl_objects"]
