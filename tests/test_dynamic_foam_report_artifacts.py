from __future__ import annotations

import json

import pytest

from research_experiments.dynamic_foam.report_artifacts import (
    PROJECT_ROOT,
    load_report_json,
    load_report_jsonl,
    parse_frame_indices,
    relative_to_project,
    validate_frame_indices,
    write_report_json,
)


def test_write_report_json_creates_parent_and_sorted_newline(tmp_path) -> None:
    path = tmp_path / "nested" / "report.json"

    write_report_json(path, {"b": 2, "a": 1})

    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_load_report_json_requires_object(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(TypeError, match="must contain a JSON object"):
        load_report_json(path)


def test_load_report_jsonl_skips_blank_rows_and_requires_objects(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

    assert load_report_jsonl(path) == [{"a": 1}, {"b": 2}]

    bad_path = tmp_path / "bad_rows.jsonl"
    bad_path.write_text('{"ok": true}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(TypeError, match="bad_rows.jsonl:2 must contain a JSON object"):
        load_report_jsonl(bad_path)


def test_load_report_jsonl_missing_ok_returns_empty_rows(tmp_path) -> None:
    assert load_report_jsonl(tmp_path / "missing.jsonl", missing_ok=True) == []


def test_relative_to_project_shortens_project_paths() -> None:
    assert relative_to_project(PROJECT_ROOT / "outputs" / "report.json") == "outputs/report.json"


def test_parse_frame_indices_supports_csv_all_and_range_validation() -> None:
    assert parse_frame_indices("0, 2,5", frame_count=6) == [0, 2, 5]
    assert parse_frame_indices("all", frame_count=3, allow_all=True) == [0, 1, 2]

    with pytest.raises(ValueError, match="At least one frame index"):
        parse_frame_indices(" , ")
    with pytest.raises(IndexError, match="frame index 3 out of range for 3 frames"):
        validate_frame_indices([0, 3], frame_count=3)
    with pytest.raises(ValueError, match="frame_count is required"):
        parse_frame_indices("all", allow_all=True)
