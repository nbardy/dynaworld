from __future__ import annotations

import json
from pathlib import Path

from train_artifacts import append_jsonl, write_csv, write_json, write_jsonl, write_resolved_config, write_text


def test_write_resolved_config_creates_output_dir_and_serializes_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"

    path = write_resolved_config(output_dir, {"logging": {"output_dir": output_dir}, "value": 3})

    assert path == output_dir / "resolved_config.json"
    assert json.loads(path.read_text()) == {
        "logging": {"output_dir": str(output_dir)},
        "value": 3,
    }


def test_append_jsonl_serializes_payloads_in_stable_key_order(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"

    append_jsonl(path, {"b": 2, "a": Path("outputs/run")})
    append_jsonl(path, {"step": 1})

    assert path.read_text().splitlines() == [
        '{"a": "outputs/run", "b": 2}',
        '{"step": 1}',
    ]


def test_write_json_creates_parent_and_serializes_paths(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "summary.json"

    written = write_json(path, {"b": Path("outputs/run"), "a": 1})

    assert written == path
    assert path.read_text() == '{\n  "a": 1,\n  "b": "outputs/run"\n}\n'


def test_write_jsonl_overwrites_with_stable_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    path.parent.mkdir()
    path.write_text("old\n", encoding="utf-8")

    written = write_jsonl(path, [{"b": 2, "a": Path("one")}, {"step": 1}])

    assert written == path
    assert path.read_text().splitlines() == [
        '{"a": "one", "b": 2}',
        '{"step": 1}',
    ]


def test_write_jsonl_can_preserve_compact_manifest_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.jsonl"

    write_jsonl(path, [{"b": 2, "a": Path("one")}], compact=True)

    assert path.read_text().splitlines() == ['{"a":"one","b":2}']


def test_write_csv_creates_parent_and_serializes_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.csv"

    written = write_csv(
        path,
        [{"b": Path("outputs/run"), "a": 1}, {"a": 2}],
        fieldnames=["a", "b"],
    )

    assert written == path
    assert path.read_text(encoding="utf-8").splitlines() == [
        "a,b",
        "1,outputs/run",
        "2,",
    ]


def test_write_text_creates_parent_and_preserves_text(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.md"

    written = write_text(path, "# Report\n\nbody\n")

    assert written == path
    assert path.read_text(encoding="utf-8") == "# Report\n\nbody\n"
