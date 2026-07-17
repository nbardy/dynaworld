from __future__ import annotations

from pathlib import Path

from research_experiments.report_artifacts import (
    load_research_json,
    load_research_jsonl,
    read_research_csv,
    write_research_csv,
    write_research_json,
    write_research_text,
)


def test_research_report_artifacts_roundtrip_absolute_paths(tmp_path: Path) -> None:
    json_path = write_research_json(tmp_path / "payload.json", {"b": 2, "a": 1})
    text_path = write_research_text(tmp_path / "note.md", "hello\n")
    csv_path = write_research_csv(
        tmp_path / "rows.csv",
        [{"name": "a", "value": 1}, {"name": "b", "value": 2}],
        fieldnames=("name", "value"),
    )

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")

    assert load_research_json(json_path) == {"a": 1, "b": 2}
    assert text_path.read_text(encoding="utf-8") == "hello\n"
    assert read_research_csv(csv_path) == [{"name": "a", "value": "1"}, {"name": "b", "value": "2"}]
    assert load_research_jsonl(jsonl_path) == [{"x": 1}, {"x": 2}]
