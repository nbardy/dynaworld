from __future__ import annotations

import json
import sys
from pathlib import Path

from research_experiments.star_uvt_feature_tubes.report_artifacts import (
    ROOT,
    distribution_stats,
    fmt_cell,
    fmt_pair,
    load_report_json,
    load_optional_report_json,
    load_optional_report_json_or_error,
    mean_timing_without_first,
    read_report_csv,
    root_path,
    run_logged_subprocess,
    split_csv_floats,
    split_csv_ints,
    split_csv_strings,
    summary_stats,
    write_report_csv,
    write_report_json,
    write_report_text,
)


def test_root_path_resolves_relative_to_dynaworld_root() -> None:
    assert root_path("outputs/example.json") == ROOT / "outputs/example.json"


def test_report_writers_preserve_absolute_paths(tmp_path: Path) -> None:
    json_path = write_report_json(tmp_path / "nested" / "row.json", {"b": 2, "a": Path("x")})
    text_path = write_report_text(tmp_path / "nested" / "row.md", "# Row\n")

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": "x", "b": 2}
    assert text_path.read_text(encoding="utf-8") == "# Row\n"


def test_write_report_csv_preserves_first_seen_column_order(tmp_path: Path) -> None:
    csv_path = write_report_csv(tmp_path / "nested" / "summary.csv", [{"b": 2, "a": "x"}, {"c": 3, "b": 4}])

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "b,a,c",
        "2,x,",
        "4,,3",
    ]
    assert read_report_csv(csv_path) == [
        {"b": "2", "a": "x", "c": ""},
        {"b": "4", "a": "", "c": "3"},
    ]


def test_load_report_json_requires_object(tmp_path: Path) -> None:
    json_path = tmp_path / "row.json"
    json_path.write_text('{"ok": true}\n', encoding="utf-8")

    assert load_report_json(json_path) == {"ok": True}


def test_load_optional_report_json_returns_none_for_missing_path(tmp_path: Path) -> None:
    assert load_optional_report_json(tmp_path / "missing.json") is None


def test_load_optional_report_json_or_error_preserves_report_error_payload(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")

    payload = load_optional_report_json_or_error(bad_json)

    assert isinstance(payload, dict)
    assert "_load_error" in payload
    assert load_optional_report_json_or_error(tmp_path / "missing.json") is None


def test_split_csv_helpers_strip_blanks_and_cast_values() -> None:
    assert split_csv_strings(" direct_atomic, gradcache ,, ") == ("direct_atomic", "gradcache")
    assert split_csv_ints("128, 256,,512") == (128, 256, 512)
    assert split_csv_floats("0.4, 1, 2.5") == (0.4, 1.0, 2.5)


def test_mean_timing_without_first_skips_warmup_step() -> None:
    row = {"step_timings_ms": [{"step_ms": 1000.0}, {"step_ms": 10.0}, {"step_ms": 14.0}]}

    assert mean_timing_without_first(row, "step_ms") == 12.0
    assert mean_timing_without_first({"step_timings_ms": [{"step_ms": 10.0}]}, "step_ms") is None


def test_summary_stats_preserves_sample_order_and_zero_empty_shape() -> None:
    assert summary_stats([3.0, 1.0, 2.0]) == {
        "samples": [3.0, 1.0, 2.0],
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert summary_stats([]) == {"samples": [], "mean": 0.0, "min": 0.0, "max": 0.0}


def test_distribution_stats_reports_count_and_none_empty_shape() -> None:
    assert distribution_stats([1.0, 3.0, 5.0]) == {
        "count": 3,
        "mean": 3.0,
        "min": 1.0,
        "max": 5.0,
        "stdev": 2.0,
        "samples": [1.0, 3.0, 5.0],
    }
    assert distribution_stats([]) == {
        "count": 0,
        "mean": None,
        "min": None,
        "max": None,
        "stdev": None,
        "samples": [],
    }


def test_report_formatters_match_table_conventions() -> None:
    assert fmt_cell(None) == ""
    assert fmt_cell(1.23456, 2) == "1.23"
    assert fmt_cell(True) == "True"
    assert fmt_pair([1.2345, None], 1) == "1.2 -> "


def test_run_logged_subprocess_writes_command_stdout_and_status(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "ok.log"

    result = run_logged_subprocess(
        (sys.executable, "-c", "print('hello report')"),
        log_path=log_path,
        cwd=tmp_path,
        timeout_sec=10,
    )

    assert result.status == "ok"
    assert result.error == ""
    text = log_path.read_text(encoding="utf-8")
    assert "$ " in text
    assert "hello report" in text


def test_run_logged_subprocess_reports_nonzero_returncode(tmp_path: Path) -> None:
    result = run_logged_subprocess(
        (sys.executable, "-c", "raise SystemExit(3)"),
        log_path=tmp_path / "fail.log",
        cwd=tmp_path,
        timeout_sec=10,
    )

    assert result.status == "failed"
    assert result.error == "returncode=3"


def test_run_logged_subprocess_supports_env_defaults_and_overrides(tmp_path: Path) -> None:
    log_path = tmp_path / "env.log"

    run_logged_subprocess(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ['REPORT_DEFAULT']); print(os.environ['REPORT_OVERRIDE'])",
        ),
        log_path=log_path,
        cwd=tmp_path,
        timeout_sec=10,
        env_defaults={"REPORT_DEFAULT": "default"},
        env_overrides={"REPORT_OVERRIDE": "override"},
    )

    text = log_path.read_text(encoding="utf-8")
    assert "default" in text
    assert "override" in text
