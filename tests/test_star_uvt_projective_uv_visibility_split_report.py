from __future__ import annotations

import json
import sys
from pathlib import Path

from research_experiments.star_uvt_feature_tubes.projective_uv_visibility_split_report import (
    SCHEMA_VERSION,
    build_uv_visibility_split_report,
    main,
)


def test_projective_uv_visibility_split_report_records_before_after_fallback() -> None:
    report = build_uv_visibility_split_report()

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "ok"
    assert report["case_count"] == 2
    assert report["summary"]["max_parent_fallback_fraction"] == 1.0
    assert report["summary"]["max_output_fallback_fraction"] == 0.0
    assert not report["summary"]["any_needs_oblique_halfspace"]

    cases = {case["name"]: case for case in report["cases"]}
    high_motion = cases["high_motion_video_centroid_line_sweep"]
    orbit = cases["orbit_parameterized_line_sweep"]

    assert high_motion["source"] == "extracted_video_motion_centroid"
    assert high_motion["reference_video_exists"]
    assert high_motion["extraction"]["frames_read"] == 16
    assert high_motion["extraction"]["sample_count"] == 3
    assert high_motion["extraction"]["pair_indices"] == (7, 8, 9)
    assert len(high_motion["extraction"]["root_positions_u"]) == 3
    assert all(0.5 <= root <= 7.5 for root in high_motion["extraction"]["root_positions_u"])
    assert high_motion["parent_uv_event_tile_samples"] == 3
    assert high_motion["parent_fallback_fraction"] == 1.0
    assert high_motion["output_tile_size"] == 4
    assert high_motion["fallback_fraction"] == 0.0
    assert high_motion["fallback_fraction_reduction"] == 1.0

    assert orbit["source"] == "synthetic_orbit_q_tan_half_angle"
    assert orbit["parent_uv_event_tile_samples"] == 3
    assert orbit["output_tile_size"] == 2
    assert orbit["fallback_fraction"] == 0.0


def test_projective_uv_visibility_split_report_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "split_report.json"
    monkeypatch.setattr(sys, "argv", ["projective_uv_visibility_split_report.py", "--output", str(output)])

    main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"]["max_output_fallback_fraction"] == 0.0
