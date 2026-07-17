from __future__ import annotations

import json
import sys
from pathlib import Path

from research_experiments.star_uvt_feature_tubes.projective_high_motion_trace_geometry_report import (
    SCHEMA_VERSION,
    build_high_motion_trace_geometry_report,
    main,
)


def test_projective_high_motion_trace_geometry_report_extracts_star_uvt_traces() -> None:
    report = build_high_motion_trace_geometry_report()

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "ok"
    assert report["case_count"] == 3
    assert report["summary"]["max_fallback_fraction"] == 0.0
    assert report["summary"]["max_interval_to_dense_tile_pair_ratio"] < 1.0
    assert report["summary"]["max_interval_to_dense_trace_sample_ratio"] < 1.0
    assert report["summary"]["trained_case_count"] == 1
    assert report["summary"]["min_train_loss_ratio"] < 1.0

    cases = {case["name"]: case for case in report["cases"]}
    zero = cases["config_faithful_zero_velocity_init"]
    moving = cases["block_match_motion_init"]
    trained = cases["block_match_motion_trained_dense_3step"]

    assert zero["source_video_exists"]
    assert zero["source"] == "star_uvt_trainer_harness_video_samples"
    assert zero["trained_checkpoint"] is None
    assert zero["trace_count"] == zero["tube_count"] == 64
    assert zero["velocity_init"] == "zero"
    assert zero["velocity_nonzero_count"] == 0
    assert zero["fallback_fraction"] == 0.0
    assert zero["interval_to_dense_tile_pair_ratio"] < 0.10

    assert moving["velocity_init"] == "block_match_gated"
    assert moving["trace_count"] == moving["tube_count"] == 64
    assert moving["velocity_nonzero_count"] > 0
    assert moving["velocity_max_px_per_frame"] > 0.0
    assert moving["cell_count"] >= zero["cell_count"]
    assert moving["fallback_fraction"] == 0.0
    assert moving["interval_to_dense_tile_pair_ratio"] < 0.5

    assert trained["velocity_init"] == "block_match_gated"
    assert trained["train_steps"] == 3
    assert trained["train_lr"] == 0.03
    assert trained["train_initial_loss"] > trained["train_final_loss"]
    assert trained["train_loss_ratio"] < 1.0
    assert trained["trained_parameter_l1_delta"] > 0.0
    assert trained["trained_parameter_l1_deltas"]["depth0"] == 0.0
    assert set(trained["trained_moved_parameter_names"]) == {
        "center_uv",
        "center_t",
        "velocity_uv",
        "raw_precision",
        "raw_opacity",
        "raw_color",
    }
    assert sum(trained["trained_parameter_l1_deltas"].values()) == trained["trained_parameter_l1_delta"]
    assert trained["fallback_fraction"] == 0.0
    assert trained["interval_to_dense_tile_pair_ratio"] < 0.5
    assert trained["interval_to_dense_trace_sample_ratio"] < 1.0


def test_projective_high_motion_trace_geometry_report_cli_writes_json(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "trace_geometry_report.json"
    monkeypatch.setattr(sys, "argv", ["projective_high_motion_trace_geometry_report.py", "--output", str(output)])

    main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"]["max_fallback_fraction"] == 0.0
    assert payload["summary"]["trained_case_count"] == 1
