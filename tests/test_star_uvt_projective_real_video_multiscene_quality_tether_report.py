from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_quality_tether_report import (
    DEFAULT_OUT_DIR,
    assert_real_video_multiscene_quality_tether_report,
    summarize,
    verify_real_video_multiscene_quality_tether_report,
)


def _row(scene_id: str = "walk_seg_000", frames: int = 4) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "frames": frames,
        "cadence_case_path": f"cases/{scene_id}_{frames}f_cadence.json",
        "measured_case_path": f"cases/{scene_id}_{frames}f_measured.json",
        "case_files_exist": True,
        "curve_length": 4,
        "max_abs_loss_curve_delta": 0.0,
        "max_abs_rgb_loss_curve_delta": 0.0,
        "start_loss_abs_delta": 0.0,
        "end_loss_abs_delta": 0.0,
        "start_psnr_abs_delta": 0.0,
        "end_psnr_abs_delta": 0.0,
        "measured_loss_decrease": 0.01,
        "cadence_loss_decrease": 0.01,
        "measured_psnr_gain": 0.05,
        "cadence_psnr_gain": 0.05,
        "measured_end_psnr": 8.0,
        "cadence_end_psnr": 8.0,
        "measured_end_loss": 0.2,
        "cadence_end_loss": 0.2,
        "measured_pass": True,
        "cadence_pass": True,
        "missing_gradient_flags": [],
        "row_errors": [],
    }


def _valid_report() -> dict[str, object]:
    rows = [
        _row("walk_seg_000", 4),
        _row("walk_seg_000", 8),
        _row("walk_seg_000", 16),
        _row("bike_seg_000", 4),
        _row("bike_seg_000", 8),
        _row("bike_seg_000", 16),
        _row("forest_seg_000", 4),
        _row("forest_seg_000", 8),
        _row("forest_seg_000", 16),
    ]
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_quality_tether",
        "base_domain": "saved source-distinct frame-scaling case payloads",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It tethers the measured live-cache "
            "projective-interval path to the cadence full-rebuild reference by comparing saved loss curves."
        ),
        "source_frame_scaling_report": "frame_scaling.json",
        "source_frame_scaling_verifier_errors": [],
        "source_frame_scaling_summary": {
            "scene_count": 3,
            "frame_count_count": 3,
            "all_measured_loss_matches_cadence": True,
        },
        "case_dir": "cases",
        "required_gradient_flags": ["center_uv_grad_seen"],
        "rows": rows,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_quality_tether_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_real_video_multiscene_quality_tether_report(report) == []
    assert_real_video_multiscene_quality_tether_report(report)
    assert report["summary"]["pair_count"] == 9  # type: ignore[index]


def test_quality_tether_rejects_loss_curve_delta() -> None:
    report = _valid_report()
    report["rows"][0]["max_abs_loss_curve_delta"] = 1.0e-4  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_quality_tether_report(report)

    assert any("measured loss curve must match cadence" in error for error in errors)


def test_quality_tether_rejects_missing_gradient_flag() -> None:
    report = _valid_report()
    report["rows"][1]["missing_gradient_flags"] = ["raw_precision_grad_seen"]  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_quality_tether_report(report)

    assert any("required gradient flags" in error for error in errors)


def test_quality_tether_rejects_psnr_regression() -> None:
    report = _valid_report()
    report["rows"][2]["measured_psnr_gain"] = 0.0  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_quality_tether_report(report)

    assert any("measured PSNR must improve" in error for error in errors)


def test_quality_tether_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["end_psnr_abs_delta"] = 0.25  # type: ignore[index]

    errors = verify_real_video_multiscene_quality_tether_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_saved_real_video_multiscene_quality_tether_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_video_multiscene_quality_tether_report(report)
