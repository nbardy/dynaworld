from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_extended_quality_tether_report import (
    DEFAULT_OUT_DIR,
    assert_real_video_multiscene_extended_quality_tether_report,
    summarize,
    verify_real_video_multiscene_extended_quality_tether_report,
)


def _row(scene_id: str = "walk_seg_000") -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "frames": 8,
        "cadence_case_path": f"cases/{scene_id}_cadence.json",
        "measured_case_path": f"cases/{scene_id}_measured.json",
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
    rows = [_row(f"scene_{idx}_seg_000") for idx in range(5)]
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_extended_quality_tether",
        "base_domain": "saved five-source trainer-matrix case payloads",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It tethers the five-source "
            "measured live-cache projective-interval path to the cadence full-rebuild reference with loss curves."
        ),
        "source_trainer_matrix_report": "extended_matrix.json",
        "source_trainer_matrix_verifier_errors": [],
        "source_trainer_matrix_summary": {
            "scene_count": 5,
            "distinct_youtube_id_count": 5,
            "all_measured_loss_matches_cadence": True,
            "max_measured_support_rebins": 0,
            "max_measured_stale_refreshes": 0,
        },
        "case_dir": "cases",
        "required_gradient_flags": ["center_uv_grad_seen"],
        "rows": rows,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_extended_quality_tether_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_real_video_multiscene_extended_quality_tether_report(report) == []
    assert_real_video_multiscene_extended_quality_tether_report(report)
    assert report["summary"]["pair_count"] == 5  # type: ignore[index]


def test_extended_quality_tether_rejects_too_few_source_scenes() -> None:
    report = _valid_report()
    report["source_trainer_matrix_summary"]["scene_count"] = 4  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_extended_quality_tether_report(report)

    assert any("at least five scenes" in error for error in errors)


def test_extended_quality_tether_rejects_loss_curve_delta() -> None:
    report = _valid_report()
    report["rows"][0]["max_abs_loss_curve_delta"] = 1.0e-4  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_extended_quality_tether_report(report)

    assert any("measured loss curve must match cadence" in error for error in errors)


def test_extended_quality_tether_rejects_missing_gradient_flag() -> None:
    report = _valid_report()
    report["rows"][1]["missing_gradient_flags"] = ["raw_precision_grad_seen"]  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_extended_quality_tether_report(report)

    assert any("required gradient flags" in error for error in errors)


def test_extended_quality_tether_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["end_psnr_abs_delta"] = 0.25  # type: ignore[index]

    errors = verify_real_video_multiscene_extended_quality_tether_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_saved_real_video_multiscene_extended_quality_tether_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_video_multiscene_extended_quality_tether_report(report)
