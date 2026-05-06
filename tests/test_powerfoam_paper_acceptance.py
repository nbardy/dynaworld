from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_experiments/dynamic_foam"))

from verify_powerfoam_paper_acceptance import (  # noqa: E402
    post_initial_paper_quality_rows,
    post_initial_raw_quality_rows,
    raw_quality_metrics,
    raw_quality_ok,
)


def append_history(path: Path, *, step: int, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": step, "metrics": metrics}) + "\n")


def passing_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "heldout_eval_psnr": 14.0,
        "heldout_eval_ssim": 0.16,
        "heldout_eval_l1": 0.15,
        "uncalibrated_heldout_eval_psnr": 12.0,
        "uncalibrated_heldout_eval_ssim": 0.12,
        "uncalibrated_heldout_eval_l1": 0.18,
        "state_mean_center_delta": 1.0e-4,
        "state_mean_feature_delta": 2.0e-4,
    }
    metrics.update(overrides)
    return metrics


def row_from_metrics(metrics: dict[str, float], *, calibration: str) -> dict[str, float | str | bool]:
    return {
        **metrics,
        "eval_color_calibration": calibration,
        "eval_color_calibration_artifact_exists": True,
    }


def test_post_initial_paper_quality_rows_require_calibration_disclosure(tmp_path: Path) -> None:
    history_path = tmp_path / "eval_metrics_history.jsonl"
    append_history(history_path, step=0, metrics=passing_metrics())
    append_history(history_path, step=1, metrics=passing_metrics())
    config = {"render": {"eval_color_calibration": "train_fit_rgb_matrix_affine"}}

    assert (
        post_initial_paper_quality_rows(
            tmp_path,
            config,
            min_clean_heldout_psnr=13.0,
            min_clean_heldout_ssim=0.15,
        )
        == []
    )

    (tmp_path / "eval_color_calibration_step_0001.json").write_text(
        json.dumps({"mode": "train_fit_rgb_matrix_affine"}),
        encoding="utf-8",
    )
    rows = post_initial_paper_quality_rows(
        tmp_path,
        config,
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
    )

    assert len(rows) == 1
    assert rows[0]["step"] == 1
    assert rows[0]["heldout_eval_psnr"] == 14.0
    assert rows[0]["heldout_eval_ssim"] == 0.16
    assert rows[0]["max_training_state_delta"] == 2.0e-4


def test_post_initial_paper_quality_rows_require_state_motion(tmp_path: Path) -> None:
    append_history(
        tmp_path / "eval_metrics_history.jsonl",
        step=1,
        metrics=passing_metrics(state_mean_center_delta=0.0, state_mean_feature_delta=0.0),
    )

    rows = post_initial_paper_quality_rows(
        tmp_path,
        {"render": {"eval_color_calibration": "none"}},
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
    )

    assert rows == []


def test_raw_quality_rejects_calibrated_pass_raw_fail() -> None:
    row = row_from_metrics(passing_metrics(), calibration="train_fit_rgb_matrix_affine")

    assert raw_quality_metrics(row)["raw_quality_source"] == "uncalibrated_heldout_eval"
    assert not raw_quality_ok(
        row,
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
    )


def test_raw_quality_accepts_raw_pass_with_calibration() -> None:
    row = row_from_metrics(
        passing_metrics(
            uncalibrated_heldout_eval_psnr=13.1,
            uncalibrated_heldout_eval_ssim=0.151,
        ),
        calibration="train_fit_rgb_matrix_affine",
    )

    assert raw_quality_ok(
        row,
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
    )


def test_raw_quality_accepts_uncalibrated_mode_from_heldout_metrics() -> None:
    row = row_from_metrics(
        {
            "heldout_eval_psnr": 13.2,
            "heldout_eval_ssim": 0.151,
            "heldout_eval_l1": 0.16,
        },
        calibration="none",
    )

    raw_metrics = raw_quality_metrics(row)
    assert raw_metrics["raw_quality_source"] == "heldout_eval"
    assert raw_metrics["raw_heldout_eval_psnr"] == 13.2
    assert raw_quality_ok(
        row,
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
    )


def test_post_initial_raw_quality_rows_ignore_calibrated_metrics(tmp_path: Path) -> None:
    append_history(tmp_path / "eval_metrics_history.jsonl", step=1, metrics=passing_metrics())
    (tmp_path / "eval_color_calibration_step_0001.json").write_text(
        json.dumps({"mode": "train_fit_rgb_matrix_affine"}),
        encoding="utf-8",
    )

    rows = post_initial_raw_quality_rows(
        tmp_path,
        {"render": {"eval_color_calibration": "train_fit_rgb_matrix_affine"}},
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
    )

    assert rows == []


def test_raw_step0_only_reports_absent_trainability(tmp_path: Path) -> None:
    append_history(
        tmp_path / "eval_metrics_history.jsonl",
        step=0,
        metrics=passing_metrics(
            uncalibrated_heldout_eval_psnr=13.1,
            uncalibrated_heldout_eval_ssim=0.151,
            state_mean_center_delta=0.0,
            state_mean_feature_delta=0.0,
        ),
    )
    config = {"render": {"eval_color_calibration": "train_fit_rgb_matrix_affine"}}
    (tmp_path / "eval_color_calibration_step_0000.json").write_text(
        json.dumps({"mode": "train_fit_rgb_matrix_affine"}),
        encoding="utf-8",
    )

    assert (
        post_initial_raw_quality_rows(
            tmp_path,
            config,
            min_clean_heldout_psnr=13.0,
            min_clean_heldout_ssim=0.15,
        )
        == []
    )
    rows = post_initial_raw_quality_rows(
        tmp_path,
        config,
        min_clean_heldout_psnr=13.0,
        min_clean_heldout_ssim=0.15,
        allow_raw_step0_acceptance=True,
    )

    assert len(rows) == 1
    assert rows[0]["step"] == 0
    assert rows[0]["trainability_evidence"] == "absent_step0_only"
