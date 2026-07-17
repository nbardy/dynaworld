from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_trained_high_motion_trace_scaling_benchmark import (
    DEFAULT_OUT_DIR,
    ROOT,
    _apply_metal_tile_env,
    _base_config,
    _make_render_config,
    _summarize,
    assert_trained_high_motion_trace_scaling_report,
    verify_trained_high_motion_trace_scaling_report,
)
from star_uvt_feature_config import resolve_config
from star_uvt_render_configs import feature_tube_render_config_from_cfg


def _row(
    *,
    label: str,
    frames: int,
    trace_count: int,
    interval_trace_entries: int,
    dense_per_frame_tile_pairs: int,
    forward_ms: float | None = None,
    backward_ms: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "frames": frames,
        "trace_count": trace_count,
        "cell_count": max(1, frames * 2),
        "interval_trace_entries": interval_trace_entries,
        "dense_trace_samples": max(interval_trace_entries, 1),
        "interval_to_dense_trace_sample_ratio": 1.0,
        "dense_per_frame_tile_pairs": dense_per_frame_tile_pairs,
        "interval_to_dense_tile_pair_ratio": interval_trace_entries / dense_per_frame_tile_pairs,
        "fallback_cells": 0,
        "fallback_fraction": 0.0,
        "fallback_reasons": [],
        "velocity_nonzero_count": 64,
        "velocity_mean_px_per_frame": 0.33,
        "velocity_max_px_per_frame": 0.86,
        "opacity_min": 0.1,
        "opacity_max": 0.9,
    }
    if forward_ms is not None:
        row["forward_ms"] = forward_ms
    if backward_ms is not None:
        row["backward_ms"] = backward_ms
    if forward_ms is not None or backward_ms is not None:
        row["grad_coeff_abs_sum"] = 1.0 + 0.01 * frames
        row["grad_opacity_abs_sum"] = 2.0 + 0.01 * frames
        row["grad_color_abs_sum"] = 3.0 + 0.01 * frames
    return row


def _valid_report() -> dict[str, object]:
    train = {
        "pass": True,
        "loss_decreased": True,
        "start_loss": 0.30,
        "end_loss": 0.29,
        "tile_overflow_sum": 0,
    }
    rows = [
        _row(
            label="trained_checkpoint",
            frames=4,
            trace_count=64,
            interval_trace_entries=392,
            dense_per_frame_tile_pairs=1542,
            forward_ms=127.0,
            backward_ms=111.0,
        ),
        _row(
            label="trained_checkpoint_per_frame",
            frames=4,
            trace_count=64,
            interval_trace_entries=392,
            dense_per_frame_tile_pairs=387,
            forward_ms=147.0,
            backward_ms=126.0,
        ),
        _row(
            label="trained_checkpoint",
            frames=8,
            trace_count=64,
            interval_trace_entries=477,
            dense_per_frame_tile_pairs=3061,
            forward_ms=20.0,
            backward_ms=50.0,
        ),
        _row(
            label="trained_checkpoint_per_frame",
            frames=8,
            trace_count=320,
            interval_trace_entries=1956,
            dense_per_frame_tile_pairs=1906,
            forward_ms=142.0,
            backward_ms=209.0,
        ),
        _row(
            label="trained_checkpoint",
            frames=16,
            trace_count=64,
            interval_trace_entries=573,
            dense_per_frame_tile_pairs=6016,
            forward_ms=57.0,
            backward_ms=60.0,
        ),
        _row(
            label="trained_checkpoint_per_frame",
            frames=16,
            trace_count=640,
            interval_trace_entries=3862,
            dense_per_frame_tile_pairs=3761,
            forward_ms=355.0,
            backward_ms=368.0,
        ),
    ]
    return {
        "status": "ok",
        "benchmark": "star_uvt_projective_trained_high_motion_trace_scaling",
        "source_video_exists": True,
        "frame_counts": [4, 8, 16],
        "trained_frames": 16,
        "size": 32,
        "steps": 4,
        "tube_count": 64,
        "tile_capacity": 128,
        "train": train,
        "summary": _summarize(rows, train),
        "rows": rows,
    }


def test_trained_high_motion_trace_scaling_verifier_accepts_sublinear_interval_report() -> None:
    report = _valid_report()

    assert verify_trained_high_motion_trace_scaling_report(report) == []
    assert_trained_high_motion_trace_scaling_report(report)


def test_trained_high_motion_benchmark_sets_nondefault_metal_tile_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STAR_UVT_TILE_CAPACITY", "128")
    cfg = resolve_config(
        _base_config(
            frames=16,
            size=96,
            steps=4,
            tube_count=256,
            tile_capacity=256,
            out_json=tmp_path / "train.json",
            checkpoint=tmp_path / "checkpoint.pt",
        )
    )
    feature_config = feature_tube_render_config_from_cfg(cfg)
    render_cfg = _make_render_config(cfg=cfg, feature_config=feature_config, frame_count=16)

    _apply_metal_tile_env(render_cfg)

    assert os.environ["STAR_UVT_TILE_X"] == "8"
    assert os.environ["STAR_UVT_TILE_Y"] == "8"
    assert os.environ["STAR_UVT_TILE_T"] == "2"
    assert os.environ["STAR_UVT_TILE_CAPACITY"] == "256"


def test_trained_high_motion_trace_scaling_verifier_rejects_bad_training_status() -> None:
    report = copy.deepcopy(_valid_report())
    report["train"]["loss_decreased"] = False

    errors = verify_trained_high_motion_trace_scaling_report(report)

    assert any("loss must decrease" in error for error in errors)


def test_trained_high_motion_trace_scaling_verifier_rejects_linear_interval_growth() -> None:
    report = copy.deepcopy(_valid_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[2]["interval_trace_entries"] = 7000
    rows[2]["interval_to_dense_tile_pair_ratio"] = 7000 / rows[2]["dense_per_frame_tile_pairs"]

    errors = verify_trained_high_motion_trace_scaling_report(report)

    assert any("trained interval entries must beat per-frame replay entries" in error for error in errors)


def test_trained_high_motion_trace_scaling_verifier_rejects_slow_shared_timing() -> None:
    report = copy.deepcopy(_valid_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[4]["forward_ms"] = 500.0

    errors = verify_trained_high_motion_trace_scaling_report(report)

    assert any("final trained forward_ms must beat per-frame replay timing" in error for error in errors)


@pytest.mark.parametrize(
    "summary_json",
    [
        DEFAULT_OUT_DIR / "summary.json",
        ROOT
        / "outputs"
        / "benchmarks"
        / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t"
        / "summary.json",
        ROOT
        / "outputs"
        / "benchmarks"
        / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256"
        / "summary.json",
    ],
)
def test_saved_trained_high_motion_trace_scaling_artifacts_satisfy_contract(summary_json: Path) -> None:
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_trained_high_motion_trace_scaling_report(report)
