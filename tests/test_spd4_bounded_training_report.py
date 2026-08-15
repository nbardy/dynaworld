from __future__ import annotations

import copy
import json
from pathlib import Path

from research_experiments.spd4_world_tubes.summarize_bounded_training import (
    MEDIA_NAMES,
    ROW_SPECS,
    assert_valid,
    summarize,
    verify,
)


def _write_fixture(root: Path) -> None:
    for spec in ROW_SPECS:
        row_dir = root / spec.directory
        row_dir.mkdir(parents=True)
        for name in MEDIA_NAMES:
            (row_dir / name).write_bytes(b"media")
        cost = {
            "optimizer_steps": 40,
            "target_frames": 160,
            "rasterized_frames": 160,
            "target_pixels": 1_597_440,
            "rasterized_pixels": 1_597_440,
            "parameter_count": spec.parameter_count,
            "trainable_parameter_count": spec.parameter_count,
            "parameter_bytes": spec.parameter_count * 4,
            "optimizer_state_bytes": spec.parameter_count * 8,
            "serialized_checkpoint_bytes": 10_000,
            "sampled_peak_current_allocated_bytes": 20_000_000,
            "sampled_peak_driver_allocated_bytes": (
                60_000_000 if spec.world_representation == "legacy_tube" else 45_000_000
            ),
            "elapsed_s": 5.0,
        }
        report = {
            "meta": {
                "seed": 17,
                "frame_count": 16,
                "image_size": [96, 128],
                "train_cameras": ["cam04", "cam09"],
                "heldout_cameras": ["cam06"],
                "uvt_backward_policy": {"name": "fast_exploration"},
            },
            "star_uvt": {
                "world_representation": spec.world_representation,
                "alpha_mode": spec.alpha_mode,
                "amplitude_convention": spec.amplitude_convention,
                "render_backend": "metal_tile",
                "steps": 40,
                "tube_count": spec.atom_count,
                "metrics": {
                    "heldout_eval_psnr": (
                        6.0 if spec.world_representation == "legacy_tube" else 7.0
                    ),
                    "heldout_eval_ssim": 0.03,
                    "heldout_eval_lpips": 0.85,
                    "heldout_eval_l1": 0.37,
                },
                "metal_stats": {
                    "rows": [{"stats": {"overflow_tile_count": 0}}],
                },
                "paper_protocol": {
                    "cost": cost,
                    "timing": {
                        "train_wall_s": 5.0,
                        "cold_compile_forward_s": 0.2,
                        "steady_forward_s": 2.0,
                        "backward_s": 2.0,
                        "optimizer_s": 0.1,
                    },
                },
            },
        }
        (row_dir / "comparison_report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )


def test_bounded_spd4_summary_validates_complete_parameter_matched_fixture(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    report = summarize(tmp_path)

    assert_valid(report)
    assert report["summary"]["parameter_count_delta_spd4_vs_legacy"] == -2
    assert report["summary"]["spd4_peak_heldout_psnr_gain_db"] == 1.0
    assert len({row["source_report_sha256"] for row in report["rows"]}) == 3


def test_bounded_spd4_verifier_rejects_protocol_and_overflow_drift(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    report = summarize(tmp_path)
    broken = copy.deepcopy(report)
    broken["rows"][1]["frame_count"] = 300
    broken["rows"][2]["max_overflow_tile_count"] = 1

    errors = verify(broken)

    assert any("frame_count" in error for error in errors)
    assert any("max_overflow_tile_count" in error for error in errors)
