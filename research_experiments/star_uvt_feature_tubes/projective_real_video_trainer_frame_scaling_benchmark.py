from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import star_uvt_feature_overfit_trainer as feature_overfit_trainer  # noqa: E402
from projective_interval_trainer_frame_scaling_benchmark import (  # noqa: E402
    _fmt,
    _row_from_payload,
    summarize,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
)


HIGH_MOTION_VIDEO = ROOT / "data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4"
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling"
)


def _base_config(
    *,
    frames: int,
    size: int,
    steps: int,
    policy: str,
    refresh_every: int,
    tile_capacity: int,
    tube_count: int,
    support_guard_padding: float,
    support_guard_policy: str,
    support_guard_bisect_steps: int,
    support_stale_overshoot_epsilon: float,
    support_stale_tail_alpha_epsilon: float,
    out_json: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "video_path": str(HIGH_MOTION_VIDEO),
            "start_seconds": None,
            "fps": None,
            "duration_seconds": None,
            "image_crop_mode": "center",
            "target_size": int(size),
            "max_frames": int(frames),
        },
        "train": {
            "steps": int(steps),
            "lr": 0.01,
            "device": "mps",
            "seed": 17,
            "frame_chunk_size": None,
            "require_loss_decrease": False,
            "require_gradient_flow": False,
            "require_no_tile_overflow": False,
        },
        "feature_uvt": {
            "tube_count": int(tube_count),
            "feature_dim": 3,
            "tile_t": 2,
            "tile_capacity": int(tile_capacity),
            "alpha_threshold": 0.01,
            "max_alpha": 0.99,
            "render_mode": "feature_direct_atomic",
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "support_guard_padding": float(support_guard_padding),
                "support_guard_policy": str(support_guard_policy),
                "support_guard_bisect_steps": int(support_guard_bisect_steps),
                "support_stale_overshoot_epsilon": float(support_stale_overshoot_epsilon),
                "support_stale_tail_alpha_epsilon": float(support_stale_tail_alpha_epsilon),
                "refresh_every": int(refresh_every),
                "refresh_policy": str(policy),
                "fallback_render_mode": "mixed",
            },
        },
        "colorize": {
            "hidden_dim": None,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": str(out_json),
            "contact_sheet": None,
            "contact_sheet_frames": int(frames),
            "contact_sheet_mode": "grid",
            "side_by_side_video": None,
            "side_by_side_fps": 8,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "unit",
            "wandb_run_name": f"projective-real-video-frame-scaling-{policy}-{frames}f",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }


def _apply_metal_tile_env(cfg: dict[str, Any]) -> None:
    backend_cfg = cfg["feature_uvt"]["projective_interval"]
    os.environ["STAR_UVT_TILE_X"] = str(int(backend_cfg["tile_size"]))
    os.environ["STAR_UVT_TILE_Y"] = str(int(backend_cfg["tile_size"]))
    os.environ["STAR_UVT_TILE_T"] = str(int(cfg["feature_uvt"]["tile_t"]))
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(int(cfg["feature_uvt"]["tile_capacity"]))


def run_case(
    *,
    frames: int,
    policy: str,
    size: int,
    steps: int,
    refresh_every: int,
    tile_capacity: int,
    tube_count: int,
    support_guard_padding: float,
    support_guard_policy: str,
    support_guard_bisect_steps: int,
    support_stale_overshoot_epsilon: float,
    support_stale_tail_alpha_epsilon: float,
    out_json: Path,
    verbose_trainer_output: bool,
) -> dict[str, Any]:
    cfg = _base_config(
        frames=frames,
        size=size,
        steps=steps,
        policy=policy,
        refresh_every=refresh_every,
        tile_capacity=tile_capacity,
        tube_count=tube_count,
        support_guard_padding=support_guard_padding,
        support_guard_policy=support_guard_policy,
        support_guard_bisect_steps=support_guard_bisect_steps,
        support_stale_overshoot_epsilon=support_stale_overshoot_epsilon,
        support_stale_tail_alpha_epsilon=support_stale_tail_alpha_epsilon,
        out_json=out_json,
    )
    _apply_metal_tile_env(cfg)
    started = time.perf_counter()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if verbose_trainer_output:
        payload = feature_overfit_trainer.run_training(cfg)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            payload = feature_overfit_trainer.run_training(cfg)
    row = _row_from_payload(frames=frames, policy=policy, elapsed_sec=time.perf_counter() - started, payload=payload)
    row.update(
        {
            "support_guard_padding": float(support_guard_padding),
            "support_guard_policy": str(support_guard_policy),
            "support_guard_bisect_steps": int(support_guard_bisect_steps),
            "support_stale_overshoot_epsilon": float(support_stale_overshoot_epsilon),
            "support_stale_tail_alpha_epsilon": float(support_stale_tail_alpha_epsilon),
            "projective_interval_effective_support_uv_padding": payload.get(
                "projective_interval_effective_support_uv_padding"
            ),
            "projective_interval_cache_last_support_missing_tile_pairs": payload.get(
                "projective_interval_cache_last_support_missing_tile_pairs"
            ),
            "projective_interval_cache_last_support_max_overshoot_px": payload.get(
                "projective_interval_cache_last_support_max_overshoot_px"
            ),
            "projective_interval_cache_max_support_max_overshoot_px": payload.get(
                "projective_interval_cache_max_support_max_overshoot_px"
            ),
            "projective_interval_cache_last_support_tail_alpha_bound": payload.get(
                "projective_interval_cache_last_support_tail_alpha_bound"
            ),
            "projective_interval_cache_max_support_tail_alpha_bound": payload.get(
                "projective_interval_cache_max_support_tail_alpha_bound"
            ),
        }
    )
    return row


def _rows_for_policy(report: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    rows = [row for row in report.get("rows", []) if row.get("policy") == policy]
    return sorted(rows, key=lambda row: int(row.get("frames", 0)))


def _finite_positive(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0.0


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _assert_summary_close(
    summary: dict[str, Any],
    expected: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > atol:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def verify_real_video_trainer_frame_scaling_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_trainer_frame_scaling":
        errors.append(f"unexpected benchmark name {report.get('benchmark')!r}")
    if not report.get("source_video_exists"):
        errors.append("source high-motion video must exist")

    raw_frame_counts = report.get("frame_counts")
    if not isinstance(raw_frame_counts, list) or len(raw_frame_counts) < 2:
        errors.append("frame_counts must contain at least two values")
        return errors
    frame_counts = [_finite_int(value, f"frame_counts[{idx}]", errors) for idx, value in enumerate(raw_frame_counts)]
    if frame_counts != sorted(frame_counts) or len(set(frame_counts)) != len(frame_counts):
        errors.append(f"frame_counts must be strictly increasing, got {frame_counts}")
    steps = _finite_int(report.get("steps"), "steps", errors)
    tile_capacity = _finite_int(report.get("tile_capacity"), "tile_capacity", errors)
    if steps <= 0:
        errors.append(f"steps must be positive, got {steps}")
    if tile_capacity <= 0:
        errors.append(f"tile_capacity must be positive, got {tile_capacity}")

    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 2 * len(frame_counts):
        errors.append("rows must contain one cadence and one measured row per frame count")
        return errors
    if any(not isinstance(row, dict) for row in raw_rows):
        errors.append("all rows must be objects")
        return errors
    rows: list[dict[str, Any]] = list(raw_rows)
    cadence_rows = _rows_for_policy(report, "cadence")
    measured_rows = _rows_for_policy(report, "measured")
    if len(cadence_rows) != len(frame_counts) or len(measured_rows) != len(frame_counts):
        errors.append("need exactly one cadence and measured row per frame count")
        return errors

    cadence_by_frame = {int(row.get("frames", 0)): row for row in cadence_rows}
    measured_by_frame = {int(row.get("frames", 0)): row for row in measured_rows}
    if sorted(cadence_by_frame) != frame_counts:
        errors.append("cadence rows must cover frame_counts exactly")
    if sorted(measured_by_frame) != frame_counts:
        errors.append("measured rows must cover frame_counts exactly")
    common_frames = frame_counts
    if set(cadence_by_frame) != set(frame_counts) or set(measured_by_frame) != set(frame_counts):
        return errors

    for frame_count in common_frames:
        cadence = cadence_by_frame[frame_count]
        measured = measured_by_frame[frame_count]
        for label, row in (("cadence", cadence), ("measured", measured)):
            prefix = f"{label} {frame_count}f"
            if not row.get("pass"):
                errors.append(f"{prefix} row must pass")
            if row.get("loss_decreased") is not True:
                errors.append(f"{prefix} row must decrease loss")
            if _finite_int(row.get("steps"), f"{prefix} steps", errors) != steps:
                errors.append(f"{prefix} steps must match report steps")
            if _finite_int(row.get("tile_overflow_sum"), f"{prefix} tile_overflow_sum", errors) != 0:
                errors.append(f"{prefix} tile_overflow_sum must be 0")
            max_tile_count = _finite_int(row.get("max_tile_count"), f"{prefix} max_tile_count", errors)
            if max_tile_count <= 0 or max_tile_count > tile_capacity:
                errors.append(f"{prefix} max_tile_count must be in (0, tile_capacity]")
            if _finite_int(
                row.get("projective_interval_cache_visibility_stratifications"),
                f"{prefix} visibility_stratifications",
                errors,
            ) != 0:
                errors.append(f"{prefix} visibility stratifications must be zero")
            if _finite_int(row.get("projective_interval_cache_fallback_marks"), f"{prefix} fallback_marks", errors) != 0:
                errors.append(f"{prefix} fallback marks must be zero")
            if _finite_float(row.get("start_loss"), f"{prefix} start_loss", errors) <= _finite_float(
                row.get("end_loss"),
                f"{prefix} end_loss",
                errors,
            ):
                errors.append(f"{prefix} end_loss must be lower than start_loss")
            if not _finite_positive(row.get("no_first_step_ms")):
                errors.append(f"{prefix} no_first_step_ms must be finite and positive")
            if not _finite_positive(row.get("mean_backward_ms")):
                errors.append(f"{prefix} mean_backward_ms must be finite and positive")
            if not _finite_positive(row.get("mean_render_forward_ms")):
                errors.append(f"{prefix} mean_render_forward_ms must be finite and positive")
        loss_delta = abs(float(measured.get("end_loss") or 0.0) - float(cadence.get("end_loss") or 0.0))
        if loss_delta >= 1.0e-5:
            errors.append(f"{frame_count}f measured/cadence end loss mismatch {loss_delta}")
        cadence_rebuilds = _finite_int(cadence.get("projective_interval_cache_rebuilds"), f"cadence {frame_count}f rebuilds", errors)
        measured_rebuilds = _finite_int(
            measured.get("projective_interval_cache_rebuilds"),
            f"measured {frame_count}f rebuilds",
            errors,
        )
        if measured_rebuilds >= cadence_rebuilds:
            errors.append(
                f"{frame_count}f measured rebuilds must be lower than cadence ({measured_rebuilds} >= {cadence_rebuilds})"
            )
        cadence_live = _finite_int(
            cadence.get("projective_interval_cache_live_updates"),
            f"cadence {frame_count}f live_updates",
            errors,
        )
        measured_live = _finite_int(
            measured.get("projective_interval_cache_live_updates"),
            f"measured {frame_count}f live_updates",
            errors,
        )
        measured_checks = _finite_int(
            measured.get("projective_interval_cache_staleness_checks"),
            f"measured {frame_count}f staleness_checks",
            errors,
        )
        measured_rebins = _finite_int(
            measured.get("projective_interval_cache_support_rebins"),
            f"measured {frame_count}f support_rebins",
            errors,
        )
        measured_refreshes = _finite_int(
            measured.get("projective_interval_cache_stale_refreshes"),
            f"measured {frame_count}f stale_refreshes",
            errors,
        )
        if measured_live <= 0:
            errors.append(f"{frame_count}f measured policy must perform live cache updates")
        if measured_live <= cadence_live:
            errors.append(f"{frame_count}f measured live updates must exceed cadence live updates")
        if measured_checks <= 0:
            errors.append(f"{frame_count}f measured policy must perform staleness checks")
        if measured_checks < measured_live:
            errors.append(f"{frame_count}f measured staleness checks must cover live updates")
        if measured_rebins != measured_refreshes:
            errors.append(f"{frame_count}f measured support rebins must equal stale refreshes")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be present")
        return errors
    expected_summary = summarize(rows)
    for key in (
        "cadence_cache_rebuilds",
        "measured_cache_rebuilds",
        "cadence_all_pass",
        "measured_all_pass",
        "cadence_all_no_overflow",
        "measured_all_no_overflow",
        "max_measured_vs_cadence_end_loss_abs_delta",
        "measured_vs_cadence_no_first_step_ms_ratios",
        "measured_vs_cadence_rebuild_ratios",
        "all_measured_loss_matches_cadence",
    ):
        _assert_summary_close(summary, expected_summary, key, errors)
    if summary.get("all_measured_loss_matches_cadence") is not True:
        errors.append("summary must report all_measured_loss_matches_cadence true")

    return errors


def assert_real_video_trainer_frame_scaling_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_trainer_frame_scaling_report(report)
    if errors:
        raise AssertionError("real-video trainer frame scaling report failed:\n- " + "\n- ".join(errors))


def verify_guarded_real_video_trainer_support_report(report: dict[str, Any]) -> list[str]:
    """Return errors for the stricter support-churn-free guarded contract."""

    errors = verify_real_video_trainer_frame_scaling_report(report)
    guard_padding = float(report.get("support_guard_padding") or 0.0)
    tail_epsilon = float(report.get("support_stale_tail_alpha_epsilon") or 0.0)
    if guard_padding <= 0.0:
        errors.append(f"support_guard_padding must be positive, got {guard_padding}")
    if report.get("support_guard_policy") != "slack_budgeted":
        errors.append("support_guard_policy must be slack_budgeted")
    if float(report.get("support_stale_overshoot_epsilon") or 0.0) != 0.0:
        errors.append("support_stale_overshoot_epsilon must be 0.0 for tail-certified guard reports")
    if tail_epsilon <= 0.0:
        errors.append(f"support_stale_tail_alpha_epsilon must be positive, got {tail_epsilon}")

    measured_rows = _rows_for_policy(report, "measured")
    if not measured_rows:
        errors.append("need measured rows for guarded support report")
    for row in measured_rows:
        frame_count = int(row.get("frames", 0))
        if int(row.get("projective_interval_cache_support_rebins") or 0) != 0:
            errors.append(f"{frame_count}f measured support rebins must be 0")
        if int(row.get("projective_interval_cache_stale_refreshes") or 0) != 0:
            errors.append(f"{frame_count}f measured stale refreshes must be 0")
        if int(row.get("projective_interval_cache_fallback_marks") or 0) != 0:
            errors.append(f"{frame_count}f measured fallback marks must be 0")
        tail_bound = float(row.get("projective_interval_cache_max_support_tail_alpha_bound") or 0.0)
        if tail_bound > tail_epsilon:
            errors.append(f"{frame_count}f measured support tail bound {tail_bound} exceeds {tail_epsilon}")
        overshoot = float(row.get("projective_interval_cache_max_support_max_overshoot_px") or 0.0)
        if overshoot != 0.0 and tail_bound <= 0.0:
            errors.append(f"{frame_count}f measured support overshoot {overshoot} lacks a positive tail certificate")

    return errors


def assert_guarded_real_video_trainer_support_report(report: dict[str, Any]) -> None:
    errors = verify_guarded_real_video_trainer_support_report(report)
    if errors:
        raise AssertionError("guarded real-video trainer support report failed:\n- " + "\n- ".join(errors))


def write_markdown(report: dict[str, Any], path: Path) -> None:
    columns = (
        "frames",
        "policy",
        "pass",
        "end_loss",
        "no_first_step_ms",
        "mean_render_forward_ms",
        "mean_backward_ms",
        "projective_interval_cache_rebuilds",
        "projective_interval_cache_live_updates",
        "projective_interval_cache_staleness_checks",
        "projective_interval_cache_stale_refreshes",
        "projective_interval_cache_support_rebins",
        "projective_interval_cache_max_support_max_overshoot_px",
        "projective_interval_cache_max_support_tail_alpha_bound",
        "tile_overflow_sum",
        "max_tile_count",
    )
    lines = [
        "# STAR UVT Real-Video Projective Interval Trainer Frame Scaling",
        "",
        "This benchmark runs the actual compatible projective-interval trainer route",
        "on the checked-in high-motion smoke video, comparing cadence rebuilds",
        "with measured live-cache reuse.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in report["rows"]:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = [int(part.strip()) for part in str(args.frame_counts).split(",") if part.strip()]
    if not frame_counts:
        raise ValueError("--frame-counts must include at least one integer")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not HIGH_MOTION_VIDEO.exists():
        return {"status": "skipped", "reason": f"missing high-motion video: {HIGH_MOTION_VIDEO}", "rows": []}
    if not torch.backends.mps.is_available():
        return {"status": "skipped", "reason": "MPS unavailable", "rows": []}
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        return {"status": "skipped", "reason": "projective interval Metal ops unavailable", "rows": []}

    rows: list[dict[str, Any]] = []
    for frame_count in frame_counts:
        for policy in ("cadence", "measured"):
            rows.append(
                run_case(
                    frames=frame_count,
                    policy=policy,
                    size=int(args.size),
                    steps=int(args.steps),
                    refresh_every=int(args.refresh_every),
                    tile_capacity=int(args.tile_capacity),
                    tube_count=int(args.tube_count),
                    support_guard_padding=float(args.support_guard_padding),
                    support_guard_policy=str(args.support_guard_policy),
                    support_guard_bisect_steps=int(args.support_guard_bisect_steps),
                    support_stale_overshoot_epsilon=float(args.support_stale_overshoot_epsilon),
                    support_stale_tail_alpha_epsilon=float(args.support_stale_tail_alpha_epsilon),
                    out_json=args.out_dir / "cases" / f"{policy}_{frame_count}f.json",
                    verbose_trainer_output=bool(args.verbose_trainer_output),
                )
            )
    return {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_trainer_frame_scaling",
        "source_video": str(HIGH_MOTION_VIDEO),
        "source_video_exists": bool(HIGH_MOTION_VIDEO.exists()),
        "frame_counts": frame_counts,
        "size": int(args.size),
        "steps": int(args.steps),
        "refresh_every": int(args.refresh_every),
        "tile_capacity": int(args.tile_capacity),
        "tube_count": int(args.tube_count),
        "support_guard_padding": float(args.support_guard_padding),
        "support_guard_policy": str(args.support_guard_policy),
        "support_guard_bisect_steps": int(args.support_guard_bisect_steps),
        "support_stale_overshoot_epsilon": float(args.support_stale_overshoot_epsilon),
        "support_stale_tail_alpha_epsilon": float(args.support_stale_tail_alpha_epsilon),
        "summary": summarize(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-counts", default="4,8,16")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--refresh-every", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--tube-count", type=int, default=128)
    parser.add_argument("--support-guard-padding", type=float, default=0.0)
    parser.add_argument("--support-guard-policy", default="fixed")
    parser.add_argument("--support-guard-bisect-steps", type=int, default=8)
    parser.add_argument("--support-stale-overshoot-epsilon", type=float, default=0.0)
    parser.add_argument("--support-stale-tail-alpha-epsilon", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verbose-trainer-output", action="store_true")
    parser.add_argument("--verify-guarded-support", action="store_true")
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        if bool(args.verify_guarded_support):
            assert_guarded_real_video_trainer_support_report(report)
        else:
            assert_real_video_trainer_frame_scaling_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_benchmark(args)
    if report.get("status") == "ok":
        if bool(args.verify_guarded_support):
            assert_guarded_real_video_trainer_support_report(report)
        else:
            assert_real_video_trainer_frame_scaling_report(report)
    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
