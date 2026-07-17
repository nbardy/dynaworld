from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import star_uvt_feature_overfit_trainer as feature_overfit_trainer  # noqa: E402
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_interval_trainer_frame_scaling"


class _SyntheticSequence:
    def __init__(self, frames: torch.Tensor) -> None:
        self.frames = frames


def _synthetic_target(frames: int, size: int, *, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, frames, dtype=torch.float32, device=device).view(frames, 1, 1, 1)
    y = torch.linspace(0.0, 1.0, size, dtype=torch.float32, device=device).view(1, 1, size, 1)
    x = torch.linspace(0.0, 1.0, size, dtype=torch.float32, device=device).view(1, 1, 1, size)
    red = (0.15 + 0.75 * x + 0.10 * t).expand(frames, 1, size, size)
    green = (0.10 + 0.70 * y + 0.15 * torch.sin(6.28318530718 * t)).expand(frames, 1, size, size)
    blue = (0.20 + 0.45 * (x - y).abs() + 0.25 * t).expand(frames, 1, size, size)
    return torch.cat((red, green, blue), dim=1).clamp(0.0, 1.0).contiguous()


def _base_config(
    *,
    frames: int,
    size: int,
    steps: int,
    policy: str,
    refresh_every: int,
    tile_capacity: int,
    tube_count: int,
    out_json: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "video_path": str(out_json.with_suffix(".synthetic.mp4")),
            "start_seconds": None,
            "fps": None,
            "duration_seconds": None,
            "image_crop_mode": "center",
            "target_size": size,
            "max_frames": frames,
        },
        "train": {
            "steps": steps,
            "lr": 0.01,
            "device": "mps",
            "seed": 11,
            "frame_chunk_size": None,
            "require_loss_decrease": False,
            "require_gradient_flow": False,
            "require_no_tile_overflow": False,
        },
        "feature_uvt": {
            "tube_count": tube_count,
            "feature_dim": 3,
            "tile_t": 2,
            "tile_capacity": tile_capacity,
            "alpha_threshold": 0.01,
            "max_alpha": 0.99,
            "render_mode": "feature_direct_atomic",
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "refresh_every": refresh_every,
                "refresh_policy": policy,
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
            "contact_sheet_frames": frames,
            "contact_sheet_mode": "grid",
            "side_by_side_video": None,
            "side_by_side_fps": 8,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "unit",
            "wandb_run_name": f"projective-interval-frame-scaling-{policy}-{frames}f",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }


def _mean_without_first(payload: dict[str, Any], key: str) -> float | None:
    timings = payload.get("step_timings_ms")
    if not isinstance(timings, list) or len(timings) <= 1:
        return None
    values = [float(item[key]) for item in timings[1:] if isinstance(item, dict) and key in item]
    return None if not values else sum(values) / float(len(values))


def _row_from_payload(*, frames: int, policy: str, elapsed_sec: float, payload: dict[str, Any]) -> dict[str, Any]:
    timing = payload.get("mean_timing_ms", {})
    return {
        "frames": frames,
        "policy": policy,
        "elapsed_sec": elapsed_sec,
        "pass": payload.get("pass"),
        "steps": payload.get("steps"),
        "start_loss": payload.get("start_loss"),
        "end_loss": payload.get("end_loss"),
        "loss_decreased": payload.get("loss_decreased"),
        "mean_step_ms": timing.get("step_ms") if isinstance(timing, dict) else None,
        "no_first_step_ms": _mean_without_first(payload, "step_ms"),
        "mean_render_forward_ms": timing.get("render_forward_ms") if isinstance(timing, dict) else None,
        "mean_backward_ms": timing.get("backward_ms") if isinstance(timing, dict) else None,
        "projective_interval_cache_rebuilds": payload.get("projective_interval_cache_rebuilds"),
        "projective_interval_cache_live_updates": payload.get("projective_interval_cache_live_updates"),
        "projective_interval_cache_staleness_checks": payload.get("projective_interval_cache_staleness_checks"),
        "projective_interval_cache_stale_refreshes": payload.get("projective_interval_cache_stale_refreshes"),
        "projective_interval_cache_support_rebins": payload.get("projective_interval_cache_support_rebins"),
        "projective_interval_cache_visibility_stratifications": payload.get(
            "projective_interval_cache_visibility_stratifications"
        ),
        "projective_interval_cache_fallback_marks": payload.get("projective_interval_cache_fallback_marks"),
        "projective_interval_cache_alpha_renders": payload.get("projective_interval_cache_alpha_renders"),
        "tile_overflow_sum": payload.get("tile_overflow_sum"),
        "max_tile_count": payload.get("tile_stats", {}).get("max_tile_count")
        if isinstance(payload.get("tile_stats"), dict)
        else None,
    }


def run_case(
    *,
    frames: int,
    policy: str,
    size: int,
    steps: int,
    refresh_every: int,
    tile_capacity: int,
    tube_count: int,
    out_json: Path,
    verbose_trainer_output: bool,
) -> dict[str, Any]:
    target_by_device: dict[torch.device, torch.Tensor] = {}

    def _load_sequence(_cfg: dict[str, Any], device: torch.device) -> _SyntheticSequence:
        if device not in target_by_device:
            target_by_device[device] = _synthetic_target(frames, size, device=device)
        return _SyntheticSequence(target_by_device[device])

    cfg = _base_config(
        frames=frames,
        size=size,
        steps=steps,
        policy=policy,
        refresh_every=refresh_every,
        tile_capacity=tile_capacity,
        tube_count=tube_count,
        out_json=out_json,
    )
    original_loader = feature_overfit_trainer._load_training_sequence
    feature_overfit_trainer._load_training_sequence = _load_sequence
    started = time.perf_counter()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        if verbose_trainer_output:
            payload = feature_overfit_trainer.run_training(cfg)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = feature_overfit_trainer.run_training(cfg)
    finally:
        feature_overfit_trainer._load_training_sequence = original_loader
    return _row_from_payload(frames=frames, policy=policy, elapsed_sec=time.perf_counter() - started, payload=payload)


def _growth(values: list[float | int | None]) -> float | None:
    if not values or values[0] in (None, 0) or values[-1] is None:
        return None
    return float(values[-1]) / float(values[0])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {
        policy: [row for row in rows if row["policy"] == policy]
        for policy in sorted({str(row["policy"]) for row in rows})
    }
    summary: dict[str, Any] = {}
    for policy, policy_rows in by_policy.items():
        summary[f"{policy}_cache_rebuilds"] = [row["projective_interval_cache_rebuilds"] for row in policy_rows]
        summary[f"{policy}_no_first_step_ms_growth"] = _growth([row["no_first_step_ms"] for row in policy_rows])
        summary[f"{policy}_mean_backward_ms_growth"] = _growth([row["mean_backward_ms"] for row in policy_rows])
        summary[f"{policy}_all_pass"] = all(bool(row["pass"]) for row in policy_rows)
        summary[f"{policy}_all_no_overflow"] = all(int(row.get("tile_overflow_sum") or 0) == 0 for row in policy_rows)
    paired_frames = sorted({int(row["frames"]) for row in rows})
    loss_deltas = []
    no_first_ratios = []
    rebuild_ratios = []
    for frame_count in paired_frames:
        cadence = next((row for row in rows if row["frames"] == frame_count and row["policy"] == "cadence"), None)
        measured = next((row for row in rows if row["frames"] == frame_count and row["policy"] == "measured"), None)
        if cadence is None or measured is None:
            continue
        loss_deltas.append(abs(float(measured["end_loss"]) - float(cadence["end_loss"])))
        if cadence.get("no_first_step_ms") not in (None, 0) and measured.get("no_first_step_ms") is not None:
            no_first_ratios.append(float(measured["no_first_step_ms"]) / float(cadence["no_first_step_ms"]))
        if cadence.get("projective_interval_cache_rebuilds") not in (None, 0):
            rebuild_ratios.append(
                float(measured["projective_interval_cache_rebuilds"])
                / float(cadence["projective_interval_cache_rebuilds"])
            )
    summary["max_measured_vs_cadence_end_loss_abs_delta"] = max(loss_deltas) if loss_deltas else None
    summary["measured_vs_cadence_no_first_step_ms_ratios"] = no_first_ratios
    summary["measured_vs_cadence_rebuild_ratios"] = rebuild_ratios
    summary["all_measured_loss_matches_cadence"] = bool(loss_deltas) and max(loss_deltas) < 1.0e-5
    return summary


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


def _rows_for_policy(report: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    raw_rows = report.get("rows", [])
    if not isinstance(raw_rows, list):
        return []
    rows = [row for row in raw_rows if isinstance(row, dict) and row.get("policy") == policy]
    return sorted(rows, key=lambda row: int(row.get("frames", 0)))


def verify_interval_trainer_frame_scaling_report(report: dict[str, Any]) -> list[str]:
    """Return contract failures for a saved synthetic trainer frame-scaling report."""

    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_interval_trainer_frame_scaling":
        errors.append(f"unexpected benchmark name {report.get('benchmark')!r}")

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
    cadence_by_frame = {int(row.get("frames", 0)): row for row in cadence_rows}
    measured_by_frame = {int(row.get("frames", 0)): row for row in measured_rows}
    if sorted(cadence_by_frame) != frame_counts:
        errors.append("cadence rows must cover frame_counts exactly")
    if sorted(measured_by_frame) != frame_counts:
        errors.append("measured rows must cover frame_counts exactly")

    no_first_ratios: list[float] = []
    rebuild_ratios: list[float] = []
    for frame_count in frame_counts:
        cadence = cadence_by_frame.get(frame_count)
        measured = measured_by_frame.get(frame_count)
        if cadence is None or measured is None:
            continue
        for label, row in (("cadence", cadence), ("measured", measured)):
            prefix = f"{label} {frame_count}f"
            if row.get("pass") is not True:
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
            for key in ("no_first_step_ms", "mean_backward_ms", "mean_render_forward_ms"):
                if _finite_float(row.get(key), f"{prefix} {key}", errors) <= 0.0:
                    errors.append(f"{prefix} {key} must be positive")

        loss_delta = abs(float(measured.get("end_loss") or 0.0) - float(cadence.get("end_loss") or 0.0))
        if loss_delta >= 1.0e-5:
            errors.append(f"{frame_count}f measured/cadence end loss mismatch {loss_delta}")
        cadence_rebuilds = _finite_int(
            cadence.get("projective_interval_cache_rebuilds"),
            f"cadence {frame_count}f rebuilds",
            errors,
        )
        measured_rebuilds = _finite_int(
            measured.get("projective_interval_cache_rebuilds"),
            f"measured {frame_count}f rebuilds",
            errors,
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
        if measured_rebuilds >= cadence_rebuilds:
            errors.append(
                f"{frame_count}f measured rebuilds must be lower than cadence ({measured_rebuilds} >= {cadence_rebuilds})"
            )
        if measured_live <= cadence_live:
            errors.append(f"{frame_count}f measured live updates must exceed cadence live updates")
        if measured_checks < measured_live:
            errors.append(f"{frame_count}f measured staleness checks must cover live updates")
        if measured_rebins != measured_refreshes:
            errors.append(f"{frame_count}f measured support rebins must equal stale refreshes")
        cadence_no_first = _finite_float(cadence.get("no_first_step_ms"), f"cadence {frame_count}f no_first", errors)
        measured_no_first = _finite_float(measured.get("no_first_step_ms"), f"measured {frame_count}f no_first", errors)
        if cadence_no_first > 0.0:
            no_first_ratios.append(measured_no_first / cadence_no_first)
        if cadence_rebuilds > 0:
            rebuild_ratios.append(measured_rebuilds / float(cadence_rebuilds))

    if not no_first_ratios or max(no_first_ratios) >= 1.0:
        errors.append("measured no-first-step timings must beat cadence for all synthetic rows")
    if not rebuild_ratios or any(ratio >= 1.0 for ratio in rebuild_ratios):
        errors.append("measured rebuild ratios must all be below 1")

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


def assert_interval_trainer_frame_scaling_report(report: dict[str, Any]) -> None:
    errors = verify_interval_trainer_frame_scaling_report(report)
    if errors:
        raise AssertionError("interval trainer frame scaling report failed:\n- " + "\n- ".join(errors))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


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
        "tile_overflow_sum",
        "max_tile_count",
    )
    lines = [
        "# STAR UVT Projective Interval Trainer Frame Scaling",
        "",
        "This benchmark runs the actual compatible projective-interval trainer route",
        "on synthetic frame tensors, comparing cadence rebuilds with measured cache reuse.",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-counts", default="4,8,16")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--refresh-every", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--tube-count", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verbose-trainer-output", action="store_true")
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_interval_trainer_frame_scaling_report(report)
        print(f"verified {args.verify_report}")
        return

    frame_counts = [int(part.strip()) for part in args.frame_counts.split(",") if part.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.backends.mps.is_available():
        report = {"status": "skipped", "reason": "MPS unavailable", "rows": [], "summary": {}}
    elif not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        report = {
            "status": "skipped",
            "reason": "projective interval Metal forward/backward ops unavailable",
            "rows": [],
            "summary": {},
        }
    else:
        rows: list[dict[str, Any]] = []
        for frame_count in frame_counts:
            for policy in ("cadence", "measured"):
                rows.append(
                    run_case(
                        frames=frame_count,
                        policy=policy,
                        size=args.size,
                        steps=args.steps,
                        refresh_every=args.refresh_every,
                        tile_capacity=args.tile_capacity,
                        tube_count=args.tube_count,
                        out_json=args.out_dir / "cases" / f"{policy}_{frame_count}f.json",
                        verbose_trainer_output=bool(args.verbose_trainer_output),
                    )
                )
        report = {
            "status": "ok",
            "benchmark": "star_uvt_projective_interval_trainer_frame_scaling",
            "frame_counts": frame_counts,
            "size": int(args.size),
            "steps": int(args.steps),
            "refresh_every": int(args.refresh_every),
            "tile_capacity": int(args.tile_capacity),
            "tube_count": int(args.tube_count),
            "summary": summarize(rows),
            "rows": rows,
        }
        assert_interval_trainer_frame_scaling_report(report)
    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
