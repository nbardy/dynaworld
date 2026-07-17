#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
VARIANT_TOOLS = VARIANT_ROOT / "tools"

for path in (THIS_DIR, VARIANT_ROOT, VARIANT_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gate4_moving_ray_slab_compiler import DEFAULT_CONFIG, SyntheticRayMotion  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402
from train_eval_owner_run_tape import RESULTS_DIR, run_train_eval  # noqa: E402


I16X3_MODE = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
I16X4_MODE = "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"
MODES = (I16X3_MODE, I16X4_MODE)


def _step_ms(row: dict[str, Any], key: str, stat: str = "mean_s") -> float:
    return float(row["step_summary"][key][stat]) * 1000.0


def _ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1.0e-12:
        return 0.0 if abs(float(numerator)) <= 1.0e-12 else float("inf")
    return float(numerator) / float(denominator)


def _rows_by_frame(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("payload missing rows list")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("payload row is not an object")
        frame_count = row.get("frame_count")
        if not isinstance(frame_count, int):
            raise ValueError("payload row missing integer frame_count")
        out[frame_count] = row
    return out


def _all_rows_ok(payload: dict[str, Any]) -> bool:
    return all(row.get("status") == "ok" for row in _rows_by_frame(payload).values())


def summarize_pair(
    *,
    i16x3: dict[str, Any],
    i16x4: dict[str, Any],
    frame_counts: tuple[int, ...],
) -> dict[str, Any]:
    rows3 = _rows_by_frame(i16x3)
    rows4 = _rows_by_frame(i16x4)
    frames = sorted(set(rows3) & set(rows4))
    if tuple(frames) != tuple(frame_counts):
        raise ValueError(f"mode rows did not cover requested frame counts: {frames} vs {list(frame_counts)}")

    ratios_by_frame: dict[str, Any] = {}
    psnr_deltas: dict[str, float] = {}
    for frame_count in frame_counts:
        row3 = rows3[frame_count]
        row4 = rows4[frame_count]
        ratios_by_frame[str(frame_count)] = {
            "i16x4_over_i16x3_total_mean": _ratio(_step_ms(row4, "total"), _step_ms(row3, "total")),
            "i16x4_over_i16x3_total_median": _ratio(_step_ms(row4, "total", "median_s"), _step_ms(row3, "total", "median_s")),
            "i16x4_over_i16x3_backward_mean": _ratio(_step_ms(row4, "backward"), _step_ms(row3, "backward")),
            "i16x4_over_i16x3_backward_median": _ratio(
                _step_ms(row4, "backward", "median_s"),
                _step_ms(row3, "backward", "median_s"),
            ),
            "i16x4_over_i16x3_storage": _ratio(
                float(row4["train_selected_tape_storage_bytes"]),
                float(row3["train_selected_tape_storage_bytes"]),
            ),
            "i16x3_total_mean_ms": _step_ms(row3, "total"),
            "i16x4_total_mean_ms": _step_ms(row4, "total"),
            "i16x3_backward_mean_ms": _step_ms(row3, "backward"),
            "i16x4_backward_mean_ms": _step_ms(row4, "backward"),
        }
        psnr_deltas[str(frame_count)] = abs(float(row4["final_heldout_psnr"]) - float(row3["final_heldout_psnr"]))

    max_total_ratio = max(float(row["i16x4_over_i16x3_total_mean"]) for row in ratios_by_frame.values())
    max_backward_ratio = max(float(row["i16x4_over_i16x3_backward_mean"]) for row in ratios_by_frame.values())
    max_psnr_delta = max(psnr_deltas.values()) if psnr_deltas else float("inf")
    frame_scale = float(frame_counts[-1]) / float(max(frame_counts[0], 1))
    summary = {
        "ratios_by_frame": ratios_by_frame,
        "psnr_delta_by_frame": psnr_deltas,
        "max_i16x4_over_i16x3_total_mean_ratio": max_total_ratio,
        "max_i16x4_over_i16x3_backward_mean_ratio": max_backward_ratio,
        "max_psnr_delta": max_psnr_delta,
        "i16x3_total_scale_first_to_last": float(i16x3["total_step_scale_first_to_last"]),
        "i16x4_total_scale_first_to_last": float(i16x4["total_step_scale_first_to_last"]),
        "i16x3_backward_scale_first_to_last": float(i16x3["backward_scale_first_to_last"]),
        "i16x4_backward_scale_first_to_last": float(i16x4["backward_scale_first_to_last"]),
        "i16x3_storage_scale_first_to_last": float(i16x3["selected_tape_storage_scale_first_to_last"]),
        "i16x4_storage_scale_first_to_last": float(i16x4["selected_tape_storage_scale_first_to_last"]),
        "frame_scale_first_to_last": frame_scale,
        "i16x3_total_sublinear": float(i16x3["total_step_scale_first_to_last"]) < frame_scale,
        "i16x4_total_sublinear": float(i16x4["total_step_scale_first_to_last"]) < frame_scale,
        "i16x3_backward_sublinear": float(i16x3["backward_scale_first_to_last"]) < frame_scale,
        "i16x4_backward_sublinear": float(i16x4["backward_scale_first_to_last"]) < frame_scale,
    }
    summary["i16x4_speed_promotion_candidate"] = (
        max_total_ratio <= 1.05
        and max_backward_ratio <= 1.05
        and max_psnr_delta <= 1.0e-4
        and bool(summary["i16x4_total_sublinear"])
        and bool(summary["i16x4_backward_sublinear"])
    )
    return summary


def _run_mode(args: argparse.Namespace, *, tape_mode: str, frame_counts: tuple[int, ...]) -> dict[str, Any]:
    return run_train_eval(
        config_path=args.config,
        frame_counts=frame_counts,
        render_size=args.render_size,
        site_count=args.site_count,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        synthetic_motion=SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        optimizer_mode="manual-vjp",
        segment_tape_vjp_mode="direct_atomic_grad_only",
        tape_mode=tape_mode,
        allow_repeat_loaded_frames=bool(args.repeat_loaded_frames),
    )


def compare(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = _parse_int_list(args.frame_counts)
    if args.prewarm_sweep:
        for mode in MODES:
            _run_mode(
                argparse.Namespace(**{**vars(args), "steps": 1, "warmup_steps": 1}),
                tape_mode=mode,
                frame_counts=frame_counts,
            )
    results = {mode: _run_mode(args, tape_mode=mode, frame_counts=frame_counts) for mode in MODES}
    summary = summarize_pair(i16x3=results[I16X3_MODE], i16x4=results[I16X4_MODE], frame_counts=frame_counts)
    failures = []
    for mode, payload in results.items():
        if not _all_rows_ok(payload):
            failures.append(f"{mode}: one or more rows did not complete")
    if not all(math.isfinite(float(value)) for value in summary["psnr_delta_by_frame"].values()):
        failures.append("non-finite PSNR delta")
    return {
        "benchmark": "world_foam_delta_framegroup_i16x3_i16x4_train_eval_compare",
        "scope": (
            "paired same-process manual-VJP comparison of framegroup fused-MSE forks; "
            "not a STAR-UVT competitiveness artifact"
        ),
        "status": "ok" if not failures else "failed",
        "frame_counts": list(frame_counts),
        "render_size": int(args.render_size),
        "site_count": int(args.site_count),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "prewarm_sweep": bool(args.prewarm_sweep),
        "repeat_loaded_frames": bool(args.repeat_loaded_frames),
        "optimizer_mode": "manual-vjp",
        "mode_statuses": {mode: results[mode].get("status") for mode in MODES},
        "modes": results,
        "summary": summary,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare i16x3 and padded i16x4 framegroup WorldFoam train/eval paths.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="16,32,64,128")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--repeat-loaded-frames", action="store_true")
    parser.add_argument(
        "--prewarm-sweep",
        action="store_true",
        help="Run a full one-step train/eval sweep before measurement. This is expensive because it rebuilds tapes.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "delta_framegroup_i16x3_i16x4_train_eval_compare.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be nonnegative")
    payload = compare(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
