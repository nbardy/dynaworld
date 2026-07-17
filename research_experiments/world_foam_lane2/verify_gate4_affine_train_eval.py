#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-18_gate4_affine_tape_train_eval_gradonly_render32_site12_2_4_8_16.json"
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _positive_finite(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _step_stat(row: dict[str, Any], phase: str, stat: str) -> float | None:
    summary = row.get("step_summary")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(stat)
    return float(value) if _finite(value) else None


def _row_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if _finite(value) else None


def _ratio(last: float, first: float) -> float:
    if abs(first) <= 1.0e-12:
        return 0.0 if abs(last) <= 1.0e-12 else float("inf")
    return last / first


def verify(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    path = Path(args.artifact)
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "failed", "artifact": str(path), "failures": [f"could not load artifact: {exc}"]}

    expected_frames = _parse_int_list(str(args.frame_counts))
    if payload.get("benchmark") != "world_foam_lane2_fused_slab_mixed_train_eval_mps":
        failures.append(f"unexpected benchmark {payload.get('benchmark')!r}")
    if payload.get("status") != "ok":
        failures.append(f"artifact status is {payload.get('status')!r}")
    if payload.get("gate") != "mixed_num32_den16_affine_moving_ray_site_rgba_train_eval":
        failures.append(f"unexpected gate {payload.get('gate')!r}")
    if payload.get("full_trainer_claim") is not False:
        failures.append("full_trainer_claim must be false")
    if payload.get("full_geometry_gradient_claim") is not False:
        failures.append("full_geometry_gradient_claim must be false")
    if payload.get("quality_claim") is not False:
        failures.append("quality_claim must be false")
    if payload.get("layout") != "per-track" or payload.get("candidate_order") != "slab-mid-depth":
        failures.append("train/eval artifact must use per-track slab-mid-depth Gate4 tape")
    if payload.get("vjp_mode") != args.vjp_mode:
        failures.append(f"vjp_mode {payload.get('vjp_mode')!r} did not match {args.vjp_mode!r}")
    alpha_aux_weight = payload.get("alpha_aux_weight")
    depth_aux_weight = payload.get("depth_aux_weight")
    if args.require_alpha_depth_aux_loss:
        if payload.get("loss_scope") != "rgb_mse_plus_optional_alpha_depth_aux":
            failures.append("loss_scope must be rgb_mse_plus_optional_alpha_depth_aux when alpha/depth aux is required")
        if not _positive_finite(alpha_aux_weight):
            failures.append("alpha_aux_weight must be positive when alpha/depth aux is required")
        if not _positive_finite(depth_aux_weight):
            failures.append("depth_aux_weight must be positive when alpha/depth aux is required")
    if tuple(payload.get("frame_counts", ())) != expected_frames:
        failures.append(f"frame_counts {payload.get('frame_counts')} did not match {list(expected_frames)}")
    if payload.get("render_size") != args.render_size:
        failures.append(f"render_size {payload.get('render_size')} did not match {args.render_size}")

    rows = payload.get("rows")
    rows_by_frame: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        failures.append("rows must be a list")
    else:
        for row in rows:
            if not isinstance(row, dict):
                failures.append("row is not an object")
                continue
            frame = row.get("frame_count")
            if not isinstance(frame, int):
                failures.append("row missing integer frame_count")
                continue
            rows_by_frame[frame] = row
    found_frames = tuple(sorted(rows_by_frame))
    if found_frames != expected_frames:
        failures.append(f"row frames {found_frames} did not match required {expected_frames}")

    time_slabs = payload.get("time_slabs")
    if not isinstance(time_slabs, int) or time_slabs <= 0:
        failures.append("time_slabs must be a positive integer")
        time_slabs = 1

    total_means: list[float] = []
    backward_means: list[float] = []
    total_medians: list[float] = []
    backward_medians: list[float] = []
    train_tape_storage: list[float] = []
    heldout_tape_storage: list[float] = []
    train_explicit_storage: list[float] = []
    heldout_explicit_storage: list[float] = []
    train_psnr_by_frame: dict[str, float] = {}
    heldout_psnr_by_frame: dict[str, float] = {}
    total_mean_ms_by_frame: dict[str, float] = {}
    backward_mean_ms_by_frame: dict[str, float] = {}
    total_median_ms_by_frame: dict[str, float] = {}
    backward_median_ms_by_frame: dict[str, float] = {}
    train_boundary_ratio_by_frame: dict[str, float] = {}
    heldout_boundary_ratio_by_frame: dict[str, float] = {}

    for frame in expected_frames:
        row = rows_by_frame.get(frame)
        if row is None:
            continue
        if row.get("status") != "ok":
            failures.append(f"frame {frame}: status is {row.get('status')!r}")
        if row.get("vjp_mode") != payload.get("vjp_mode"):
            failures.append(
                f"frame {frame}: row vjp_mode {row.get('vjp_mode')!r} "
                f"did not match top-level {payload.get('vjp_mode')!r}"
            )
        if args.require_alpha_depth_aux_loss:
            loss_terms = row.get("loss_terms")
            if not isinstance(loss_terms, dict) or loss_terms.get("alpha_depth_aux_active") is not True:
                failures.append(f"frame {frame}: alpha/depth aux loss_terms must be active")
            else:
                if not _positive_finite(loss_terms.get("alpha_aux_weight")):
                    failures.append(f"frame {frame}: loss_terms.alpha_aux_weight must be positive")
                if not _positive_finite(loss_terms.get("depth_aux_weight")):
                    failures.append(f"frame {frame}: loss_terms.depth_aux_weight must be positive")
            if not _positive_finite(row.get("first_alpha_output_grad_abs_sum")):
                failures.append(f"frame {frame}: first_alpha_output_grad_abs_sum must be positive")
            if not _positive_finite(row.get("first_depth_output_grad_abs_sum")):
                failures.append(f"frame {frame}: first_depth_output_grad_abs_sum must be positive")
        if row.get("render_size") != args.render_size:
            failures.append(f"frame {frame}: render_size {row.get('render_size')} did not match {args.render_size}")
        if row.get("site_count") != args.site_count:
            failures.append(f"frame {frame}: site_count {row.get('site_count')} did not match {args.site_count}")

        acceptance = row.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append(f"frame {frame}: missing acceptance map")
        else:
            for key, value in sorted(acceptance.items()):
                if value is not True:
                    failures.append(f"frame {frame}: acceptance {key} is not true")

        train_psnr = _row_value(row, "final_train_psnr")
        heldout_psnr = _row_value(row, "final_heldout_psnr")
        if train_psnr is None or train_psnr < args.min_train_psnr:
            failures.append(f"frame {frame}: final_train_psnr {train_psnr} below {args.min_train_psnr}")
        else:
            train_psnr_by_frame[str(frame)] = train_psnr
        if heldout_psnr is None or heldout_psnr < args.min_heldout_psnr:
            failures.append(f"frame {frame}: final_heldout_psnr {heldout_psnr} below {args.min_heldout_psnr}")
        else:
            heldout_psnr_by_frame[str(frame)] = heldout_psnr
        for key in ("first_grad_abs_sum", "parameter_update_abs_max"):
            if not _positive_finite(row.get(key)):
                failures.append(f"frame {frame}: {key} must be positive finite")

        for phase in ("render", "loss_eval", "backward", "optimizer", "total"):
            value = _step_stat(row, phase, "mean_s")
            if value is None or value <= 0.0:
                failures.append(f"frame {frame}: step_summary.{phase}.mean_s must be positive finite")
            elif phase == "total":
                total_means.append(value)
                total_mean_ms_by_frame[str(frame)] = value * 1000.0
            elif phase == "backward":
                backward_means.append(value)
                backward_mean_ms_by_frame[str(frame)] = value * 1000.0
            median_value = _step_stat(row, phase, "median_s")
            if args.require_median_timing and (median_value is None or median_value <= 0.0):
                failures.append(f"frame {frame}: step_summary.{phase}.median_s must be positive finite")
            if phase not in {"total", "backward"} or median_value is None or median_value <= 0.0:
                continue
            if phase == "total":
                total_medians.append(median_value)
                total_median_ms_by_frame[str(frame)] = median_value * 1000.0
            else:
                backward_medians.append(median_value)
                backward_median_ms_by_frame[str(frame)] = median_value * 1000.0
            if value is not None and value > 0.0:
                mean_to_median = value / median_value
                if mean_to_median > args.max_row_mean_to_median:
                    failures.append(
                        f"frame {frame}: step_summary.{phase} mean/median {mean_to_median:.3f} "
                        f"exceeds {args.max_row_mean_to_median:.3f}"
                    )
            max_value = _step_stat(row, phase, "max_s")
            if max_value is not None and max_value > 0.0:
                max_to_median = max_value / median_value
                if max_to_median > args.max_row_max_to_median:
                    failures.append(
                        f"frame {frame}: step_summary.{phase} max/median {max_to_median:.3f} "
                        f"exceeds {args.max_row_max_to_median:.3f}"
                    )

        expected_boundary_ratio = float(time_slabs) / float(frame)
        for key, out in (
            ("train_compiled_boundary_test_ratio", train_boundary_ratio_by_frame),
            ("heldout_compiled_boundary_test_ratio", heldout_boundary_ratio_by_frame),
        ):
            value = _row_value(row, key)
            if value is None or abs(value - expected_boundary_ratio) > args.boundary_ratio_tolerance:
                failures.append(f"frame {frame}: {key} {value} did not match {expected_boundary_ratio:.6g}")
            else:
                out[str(frame)] = value

        for key, out in (
            ("train_mixed_tape_storage_bytes", train_tape_storage),
            ("heldout_mixed_tape_storage_bytes", heldout_tape_storage),
            ("train_explicit_ray_storage_bytes", train_explicit_storage),
            ("heldout_explicit_ray_storage_bytes", heldout_explicit_storage),
        ):
            value = _row_value(row, key)
            if value is None or value <= 0.0:
                failures.append(f"frame {frame}: {key} must be positive finite")
            else:
                out.append(value)

    frame_scale = expected_frames[-1] / expected_frames[0]
    total_scale = _ratio(total_means[-1], total_means[0]) if len(total_means) == len(expected_frames) else float("inf")
    backward_scale = (
        _ratio(backward_means[-1], backward_means[0])
        if len(backward_means) == len(expected_frames)
        else float("inf")
    )
    total_median_scale = (
        _ratio(total_medians[-1], total_medians[0])
        if len(total_medians) == len(expected_frames)
        else float("inf")
    )
    backward_median_scale = (
        _ratio(backward_medians[-1], backward_medians[0])
        if len(backward_medians) == len(expected_frames)
        else float("inf")
    )
    train_tape_storage_scale = (
        _ratio(train_tape_storage[-1], train_tape_storage[0])
        if len(train_tape_storage) == len(expected_frames)
        else float("inf")
    )
    heldout_tape_storage_scale = (
        _ratio(heldout_tape_storage[-1], heldout_tape_storage[0])
        if len(heldout_tape_storage) == len(expected_frames)
        else float("inf")
    )
    train_explicit_storage_scale = (
        _ratio(train_explicit_storage[-1], train_explicit_storage[0])
        if len(train_explicit_storage) == len(expected_frames)
        else float("nan")
    )
    heldout_explicit_storage_scale = (
        _ratio(heldout_explicit_storage[-1], heldout_explicit_storage[0])
        if len(heldout_explicit_storage) == len(expected_frames)
        else float("nan")
    )

    if total_scale > args.max_total_scale:
        failures.append(f"total mean scale {total_scale:.3f} exceeds {args.max_total_scale:.3f}")
    if total_scale >= frame_scale:
        failures.append(f"total mean scale {total_scale:.3f} is not sublinear versus frame scale {frame_scale:.3f}")
    if backward_scale > args.max_backward_scale:
        failures.append(f"backward mean scale {backward_scale:.3f} exceeds {args.max_backward_scale:.3f}")
    if backward_scale >= frame_scale:
        failures.append(f"backward mean scale {backward_scale:.3f} is not sublinear versus frame scale {frame_scale:.3f}")
    if args.require_median_timing:
        if total_median_scale > args.max_total_median_scale:
            failures.append(
                f"total median scale {total_median_scale:.3f} exceeds {args.max_total_median_scale:.3f}"
            )
        if total_median_scale >= frame_scale:
            failures.append(
                f"total median scale {total_median_scale:.3f} is not sublinear versus frame scale {frame_scale:.3f}"
            )
        if backward_median_scale > args.max_backward_median_scale:
            failures.append(
                f"backward median scale {backward_median_scale:.3f} exceeds {args.max_backward_median_scale:.3f}"
            )
        if backward_median_scale >= frame_scale:
            failures.append(
                f"backward median scale {backward_median_scale:.3f} is not sublinear versus frame scale {frame_scale:.3f}"
            )
    if train_tape_storage_scale > args.max_tape_storage_scale:
        failures.append(f"train mixed tape storage scale {train_tape_storage_scale:.3f} exceeds {args.max_tape_storage_scale:.3f}")
    if heldout_tape_storage_scale > args.max_tape_storage_scale:
        failures.append(
            f"heldout mixed tape storage scale {heldout_tape_storage_scale:.3f} exceeds {args.max_tape_storage_scale:.3f}"
        )
    for name, scale in (
        ("train explicit ray storage", train_explicit_storage_scale),
        ("heldout explicit ray storage", heldout_explicit_storage_scale),
    ):
        if _finite(scale) and abs(scale - frame_scale) > 1.0e-6:
            failures.append(f"{name} scale {scale:.3f} did not match frame scale {frame_scale:.3f}")

    return {
        "status": "failed" if failures else "ok",
        "artifact": str(path),
        "failures": failures,
        "frame_counts": list(expected_frames),
        "gradient_scope": payload.get("gradient_scope"),
        "loss_scope": payload.get("loss_scope"),
        "vjp_mode": payload.get("vjp_mode"),
        "alpha_aux_weight": alpha_aux_weight,
        "depth_aux_weight": depth_aux_weight,
        "frame_scale_first_to_last": frame_scale,
        "total_mean_scale_first_to_last": total_scale,
        "backward_mean_scale_first_to_last": backward_scale,
        "total_median_scale_first_to_last": total_median_scale,
        "backward_median_scale_first_to_last": backward_median_scale,
        "train_mixed_tape_storage_scale_first_to_last": train_tape_storage_scale,
        "heldout_mixed_tape_storage_scale_first_to_last": heldout_tape_storage_scale,
        "train_explicit_ray_storage_scale_first_to_last": train_explicit_storage_scale,
        "heldout_explicit_ray_storage_scale_first_to_last": heldout_explicit_storage_scale,
        "train_psnr_by_frame": train_psnr_by_frame,
        "heldout_psnr_by_frame": heldout_psnr_by_frame,
        "total_mean_ms_by_frame": total_mean_ms_by_frame,
        "backward_mean_ms_by_frame": backward_mean_ms_by_frame,
        "total_median_ms_by_frame": total_median_ms_by_frame,
        "backward_median_ms_by_frame": backward_median_ms_by_frame,
        "train_boundary_ratio_by_frame": train_boundary_ratio_by_frame,
        "heldout_boundary_ratio_by_frame": heldout_boundary_ratio_by_frame,
        "scope": "Gate4 affine moving-camera frozen-geometry site-RGBA train/eval; not full trainer or STAR-UVT quality/capacity proof",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Gate4 affine moving-camera train/eval artifacts.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--vjp-mode", default="direct_atomic_grad_only")
    parser.add_argument("--min-train-psnr", type=float, default=8.0)
    parser.add_argument("--min-heldout-psnr", type=float, default=8.0)
    parser.add_argument("--max-total-scale", type=float, default=2.0)
    parser.add_argument("--max-backward-scale", type=float, default=2.5)
    parser.add_argument("--require-alpha-depth-aux-loss", action="store_true")
    parser.add_argument("--require-median-timing", action="store_true")
    parser.add_argument("--max-total-median-scale", type=float, default=2.0)
    parser.add_argument("--max-backward-median-scale", type=float, default=2.5)
    parser.add_argument("--max-row-mean-to-median", type=float, default=2.5)
    parser.add_argument("--max-row-max-to-median", type=float, default=8.0)
    parser.add_argument("--max-tape-storage-scale", type=float, default=1.10)
    parser.add_argument("--boundary-ratio-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
