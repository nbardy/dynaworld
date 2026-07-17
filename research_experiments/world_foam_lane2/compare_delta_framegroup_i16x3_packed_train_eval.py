#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
PACKED_MODE = "endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse"
AUTO_MODE = "endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse"
AUTO_PACKED_MAX_FRAME_COUNT = 64
DEFAULT_MODES = (I16X3_MODE, PACKED_MODE)


def _modes_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    return (*DEFAULT_MODES, AUTO_MODE) if bool(args.include_auto_selector) else DEFAULT_MODES


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


def _safe_scale(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1.0e-12:
        return 0.0 if abs(float(numerator)) <= 1.0e-12 else float("inf")
    return float(numerator) / float(denominator)


def _row_scale(rows: list[dict[str, Any]], key: str) -> float:
    return _safe_scale(
        float(rows[-1]["step_summary"][key]["mean_s"]),
        float(rows[0]["step_summary"][key]["mean_s"]),
    )


def _field_scale(rows: list[dict[str, Any]], key: str) -> float:
    return _safe_scale(float(rows[-1][key]), float(rows[0][key]))


def _combine_single_frame_payloads(
    *,
    tape_mode: str,
    frame_counts: tuple[int, ...],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(payloads) != len(frame_counts):
        raise ValueError(f"expected {len(frame_counts)} payloads for {tape_mode}, got {len(payloads)}")
    combined = copy.deepcopy(payloads[0])
    rows: list[dict[str, Any]] = []
    for frame_count, payload in zip(frame_counts, payloads, strict=True):
        frame_rows = payload.get("rows")
        if not isinstance(frame_rows, list) or len(frame_rows) != 1 or not isinstance(frame_rows[0], dict):
            raise ValueError(f"{tape_mode} frame {frame_count} payload must contain exactly one row")
        row = copy.deepcopy(frame_rows[0])
        if row.get("frame_count") != frame_count:
            raise ValueError(f"{tape_mode} row frame_count {row.get('frame_count')} did not match {frame_count}")
        rows.append(row)

    frame_scale = float(frame_counts[-1]) / float(max(frame_counts[0], 1))
    total_scale = _row_scale(rows, "total")
    backward_scale = _row_scale(rows, "backward")
    render_scale = _row_scale(rows, "render")
    selected_segment_scale = _field_scale(rows, "train_selected_tape_segments")
    selected_storage_scale = _field_scale(rows, "train_selected_tape_storage_bytes")
    owner_segment_scale = _field_scale(rows, "train_owner_run_segments")
    endpoint_edit_ops_scale = _field_scale(rows, "train_endpoint_record_edit_ops")
    max_row = rows[-1]
    acceptance = {
        "all_rows_ok": all(row.get("status") == "ok" for row in rows),
        "total_step_sublinear_vs_frames": total_scale < frame_scale,
        "backward_sublinear_vs_frames": backward_scale < frame_scale,
        "render_sublinear_vs_frames": render_scale < frame_scale,
        "selected_tape_segments_below_full_at_max_frame": float(max_row["train_selected_tape_segments"])
        < float(max_row["train_full_segments"]),
        "selected_tape_storage_below_full_at_max_frame": float(max_row["train_selected_tape_storage_bytes"])
        < float(max_row["train_full_storage_bytes"]),
        "owner_run_segments_below_full_at_max_frame": float(max_row["train_owner_run_segments"])
        < float(max_row["train_full_segments"]),
    }
    combined.update(
        {
            "status": "ok" if all(payload.get("status") == "ok" for payload in payloads) and all(acceptance.values()) else "failed",
            "frame_counts": list(frame_counts),
            "rows": rows,
            "acceptance": acceptance,
            "frame_scale_first_to_last": frame_scale,
            "total_step_scale_first_to_last": total_scale,
            "backward_scale_first_to_last": backward_scale,
            "render_scale_first_to_last": render_scale,
            "selected_tape_segment_scale_first_to_last": selected_segment_scale,
            "selected_tape_storage_scale_first_to_last": selected_storage_scale,
            "owner_run_segment_scale_first_to_last": owner_segment_scale,
            "endpoint_record_edit_op_scale_first_to_last": endpoint_edit_ops_scale,
            "tape_mode": tape_mode,
        }
    )
    return combined


def summarize_pair(
    *,
    i16x3: dict[str, Any],
    packed: dict[str, Any],
    frame_counts: tuple[int, ...],
) -> dict[str, Any]:
    rows3 = _rows_by_frame(i16x3)
    rows_packed = _rows_by_frame(packed)
    frames = sorted(set(rows3) & set(rows_packed))
    if tuple(frames) != tuple(frame_counts):
        raise ValueError(f"mode rows did not cover requested frame counts: {frames} vs {list(frame_counts)}")

    ratios_by_frame: dict[str, Any] = {}
    psnr_deltas: dict[str, float] = {}
    for frame_count in frame_counts:
        row3 = rows3[frame_count]
        row_packed = rows_packed[frame_count]
        ratios_by_frame[str(frame_count)] = {
            "packed_over_i16x3_total_mean": _ratio(_step_ms(row_packed, "total"), _step_ms(row3, "total")),
            "packed_over_i16x3_total_median": _ratio(
                _step_ms(row_packed, "total", "median_s"),
                _step_ms(row3, "total", "median_s"),
            ),
            "packed_over_i16x3_backward_mean": _ratio(_step_ms(row_packed, "backward"), _step_ms(row3, "backward")),
            "packed_over_i16x3_backward_median": _ratio(
                _step_ms(row_packed, "backward", "median_s"),
                _step_ms(row3, "backward", "median_s"),
            ),
            "packed_over_i16x3_storage": _ratio(
                float(row_packed["train_selected_tape_storage_bytes"]),
                float(row3["train_selected_tape_storage_bytes"]),
            ),
            "i16x3_total_mean_ms": _step_ms(row3, "total"),
            "packed_total_mean_ms": _step_ms(row_packed, "total"),
            "i16x3_backward_mean_ms": _step_ms(row3, "backward"),
            "packed_backward_mean_ms": _step_ms(row_packed, "backward"),
        }
        psnr_deltas[str(frame_count)] = abs(
            float(row_packed["final_heldout_psnr"]) - float(row3["final_heldout_psnr"])
        )

    max_total_ratio = max(float(row["packed_over_i16x3_total_mean"]) for row in ratios_by_frame.values())
    max_backward_ratio = max(float(row["packed_over_i16x3_backward_mean"]) for row in ratios_by_frame.values())
    max_storage_ratio = max(float(row["packed_over_i16x3_storage"]) for row in ratios_by_frame.values())
    max_psnr_delta = max(psnr_deltas.values()) if psnr_deltas else float("inf")
    frame_scale = float(frame_counts[-1]) / float(max(frame_counts[0], 1))
    summary = {
        "ratios_by_frame": ratios_by_frame,
        "psnr_delta_by_frame": psnr_deltas,
        "max_packed_over_i16x3_total_mean_ratio": max_total_ratio,
        "max_packed_over_i16x3_backward_mean_ratio": max_backward_ratio,
        "max_packed_over_i16x3_storage_ratio": max_storage_ratio,
        "max_psnr_delta": max_psnr_delta,
        "i16x3_total_scale_first_to_last": float(i16x3["total_step_scale_first_to_last"]),
        "packed_total_scale_first_to_last": float(packed["total_step_scale_first_to_last"]),
        "i16x3_backward_scale_first_to_last": float(i16x3["backward_scale_first_to_last"]),
        "packed_backward_scale_first_to_last": float(packed["backward_scale_first_to_last"]),
        "i16x3_storage_scale_first_to_last": float(i16x3["selected_tape_storage_scale_first_to_last"]),
        "packed_storage_scale_first_to_last": float(packed["selected_tape_storage_scale_first_to_last"]),
        "frame_scale_first_to_last": frame_scale,
        "i16x3_total_sublinear": float(i16x3["total_step_scale_first_to_last"]) < frame_scale,
        "packed_total_sublinear": float(packed["total_step_scale_first_to_last"]) < frame_scale,
        "i16x3_backward_sublinear": float(i16x3["backward_scale_first_to_last"]) < frame_scale,
        "packed_backward_sublinear": float(packed["backward_scale_first_to_last"]) < frame_scale,
        "packed_storage_below_i16x3": max_storage_ratio < 1.0,
    }
    summary["packed_speed_promotion_candidate"] = (
        max_total_ratio <= 1.05
        and max_backward_ratio <= 1.05
        and max_psnr_delta <= 1.0e-4
        and bool(summary["packed_total_sublinear"])
        and bool(summary["packed_backward_sublinear"])
    )
    return summary


def _expected_auto_mode(frame_count: int) -> str:
    return PACKED_MODE if int(frame_count) <= AUTO_PACKED_MAX_FRAME_COUNT else I16X3_MODE


def summarize_auto_selector(
    *,
    i16x3: dict[str, Any],
    packed: dict[str, Any],
    auto: dict[str, Any],
    frame_counts: tuple[int, ...],
) -> dict[str, Any]:
    rows3 = _rows_by_frame(i16x3)
    rows_packed = _rows_by_frame(packed)
    rows_auto = _rows_by_frame(auto)
    frames = sorted(set(rows3) & set(rows_packed) & set(rows_auto))
    if tuple(frames) != tuple(frame_counts):
        raise ValueError(f"auto-selector rows did not cover requested frame counts: {frames} vs {list(frame_counts)}")

    ratios_by_frame: dict[str, Any] = {}
    resolved_modes_by_frame: dict[str, str] = {}
    expected_modes_by_frame: dict[str, str] = {}
    for frame_count in frame_counts:
        row3 = rows3[frame_count]
        row_packed = rows_packed[frame_count]
        row_auto = rows_auto[frame_count]
        best_total = min(_step_ms(row3, "total"), _step_ms(row_packed, "total"))
        best_backward = min(_step_ms(row3, "backward"), _step_ms(row_packed, "backward"))
        best_storage = min(
            float(row3["train_selected_tape_storage_bytes"]),
            float(row_packed["train_selected_tape_storage_bytes"]),
        )
        ratios_by_frame[str(frame_count)] = {
            "auto_over_i16x3_total_mean": _ratio(_step_ms(row_auto, "total"), _step_ms(row3, "total")),
            "auto_over_packed_total_mean": _ratio(_step_ms(row_auto, "total"), _step_ms(row_packed, "total")),
            "auto_over_best_component_total_mean": _ratio(_step_ms(row_auto, "total"), best_total),
            "auto_over_i16x3_backward_mean": _ratio(_step_ms(row_auto, "backward"), _step_ms(row3, "backward")),
            "auto_over_packed_backward_mean": _ratio(_step_ms(row_auto, "backward"), _step_ms(row_packed, "backward")),
            "auto_over_best_component_backward_mean": _ratio(_step_ms(row_auto, "backward"), best_backward),
            "auto_over_i16x3_storage": _ratio(
                float(row_auto["train_selected_tape_storage_bytes"]),
                float(row3["train_selected_tape_storage_bytes"]),
            ),
            "auto_over_packed_storage": _ratio(
                float(row_auto["train_selected_tape_storage_bytes"]),
                float(row_packed["train_selected_tape_storage_bytes"]),
            ),
            "auto_over_best_component_storage": _ratio(
                float(row_auto["train_selected_tape_storage_bytes"]),
                best_storage,
            ),
            "auto_total_mean_ms": _step_ms(row_auto, "total"),
            "auto_backward_mean_ms": _step_ms(row_auto, "backward"),
        }
        resolved_modes_by_frame[str(frame_count)] = str(row_auto.get("tape_mode_resolved", auto.get("tape_mode", "")))
        expected_modes_by_frame[str(frame_count)] = _expected_auto_mode(frame_count)

    max_auto_over_best_total = max(
        float(row["auto_over_best_component_total_mean"]) for row in ratios_by_frame.values()
    )
    max_auto_over_best_backward = max(
        float(row["auto_over_best_component_backward_mean"]) for row in ratios_by_frame.values()
    )
    max_auto_over_best_storage = max(
        float(row["auto_over_best_component_storage"]) for row in ratios_by_frame.values()
    )
    max_psnr_delta_vs_i16x3 = max(
        abs(float(rows_auto[frame]["final_heldout_psnr"]) - float(rows3[frame]["final_heldout_psnr"]))
        for frame in frame_counts
    )
    max_psnr_delta_vs_packed = max(
        abs(float(rows_auto[frame]["final_heldout_psnr"]) - float(rows_packed[frame]["final_heldout_psnr"]))
        for frame in frame_counts
    )
    return {
        "policy": {
            "packed_mode": PACKED_MODE,
            "i16x3_mode": I16X3_MODE,
            "packed_max_frame_count": AUTO_PACKED_MAX_FRAME_COUNT,
        },
        "ratios_by_frame": ratios_by_frame,
        "resolved_modes_by_frame": resolved_modes_by_frame,
        "expected_modes_by_frame": expected_modes_by_frame,
        "auto_matches_expected_policy": resolved_modes_by_frame == expected_modes_by_frame,
        "max_auto_over_best_component_total_mean_ratio": max_auto_over_best_total,
        "max_auto_over_best_component_backward_mean_ratio": max_auto_over_best_backward,
        "max_auto_over_best_component_storage_ratio": max_auto_over_best_storage,
        "max_auto_psnr_delta_vs_i16x3": max_psnr_delta_vs_i16x3,
        "max_auto_psnr_delta_vs_packed": max_psnr_delta_vs_packed,
        "auto_total_scale_first_to_last": float(auto["total_step_scale_first_to_last"]),
        "auto_backward_scale_first_to_last": float(auto["backward_scale_first_to_last"]),
        "auto_storage_scale_first_to_last": float(auto["selected_tape_storage_scale_first_to_last"]),
        "auto_oracle_candidate": (
            max_auto_over_best_total <= 1.10
            and max_auto_over_best_backward <= 1.10
            and max_auto_over_best_storage <= 1.05
            and resolved_modes_by_frame == expected_modes_by_frame
        ),
    }


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


def _partial_path(args: argparse.Namespace) -> Path | None:
    if args.partial_out_json is not None:
        return args.partial_out_json
    if args.out_json is not None:
        return args.out_json.with_suffix(args.out_json.suffix + ".partial")
    return None


def _write_partial(
    args: argparse.Namespace,
    *,
    phase: str,
    frame_counts: tuple[int, ...],
    results: dict[str, dict[str, Any]],
) -> None:
    path = _partial_path(args)
    if path is None:
        return
    payload = {
        "benchmark": "world_foam_delta_framegroup_i16x3_packed_train_eval_compare",
        "status": "partial",
        "phase": phase,
        "frame_counts": list(frame_counts),
        "render_size": int(args.render_size),
        "site_count": int(args.site_count),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "prewarm_sweep": bool(args.prewarm_sweep),
        "repeat_loaded_frames": bool(args.repeat_loaded_frames),
        "interleave_modes": bool(args.interleave_modes),
        "modes": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_interleaved(
    args: argparse.Namespace,
    *,
    frame_counts: tuple[int, ...],
    modes: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    per_mode_payloads: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    partial_results: dict[str, dict[str, Any]] = {}
    for frame_index, frame_count in enumerate(frame_counts):
        mode_order = modes if frame_index % 2 == 0 else tuple(reversed(modes))
        for mode in mode_order:
            payload = _run_mode(args, tape_mode=mode, frame_counts=(frame_count,))
            per_mode_payloads[mode].append(payload)
            partial_results[mode] = _combine_single_frame_payloads(
                tape_mode=mode,
                frame_counts=tuple(frame_counts[: len(per_mode_payloads[mode])]),
                payloads=per_mode_payloads[mode],
            )
            _write_partial(args, phase=f"measured_{mode}_{frame_count}f", frame_counts=frame_counts, results=partial_results)
    return {
        mode: _combine_single_frame_payloads(tape_mode=mode, frame_counts=frame_counts, payloads=payloads)
        for mode, payloads in per_mode_payloads.items()
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = _parse_int_list(args.frame_counts)
    modes = _modes_for_args(args)
    if args.prewarm_sweep:
        prewarm_args = argparse.Namespace(**{**vars(args), "steps": 1, "warmup_steps": 1})
        if args.interleave_modes:
            for frame_index, frame_count in enumerate(frame_counts):
                mode_order = modes if frame_index % 2 == 0 else tuple(reversed(modes))
                for mode in mode_order:
                    _run_mode(prewarm_args, tape_mode=mode, frame_counts=(frame_count,))
        else:
            for mode in modes:
                _run_mode(prewarm_args, tape_mode=mode, frame_counts=frame_counts)
    if args.interleave_modes:
        results = _run_interleaved(args, frame_counts=frame_counts, modes=modes)
    else:
        results = {mode: _run_mode(args, tape_mode=mode, frame_counts=frame_counts) for mode in modes}
    summary = summarize_pair(i16x3=results[I16X3_MODE], packed=results[PACKED_MODE], frame_counts=frame_counts)
    auto_summary = (
        summarize_auto_selector(
            i16x3=results[I16X3_MODE],
            packed=results[PACKED_MODE],
            auto=results[AUTO_MODE],
            frame_counts=frame_counts,
        )
        if AUTO_MODE in results
        else None
    )
    failures = []
    for mode, payload in results.items():
        if not _all_rows_ok(payload):
            failures.append(f"{mode}: one or more rows did not complete")
    if not all(math.isfinite(float(value)) for value in summary["psnr_delta_by_frame"].values()):
        failures.append("non-finite PSNR delta")
    return {
        "benchmark": "world_foam_delta_framegroup_i16x3_packed_train_eval_compare",
        "scope": (
            "paired same-process manual-VJP comparison of selected i16x3 and packed-record framegroup fused-MSE "
            "forks; not a STAR-UVT competitiveness artifact"
        ),
        "status": "ok" if not failures else "failed",
        "frame_counts": list(frame_counts),
        "render_size": int(args.render_size),
        "site_count": int(args.site_count),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "prewarm_sweep": bool(args.prewarm_sweep),
        "interleave_modes": bool(args.interleave_modes),
        "repeat_loaded_frames": bool(args.repeat_loaded_frames),
        "optimizer_mode": "manual-vjp",
        "mode_statuses": {mode: results[mode].get("status") for mode in modes},
        "modes": results,
        "summary": summary,
        "auto_selector_summary": auto_summary,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare i16x3 and packed-record framegroup WorldFoam train/eval paths.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="16,32")
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
    parser.add_argument("--steps", type=int, default=5)
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
        "--interleave-modes",
        action="store_true",
        help="Measure each frame count as alternating single-frame mode runs instead of one whole mode at a time.",
    )
    parser.add_argument(
        "--include-auto-selector",
        action="store_true",
        help="Also run the packed<=64/i16x3>64 auto-selector sidecar mode.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "delta_framegroup_i16x3_packed_train_eval_compare.json",
    )
    parser.add_argument("--partial-out-json", type=Path)
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
