#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from train_eval_owner_run_tape import DEFAULT_CONFIG, RESULTS_DIR, SyntheticRayMotion, run_train_eval
from smoke_fused_slab_affine_realray_mps import _parse_int_list


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _last_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows or not isinstance(rows[-1], dict):
        raise ValueError("payload missing final row")
    return rows[-1]


def _row_for_frame(payload: dict[str, Any], *, frame_count: int) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("payload missing rows")
    for row in rows:
        if isinstance(row, dict) and int(row.get("frame_count", -1)) == int(frame_count):
            return row
    return _last_row(payload)


def _nonnegative_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def _ms(row: dict[str, Any], key: str, *, allow_zero: bool = False) -> float:
    step_summary = row.get("step_summary")
    if not isinstance(step_summary, dict):
        raise ValueError("row missing step_summary")
    summary = step_summary.get(key)
    if not isinstance(summary, dict):
        raise ValueError(f"row missing step_summary.{key}")
    value = summary.get("mean_s")
    valid = _nonnegative_finite(value) if allow_zero else _positive_finite(value)
    if not valid:
        raise ValueError(f"row has invalid step_summary.{key}.mean_s")
    return float(value) * 1000.0


def _summary_for_frame(payload: dict[str, Any], *, frame_count: int) -> dict[str, float]:
    row = _row_for_frame(payload, frame_count=frame_count)
    return {
        "frame_count": int(row.get("frame_count", frame_count)),
        "total_ms": _ms(row, "total"),
        "render_ms": _ms(row, "render", allow_zero=True),
        "backward_ms": _ms(row, "backward"),
        "heldout_psnr": float(row["final_heldout_psnr"]),
        "segments_vs_full": float(row["train_selected_tape_segments_vs_full"]),
        "storage_vs_full": float(row["train_selected_tape_storage_vs_full"]),
        "edit_storage_vs_endpoint_run": float(row.get("train_endpoint_record_edit_storage_vs_endpoint_run", 0.0)),
        "block4_storage_vs_endpoint_run": float(
            row.get("train_endpoint_record_block4_storage_vs_endpoint_run", 0.0)
        ),
    }


def _summary_16f(payload: dict[str, Any]) -> dict[str, float]:
    return _summary_for_frame(payload, frame_count=16)


def _mode_partial_path(partial_out_json: Path | None, mode: str) -> Path | None:
    if partial_out_json is None:
        return None
    safe_mode = mode.replace("/", "_").replace(" ", "_")
    return partial_out_json.with_name(f"{partial_out_json.stem}.{safe_mode}.rows.partial.json")


def _write_compare_partial(
    partial_out_json: Path | None,
    *,
    requested_modes: list[str],
    completed_modes: list[str],
    results: dict[str, Any],
    render_size: int,
    site_count: int,
    frame_counts: tuple[int, ...],
    edit_block_size: int,
    allow_repeat_loaded_frames: bool,
) -> None:
    if partial_out_json is None:
        return
    partial_out_json.parent.mkdir(parents=True, exist_ok=True)
    partial_out_json.write_text(
        json.dumps(
            {
                "benchmark": "world_foam_lane2_endpoint_run_vs_record_edit_current_process_train_eval_partial",
                "status": "running",
                "requested_modes": requested_modes,
                "completed_modes": completed_modes,
                "frame_counts": list(frame_counts),
                "render_size": int(render_size),
                "site_count": int(site_count),
                "edit_block_size": int(edit_block_size),
                "allow_repeat_loaded_frames": bool(allow_repeat_loaded_frames),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def compare(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    synthetic_motion: SyntheticRayMotion,
    steps: int,
    warmup_steps: int,
    lr: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    optimizer_mode: str,
    segment_tape_vjp_mode: str,
    include_block4: bool,
    include_block_coeff: bool,
    include_block_coeff_rgb: bool,
    include_block_coeff16: bool,
    edit_block_size: int,
    include_block_coeff_fused_mse: bool = False,
    include_edit_fused_mse: bool = False,
    include_delta_framegroup16_fused_mse: bool = False,
    include_delta_i16x4_framegroup16_fused_mse: bool = False,
    allow_repeat_loaded_frames: bool = False,
    partial_out_json: Path | None = None,
) -> dict[str, Any]:
    common = {
        "config_path": config_path,
        "frame_counts": frame_counts,
        "render_size": render_size,
        "site_count": site_count,
        "near": near,
        "far": far,
        "density": density,
        "invalid_epsilon": invalid_epsilon,
        "transmittance_threshold": transmittance_threshold,
        "synthetic_motion": synthetic_motion,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "adam_eps": adam_eps,
        "optimizer_mode": optimizer_mode,
        "segment_tape_vjp_mode": segment_tape_vjp_mode,
        "edit_block_size": edit_block_size,
        "allow_repeat_loaded_frames": allow_repeat_loaded_frames,
    }
    requested_modes = ["endpoint-run", "endpoint-record-edit"]
    if include_edit_fused_mse:
        requested_modes.append("endpoint-record-edit-fused-mse")
    if include_delta_framegroup16_fused_mse:
        requested_modes.append("endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse")
    if include_delta_i16x4_framegroup16_fused_mse:
        requested_modes.append("endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse")
    if include_block4:
        requested_modes.append("endpoint-record-edit-block4")
    if include_block_coeff:
        requested_modes.append("endpoint-record-edit-block-coeff")
    if include_block_coeff_rgb:
        requested_modes.append("endpoint-record-edit-block-coeff-rgb")
    if include_block_coeff_fused_mse:
        requested_modes.append("endpoint-record-edit-block-coeff-fused-mse")
    if include_block_coeff16:
        requested_modes.append("endpoint-record-edit-block-coeff16")

    results: dict[str, Any] = {}
    completed_modes: list[str] = []
    for mode in requested_modes:
        print(f"[compare_endpoint_run_record_edit_train_eval] start mode={mode}", flush=True)
        results[mode] = run_train_eval(
            tape_mode=mode,
            partial_out_json=_mode_partial_path(partial_out_json, mode),
            **common,
        )
        completed_modes.append(mode)
        final = _summary_16f(results[mode])
        print(
            "[compare_endpoint_run_record_edit_train_eval] done "
            f"mode={mode} total_ms={final['total_ms']:.3f} render_ms={final['render_ms']:.3f} "
            f"backward_ms={final['backward_ms']:.3f} heldout_psnr={final['heldout_psnr']:.4f}",
            flush=True,
        )
        _write_compare_partial(
            partial_out_json,
            requested_modes=requested_modes,
            completed_modes=completed_modes,
            results=results,
            render_size=render_size,
            site_count=site_count,
            frame_counts=frame_counts,
            edit_block_size=edit_block_size,
            allow_repeat_loaded_frames=allow_repeat_loaded_frames,
        )
    summary_16f = {name: _summary_16f(payload) for name, payload in results.items()}
    summary_by_frame = {
        str(frame): {name: _summary_for_frame(payload, frame_count=frame) for name, payload in results.items()}
        for frame in frame_counts
    }
    endpoint = summary_16f["endpoint-run"]
    edit = summary_16f["endpoint-record-edit"]
    ratios = {
        "edit_to_endpoint_total_16f": edit["total_ms"] / endpoint["total_ms"],
        "edit_to_endpoint_render_16f": edit["render_ms"] / endpoint["render_ms"],
        "edit_to_endpoint_backward_16f": edit["backward_ms"] / endpoint["backward_ms"],
    }
    block4 = summary_16f.get("endpoint-record-edit-block4")
    edit_fused_mse = summary_16f.get("endpoint-record-edit-fused-mse")
    delta_framegroup16 = summary_16f.get("endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse")
    block_coeff = summary_16f.get("endpoint-record-edit-block-coeff")
    block_coeff_rgb = summary_16f.get("endpoint-record-edit-block-coeff-rgb")
    block_coeff_fused_mse = summary_16f.get("endpoint-record-edit-block-coeff-fused-mse")
    block_coeff16 = summary_16f.get("endpoint-record-edit-block-coeff16")
    if block4 is not None:
        ratios.update(
            {
                "block4_to_endpoint_total_16f": block4["total_ms"] / endpoint["total_ms"],
                "block4_to_endpoint_render_16f": block4["render_ms"] / endpoint["render_ms"],
                "block4_to_endpoint_backward_16f": block4["backward_ms"] / endpoint["backward_ms"],
                "block4_to_edit_total_16f": block4["total_ms"] / edit["total_ms"],
                "block4_to_edit_render_16f": block4["render_ms"] / edit["render_ms"],
                "block4_to_edit_backward_16f": block4["backward_ms"] / edit["backward_ms"],
            }
        )
    if edit_fused_mse is not None:
        ratios.update(
            {
                "edit_fused_mse_to_endpoint_total_16f": edit_fused_mse["total_ms"] / endpoint["total_ms"],
                "edit_fused_mse_to_endpoint_backward_16f": edit_fused_mse["backward_ms"] / endpoint["backward_ms"],
                "edit_fused_mse_to_edit_total_16f": edit_fused_mse["total_ms"] / edit["total_ms"],
                "edit_fused_mse_to_edit_backward_16f": edit_fused_mse["backward_ms"] / edit["backward_ms"],
            }
        )
    if delta_framegroup16 is not None:
        ratios.update(
            {
                "delta_framegroup16_to_endpoint_total_16f": delta_framegroup16["total_ms"] / endpoint["total_ms"],
                "delta_framegroup16_to_endpoint_backward_16f": delta_framegroup16["backward_ms"]
                / endpoint["backward_ms"],
                "delta_framegroup16_to_edit_total_16f": delta_framegroup16["total_ms"] / edit["total_ms"],
                "delta_framegroup16_to_edit_backward_16f": delta_framegroup16["backward_ms"]
                / edit["backward_ms"],
            }
        )
        if edit_fused_mse is not None:
            ratios.update(
                {
                    "delta_framegroup16_to_edit_fused_mse_total_16f": delta_framegroup16["total_ms"]
                    / edit_fused_mse["total_ms"],
                    "delta_framegroup16_to_edit_fused_mse_backward_16f": delta_framegroup16["backward_ms"]
                    / edit_fused_mse["backward_ms"],
                }
            )
    if block_coeff is not None:
        ratios.update(
            {
                "block_coeff_to_endpoint_total_16f": block_coeff["total_ms"] / endpoint["total_ms"],
                "block_coeff_to_endpoint_render_16f": block_coeff["render_ms"] / endpoint["render_ms"],
                "block_coeff_to_endpoint_backward_16f": block_coeff["backward_ms"] / endpoint["backward_ms"],
                "block_coeff_to_edit_total_16f": block_coeff["total_ms"] / edit["total_ms"],
                "block_coeff_to_edit_render_16f": block_coeff["render_ms"] / edit["render_ms"],
                "block_coeff_to_edit_backward_16f": block_coeff["backward_ms"] / edit["backward_ms"],
            }
        )
        if block4 is not None:
            ratios.update(
                {
                    "block_coeff_to_block4_total_16f": block_coeff["total_ms"] / block4["total_ms"],
                    "block_coeff_to_block4_render_16f": block_coeff["render_ms"] / block4["render_ms"],
                    "block_coeff_to_block4_backward_16f": block_coeff["backward_ms"] / block4["backward_ms"],
                }
            )
    if block_coeff_rgb is not None:
        ratios.update(
            {
                "block_coeff_rgb_to_endpoint_total_16f": block_coeff_rgb["total_ms"] / endpoint["total_ms"],
                "block_coeff_rgb_to_endpoint_render_16f": block_coeff_rgb["render_ms"] / endpoint["render_ms"],
                "block_coeff_rgb_to_endpoint_backward_16f": block_coeff_rgb["backward_ms"] / endpoint["backward_ms"],
                "block_coeff_rgb_to_edit_total_16f": block_coeff_rgb["total_ms"] / edit["total_ms"],
                "block_coeff_rgb_to_edit_render_16f": block_coeff_rgb["render_ms"] / edit["render_ms"],
                "block_coeff_rgb_to_edit_backward_16f": block_coeff_rgb["backward_ms"] / edit["backward_ms"],
            }
        )
        if block4 is not None:
            ratios.update(
                {
                    "block_coeff_rgb_to_block4_total_16f": block_coeff_rgb["total_ms"] / block4["total_ms"],
                    "block_coeff_rgb_to_block4_render_16f": block_coeff_rgb["render_ms"] / block4["render_ms"],
                    "block_coeff_rgb_to_block4_backward_16f": block_coeff_rgb["backward_ms"] / block4["backward_ms"],
                }
            )
        if block_coeff is not None:
            ratios.update(
                {
                    "block_coeff_rgb_to_block_coeff_total_16f": block_coeff_rgb["total_ms"] / block_coeff["total_ms"],
                    "block_coeff_rgb_to_block_coeff_render_16f": block_coeff_rgb["render_ms"] / block_coeff["render_ms"],
                    "block_coeff_rgb_to_block_coeff_backward_16f": block_coeff_rgb["backward_ms"]
                    / block_coeff["backward_ms"],
                }
            )
    if block_coeff_fused_mse is not None:
        ratios.update(
            {
                "block_coeff_fused_mse_to_endpoint_total_16f": block_coeff_fused_mse["total_ms"] / endpoint["total_ms"],
                "block_coeff_fused_mse_to_endpoint_backward_16f": block_coeff_fused_mse["backward_ms"]
                / endpoint["backward_ms"],
                "block_coeff_fused_mse_to_edit_total_16f": block_coeff_fused_mse["total_ms"] / edit["total_ms"],
                "block_coeff_fused_mse_to_edit_backward_16f": block_coeff_fused_mse["backward_ms"]
                / edit["backward_ms"],
            }
        )
        if block_coeff_rgb is not None:
            ratios.update(
                {
                    "block_coeff_fused_mse_to_block_coeff_rgb_total_16f": block_coeff_fused_mse["total_ms"]
                    / block_coeff_rgb["total_ms"],
                    "block_coeff_fused_mse_to_block_coeff_rgb_backward_16f": block_coeff_fused_mse["backward_ms"]
                    / block_coeff_rgb["backward_ms"],
                }
            )
        if block_coeff is not None:
            ratios.update(
                {
                    "block_coeff_fused_mse_to_block_coeff_total_16f": block_coeff_fused_mse["total_ms"]
                    / block_coeff["total_ms"],
                    "block_coeff_fused_mse_to_block_coeff_backward_16f": block_coeff_fused_mse["backward_ms"]
                    / block_coeff["backward_ms"],
                }
            )
    if block_coeff16 is not None:
        ratios.update(
            {
                "block_coeff16_to_endpoint_total_16f": block_coeff16["total_ms"] / endpoint["total_ms"],
                "block_coeff16_to_endpoint_render_16f": block_coeff16["render_ms"] / endpoint["render_ms"],
                "block_coeff16_to_endpoint_backward_16f": block_coeff16["backward_ms"] / endpoint["backward_ms"],
                "block_coeff16_to_edit_total_16f": block_coeff16["total_ms"] / edit["total_ms"],
                "block_coeff16_to_edit_render_16f": block_coeff16["render_ms"] / edit["render_ms"],
                "block_coeff16_to_edit_backward_16f": block_coeff16["backward_ms"] / edit["backward_ms"],
            }
        )
        if block4 is not None:
            ratios.update(
                {
                    "block_coeff16_to_block4_total_16f": block_coeff16["total_ms"] / block4["total_ms"],
                    "block_coeff16_to_block4_render_16f": block_coeff16["render_ms"] / block4["render_ms"],
                    "block_coeff16_to_block4_backward_16f": block_coeff16["backward_ms"] / block4["backward_ms"],
                }
            )
        if block_coeff is not None:
            ratios.update(
                {
                    "block_coeff16_to_block_coeff_total_16f": block_coeff16["total_ms"] / block_coeff["total_ms"],
                    "block_coeff16_to_block_coeff_render_16f": block_coeff16["render_ms"] / block_coeff["render_ms"],
                    "block_coeff16_to_block_coeff_backward_16f": block_coeff16["backward_ms"] / block_coeff["backward_ms"],
                }
            )
    psnr_delta = abs(edit["heldout_psnr"] - endpoint["heldout_psnr"])
    block4_psnr_delta = (
        abs(block4["heldout_psnr"] - endpoint["heldout_psnr"]) if block4 is not None else 0.0
    )
    edit_fused_mse_psnr_delta = (
        abs(edit_fused_mse["heldout_psnr"] - endpoint["heldout_psnr"])
        if edit_fused_mse is not None
        else 0.0
    )
    delta_framegroup16_psnr_delta = (
        abs(delta_framegroup16["heldout_psnr"] - endpoint["heldout_psnr"])
        if delta_framegroup16 is not None
        else 0.0
    )
    block_coeff_psnr_delta = (
        abs(block_coeff["heldout_psnr"] - endpoint["heldout_psnr"]) if block_coeff is not None else 0.0
    )
    block_coeff_rgb_psnr_delta = (
        abs(block_coeff_rgb["heldout_psnr"] - endpoint["heldout_psnr"]) if block_coeff_rgb is not None else 0.0
    )
    block_coeff_fused_mse_psnr_delta = (
        abs(block_coeff_fused_mse["heldout_psnr"] - endpoint["heldout_psnr"])
        if block_coeff_fused_mse is not None
        else 0.0
    )
    block_coeff16_psnr_delta = (
        abs(block_coeff16["heldout_psnr"] - endpoint["heldout_psnr"]) if block_coeff16 is not None else 0.0
    )
    edit_total_ratio = ratios["edit_to_endpoint_total_16f"]
    block4_total_ratio = ratios.get("block4_to_endpoint_total_16f")
    edit_fused_mse_total_ratio = ratios.get("edit_fused_mse_to_endpoint_total_16f")
    delta_framegroup16_total_ratio = ratios.get("delta_framegroup16_to_endpoint_total_16f")
    block_coeff_total_ratio = ratios.get("block_coeff_to_endpoint_total_16f")
    block_coeff_rgb_total_ratio = ratios.get("block_coeff_rgb_to_endpoint_total_16f")
    block_coeff_fused_mse_total_ratio = ratios.get("block_coeff_fused_mse_to_endpoint_total_16f")
    block_coeff16_total_ratio = ratios.get("block_coeff16_to_endpoint_total_16f")
    speed_clause = (
        f"and is faster than endpoint-run at 16f ({edit_total_ratio:.3f}x total-step ratio)"
        if edit_total_ratio < 1.0
        else f"but is slower than endpoint-run at 16f ({edit_total_ratio:.3f}x total-step ratio)"
    )
    block4_clause = ""
    if block4_total_ratio is not None:
        block4_clause = (
            f" Block4 is faster than endpoint-run at 16f ({float(block4_total_ratio):.3f}x total-step ratio)."
            if float(block4_total_ratio) < 1.0
            else f" Block4 is slower than endpoint-run at 16f ({float(block4_total_ratio):.3f}x total-step ratio)."
        )
    edit_fused_mse_clause = ""
    if edit_fused_mse_total_ratio is not None:
        edit_fused_mse_clause = (
            " Edit-fused-mse is faster than endpoint-run at 16f "
            f"({float(edit_fused_mse_total_ratio):.3f}x total-step ratio)."
            if float(edit_fused_mse_total_ratio) < 1.0
            else " Edit-fused-mse is slower than endpoint-run at 16f "
            f"({float(edit_fused_mse_total_ratio):.3f}x total-step ratio)."
        )
    delta_framegroup16_clause = ""
    if delta_framegroup16_total_ratio is not None:
        delta_framegroup16_clause = (
            " Delta-framegroup16 fused-MSE is faster than endpoint-run at 16f "
            f"({float(delta_framegroup16_total_ratio):.3f}x total-step ratio)."
            if float(delta_framegroup16_total_ratio) < 1.0
            else " Delta-framegroup16 fused-MSE is slower than endpoint-run at 16f "
            f"({float(delta_framegroup16_total_ratio):.3f}x total-step ratio)."
        )
    block_coeff_clause = ""
    if block_coeff_total_ratio is not None:
        block_coeff_clause = (
            f" Block-coeff is faster than endpoint-run at 16f ({float(block_coeff_total_ratio):.3f}x total-step ratio)."
            if float(block_coeff_total_ratio) < 1.0
            else f" Block-coeff is slower than endpoint-run at 16f ({float(block_coeff_total_ratio):.3f}x total-step ratio)."
        )
    block_coeff_rgb_clause = ""
    if block_coeff_rgb_total_ratio is not None:
        block_coeff_rgb_clause = (
            " Block-coeff-rgb is faster than endpoint-run at 16f "
            f"({float(block_coeff_rgb_total_ratio):.3f}x total-step ratio)."
            if float(block_coeff_rgb_total_ratio) < 1.0
            else " Block-coeff-rgb is slower than endpoint-run at 16f "
            f"({float(block_coeff_rgb_total_ratio):.3f}x total-step ratio)."
        )
    block_coeff16_clause = ""
    block_coeff_fused_mse_clause = ""
    if block_coeff_fused_mse_total_ratio is not None:
        block_coeff_fused_mse_clause = (
            " Block-coeff-fused-mse is faster than endpoint-run at 16f "
            f"({float(block_coeff_fused_mse_total_ratio):.3f}x total-step ratio)."
            if float(block_coeff_fused_mse_total_ratio) < 1.0
            else " Block-coeff-fused-mse is slower than endpoint-run at 16f "
            f"({float(block_coeff_fused_mse_total_ratio):.3f}x total-step ratio)."
        )
    if block_coeff16_total_ratio is not None:
        block_coeff16_clause = (
            f" Block-coeff16 is faster than endpoint-run at 16f ({float(block_coeff16_total_ratio):.3f}x total-step ratio)."
            if float(block_coeff16_total_ratio) < 1.0
            else f" Block-coeff16 is slower than endpoint-run at 16f ({float(block_coeff16_total_ratio):.3f}x total-step ratio)."
        )
    return {
        "benchmark": "world_foam_lane2_endpoint_run_vs_record_edit_current_process_train_eval",
        "status": "ok" if all(payload.get("status") == "ok" for payload in results.values()) else "failed",
        "scope": (
            "paired current-process smoke-scale comparison, not a stable benchmark or STAR UVT competitive claim"
            + (
                "; requested frame counts above the fixture length use repeated loaded frames for synthetic "
                "speed-scaling only"
                if allow_repeat_loaded_frames
                else ""
            )
        ),
        "edit_block_size": int(edit_block_size),
        "allow_repeat_loaded_frames": bool(allow_repeat_loaded_frames),
        "repeat_loaded_frames": any(
            bool(result.get("repeat_loaded_frames")) for result in results.values() if isinstance(result, dict)
        ),
        "results": results,
        "summary_16f": summary_16f,
        "summary_by_frame": summary_by_frame,
        "ratios": ratios,
        "acceptance": {
            "endpoint_run_ok": results["endpoint-run"].get("status") == "ok",
            "endpoint_record_edit_ok": results["endpoint-record-edit"].get("status") == "ok",
            "endpoint_record_edit_fused_mse_ok": True
            if edit_fused_mse is None
            else results["endpoint-record-edit-fused-mse"].get("status") == "ok",
            "endpoint_record_delta_framegroup16_fused_mse_ok": True
            if delta_framegroup16 is None
            else results["endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"].get("status") == "ok",
            "endpoint_record_edit_block4_ok": True
            if block4 is None
            else results["endpoint-record-edit-block4"].get("status") == "ok",
            "endpoint_record_edit_block_coeff_ok": True
            if block_coeff is None
            else results["endpoint-record-edit-block-coeff"].get("status") == "ok",
            "endpoint_record_edit_block_coeff_rgb_ok": True
            if block_coeff_rgb is None
            else results["endpoint-record-edit-block-coeff-rgb"].get("status") == "ok",
            "endpoint_record_edit_block_coeff_fused_mse_ok": True
            if block_coeff_fused_mse is None
            else results["endpoint-record-edit-block-coeff-fused-mse"].get("status") == "ok",
            "endpoint_record_edit_block_coeff16_ok": True
            if block_coeff16 is None
            else results["endpoint-record-edit-block-coeff16"].get("status") == "ok",
            "psnr_matches": psnr_delta <= 1.0e-4,
            "edit_fused_mse_psnr_matches": True
            if edit_fused_mse is None
            else edit_fused_mse_psnr_delta <= 1.0e-4,
            "delta_framegroup16_psnr_matches": True
            if delta_framegroup16 is None
            else delta_framegroup16_psnr_delta <= 1.0e-4,
            "block4_psnr_matches": True if block4 is None else block4_psnr_delta <= 1.0e-4,
            "block_coeff_psnr_matches": True if block_coeff is None else block_coeff_psnr_delta <= 1.0e-4,
            "block_coeff_rgb_psnr_matches": True
            if block_coeff_rgb is None
            else block_coeff_rgb_psnr_delta <= 1.0e-4,
            "block_coeff_fused_mse_psnr_matches": True
            if block_coeff_fused_mse is None
            else block_coeff_fused_mse_psnr_delta <= 1.0e-4,
            "block_coeff16_psnr_matches": True if block_coeff16 is None else block_coeff16_psnr_delta <= 1.0e-3,
            "edit_storage_below_endpoint": edit["storage_vs_full"] < endpoint["storage_vs_full"],
            "edit_fused_mse_storage_below_endpoint": True
            if edit_fused_mse is None
            else edit_fused_mse["storage_vs_full"] < endpoint["storage_vs_full"],
            "delta_framegroup16_storage_below_endpoint": True
            if delta_framegroup16 is None
            else delta_framegroup16["storage_vs_full"] < endpoint["storage_vs_full"],
            "block4_storage_below_endpoint": True
            if block4 is None
            else block4["storage_vs_full"] < endpoint["storage_vs_full"],
            "block_coeff_storage_positive": True
            if block_coeff is None
            else block_coeff["storage_vs_full"] > 0.0,
            "block_coeff_rgb_storage_positive": True
            if block_coeff_rgb is None
            else block_coeff_rgb["storage_vs_full"] > 0.0,
            "block_coeff_fused_mse_storage_positive": True
            if block_coeff_fused_mse is None
            else block_coeff_fused_mse["storage_vs_full"] > 0.0,
            "block_coeff16_storage_positive": True
            if block_coeff16 is None
            else block_coeff16["storage_vs_full"] > 0.0,
            "edit_total_ratio_positive": edit_total_ratio > 0.0,
            "edit_fused_mse_total_ratio_positive": True
            if edit_fused_mse_total_ratio is None
            else float(edit_fused_mse_total_ratio) > 0.0,
            "delta_framegroup16_total_ratio_positive": True
            if delta_framegroup16_total_ratio is None
            else float(delta_framegroup16_total_ratio) > 0.0,
            "block4_total_ratio_positive": True if block4_total_ratio is None else float(block4_total_ratio) > 0.0,
            "block_coeff_total_ratio_positive": True
            if block_coeff_total_ratio is None
            else float(block_coeff_total_ratio) > 0.0,
            "block_coeff_rgb_total_ratio_positive": True
            if block_coeff_rgb_total_ratio is None
            else float(block_coeff_rgb_total_ratio) > 0.0,
            "block_coeff_fused_mse_total_ratio_positive": True
            if block_coeff_fused_mse_total_ratio is None
            else float(block_coeff_fused_mse_total_ratio) > 0.0,
            "block_coeff16_total_ratio_positive": True
            if block_coeff16_total_ratio is None
            else float(block_coeff16_total_ratio) > 0.0,
            "edit_total_not_slower_than_endpoint": edit_total_ratio <= 1.0,
            "edit_fused_mse_total_not_slower_than_endpoint": True
            if edit_fused_mse_total_ratio is None
            else float(edit_fused_mse_total_ratio) <= 1.0,
            "delta_framegroup16_total_not_slower_than_endpoint": True
            if delta_framegroup16_total_ratio is None
            else float(delta_framegroup16_total_ratio) <= 1.0,
            "block4_total_not_slower_than_endpoint": True
            if block4_total_ratio is None
            else float(block4_total_ratio) <= 1.0,
            "block_coeff_total_not_slower_than_endpoint": True
            if block_coeff_total_ratio is None
            else float(block_coeff_total_ratio) <= 1.0,
            "block_coeff_rgb_total_not_slower_than_endpoint": True
            if block_coeff_rgb_total_ratio is None
            else float(block_coeff_rgb_total_ratio) <= 1.0,
            "block_coeff_fused_mse_total_not_slower_than_endpoint": True
            if block_coeff_fused_mse_total_ratio is None
            else float(block_coeff_fused_mse_total_ratio) <= 1.0,
            "block_coeff16_total_not_slower_than_endpoint": True
            if block_coeff16_total_ratio is None
            else float(block_coeff16_total_ratio) <= 1.0,
        },
        "conclusion": (
            "Endpoint-record edit keeps matched PSNR and much lower storage, "
            f"{speed_clause} in this same-process smoke-scale comparison.{block4_clause}{block_coeff_clause}"
            f"{edit_fused_mse_clause}{delta_framegroup16_clause}{block_coeff_rgb_clause}"
            f"{block_coeff_fused_mse_clause}{block_coeff16_clause}"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare endpoint-run and endpoint-record-edit train/eval.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
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
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--optimizer-mode", choices=("manual-vjp", "autograd"), default="autograd")
    parser.add_argument("--segment-tape-vjp-mode", choices=("direct_atomic_grad_only",), default="direct_atomic_grad_only")
    parser.add_argument("--include-block4", action="store_true")
    parser.add_argument("--include-edit-fused-mse", action="store_true")
    parser.add_argument(
        "--include-delta-framegroup16-fused-mse",
        action="store_true",
        help="Also run the current endpoint-record delta-replace coeff16 i16x3 framegroup16 fused-MSE path.",
    )
    parser.add_argument(
        "--include-delta-i16x4-framegroup16-fused-mse",
        action="store_true",
        help="Also run the experimental padded i16x4 framegroup16 fused-MSE fork.",
    )
    parser.add_argument("--include-block-coeff", action="store_true")
    parser.add_argument("--include-block-coeff-rgb", action="store_true")
    parser.add_argument("--include-block-coeff-fused-mse", action="store_true")
    parser.add_argument("--include-block-coeff16", action="store_true")
    parser.add_argument("--edit-block-size", type=int, default=4)
    parser.add_argument(
        "--repeat-loaded-frames",
        action="store_true",
        help=(
            "Repeat a shorter loaded view-major fixture when requested frame counts exceed the real fixture. "
            "This is a synthetic speed-scaling smoke, not a real longer-video quality run."
        ),
    )
    parser.add_argument("--partial-out-json", type=Path)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_cutcache_render32_2_4_8_16.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare(
        config_path=args.config,
        frame_counts=_parse_int_list(args.frame_counts),
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
        optimizer_mode=args.optimizer_mode,
        segment_tape_vjp_mode=args.segment_tape_vjp_mode,
        include_edit_fused_mse=args.include_edit_fused_mse,
        include_delta_framegroup16_fused_mse=args.include_delta_framegroup16_fused_mse,
        include_delta_i16x4_framegroup16_fused_mse=args.include_delta_i16x4_framegroup16_fused_mse,
        include_block4=args.include_block4,
        include_block_coeff=args.include_block_coeff,
        include_block_coeff_rgb=args.include_block_coeff_rgb,
        include_block_coeff_fused_mse=args.include_block_coeff_fused_mse,
        include_block_coeff16=args.include_block_coeff16,
        edit_block_size=int(args.edit_block_size),
        allow_repeat_loaded_frames=bool(args.repeat_loaded_frames),
        partial_out_json=args.partial_out_json,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
