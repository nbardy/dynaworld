from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


from benchmark_bootstrap import ROOT, ensure_sys_path
from train_artifacts import write_json

WORLD_FOAM_DIR = ROOT / "research_experiments" / "world_foam_lane2"
ensure_sys_path(WORLD_FOAM_DIR)

from gate0_beam_toy import ToyConfig  # noqa: E402
from gate0_event_sharing_benchmark import build_payload as build_world_foam_payload  # noqa: E402
from gate0_event_sharing_benchmark import parse_frame_counts, parse_float_list  # noqa: E402
from gate0_shared_forward_backward import run as run_world_foam_gradient_reference  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _num(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return value
    return None


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_num(row.get(field)) for row in rows]
    numeric = [float(value) for value in values if value is not None]
    return mean(numeric) if numeric else None


def _sum_field(rows: list[dict[str, Any]], field: str) -> int | None:
    values = [_num(row.get(field)) for row in rows]
    numeric = [int(value) for value in values if value is not None]
    return sum(numeric) if numeric else None


def _find_world_row(world_payload: dict[str, Any], summary_frame: int) -> dict[str, Any]:
    rows = world_payload["world_foam_rows"]
    if not isinstance(rows, list):
        raise ValueError("world_foam_rows must be a list")
    for row in rows:
        if int(row["frames"]) == int(summary_frame):
            return row
    raise ValueError(f"summary frame {summary_frame} not present in World Foam rows")


def _world_rows(world_payload: dict[str, Any], summary_frame: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = _find_world_row(world_payload, summary_frame)
    reference_count = int(row["per_frame_event_sum"])
    beam_count = int(row["beam_slab_event_sum"])
    normalized = [
        {
            "row_id": f"world_foam_per_frame_{summary_frame}f",
            "method": "per_frame_foam",
            "comparison_unit": "power_boundary_events",
            "frames": summary_frame,
            "reference_count": reference_count,
            "candidate_count": reference_count,
            "ratio_vs_per_frame": 1.0,
        },
        {
            "row_id": f"world_foam_beam_{summary_frame}f",
            "method": "beam_foam",
            "comparison_unit": "power_boundary_events",
            "frames": summary_frame,
            "reference_count": reference_count,
            "candidate_count": beam_count,
            "ratio_vs_per_frame": float(row["event_sharing_ratio"]),
            "missing_sample_events": int(row["missing_sample_events"]),
            "extra_candidate_events": int(row["extra_candidate_events"]),
            "invalid_denominator_count": int(row["invalid_denominator_count"]),
        },
    ]
    summary = {
        "summary_frame": summary_frame,
        "per_frame_event_sum": reference_count,
        "beam_slab_event_sum": beam_count,
        "event_sharing_ratio": float(row["event_sharing_ratio"]),
        "backward_status": world_payload["backward_status"],
        "per_frame_backward_replay_event_sum": reference_count,
        "beam_backward_replay_event_sum": beam_count,
        "backward_replay_ratio": float(row["event_sharing_ratio"]),
        "missing_sample_events": int(row["missing_sample_events"]),
        "sublinear_event_growth": bool(world_payload["growth"]["sublinear_event_growth"]),
        "all_sweeps_zero_missing": bool(world_payload["acceptance"]["all_rows_zero_missing"]),
        "all_sweeps_sublinear": bool(world_payload["acceptance"]["sublinear_event_growth"]),
    }
    return normalized, summary


def _world_gradient_reference(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    camera_velocity_x = parse_float_list(args.camera_velocities)[0]
    config = ToyConfig(
        frame_counts=parse_frame_counts(args.frame_counts),
        u_samples=args.u_samples,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        camera_velocity_x=camera_velocity_x,
        invalid_epsilon=args.invalid_epsilon,
    )
    payload = run_world_foam_gradient_reference(config)
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("gradient reference rows must be a list")
    row = next((item for item in rows if int(item["frames"]) == int(args.summary_frame)), None)
    if row is None:
        raise ValueError(f"summary frame {args.summary_frame} not present in gradient reference rows")
    normalized = [
        {
            "row_id": f"world_foam_shared_backward_{args.summary_frame}f",
            "method": "beam_foam_shared_forward_backward",
            "comparison_unit": "site_signal_gradient_cpu_reference",
            "frames": args.summary_frame,
            "gradient_scope": payload["gradient_scope"],
            "direct_forward_boundary_scans": int(row["direct_forward_boundary_scans"]),
            "direct_backward_boundary_scans": int(row["direct_backward_boundary_scans"]),
            "shared_forward_boundary_scans": int(row["shared_forward_boundary_scans"]),
            "shared_backward_boundary_scans": int(row["shared_backward_boundary_scans"]),
            "shared_forward_backward_boundary_scan_ratio": float(
                row["shared_forward_backward_boundary_scan_ratio"]
            ),
            "max_output_abs_error": float(row["max_output_abs_error"]),
            "signal_gradient_max_abs_error": float(row["signal_gradient_max_abs_error"]),
            "finite_difference_max_abs_error": float(payload["finite_difference_max_abs_error"]),
        }
    ]
    summary = {
        "status": payload["status"],
        "gradient_scope": payload["gradient_scope"],
        "acceptance": payload["acceptance"],
        "summary_frame": args.summary_frame,
        "shared_forward_backward_boundary_scan_ratio": float(
            row["shared_forward_backward_boundary_scan_ratio"]
        ),
        "max_output_abs_error": float(row["max_output_abs_error"]),
        "signal_gradient_max_abs_error": float(row["signal_gradient_max_abs_error"]),
        "finite_difference_max_abs_error": float(payload["finite_difference_max_abs_error"]),
    }
    return normalized, summary


def _world_mps_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate0_mps_power_boundary_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not have a rows list")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_power_boundary_events",
        "status": payload.get("status"),
        "boundary_count": payload.get("boundary_count"),
        "beam_count": payload.get("beam_count"),
        "all_rows_match_cpu_fixture": all(
            bool(row.get("matches_cpu_fixture")) for row in rows if isinstance(row, dict)
        ),
        "rows": rows,
    }


def _world_mps_shared_replay_summary(summary_frame: int) -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate0_6_mps_shared_replay_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not have a rows list")
    row = next((item for item in rows if isinstance(item, dict) and int(item["frames"]) == summary_frame), None)
    if row is None:
        raise ValueError(f"{path} does not have a {summary_frame}-frame row")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_shared_forward_backward_replay",
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "gradient_scope": payload.get("gradient_scope"),
        "summary_frame": summary_frame,
        "acceptance": acceptance,
        "all_rows_match_cpu_reference": all(bool(value) for value in acceptance.values()),
        "row_count": len(rows),
        "rows": rows,
        "direct_forward_boundary_scans": int(row["direct_forward_boundary_scans"]),
        "direct_backward_boundary_scans": int(row["direct_backward_boundary_scans"]),
        "shared_forward_boundary_scans": int(row["shared_forward_boundary_scans"]),
        "shared_backward_boundary_scans": int(row["shared_backward_boundary_scans"]),
        "shared_forward_backward_boundary_scan_ratio": float(row["shared_forward_backward_boundary_scan_ratio"]),
        "max_output_abs_error": float(row["max_output_abs_error"]),
        "signal_gradient_max_abs_error": float(row["signal_gradient_max_abs_error"]),
        "loss_abs_error": float(row["loss_abs_error"]),
        "mps_shared_replay_wall_clock_ms": float(row["mps_shared_replay_wall_clock_ms"]),
        "timing_iters": int(row["timing_iters"]),
    }


def _world_mps_rgb_strip_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate0_7_mps_rgb_strip_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_rgb_strip_toy_image",
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "acceptance": acceptance,
        "frames": payload.get("frames"),
        "height": payload.get("height"),
        "width": payload.get("width"),
        "pixel_ray_count": payload.get("pixel_ray_count"),
        "strip_image_shape": payload.get("strip_image_shape"),
        "ppm_out": payload.get("ppm_out"),
        "max_rgb_abs_error": payload.get("max_rgb_abs_error"),
        "mean_rgb_abs_error": payload.get("mean_rgb_abs_error"),
        "color_gradient_max_abs_error": payload.get("color_gradient_max_abs_error"),
        "loss_abs_error": payload.get("loss_abs_error"),
        "mps_rgb_strip_wall_clock_ms": payload.get("mps_rgb_strip_wall_clock_ms"),
        "rgb_min": payload.get("rgb_min"),
        "rgb_max": payload.get("rgb_max"),
        "rgb_std": payload.get("rgb_std"),
    }


def _world_mps_composite_strip_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate0_8_mps_composite_strip_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_rgb_alpha_depth_composite_toy_image",
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "acceptance": acceptance,
        "frames": payload.get("frames"),
        "height": payload.get("height"),
        "width": payload.get("width"),
        "pixel_ray_count": payload.get("pixel_ray_count"),
        "rgb_shape": payload.get("rgb_shape"),
        "alpha_shape": payload.get("alpha_shape"),
        "depth_shape": payload.get("depth_shape"),
        "ppm_out": payload.get("ppm_out"),
        "max_rgb_abs_error": payload.get("max_rgb_abs_error"),
        "max_alpha_abs_error": payload.get("max_alpha_abs_error"),
        "max_depth_abs_error": payload.get("max_depth_abs_error"),
        "mps_composite_wall_clock_ms": payload.get("mps_composite_wall_clock_ms"),
        "shared_forward_boundary_scan_ratio": payload.get("shared_forward_boundary_scan_ratio"),
        "rgb_min": payload.get("rgb_min"),
        "rgb_max": payload.get("rgb_max"),
        "rgb_std": payload.get("rgb_std"),
        "alpha_min": payload.get("alpha_min"),
        "alpha_max": payload.get("alpha_max"),
        "alpha_std": payload.get("alpha_std"),
        "depth_min": payload.get("depth_min"),
        "depth_max": payload.get("depth_max"),
        "depth_std": payload.get("depth_std"),
    }


def _world_mps_composite_vjp_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate0_9_mps_composite_vjp_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_fixed_segment_composite_vjp",
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "acceptance": acceptance,
        "frames": payload.get("frames"),
        "width": payload.get("width"),
        "pixel_ray_count": payload.get("pixel_ray_count"),
        "gradient_shape": payload.get("gradient_shape"),
        "max_rgb_abs_error": payload.get("max_rgb_abs_error"),
        "max_alpha_abs_error": payload.get("max_alpha_abs_error"),
        "max_depth_abs_error": payload.get("max_depth_abs_error"),
        "max_rgba_gradient_abs_error": payload.get("max_rgba_gradient_abs_error"),
        "finite_difference_max_abs_error": payload.get("finite_difference_max_abs_error"),
        "loss_abs_error": payload.get("loss_abs_error"),
        "mps_composite_vjp_wall_clock_ms": payload.get("mps_composite_vjp_wall_clock_ms"),
        "shared_forward_boundary_scan_ratio": payload.get("shared_forward_boundary_scan_ratio"),
    }


def _world_mps_composite_vjp_slab_mask_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate0_95_mps_composite_vjp_slab_mask_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not have a rows list")
    summary_row = max(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: int(row["time_slabs"]),
        default=None,
    )
    if summary_row is None:
        raise ValueError(f"{path} does not have any row objects")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_slab_indexed_fixed_segment_composite_vjp",
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "time_slabs": payload.get("time_slabs"),
        "frames": payload.get("frames"),
        "width": payload.get("width"),
        "row_count": len(rows),
        "rows": rows,
        "summary_time_slabs": summary_row.get("time_slabs"),
        "summary_candidate_mask_shape": summary_row.get("candidate_mask_shape"),
        "summary_total_candidates": summary_row.get("total_candidates"),
        "summary_max_candidates_per_slab": summary_row.get("max_candidates_per_slab"),
        "summary_max_segment_count": summary_row.get("max_segment_count"),
        "summary_segment_overflow_count": summary_row.get("segment_overflow_count"),
        "summary_max_rgb_abs_error": summary_row.get("max_rgb_abs_error"),
        "summary_max_alpha_abs_error": summary_row.get("max_alpha_abs_error"),
        "summary_max_depth_abs_error": summary_row.get("max_depth_abs_error"),
        "summary_max_rgba_gradient_abs_error": summary_row.get("max_rgba_gradient_abs_error"),
        "summary_finite_difference_max_abs_error": summary_row.get("finite_difference_max_abs_error"),
        "summary_loss_abs_error": summary_row.get("loss_abs_error"),
        "summary_mps_composite_vjp_wall_clock_ms": summary_row.get("mps_composite_vjp_wall_clock_ms"),
        "summary_shared_forward_boundary_scan_ratio": summary_row.get("shared_forward_boundary_scan_ratio"),
    }


def _world_mps_full_frame_vjp_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate1_mps_full_frame_vjp_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_toy_full_frame_fixed_segment_composite_vjp",
        "quality_claim": False,
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "projection_scope": payload.get("projection_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "acceptance": acceptance,
        "frames": payload.get("frames"),
        "height": payload.get("height"),
        "width": payload.get("width"),
        "time_slabs": payload.get("time_slabs"),
        "pixel_ray_count": payload.get("pixel_ray_count"),
        "candidate_mask_shape": payload.get("candidate_mask_shape"),
        "total_candidates": payload.get("total_candidates"),
        "max_candidates_per_slab": payload.get("max_candidates_per_slab"),
        "max_segment_count": payload.get("max_segment_count"),
        "segment_overflow_count": payload.get("segment_overflow_count"),
        "rgb_shape": payload.get("rgb_shape"),
        "alpha_shape": payload.get("alpha_shape"),
        "depth_shape": payload.get("depth_shape"),
        "ppm_out": payload.get("ppm_out"),
        "max_rgb_abs_error": payload.get("max_rgb_abs_error"),
        "max_alpha_abs_error": payload.get("max_alpha_abs_error"),
        "max_depth_abs_error": payload.get("max_depth_abs_error"),
        "max_rgba_gradient_abs_error": payload.get("max_rgba_gradient_abs_error"),
        "finite_difference_max_abs_error": payload.get("finite_difference_max_abs_error"),
        "loss_abs_error": payload.get("loss_abs_error"),
        "mps_full_frame_vjp_wall_clock_ms": payload.get("mps_full_frame_vjp_wall_clock_ms"),
        "shared_forward_boundary_scan_ratio": payload.get("shared_forward_boundary_scan_ratio"),
        "world_foam_renderer_status": "toy_full_frame_image_shape_only_existing_u_t_replay_op_no_true_u_v_t_camera_rays",
    }


def _world_realray_per_sample_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate1_realray_per_sample_reference.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    heldout = payload.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} must have train and heldout objects")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "cpu_real_camera_ray_per_sample_reference",
        "quality_claim": False,
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "heldout_views": payload.get("heldout_views"),
        "pose_source": payload.get("pose_source"),
        "frame_count": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        "train_rgb_shape": train.get("rgb_shape"),
        "heldout_rgb_shape": heldout.get("rgb_shape"),
        "train_pixel_ray_count": train.get("pixel_ray_count"),
        "heldout_pixel_ray_count": heldout.get("pixel_ray_count"),
        "train_linear_boundary_scans": train.get("linear_boundary_scans"),
        "heldout_linear_boundary_scans": heldout.get("linear_boundary_scans"),
        "train_render_elapsed_s": train.get("render_elapsed_s"),
        "heldout_render_elapsed_s": heldout.get("render_elapsed_s"),
        "train_target_l1": train.get("target_l1"),
        "train_target_mse": train.get("target_mse"),
        "train_target_psnr": train.get("target_psnr"),
        "heldout_target_l1": heldout.get("target_l1"),
        "heldout_target_mse": heldout.get("target_mse"),
        "heldout_target_psnr": heldout.get("target_psnr"),
        "train_max_segments_per_ray": train.get("max_segments_per_ray"),
        "heldout_max_segments_per_ray": heldout.get("max_segments_per_ray"),
        "train_invalid_denominator_count": train.get("invalid_denominator_count"),
        "heldout_invalid_denominator_count": heldout.get("invalid_denominator_count"),
        "proof_images": payload.get("proof_images"),
    }


def _world_mps_realray_replay_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate1_mps_realray_replay_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    heldout = payload.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} must have train and heldout objects")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_real_camera_ray_per_sample_forward",
        "quality_claim": False,
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "heldout_views": payload.get("heldout_views"),
        "pose_source": payload.get("pose_source"),
        "frame_count": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        "train_rgb_shape": train.get("rgb_shape"),
        "heldout_rgb_shape": heldout.get("rgb_shape"),
        "train_pixel_ray_count": train.get("pixel_ray_count"),
        "heldout_pixel_ray_count": heldout.get("pixel_ray_count"),
        "train_linear_boundary_scans": train.get("linear_boundary_scans"),
        "heldout_linear_boundary_scans": heldout.get("linear_boundary_scans"),
        "train_mps_realray_replay_wall_clock_ms": train.get("mps_realray_replay_wall_clock_ms"),
        "heldout_mps_realray_replay_wall_clock_ms": heldout.get("mps_realray_replay_wall_clock_ms"),
        "train_max_rgb_abs_error": train.get("max_rgb_abs_error"),
        "train_max_alpha_abs_error": train.get("max_alpha_abs_error"),
        "train_max_depth_abs_error": train.get("max_depth_abs_error"),
        "heldout_max_rgb_abs_error": heldout.get("max_rgb_abs_error"),
        "heldout_max_alpha_abs_error": heldout.get("max_alpha_abs_error"),
        "heldout_max_depth_abs_error": heldout.get("max_depth_abs_error"),
        "train_target_l1": train.get("target_l1"),
        "train_target_mse": train.get("target_mse"),
        "train_target_psnr": train.get("target_psnr"),
        "heldout_target_l1": heldout.get("target_l1"),
        "heldout_target_mse": heldout.get("target_mse"),
        "heldout_target_psnr": heldout.get("target_psnr"),
        "proof_images": payload.get("proof_images"),
    }


def _world_mps_shared_realray_forward_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate2_mps_shared_realray_forward_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    heldout = payload.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} must have train and heldout objects")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_real_camera_ray_time_slab_shared_forward",
        "quality_claim": False,
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "heldout_views": payload.get("heldout_views"),
        "pose_source": payload.get("pose_source"),
        "frame_counts": payload.get("frame_counts"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        "summary_frame": payload.get("frame_count"),
        "summary_per_frame_event_sum": train.get("per_frame_event_sum"),
        "summary_shared_slab_event_sum": train.get("shared_slab_event_sum"),
        "summary_event_sharing_ratio": train.get("event_sharing_ratio"),
        "summary_missing_sample_events": train.get("missing_sample_events"),
        "summary_extra_candidate_events": train.get("extra_candidate_events"),
        "summary_direct_forward_boundary_scans": train.get("direct_forward_boundary_scans"),
        "summary_shared_forward_boundary_scans": train.get("shared_forward_boundary_scans"),
        "summary_shared_forward_boundary_scan_ratio": train.get("shared_forward_boundary_scan_ratio"),
        "train_rgb_shape": train.get("rgb_shape"),
        "heldout_rgb_shape": heldout.get("rgb_shape"),
        "train_pixel_ray_count": train.get("pixel_rays"),
        "heldout_pixel_ray_count": heldout.get("pixel_rays"),
        "train_candidate_mask_shape": train.get("candidate_mask_shape"),
        "heldout_candidate_mask_shape": heldout.get("candidate_mask_shape"),
        "train_max_rgb_abs_error": train.get("max_rgb_abs_error"),
        "train_max_alpha_abs_error": train.get("max_alpha_abs_error"),
        "train_max_depth_abs_error": train.get("max_depth_abs_error"),
        "heldout_max_rgb_abs_error": heldout.get("max_rgb_abs_error"),
        "heldout_max_alpha_abs_error": heldout.get("max_alpha_abs_error"),
        "heldout_max_depth_abs_error": heldout.get("max_depth_abs_error"),
        "train_mps_shared_realray_forward_wall_clock_ms": train.get(
            "mps_shared_realray_forward_wall_clock_ms"
        ),
        "heldout_mps_shared_realray_forward_wall_clock_ms": heldout.get(
            "mps_shared_realray_forward_wall_clock_ms"
        ),
        "proof_images": payload.get("proof_images"),
    }


def _world_mps_shared_realray_vjp_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate2c_mps_shared_realray_vjp_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    heldout = payload.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} must have train and heldout objects")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_real_camera_ray_time_slab_shared_fixed_segment_vjp",
        "quality_claim": False,
        "training_claim": False,
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "heldout_views": payload.get("heldout_views"),
        "pose_source": payload.get("pose_source"),
        "frame_counts": payload.get("frame_counts"),
        "summary_frame": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        "summary_per_frame_event_sum": train.get("per_frame_event_sum"),
        "summary_shared_slab_event_sum": train.get("shared_slab_event_sum"),
        "summary_event_sharing_ratio": train.get("event_sharing_ratio"),
        "summary_missing_sample_events": train.get("missing_sample_events"),
        "summary_direct_forward_boundary_scans": train.get("direct_forward_boundary_scans"),
        "summary_shared_forward_boundary_scans": train.get("shared_forward_boundary_scans"),
        "summary_shared_forward_boundary_scan_ratio": train.get("shared_forward_boundary_scan_ratio"),
        "train_rgb_shape": train.get("rgb_shape"),
        "heldout_rgb_shape": heldout.get("rgb_shape"),
        "train_gradient_shape": train.get("gradient_shape"),
        "heldout_gradient_shape": heldout.get("gradient_shape"),
        "train_pixel_ray_count": train.get("pixel_rays"),
        "heldout_pixel_ray_count": heldout.get("pixel_rays"),
        "train_candidate_mask_shape": train.get("candidate_mask_shape"),
        "heldout_candidate_mask_shape": heldout.get("candidate_mask_shape"),
        "train_max_rgb_abs_error": train.get("max_rgb_abs_error"),
        "train_max_alpha_abs_error": train.get("max_alpha_abs_error"),
        "train_max_depth_abs_error": train.get("max_depth_abs_error"),
        "train_max_rgba_gradient_abs_error": train.get("max_rgba_gradient_abs_error"),
        "train_loss_abs_error": train.get("loss_abs_error"),
        "heldout_max_rgb_abs_error": heldout.get("max_rgb_abs_error"),
        "heldout_max_alpha_abs_error": heldout.get("max_alpha_abs_error"),
        "heldout_max_depth_abs_error": heldout.get("max_depth_abs_error"),
        "heldout_max_rgba_gradient_abs_error": heldout.get("max_rgba_gradient_abs_error"),
        "heldout_loss_abs_error": heldout.get("loss_abs_error"),
        "train_mps_shared_realray_vjp_wall_clock_ms": train.get("mps_shared_realray_vjp_wall_clock_ms"),
        "heldout_mps_shared_realray_vjp_wall_clock_ms": heldout.get("mps_shared_realray_vjp_wall_clock_ms"),
    }


def _world_mps_shared_realray_reduced_vjp_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate2d_mps_shared_realray_reduced_vjp_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    heldout = payload.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} must have train and heldout objects")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_real_camera_ray_time_slab_shared_reduced_fixed_segment_vjp",
        "quality_claim": False,
        "training_claim": False,
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "reduction_impl": payload.get("reduction_impl"),
        "reduction_chunk_size": payload.get("reduction_chunk_size"),
        "partial_reduction_materializes_chunk_site_gradients": payload.get(
            "partial_reduction_materializes_chunk_site_gradients"
        ),
        "autograd_wrapper": payload.get("autograd_wrapper"),
        "reduced_op_materializes_sample_gradients": payload.get("reduced_op_materializes_sample_gradients"),
        "oracle_materializes_sample_gradients_for_validation": payload.get(
            "oracle_materializes_sample_gradients_for_validation"
        ),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "heldout_views": payload.get("heldout_views"),
        "pose_source": payload.get("pose_source"),
        "frame_counts": payload.get("frame_counts"),
        "summary_frame": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        "summary_per_frame_event_sum": train.get("per_frame_event_sum"),
        "summary_shared_slab_event_sum": train.get("shared_slab_event_sum"),
        "summary_event_sharing_ratio": train.get("event_sharing_ratio"),
        "summary_missing_sample_events": train.get("missing_sample_events"),
        "summary_direct_forward_boundary_scans": train.get("direct_forward_boundary_scans"),
        "summary_shared_forward_boundary_scans": train.get("shared_forward_boundary_scans"),
        "summary_shared_forward_boundary_scan_ratio": train.get("shared_forward_boundary_scan_ratio"),
        "train_rgb_shape": train.get("rgb_shape"),
        "heldout_rgb_shape": heldout.get("rgb_shape"),
        "train_gradient_shape": train.get("gradient_shape"),
        "heldout_gradient_shape": heldout.get("gradient_shape"),
        "train_sample_gradient_oracle_shape": train.get("sample_gradient_oracle_shape"),
        "heldout_sample_gradient_oracle_shape": heldout.get("sample_gradient_oracle_shape"),
        "train_partial_gradient_shape": train.get("partial_gradient_shape"),
        "heldout_partial_gradient_shape": heldout.get("partial_gradient_shape"),
        "train_partial_vs_oracle_gradient_float_ratio": train.get("partial_vs_oracle_gradient_float_ratio"),
        "heldout_partial_vs_oracle_gradient_float_ratio": heldout.get("partial_vs_oracle_gradient_float_ratio"),
        "train_pixel_ray_count": train.get("pixel_rays"),
        "heldout_pixel_ray_count": heldout.get("pixel_rays"),
        "train_candidate_mask_shape": train.get("candidate_mask_shape"),
        "heldout_candidate_mask_shape": heldout.get("candidate_mask_shape"),
        "train_max_rgb_abs_error": train.get("max_rgb_abs_error"),
        "train_max_alpha_abs_error": train.get("max_alpha_abs_error"),
        "train_max_depth_abs_error": train.get("max_depth_abs_error"),
        "train_max_reduced_rgba_gradient_abs_error": train.get("max_reduced_rgba_gradient_abs_error"),
        "train_max_reduced_vs_unreduced_mps_sum_abs_error": train.get(
            "max_reduced_vs_unreduced_mps_sum_abs_error"
        ),
        "train_max_autograd_rgba_gradient_abs_error": train.get("max_autograd_rgba_gradient_abs_error"),
        "train_autograd_loss_abs_error": train.get("autograd_loss_abs_error"),
        "train_loss_abs_error": train.get("loss_abs_error"),
        "heldout_max_rgb_abs_error": heldout.get("max_rgb_abs_error"),
        "heldout_max_alpha_abs_error": heldout.get("max_alpha_abs_error"),
        "heldout_max_depth_abs_error": heldout.get("max_depth_abs_error"),
        "heldout_max_reduced_rgba_gradient_abs_error": heldout.get("max_reduced_rgba_gradient_abs_error"),
        "heldout_max_reduced_vs_unreduced_mps_sum_abs_error": heldout.get(
            "max_reduced_vs_unreduced_mps_sum_abs_error"
        ),
        "heldout_max_autograd_rgba_gradient_abs_error": heldout.get("max_autograd_rgba_gradient_abs_error"),
        "heldout_autograd_loss_abs_error": heldout.get("autograd_loss_abs_error"),
        "heldout_loss_abs_error": heldout.get("loss_abs_error"),
        "train_mps_shared_realray_reduced_vjp_wall_clock_ms": train.get(
            "mps_shared_realray_reduced_vjp_wall_clock_ms"
        ),
        "train_mps_shared_realray_autograd_backward_wall_clock_ms": train.get(
            "mps_shared_realray_autograd_backward_wall_clock_ms"
        ),
        "heldout_mps_shared_realray_reduced_vjp_wall_clock_ms": heldout.get(
            "mps_shared_realray_reduced_vjp_wall_clock_ms"
        ),
        "heldout_mps_shared_realray_autograd_backward_wall_clock_ms": heldout.get(
            "mps_shared_realray_autograd_backward_wall_clock_ms"
        ),
    }


def _world_mps_shared_realray_csr_candidate_storage_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate2f_mps_shared_realray_csr_candidate_storage_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    heldout = payload.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} must have train and heldout objects")

    def split(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
        per_track = row.get("per_track_csr")
        tiled = row.get("tiled_csr")
        per_track_valid = row.get("per_track_csr_valid")
        tiled_valid = row.get("tiled_csr_valid")
        if not isinstance(per_track, dict) or not isinstance(tiled, dict):
            raise ValueError(f"{path} {prefix} must have per_track_csr and tiled_csr objects")
        if not isinstance(per_track_valid, dict) or not isinstance(tiled_valid, dict):
            raise ValueError(f"{path} {prefix} must have CSR validity objects")
        return {
            f"{prefix}_rgb_shape": row.get("rgb_shape"),
            f"{prefix}_pixel_ray_count": row.get("pixel_rays"),
            f"{prefix}_pixel_track_count": row.get("pixel_tracks"),
            f"{prefix}_candidate_mask_shape": row.get("candidate_mask_shape"),
            f"{prefix}_mask_word_count": row.get("mask_word_count"),
            f"{prefix}_bitset_storage_bytes": row.get("bitset_storage_bytes"),
            f"{prefix}_partial_gradient_bytes": row.get("partial_gradient_bytes"),
            f"{prefix}_reduced_gradient_bytes": row.get("reduced_gradient_bytes"),
            f"{prefix}_direct_forward_boundary_scans": row.get("direct_forward_boundary_scans"),
            f"{prefix}_shared_forward_boundary_scans": row.get("shared_forward_boundary_scans"),
            f"{prefix}_shared_forward_boundary_scan_ratio": row.get("shared_forward_boundary_scan_ratio"),
            f"{prefix}_per_frame_event_sum": row.get("per_frame_event_sum"),
            f"{prefix}_shared_slab_event_sum": row.get("shared_slab_event_sum"),
            f"{prefix}_event_sharing_ratio": row.get("event_sharing_ratio"),
            f"{prefix}_missing_sample_events": row.get("missing_sample_events"),
            f"{prefix}_extra_candidate_events": row.get("extra_candidate_events"),
            f"{prefix}_max_bitset_rgb_abs_error_vs_cpu": row.get("max_bitset_rgb_abs_error_vs_cpu"),
            f"{prefix}_per_track_csr_valid": per_track_valid,
            f"{prefix}_per_track_csr_row_count": per_track.get("row_count"),
            f"{prefix}_per_track_csr_row_offsets_shape": per_track.get("row_offsets_shape"),
            f"{prefix}_per_track_csr_candidate_ids_shape": per_track.get("candidate_ids_shape"),
            f"{prefix}_per_track_csr_candidate_count": per_track.get("candidate_count"),
            f"{prefix}_per_track_csr_max_candidates_per_row": per_track.get("max_candidates_per_row"),
            f"{prefix}_per_track_csr_empty_row_count": per_track.get("empty_row_count"),
            f"{prefix}_per_track_csr_storage_bytes": per_track.get("storage_bytes"),
            f"{prefix}_per_track_csr_storage_vs_bitset_ratio": per_track.get("storage_vs_bitset_ratio"),
            f"{prefix}_per_track_csr_candidate_iterations": per_track.get("candidate_iterations"),
            f"{prefix}_per_track_csr_exact_candidate_sets": per_track.get("exact_candidate_sets"),
            f"{prefix}_per_track_csr_superset_candidate_sets": per_track.get("superset_candidate_sets"),
            f"{prefix}_per_track_csr_extra_candidate_refs_vs_per_track_slab": per_track.get(
                "extra_candidate_refs_vs_per_track_slab"
            ),
            f"{prefix}_tiled_csr_valid": tiled_valid,
            f"{prefix}_tiled_csr_row_count": tiled.get("row_count"),
            f"{prefix}_tiled_csr_row_offsets_shape": tiled.get("row_offsets_shape"),
            f"{prefix}_tiled_csr_candidate_ids_shape": tiled.get("candidate_ids_shape"),
            f"{prefix}_tiled_csr_candidate_count": tiled.get("candidate_count"),
            f"{prefix}_tiled_csr_max_candidates_per_row": tiled.get("max_candidates_per_row"),
            f"{prefix}_tiled_csr_empty_row_count": tiled.get("empty_row_count"),
            f"{prefix}_tiled_csr_storage_bytes": tiled.get("storage_bytes"),
            f"{prefix}_tiled_csr_storage_vs_bitset_ratio": tiled.get("storage_vs_bitset_ratio"),
            f"{prefix}_tiled_csr_candidate_iterations": tiled.get("candidate_iterations"),
            f"{prefix}_tiled_csr_exact_candidate_sets": tiled.get("exact_candidate_sets"),
            f"{prefix}_tiled_csr_superset_candidate_sets": tiled.get("superset_candidate_sets"),
            f"{prefix}_tiled_csr_extra_candidate_refs_vs_per_track_slab": tiled.get(
                "extra_candidate_refs_vs_per_track_slab"
            ),
            f"{prefix}_per_track_max_csr_vs_bitset_rgb_abs_error": row.get(
                "per_track_max_csr_vs_bitset_rgb_abs_error"
            ),
            f"{prefix}_per_track_max_csr_vs_bitset_alpha_abs_error": row.get(
                "per_track_max_csr_vs_bitset_alpha_abs_error"
            ),
            f"{prefix}_per_track_max_csr_vs_bitset_depth_abs_error": row.get(
                "per_track_max_csr_vs_bitset_depth_abs_error"
            ),
            f"{prefix}_per_track_max_csr_vs_bitset_rgba_gradient_abs_error": row.get(
                "per_track_max_csr_vs_bitset_rgba_gradient_abs_error"
            ),
            f"{prefix}_tiled_max_csr_vs_bitset_rgb_abs_error": row.get(
                "tiled_max_csr_vs_bitset_rgb_abs_error"
            ),
            f"{prefix}_tiled_max_csr_vs_bitset_alpha_abs_error": row.get(
                "tiled_max_csr_vs_bitset_alpha_abs_error"
            ),
            f"{prefix}_tiled_max_csr_vs_bitset_depth_abs_error": row.get(
                "tiled_max_csr_vs_bitset_depth_abs_error"
            ),
            f"{prefix}_tiled_max_csr_vs_bitset_rgba_gradient_abs_error": row.get(
                "tiled_max_csr_vs_bitset_rgba_gradient_abs_error"
            ),
            f"{prefix}_per_track_mps_shared_realray_csr_reduced_vjp_wall_clock_ms": row.get(
                "per_track_mps_shared_realray_csr_reduced_vjp_wall_clock_ms"
            ),
            f"{prefix}_tiled_mps_shared_realray_csr_reduced_vjp_wall_clock_ms": row.get(
                "tiled_mps_shared_realray_csr_reduced_vjp_wall_clock_ms"
            ),
        }

    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_real_camera_ray_time_slab_shared_csr_candidate_storage",
        "quality_claim": False,
        "training_claim": False,
        "csr_candidate_storage_claim": payload.get("csr_candidate_storage_claim"),
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "candidate_storage_format": payload.get("candidate_storage_format"),
        "bitset_reference_storage_format": payload.get("bitset_reference_storage_format"),
        "csr_offset_dtype": payload.get("csr_offset_dtype"),
        "csr_index_dtype": payload.get("csr_index_dtype"),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "heldout_views": payload.get("heldout_views"),
        "pose_source": payload.get("pose_source"),
        "frame_counts": payload.get("frame_counts"),
        "summary_frame": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "tile_shape": payload.get("tile_shape"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        **split("train", train),
        **split("heldout", heldout),
    }


def _world_mps_shared_realray_csr_scaling_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate2g_mps_shared_realray_csr_scaling_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not have nonempty rows")
    summary_row = rows[-1]
    if not isinstance(summary_row, dict):
        raise ValueError(f"{path} summary row must be an object")
    train = summary_row.get("train")
    heldout = summary_row.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} summary row must have train and heldout objects")

    def split(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
        tiled = row.get("tiled_csr")
        per_track = row.get("per_track_csr")
        if not isinstance(tiled, dict) or not isinstance(per_track, dict):
            raise ValueError(f"{path} {prefix} must have tiled_csr and per_track_csr objects")
        return {
            f"{prefix}_pixel_tracks": row.get("pixel_tracks"),
            f"{prefix}_pixel_rays": row.get("pixel_rays"),
            f"{prefix}_bitset_storage_bytes": row.get("bitset_storage_bytes"),
            f"{prefix}_tiled_csr_storage_bytes": tiled.get("storage_bytes"),
            f"{prefix}_tiled_csr_storage_vs_bitset_ratio": tiled.get("storage_vs_bitset_ratio"),
            f"{prefix}_per_track_csr_storage_vs_bitset_ratio": per_track.get("storage_vs_bitset_ratio"),
            f"{prefix}_tiled_csr_candidate_iterations": tiled.get("candidate_iterations"),
            f"{prefix}_tiled_candidate_iteration_vs_direct_scan_ratio": row.get(
                "tiled_candidate_iteration_vs_direct_scan_ratio"
            ),
            f"{prefix}_direct_forward_boundary_scans": row.get("direct_forward_boundary_scans"),
            f"{prefix}_shared_forward_boundary_scans": row.get("shared_forward_boundary_scans"),
            f"{prefix}_shared_forward_boundary_scan_ratio": row.get("shared_forward_boundary_scan_ratio"),
            f"{prefix}_missing_sample_events": row.get("missing_sample_events"),
            f"{prefix}_event_sharing_ratio": row.get("event_sharing_ratio"),
            f"{prefix}_tiled_max_csr_vs_bitset_rgb_abs_error": row.get(
                "tiled_max_csr_vs_bitset_rgb_abs_error"
            ),
            f"{prefix}_tiled_max_csr_vs_bitset_rgba_gradient_abs_error": row.get(
                "tiled_max_csr_vs_bitset_rgba_gradient_abs_error"
            ),
            f"{prefix}_tiled_mps_shared_realray_csr_reduced_vjp_wall_clock_ms": row.get(
                "tiled_mps_shared_realray_csr_reduced_vjp_wall_clock_ms"
            ),
        }

    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": payload.get("comparison_unit"),
        "quality_claim": payload.get("quality_claim"),
        "training_claim": payload.get("training_claim"),
        "large_scale_claim": payload.get("large_scale_claim"),
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "frame_counts": payload.get("frame_counts"),
        "summary_frame": summary_row.get("frames"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "tile_shape": payload.get("tile_shape"),
        "site_count": payload.get("site_count"),
        "acceptance": acceptance,
        "growth": payload.get("growth"),
        "max_tiled_csr_vs_bitset_mps_error": payload.get("max_tiled_csr_vs_bitset_mps_error"),
        "timing_iters": payload.get("timing_iters"),
        **split("train", train),
        **split("heldout", heldout),
    }


def _world_mps_shared_realray_autograd_overfit_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate2e_mps_shared_realray_autograd_overfit_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    if not isinstance(train, dict):
        raise ValueError(f"{path} must have a train object")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_real_camera_ray_frozen_geometry_site_rgba_teacher_overfit",
        "quality_claim": False,
        "trainer_claim": False,
        "parameter_update_claim": payload.get("parameter_update_claim"),
        "teacher_target_claim": payload.get("teacher_target_claim"),
        "real_target_training_claim": payload.get("real_target_training_claim"),
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "autograd_wrapper": payload.get("autograd_wrapper"),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "pose_source": payload.get("pose_source"),
        "frame_counts": payload.get("frame_counts"),
        "summary_frame": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        "steps": payload.get("steps"),
        "lr": payload.get("lr"),
        "initial_loss": payload.get("initial_loss"),
        "final_loss": payload.get("final_loss"),
        "loss_ratio": payload.get("loss_ratio"),
        "initial_mean_abs_rgba_error_to_teacher": payload.get("initial_mean_abs_rgba_error_to_teacher"),
        "final_mean_abs_rgba_error_to_teacher": payload.get("final_mean_abs_rgba_error_to_teacher"),
        "parameter_update_abs_max": payload.get("parameter_update_abs_max"),
        "first_grad_abs_sum": payload.get("first_grad_abs_sum"),
        "first_grad_abs_max": payload.get("first_grad_abs_max"),
        "elapsed_s": payload.get("elapsed_s"),
        "train_rgb_shape": train.get("rgb_shape"),
        "train_target_rgb_shape": train.get("target_rgb_shape"),
        "train_pixel_ray_count": train.get("pixel_rays"),
        "train_candidate_mask_shape": train.get("candidate_mask_shape"),
        "summary_per_frame_event_sum": train.get("per_frame_event_sum"),
        "summary_shared_slab_event_sum": train.get("shared_slab_event_sum"),
        "summary_event_sharing_ratio": train.get("event_sharing_ratio"),
        "summary_missing_sample_events": train.get("missing_sample_events"),
        "summary_direct_forward_boundary_scans": train.get("direct_forward_boundary_scans"),
        "summary_shared_forward_boundary_scans": train.get("shared_forward_boundary_scans"),
        "summary_shared_forward_boundary_scan_ratio": train.get("shared_forward_boundary_scan_ratio"),
        "train_max_final_rgb_abs_error_to_teacher": train.get("max_final_rgb_abs_error_to_teacher"),
        "train_max_final_alpha_abs_error_to_teacher": train.get("max_final_alpha_abs_error_to_teacher"),
        "train_max_final_depth_abs_error_to_teacher": train.get("max_final_depth_abs_error_to_teacher"),
    }


def _world_mps_shared_realray_real_target_train_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate3_mps_shared_realray_real_target_train_smoke.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    train = payload.get("train")
    if not isinstance(train, dict):
        raise ValueError(f"{path} must have a train object")
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": "mps_real_camera_ray_frozen_geometry_site_rgba_real_target_train",
        "quality_claim": False,
        "real_target_training_smoke_claim": payload.get("real_target_training_smoke_claim"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "parameter_update_claim": payload.get("parameter_update_claim"),
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "autograd_wrapper": payload.get("autograd_wrapper"),
        "sample_id": payload.get("sample_id"),
        "train_views": payload.get("train_views"),
        "pose_source": payload.get("pose_source"),
        "frame_counts": payload.get("frame_counts"),
        "summary_frame": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "acceptance": acceptance,
        "steps": payload.get("steps"),
        "lr": payload.get("lr"),
        "initial_rgb_mse": payload.get("initial_rgb_mse"),
        "final_rgb_mse": payload.get("final_rgb_mse"),
        "loss_ratio": payload.get("loss_ratio"),
        "initial_train_psnr": payload.get("initial_train_psnr"),
        "final_train_psnr": payload.get("final_train_psnr"),
        "train_psnr_delta": payload.get("train_psnr_delta"),
        "parameter_update_abs_max": payload.get("parameter_update_abs_max"),
        "first_grad_abs_sum": payload.get("first_grad_abs_sum"),
        "first_grad_abs_max": payload.get("first_grad_abs_max"),
        "elapsed_s": payload.get("elapsed_s"),
        "train_rgb_shape": train.get("rgb_shape"),
        "train_target_rgb_shape": train.get("target_rgb_shape"),
        "train_pixel_ray_count": train.get("pixel_rays"),
        "train_candidate_mask_shape": train.get("candidate_mask_shape"),
        "summary_per_frame_event_sum": train.get("per_frame_event_sum"),
        "summary_shared_slab_event_sum": train.get("shared_slab_event_sum"),
        "summary_event_sharing_ratio": train.get("event_sharing_ratio"),
        "summary_missing_sample_events": train.get("missing_sample_events"),
        "summary_direct_forward_boundary_scans": train.get("direct_forward_boundary_scans"),
        "summary_shared_forward_boundary_scans": train.get("shared_forward_boundary_scans"),
        "summary_shared_forward_boundary_scan_ratio": train.get("shared_forward_boundary_scan_ratio"),
        "train_max_final_rgb_abs_error_to_target": train.get("max_final_rgb_abs_error_to_target"),
        "train_final_alpha_min": train.get("final_alpha_min"),
        "train_final_alpha_max": train.get("final_alpha_max"),
        "train_final_depth_min": train.get("final_depth_min"),
        "train_final_depth_max": train.get("final_depth_max"),
    }


def _world_mps_shared_realray_csr_quality_256px_16f_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate3_mps_shared_realray_csr_quality_256px_16f.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} does not have a metrics object")
    train = payload.get("train")
    heldout = payload.get("heldout")
    if not isinstance(train, dict) or not isinstance(heldout, dict):
        raise ValueError(f"{path} must have train and heldout objects")
    train_tiled = train.get("tiled_csr")
    train_tiled = train_tiled if isinstance(train_tiled, dict) else {}
    heldout_tiled = heldout.get("tiled_csr")
    heldout_tiled = heldout_tiled if isinstance(heldout_tiled, dict) else {}
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": payload.get("comparison_unit"),
        "quality_claim": payload.get("quality_claim"),
        "heldout_quality_metric_claim": payload.get("heldout_quality_metric_claim"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "renderer_scope": payload.get("renderer_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "sharing_scope": payload.get("sharing_scope"),
        "world_foam_renderer_status": payload.get("world_foam_renderer_status"),
        "autograd_wrapper": payload.get("autograd_wrapper"),
        "sample_id": payload.get("sample_id"),
        "pose_source": payload.get("pose_source"),
        "train_views": payload.get("train_views"),
        "heldout_views": payload.get("heldout_views"),
        "frames": payload.get("frame_count"),
        "frame_counts": payload.get("frame_counts"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "tile_shape": payload.get("tile_shape"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "steps": payload.get("steps"),
        "lr": payload.get("lr"),
        "train_loop_elapsed_s": payload.get("train_loop_elapsed_s"),
        "total_elapsed_s": payload.get("total_elapsed_s"),
        "eval_l1": metrics.get("eval_l1"),
        "eval_mse": metrics.get("eval_mse"),
        "eval_psnr": metrics.get("eval_psnr"),
        "eval_ssim": metrics.get("eval_ssim"),
        "heldout_eval_l1": metrics.get("heldout_eval_l1"),
        "heldout_eval_mse": metrics.get("heldout_eval_mse"),
        "heldout_eval_psnr": metrics.get("heldout_eval_psnr"),
        "heldout_eval_ssim": metrics.get("heldout_eval_ssim"),
        "eval_render_only_elapsed_s": metrics.get("eval_render_only_elapsed_s"),
        "eval_train_render_only_elapsed_s": metrics.get("eval_train_render_only_elapsed_s"),
        "eval_heldout_render_only_elapsed_s": metrics.get("eval_heldout_render_only_elapsed_s"),
        "selected_step": payload.get("selected_step"),
        "selected_uses_heldout_for_selection": payload.get("selected_uses_heldout_for_selection"),
        "selection_metric": payload.get("selection_metric"),
        "initial_train_mse": payload.get("initial_train_mse"),
        "final_train_mse": payload.get("final_train_mse"),
        "initial_train_psnr": payload.get("initial_train_psnr"),
        "final_train_psnr": payload.get("final_train_psnr"),
        "final_heldout_psnr": payload.get("final_heldout_psnr"),
        "train_psnr_delta": payload.get("train_psnr_delta"),
        "loss_ratio": payload.get("loss_ratio"),
        "parameter_update_abs_max": payload.get("parameter_update_abs_max"),
        "first_grad_abs_sum": payload.get("first_grad_abs_sum"),
        "first_grad_abs_max": payload.get("first_grad_abs_max"),
        "train_rgb_shape": train.get("rgb_shape"),
        "heldout_rgb_shape": heldout.get("rgb_shape"),
        "train_pixel_ray_count": train.get("pixel_rays"),
        "heldout_pixel_ray_count": heldout.get("pixel_rays"),
        "train_shared_forward_boundary_scan_ratio": train.get("shared_forward_boundary_scan_ratio"),
        "heldout_shared_forward_boundary_scan_ratio": heldout.get("shared_forward_boundary_scan_ratio"),
        "train_tiled_candidate_iteration_vs_direct_scan_ratio": train.get(
            "tiled_candidate_iteration_vs_direct_scan_ratio"
        ),
        "heldout_tiled_candidate_iteration_vs_direct_scan_ratio": heldout.get(
            "tiled_candidate_iteration_vs_direct_scan_ratio"
        ),
        "train_tiled_csr_storage_vs_bitset_ratio": train_tiled.get("storage_vs_bitset_ratio"),
        "heldout_tiled_csr_storage_vs_bitset_ratio": heldout_tiled.get("storage_vs_bitset_ratio"),
        "train_missing_sample_events": train.get("missing_sample_events"),
        "heldout_missing_sample_events": heldout.get("missing_sample_events"),
        "target_shape": payload.get("target_shape"),
        "acceptance": acceptance,
        "proof_images": payload.get("proof_images"),
        "ssim_method": payload.get("ssim_method"),
    }


def _world_gate2_realray_event_sharing_summary() -> dict[str, Any] | None:
    path = WORLD_FOAM_DIR / "results" / "gate2_realray_event_sharing.json"
    if not path.exists():
        return None
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not have a rows list")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"{path} does not have an acceptance object")
    summary_row = max((row for row in rows if isinstance(row, dict)), key=lambda row: int(row["frames"]), default=None)
    if summary_row is None:
        raise ValueError(f"{path} does not have any row objects")
    growth = payload.get("growth")
    growth = growth if isinstance(growth, dict) else {}
    return {
        "source": str(path.relative_to(ROOT)),
        "comparison_unit": payload.get("comparison_unit"),
        "quality_claim": False,
        "status": payload.get("status"),
        "device": payload.get("device"),
        "gate": payload.get("gate"),
        "sharing_scope": payload.get("sharing_scope"),
        "gradient_scope": payload.get("gradient_scope"),
        "frame_counts": payload.get("frame_counts"),
        "render_size": payload.get("render_size"),
        "time_slabs": payload.get("time_slabs"),
        "site_count": payload.get("site_count"),
        "acceptance": acceptance,
        "growth": growth,
        "summary_frame": summary_row.get("frames"),
        "summary_per_frame_event_sum": summary_row.get("per_frame_event_sum"),
        "summary_shared_slab_event_sum": summary_row.get("shared_slab_event_sum"),
        "summary_event_sharing_ratio": summary_row.get("event_sharing_ratio"),
        "summary_missing_sample_events": summary_row.get("missing_sample_events"),
        "summary_extra_candidate_events": summary_row.get("extra_candidate_events"),
        "summary_direct_forward_boundary_scans": summary_row.get("direct_forward_boundary_scans"),
        "summary_shared_forward_boundary_scans": summary_row.get("shared_forward_boundary_scans"),
        "summary_shared_forward_boundary_scan_ratio": summary_row.get("shared_forward_boundary_scan_ratio"),
        "rows": rows,
    }


def _extract_star_rows_from_pair_json(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not have a rows list")
    return [row for row in rows if isinstance(row, dict)]


def _extract_star_rows_from_comparison(payload: dict[str, Any]) -> list[dict[str, Any]]:
    star = payload.get("star_uvt")
    if not isinstance(star, dict):
        return []
    metal_stats = star.get("metal_stats")
    if not isinstance(metal_stats, dict):
        return []
    rows = metal_stats.get("rows")
    if not isinstance(rows, list):
        return []
    extracted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        stats = row.get("stats")
        if not isinstance(stats, dict):
            continue
        extracted.append({**stats, "camera": row.get("camera"), "split": row.get("split")})
    return extracted


def _star_summary(rows: list[dict[str, Any]], *, source: str) -> dict[str, Any] | None:
    if not rows:
        return None
    uvt_pairs = _sum_field(rows, "uvt_tile_tube_pairs")
    per_frame_pairs = _sum_field(rows, "summed_per_frame_tile_splat_pairs")
    pair_ratio = float(uvt_pairs) / float(per_frame_pairs) if uvt_pairs is not None and per_frame_pairs else None
    return {
        "source": source,
        "row_count": len(rows),
        "comparison_unit": "uvt_tile_tube_pairs",
        "uvt_tile_tube_pairs": uvt_pairs,
        "summed_per_frame_tile_splat_pairs": per_frame_pairs,
        "pair_ratio": pair_ratio if pair_ratio is not None else _mean_field(rows, "pair_ratio"),
        "mean_pair_ratio": _mean_field(rows, "pair_ratio"),
        "mean_effective_pair_ratio_after_unstable_fallback": _mean_field(
            rows, "effective_pair_ratio_after_unstable_fallback"
        ),
        "mean_stable_tile_fraction": _mean_field(rows, "stable_tile_fraction"),
        "mean_unstable_tile_fraction": _mean_field(rows, "unstable_tile_fraction"),
        "overflow_tile_count": _sum_field(rows, "overflow_tile_count"),
        "mean_forward_wall_clock_ms": _mean_field(rows, "forward_wall_clock_ms"),
        "max_rgb_error": max(
            (float(row["max_rgb_error"]) for row in rows if _num(row.get("max_rgb_error")) is not None),
            default=None,
        ),
    }


def _star_quality_summary(comparison_payload: dict[str, Any] | None) -> dict[str, Any]:
    if comparison_payload is None:
        return {}
    star = comparison_payload.get("star_uvt")
    if not isinstance(star, dict):
        return {}
    metrics = star.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    selected = comparison_payload.get("star_uvt_selected")
    selected = selected if isinstance(selected, dict) else {}
    selected_metrics = selected.get("metrics")
    selected_metrics = selected_metrics if isinstance(selected_metrics, dict) else {}
    quality: dict[str, Any] = {
        "steps": star.get("steps"),
        "train_loop_elapsed_s": star.get("train_loop_elapsed_s"),
        "render_backend": star.get("render_backend"),
        "tube_count": star.get("tube_count"),
        "eval_psnr": metrics.get("eval_psnr"),
        "heldout_eval_psnr": metrics.get("heldout_eval_psnr"),
        "eval_ssim": metrics.get("eval_ssim"),
        "heldout_eval_ssim": metrics.get("heldout_eval_ssim"),
        "eval_render_only_elapsed_s": metrics.get("eval_render_only_elapsed_s"),
        "eval_train_render_only_elapsed_s": metrics.get("eval_train_render_only_elapsed_s"),
        "eval_heldout_render_only_elapsed_s": metrics.get("eval_heldout_render_only_elapsed_s"),
    }
    if selected_metrics:
        quality.update(
            {
                "selected_heldout_eval_psnr": selected_metrics.get("heldout_eval_psnr"),
                "selected_eval_psnr": selected_metrics.get("eval_psnr"),
                "selected_eval_render_only_elapsed_s": selected_metrics.get("eval_render_only_elapsed_s"),
                "selected_uses_heldout_for_selection": selected.get("uses_heldout_for_selection"),
                "selected_step": selected.get("selected_step"),
            }
        )
    return quality


def _dynamic_summary(comparison_payload: dict[str, Any]) -> dict[str, Any] | None:
    free = comparison_payload.get("free_dynamic_splats")
    if not isinstance(free, dict):
        return None
    meta = comparison_payload.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    metrics = free.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    max_frames = int(meta["max_frames"]) if isinstance(meta.get("max_frames"), int) else None
    splat_count = int(free["splat_count"]) if isinstance(free.get("splat_count"), int) else None
    total_splats = splat_count * max_frames if splat_count is not None and max_frames is not None else None
    return {
        "source": "star_comparison_json.free_dynamic_splats",
        "comparison_unit": "per_frame_dynamic_splats",
        "renderer": free.get("renderer"),
        "splat_count": splat_count,
        "frames": max_frames,
        "total_splats": total_splats,
        "steps": free.get("steps"),
        "train_loop_elapsed_s": free.get("train_loop_elapsed_s"),
        "eval_psnr": metrics.get("eval_psnr"),
        "heldout_eval_psnr": metrics.get("heldout_eval_psnr"),
        "eval_render_only_elapsed_s": metrics.get("eval_render_only_elapsed_s"),
        "eval_train_render_only_elapsed_s": metrics.get("eval_train_render_only_elapsed_s"),
        "eval_heldout_render_only_elapsed_s": metrics.get("eval_heldout_render_only_elapsed_s"),
    }


def _paired_status(comparison_payload: dict[str, Any] | None) -> dict[str, Any]:
    if comparison_payload is None:
        return {
            "paired": False,
            "reason": "no STAR comparison report supplied",
        }
    meta = comparison_payload.get("meta")
    star = comparison_payload.get("star_uvt")
    free = comparison_payload.get("free_dynamic_splats")
    paired = isinstance(meta, dict) and isinstance(star, dict) and isinstance(free, dict)
    return {
        "paired": paired,
        "reason": "same comparison_report.json includes meta, star_uvt, and free_dynamic_splats"
        if paired
        else "comparison report lacks one of meta/star_uvt/free_dynamic_splats",
        "target_size": meta.get("target_size") if isinstance(meta, dict) else None,
        "max_frames": meta.get("max_frames") if isinstance(meta, dict) else None,
        "device": meta.get("device") if isinstance(meta, dict) else None,
        "train_cameras": meta.get("train_cameras") if isinstance(meta, dict) else None,
        "heldout_cameras": meta.get("heldout_cameras") if isinstance(meta, dict) else None,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    world_payload = build_world_foam_payload(args)
    world_rows, world_summary = _world_rows(world_payload, args.summary_frame)
    world_gradient_rows, world_gradient_summary = _world_gradient_reference(args)
    world_mps = _world_mps_summary()
    world_mps_shared_replay = _world_mps_shared_replay_summary(args.summary_frame)
    world_mps_rgb_strip = _world_mps_rgb_strip_summary()
    world_mps_composite_strip = _world_mps_composite_strip_summary()
    world_mps_composite_vjp = _world_mps_composite_vjp_summary()
    world_mps_composite_vjp_slab_mask = _world_mps_composite_vjp_slab_mask_summary()
    world_mps_full_frame_vjp = _world_mps_full_frame_vjp_summary()
    world_realray_per_sample = _world_realray_per_sample_summary()
    world_mps_realray_replay = _world_mps_realray_replay_summary()
    world_mps_shared_realray_forward = _world_mps_shared_realray_forward_summary()
    world_mps_shared_realray_vjp = _world_mps_shared_realray_vjp_summary()
    world_mps_shared_realray_reduced_vjp = _world_mps_shared_realray_reduced_vjp_summary()
    world_mps_shared_realray_csr_candidate_storage = (
        _world_mps_shared_realray_csr_candidate_storage_summary()
    )
    world_mps_shared_realray_csr_scaling = _world_mps_shared_realray_csr_scaling_summary()
    world_mps_shared_realray_autograd_overfit = _world_mps_shared_realray_autograd_overfit_summary()
    world_mps_shared_realray_real_target_train = _world_mps_shared_realray_real_target_train_summary()
    world_mps_shared_realray_csr_quality_256px_16f = (
        _world_mps_shared_realray_csr_quality_256px_16f_summary()
    )
    world_gate2_realray_event_sharing = _world_gate2_realray_event_sharing_summary()

    comparison_payload = _read_json(args.star_comparison_json) if args.star_comparison_json else None
    star_rows: list[dict[str, Any]] = []
    star_source = "none"
    if comparison_payload is not None:
        star_rows = _extract_star_rows_from_comparison(comparison_payload)
        star_source = str(args.star_comparison_json)
    if args.star_pair_json is not None:
        star_rows = _extract_star_rows_from_pair_json(args.star_pair_json)
        star_source = str(args.star_pair_json)

    star = _star_summary(star_rows, source=star_source)
    star_quality = _star_quality_summary(comparison_payload)
    if star is not None:
        star = {**star, **star_quality}
    dynamic = _dynamic_summary(comparison_payload) if comparison_payload is not None else None
    paired = _paired_status(comparison_payload)

    rows = [*world_rows, *world_gradient_rows]
    if world_mps is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_power_boundary_smoke",
                "method": "beam_foam_mps_count",
                **world_mps,
            }
        )
    if world_mps_shared_replay is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_replay_smoke",
                "method": "beam_foam_mps_shared_forward_backward_replay",
                **world_mps_shared_replay,
            }
        )
    if world_mps_rgb_strip is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_rgb_strip_smoke",
                "method": "beam_foam_mps_rgb_strip",
                **world_mps_rgb_strip,
            }
        )
    if world_mps_composite_strip is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_composite_strip_smoke",
                "method": "beam_foam_mps_composite_strip",
                **world_mps_composite_strip,
            }
        )
    if world_mps_composite_vjp is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_composite_vjp_smoke",
                "method": "beam_foam_mps_composite_vjp",
                **world_mps_composite_vjp,
            }
        )
    if world_mps_composite_vjp_slab_mask is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_composite_vjp_slab_mask_smoke",
                "method": "beam_foam_mps_composite_vjp_slab_mask",
                **world_mps_composite_vjp_slab_mask,
            }
        )
    if world_mps_full_frame_vjp is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_full_frame_vjp_smoke",
                "method": "beam_foam_mps_full_frame_vjp_toy",
                **world_mps_full_frame_vjp,
            }
        )
    if world_realray_per_sample is not None:
        rows.append(
            {
                "row_id": "world_foam_realray_per_sample_reference",
                "method": "world_foam_cpu_realray_per_sample_reference",
                **world_realray_per_sample,
            }
        )
    if world_mps_realray_replay is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_realray_replay_smoke",
                "method": "world_foam_mps_realray_per_sample_forward",
                **world_mps_realray_replay,
            }
        )
    if world_gate2_realray_event_sharing is not None:
        rows.append(
            {
                "row_id": "world_foam_gate2_realray_event_sharing",
                "method": "world_foam_cpu_realray_time_slab_sharing",
                **world_gate2_realray_event_sharing,
            }
        )
    if world_mps_shared_realray_forward is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_forward_smoke",
                "method": "world_foam_mps_realray_time_slab_shared_forward",
                **world_mps_shared_realray_forward,
            }
        )
    if world_mps_shared_realray_vjp is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_vjp_smoke",
                "method": "world_foam_mps_realray_time_slab_shared_fixed_segment_vjp",
                **world_mps_shared_realray_vjp,
            }
        )
    if world_mps_shared_realray_reduced_vjp is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_reduced_vjp_smoke",
                "method": "world_foam_mps_realray_time_slab_shared_reduced_fixed_segment_vjp",
                **world_mps_shared_realray_reduced_vjp,
            }
        )
    if world_mps_shared_realray_csr_candidate_storage is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_csr_candidate_storage_smoke",
                "method": "world_foam_mps_realray_time_slab_shared_csr_candidate_storage",
                **world_mps_shared_realray_csr_candidate_storage,
            }
        )
    if world_mps_shared_realray_csr_scaling is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_csr_scaling_smoke",
                "method": "world_foam_mps_realray_time_slab_shared_csr_scaling",
                **world_mps_shared_realray_csr_scaling,
            }
        )
    if world_mps_shared_realray_autograd_overfit is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_autograd_overfit_smoke",
                "method": "world_foam_mps_realray_frozen_geometry_site_rgba_teacher_overfit",
                **world_mps_shared_realray_autograd_overfit,
            }
        )
    if world_mps_shared_realray_real_target_train is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_real_target_train_smoke",
                "method": "world_foam_mps_realray_frozen_geometry_site_rgba_real_target_train",
                **world_mps_shared_realray_real_target_train,
            }
        )
    if world_mps_shared_realray_csr_quality_256px_16f is not None:
        rows.append(
            {
                "row_id": "world_foam_mps_shared_realray_csr_quality_256px_16f",
                "method": "world_foam_mps_shared_realray_csr_quality",
                **world_mps_shared_realray_csr_quality_256px_16f,
            }
        )
    if star is not None:
        rows.append({"row_id": "star_uvt_pair", "method": "star_uvt", **star})
    if dynamic is not None:
        rows.append({"row_id": "dynamic_splat_baseline", "method": "dynamic_splats", **dynamic})

    return {
        "benchmark": "world_foam_lane2_paired_report",
        "schema_version": 1,
        "status": "ok",
        "meta": {
            "frame_counts": list(world_payload["world_foam_rows"][i]["frames"] for i in range(len(world_payload["world_foam_rows"]))),
            "summary_frame": args.summary_frame,
            "world_foam_status": world_payload["status"],
            "world_foam_backward_status": world_payload["backward_status"],
            "world_foam_gradient_reference_status": world_gradient_summary["status"],
            "world_foam_mps_shared_replay_status": world_mps_shared_replay["status"]
            if world_mps_shared_replay is not None
            else None,
            "world_foam_mps_rgb_strip_status": world_mps_rgb_strip["status"]
            if world_mps_rgb_strip is not None
            else None,
            "world_foam_mps_composite_strip_status": world_mps_composite_strip["status"]
            if world_mps_composite_strip is not None
            else None,
            "world_foam_mps_composite_vjp_status": world_mps_composite_vjp["status"]
            if world_mps_composite_vjp is not None
            else None,
            "world_foam_mps_composite_vjp_slab_mask_status": world_mps_composite_vjp_slab_mask["status"]
            if world_mps_composite_vjp_slab_mask is not None
            else None,
            "world_foam_mps_full_frame_vjp_status": world_mps_full_frame_vjp["status"]
            if world_mps_full_frame_vjp is not None
            else None,
            "world_foam_realray_per_sample_status": world_realray_per_sample["status"]
            if world_realray_per_sample is not None
            else None,
            "world_foam_mps_realray_replay_status": world_mps_realray_replay["status"]
            if world_mps_realray_replay is not None
            else None,
            "world_foam_gate2_realray_event_sharing_status": world_gate2_realray_event_sharing["status"]
            if world_gate2_realray_event_sharing is not None
            else None,
            "world_foam_mps_shared_realray_forward_status": world_mps_shared_realray_forward["status"]
            if world_mps_shared_realray_forward is not None
            else None,
            "world_foam_mps_shared_realray_vjp_status": world_mps_shared_realray_vjp["status"]
            if world_mps_shared_realray_vjp is not None
            else None,
            "world_foam_mps_shared_realray_reduced_vjp_status": world_mps_shared_realray_reduced_vjp["status"]
            if world_mps_shared_realray_reduced_vjp is not None
            else None,
            "world_foam_mps_shared_realray_csr_candidate_storage_status": world_mps_shared_realray_csr_candidate_storage[
                "status"
            ]
            if world_mps_shared_realray_csr_candidate_storage is not None
            else None,
            "world_foam_mps_shared_realray_csr_scaling_status": world_mps_shared_realray_csr_scaling[
                "status"
            ]
            if world_mps_shared_realray_csr_scaling is not None
            else None,
            "world_foam_mps_shared_realray_autograd_overfit_status": world_mps_shared_realray_autograd_overfit[
                "status"
            ]
            if world_mps_shared_realray_autograd_overfit is not None
            else None,
            "world_foam_mps_shared_realray_real_target_train_status": world_mps_shared_realray_real_target_train[
                "status"
            ]
            if world_mps_shared_realray_real_target_train is not None
            else None,
            "world_foam_mps_shared_realray_csr_quality_256px_16f_status": world_mps_shared_realray_csr_quality_256px_16f[
                "status"
            ]
            if world_mps_shared_realray_csr_quality_256px_16f is not None
            else None,
            "star_comparison_json": str(args.star_comparison_json) if args.star_comparison_json else None,
            "star_pair_json": str(args.star_pair_json) if args.star_pair_json else None,
            **paired,
        },
        "world_foam_sweep_rows": world_payload["world_foam_rows"],
        "world_foam_backward_replay_rows": world_payload["world_foam_backward_replay_rows"],
        "world_foam_gradient_reference": world_gradient_summary,
        "world_foam_mps_power_boundary_smoke": world_mps,
        "world_foam_mps_shared_replay_smoke": world_mps_shared_replay,
        "world_foam_mps_rgb_strip_smoke": world_mps_rgb_strip,
        "world_foam_mps_composite_strip_smoke": world_mps_composite_strip,
        "world_foam_mps_composite_vjp_smoke": world_mps_composite_vjp,
        "world_foam_mps_composite_vjp_slab_mask_smoke": world_mps_composite_vjp_slab_mask,
        "world_foam_mps_full_frame_vjp_smoke": world_mps_full_frame_vjp,
        "world_foam_realray_per_sample_reference": world_realray_per_sample,
        "world_foam_mps_realray_replay_smoke": world_mps_realray_replay,
        "world_foam_mps_shared_realray_forward_smoke": world_mps_shared_realray_forward,
        "world_foam_mps_shared_realray_vjp_smoke": world_mps_shared_realray_vjp,
        "world_foam_mps_shared_realray_reduced_vjp_smoke": world_mps_shared_realray_reduced_vjp,
        "world_foam_mps_shared_realray_csr_candidate_storage_smoke": world_mps_shared_realray_csr_candidate_storage,
        "world_foam_mps_shared_realray_csr_scaling_smoke": world_mps_shared_realray_csr_scaling,
        "world_foam_mps_shared_realray_autograd_overfit_smoke": world_mps_shared_realray_autograd_overfit,
        "world_foam_mps_shared_realray_real_target_train_smoke": world_mps_shared_realray_real_target_train,
        "world_foam_mps_shared_realray_csr_quality_256px_16f": world_mps_shared_realray_csr_quality_256px_16f,
        "world_foam_gate2_realray_event_sharing": world_gate2_realray_event_sharing,
        "world_foam_sweeps": world_payload["sweeps"],
        "rows": rows,
        "summary": {
            "world_foam": world_summary,
            "world_foam_gradient_reference": world_gradient_summary,
            "world_foam_mps_power_boundary_smoke": world_mps,
            "world_foam_mps_shared_replay_smoke": world_mps_shared_replay,
            "world_foam_mps_rgb_strip_smoke": world_mps_rgb_strip,
            "world_foam_mps_composite_strip_smoke": world_mps_composite_strip,
            "world_foam_mps_composite_vjp_smoke": world_mps_composite_vjp,
            "world_foam_mps_composite_vjp_slab_mask_smoke": world_mps_composite_vjp_slab_mask,
            "world_foam_mps_full_frame_vjp_smoke": world_mps_full_frame_vjp,
            "world_foam_realray_per_sample_reference": world_realray_per_sample,
            "world_foam_mps_realray_replay_smoke": world_mps_realray_replay,
            "world_foam_mps_shared_realray_forward_smoke": world_mps_shared_realray_forward,
            "world_foam_mps_shared_realray_vjp_smoke": world_mps_shared_realray_vjp,
            "world_foam_mps_shared_realray_reduced_vjp_smoke": world_mps_shared_realray_reduced_vjp,
            "world_foam_mps_shared_realray_csr_candidate_storage_smoke": world_mps_shared_realray_csr_candidate_storage,
            "world_foam_mps_shared_realray_csr_scaling_smoke": world_mps_shared_realray_csr_scaling,
            "world_foam_mps_shared_realray_autograd_overfit_smoke": world_mps_shared_realray_autograd_overfit,
            "world_foam_mps_shared_realray_real_target_train_smoke": world_mps_shared_realray_real_target_train,
            "world_foam_mps_shared_realray_csr_quality_256px_16f": world_mps_shared_realray_csr_quality_256px_16f,
            "world_foam_gate2_realray_event_sharing": world_gate2_realray_event_sharing,
            "star_uvt_pair": star,
            "dynamic_splat_baseline": dynamic,
        },
        "notes": [
            "World Foam event counts and STAR tile-pair counts use different comparison_unit values.",
            "Treat paired=false reports as routing artifacts, not apples-to-apples quality benchmarks.",
            "World Foam Gate 0.7 is RGB strip only.",
            "World Foam Gate 0.8 is alpha/depth forward only.",
            "World Foam Gate 0.9 is fixed-segment RGBA/depth VJP only; it has no geometry/topology gradients or real-video heldout quality.",
            "World Foam Gate 0.95 proves slab-indexed bitmask candidate masks for the fixed-segment VJP at time_slabs=1,2,4; it is still toy strip output.",
            "World Foam Gate 1 image-shape smoke is toy full-frame output from the existing u/t replay op; it is not true u/v/t camera-ray rendering or a quality claim.",
            "World Foam Gate 1B consumes real train and heldout camera rays, but it is a CPU linear per-sample reference with no Metal sharing, backward pass, or training.",
            "World Foam Gate 1C consumes real train and heldout camera rays on MPS, but it is still linear per sample with no temporal sharing, backward pass, or training.",
            "World Foam Gate 2 proves real-ray CPU time-slab event sharing only; it is not a Metal renderer.",
            "World Foam Gate 2B consumes real train and heldout camera rays on MPS with time-slab shared forward candidates, but it is forward-only with no backward pass, no training, and no quality claim.",
            "World Foam Gate 2C consumes real train and heldout camera rays on MPS with time-slab shared fixed-segment VJP, but it only differentiates fixed site RGBA/density samples; it has no geometry/topology gradients, no trainer, no parameter update, and no quality claim.",
            "World Foam Gate 2D consumes real train and heldout camera rays on MPS with a chunked partial-reduction fixed-segment VJP that returns [S,4] site RGBA/density gradients instead of materializing [K,T,S,4] samples; it has frozen-geometry site-RGBA autograd, but still has no geometry/topology gradients, trainer, parameter update, or quality claim.",
            "World Foam Gate 2F proves CSR candidate storage parity for the shared real-ray reduced VJP path; exact per-track CSR is a parity oracle, tiled CSR is a storage-format smoke, and neither is a training or quality claim.",
            "World Foam Gate 2G sweeps tiled CSR shared real-ray reduced VJP at 32px over 2,4,8 frames; it reports storage/work scaling and MPS parity against the bitset oracle, but it is still fixed-geometry and not a heldout-quality comparison.",
            "World Foam Gate 2E uses the frozen-geometry site-RGBA autograd wrapper for a teacher-target parameter-update smoke on real train camera rays; it is not real-target training, a full trainer, or a quality claim.",
            "World Foam Gate 3 optimizes frozen site RGBA/density against real train RGB targets on the shared real-ray MPS path; it is a tiny 16px/2f training smoke, not a 256px/16f heldout baseline or full trainer.",
            "World Foam Gate 3 CSR quality emits same-split 256px/16f train and heldout metrics through tiled CSR frozen-geometry site-RGBA autograd; it still has no geometry/topology gradients and is not a full trainer.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Lane 2 paired benchmark report.")
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--summary-frame", type=int, default=16)
    parser.add_argument("--u-samples", type=int, default=17)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.25)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--camera-velocities", default="0.35,0.7")
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--star-comparison-json", type=Path)
    parser.add_argument("--star-pair-json", type=Path)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json is not None:
        write_json(args.out_json, report)
    print(text)


if __name__ == "__main__":
    main()
