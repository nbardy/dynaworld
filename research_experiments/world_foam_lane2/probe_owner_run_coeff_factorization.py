#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
VARIANT_TOOLS = VARIANT_ROOT / "tools"
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"

for path in (VARIANT_ROOT, VARIANT_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    boundary_depth_coefficients,
    fit_linear_ray_track,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from probe_endpoint_record_delta_replay import _boundary_tensor, _track_frame_rays  # noqa: E402
from probe_endpoint_record_edit_replay import _track_boundary_coefficients  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402


def _track_ray_linear_coefficients(*, track_rays: torch.Tensor, frame_t: torch.Tensor) -> torch.Tensor:
    if track_rays.ndim != 3 or track_rays.shape[2] != 6:
        raise ValueError("track_rays must have shape [track_count, frame_count, 6]")
    times = frame_t.to(dtype=torch.float64).cpu()
    rows: list[tuple[float, ...]] = []
    for track_id in range(int(track_rays.shape[0])):
        track = fit_linear_ray_track(track_rays[track_id].cpu(), times)
        rows.append(
            (
                *track.origin_base,
                *track.origin_slope,
                *track.direction_base,
                *track.direction_slope,
            )
        )
    return torch.tensor(rows, dtype=torch.float32)


def _factorized_boundary_coefficients(
    *,
    boundary_f32: torch.Tensor,
    track_ray_coeff_f32: torch.Tensor,
) -> torch.Tensor:
    if boundary_f32.ndim != 2 or boundary_f32.shape[1] != 5:
        raise ValueError("boundary_f32 must have shape [boundary_count, 5]")
    if track_ray_coeff_f32.ndim != 2 or track_ray_coeff_f32.shape[1] != 12:
        raise ValueError("track_ray_coeff_f32 must have shape [track_count, 12]")
    boundary = boundary_f32.to(dtype=torch.float64)
    track = track_ray_coeff_f32.to(dtype=torch.float64)
    normal = boundary[:, 0:3]
    nt = boundary[:, 3]
    b = boundary[:, 4]
    origin_base = track[:, 0:3]
    origin_slope = track[:, 3:6]
    direction_base = track[:, 6:9]
    direction_slope = track[:, 9:12]
    numer_base = -(origin_base @ normal.T + b.reshape(1, -1))
    numer_slope = -(origin_slope @ normal.T + nt.reshape(1, -1))
    denom_base = direction_base @ normal.T
    denom_slope = direction_slope @ normal.T
    return torch.stack((numer_base, numer_slope, denom_base, denom_slope), dim=2).reshape(-1, 4).to(
        dtype=torch.float32
    )


def _max_cut_depth_error(
    *,
    coeff_a: torch.Tensor,
    coeff_b: torch.Tensor,
    frame_t: torch.Tensor,
    invalid_epsilon: float,
    near: float,
    far: float,
) -> dict[str, float]:
    coeff_a = coeff_a.to(dtype=torch.float64).reshape(-1, 4)
    coeff_b = coeff_b.to(dtype=torch.float64).reshape(-1, 4)
    max_depth_diff = 0.0
    max_validity_diff = 0
    valid_samples = 0
    for t_raw in frame_t.to(dtype=torch.float64).tolist():
        t = float(t_raw)
        a_den = coeff_a[:, 2] + coeff_a[:, 3] * t
        b_den = coeff_b[:, 2] + coeff_b[:, 3] * t
        a_valid = a_den.abs() >= invalid_epsilon
        b_valid = b_den.abs() >= invalid_epsilon
        a_depth = (coeff_a[:, 0] + coeff_a[:, 1] * t) / torch.where(a_valid, a_den, torch.ones_like(a_den))
        b_depth = (coeff_b[:, 0] + coeff_b[:, 1] * t) / torch.where(b_valid, b_den, torch.ones_like(b_den))
        a_valid = a_valid & torch.isfinite(a_depth) & (a_depth >= near) & (a_depth <= far)
        b_valid = b_valid & torch.isfinite(b_depth) & (b_depth >= near) & (b_depth <= far)
        max_validity_diff += int((a_valid != b_valid).sum().item())
        both = a_valid & b_valid
        if bool(both.any().item()):
            max_depth_diff = max(max_depth_diff, float((a_depth[both] - b_depth[both]).abs().max().item()))
            valid_samples += int(both.sum().item())
    return {
        "max_valid_depth_abs_error": float(max_depth_diff),
        "validity_mismatches": int(max_validity_diff),
        "valid_depth_samples": int(valid_samples),
    }


def _profile_frame_count(
    *,
    config_path: Path,
    frame_count: int,
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    synthetic_motion: SyntheticRayMotion,
) -> dict[str, Any]:
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    rays = apply_synthetic_ray_motion(
        rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    boundaries = make_boundaries_4d(sites)
    track_rays, frame_t = _track_frame_rays(rays, frame_indices, frame_count=frame_count)
    boundary_f32 = _boundary_tensor(boundaries)
    track_ray_coeff_f32 = _track_ray_linear_coefficients(track_rays=track_rays, frame_t=frame_t)
    dense_coeff_f32 = _track_boundary_coefficients(
        boundaries=boundaries,
        track_rays=track_rays,
        frame_t=frame_t,
    )
    factorized_coeff_f32 = _factorized_boundary_coefficients(
        boundary_f32=boundary_f32,
        track_ray_coeff_f32=track_ray_coeff_f32,
    )
    coeff_abs_error = (dense_coeff_f32 - factorized_coeff_f32).abs()
    dense_coeff_f16 = dense_coeff_f32.to(dtype=torch.float16).to(dtype=torch.float32)
    depth_vs_dense = _max_cut_depth_error(
        coeff_a=dense_coeff_f32,
        coeff_b=factorized_coeff_f32,
        frame_t=frame_t,
        invalid_epsilon=invalid_epsilon,
        near=near,
        far=far,
    )
    depth_vs_coeff16 = _max_cut_depth_error(
        coeff_a=dense_coeff_f16,
        coeff_b=factorized_coeff_f32,
        frame_t=frame_t,
        invalid_epsilon=invalid_epsilon,
        near=near,
        far=far,
    )
    current_coeff16_bytes = int(dense_coeff_f32.numel() * 2)
    factorized_f32_bytes = int(boundary_f32.numel() * 4 + track_ray_coeff_f32.numel() * 4)
    factorized_f16_bytes = int(boundary_f32.numel() * 2 + track_ray_coeff_f32.numel() * 2)
    return {
        "frames": int(frame_count),
        "render_size": int(render_size),
        "site_count": int(site_count),
        "track_count": int(track_rays.shape[0]),
        "boundary_count": int(len(boundaries)),
        "dense_coeff_rows": int(dense_coeff_f32.shape[0]),
        "current_coeff16_bytes": current_coeff16_bytes,
        "factorized_boundary_track_f32_bytes": factorized_f32_bytes,
        "factorized_boundary_track_f16_bytes": factorized_f16_bytes,
        "factorized_f32_vs_current_coeff16": float(factorized_f32_bytes) / float(max(current_coeff16_bytes, 1)),
        "factorized_f16_vs_current_coeff16": float(factorized_f16_bytes) / float(max(current_coeff16_bytes, 1)),
        "max_coeff_abs_error_vs_dense": float(coeff_abs_error.max().item()) if coeff_abs_error.numel() else 0.0,
        "mean_coeff_abs_error_vs_dense": float(coeff_abs_error.mean().item()) if coeff_abs_error.numel() else 0.0,
        "depth_vs_dense_factorized": depth_vs_dense,
        "depth_vs_coeff16_factorized": depth_vs_coeff16,
    }


def run_probe(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    synthetic_motion: SyntheticRayMotion,
) -> dict[str, Any]:
    rows = [
        _profile_frame_count(
            config_path=config_path,
            frame_count=frame_count,
            render_size=render_size,
            site_count=site_count,
            near=near,
            far=far,
            density=density,
            invalid_epsilon=invalid_epsilon,
            synthetic_motion=synthetic_motion,
        )
        for frame_count in frame_counts
    ]
    first = rows[0]
    last = rows[-1]
    coeff_scale = float(last["current_coeff16_bytes"]) / float(max(first["current_coeff16_bytes"], 1))
    factorized_scale = float(last["factorized_boundary_track_f32_bytes"]) / float(
        max(first["factorized_boundary_track_f32_bytes"], 1)
    )
    max_depth_error = max(float(row["depth_vs_dense_factorized"]["max_valid_depth_abs_error"]) for row in rows)
    max_validity_mismatches = max(int(row["depth_vs_dense_factorized"]["validity_mismatches"]) for row in rows)
    max_factorized_ratio = max(float(row["factorized_f32_vs_current_coeff16"]) for row in rows)
    return {
        "benchmark": "world_foam_lane2_owner_run_coeff_factorization_probe",
        "completion_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "frame_scale_first_to_last": float(frame_counts[-1]) / float(max(frame_counts[0], 1)),
        "current_coeff16_storage_scale_first_to_last": coeff_scale,
        "factorized_f32_storage_scale_first_to_last": factorized_scale,
        "max_factorized_f32_vs_current_coeff16": max_factorized_ratio,
        "max_depth_error_vs_dense_factorized": max_depth_error,
        "max_validity_mismatches_vs_dense_factorized": max_validity_mismatches,
        "acceptance": {
            "factorized_coefficients_match_dense_coefficients": max(
                float(row["max_coeff_abs_error_vs_dense"]) for row in rows
            )
            <= 5.0e-6,
            "factorized_depth_matches_dense_coefficients": max_depth_error <= 1.0e-4
            and max_validity_mismatches == 0,
            "factorized_f32_storage_below_current_coeff16_at_max_frame": float(
                last["factorized_boundary_track_f32_bytes"]
            )
            < float(last["current_coeff16_bytes"]),
            "factorized_f32_storage_sublinear_vs_frames": factorized_scale
            < float(frame_counts[-1]) / float(max(frame_counts[0], 1)),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe factorized coefficient storage for owner-run packed delta.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=16)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-19_owner_run_coeff_factorization_probe_render16_site24_2_4_8_16.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_probe(
        config_path=args.config,
        frame_counts=_parse_int_list(args.frame_counts),
        render_size=args.render_size,
        site_count=args.site_count,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        synthetic_motion=SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(payload["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
