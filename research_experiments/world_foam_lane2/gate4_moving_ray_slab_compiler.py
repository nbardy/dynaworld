#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
TRAIN_SRC = DYNAWORLD / "src" / "train"
DEFAULT_CONFIG = (
    DYNAWORLD
    / "src"
    / "train_configs"
    / "local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke.jsonc"
)

if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from gate1_realray_per_sample_reference import (  # noqa: E402
    Boundary4D,
    SITE_INITIALIZATION_CHOICES,
    SITE_INITIALIZATION_LEGACY_SPARSE,
    _frame_time,
    _load_config,
    crossing_depth_4d,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)


@dataclass(frozen=True)
class LinearRayTrack:
    origin_base: tuple[float, float, float]
    origin_slope: tuple[float, float, float]
    direction_base: tuple[float, float, float]
    direction_slope: tuple[float, float, float]
    max_origin_residual: float
    max_direction_residual: float


@dataclass(frozen=True)
class SyntheticRayMotion:
    origin_velocity: tuple[float, float, float]
    direction_velocity: tuple[float, float, float]

    @property
    def active(self) -> bool:
        return any(abs(value) > 0.0 for value in (*self.origin_velocity, *self.direction_velocity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "origin_velocity": list(self.origin_velocity),
            "direction_velocity": list(self.direction_velocity),
            "model": "affine_ray_space_motion_centered_at_t_0_5",
        }


def parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one integer")
    return out


def slab_ranges(count: int) -> list[tuple[float, float]]:
    if count <= 0:
        raise ValueError("time_slab_count must be positive")
    step = 1.0 / float(count)
    return [(index * step, (index + 1) * step) for index in range(count)]


def _ray_tuple(ray: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
        (float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
    )


def _track_tuple(values: torch.Tensor) -> tuple[float, float, float]:
    return (float(values[0].item()), float(values[1].item()), float(values[2].item()))


def apply_synthetic_ray_motion(
    rays: torch.Tensor,
    *,
    frame_indices: torch.Tensor,
    frame_count: int,
    motion: SyntheticRayMotion,
) -> torch.Tensor:
    if not motion.active:
        return rays
    moved = rays.clone()
    times = torch.tensor(
        [_frame_time(int(index.item()), frame_count) - 0.5 for index in frame_indices],
        dtype=moved.dtype,
        device=moved.device,
    )
    origin_velocity = torch.tensor(motion.origin_velocity, dtype=moved.dtype, device=moved.device)
    direction_velocity = torch.tensor(motion.direction_velocity, dtype=moved.dtype, device=moved.device)
    moved[..., :3] = moved[..., :3] + times[:, None, None, None] * origin_velocity
    moved[..., 3:] = moved[..., 3:] + times[:, None, None, None] * direction_velocity
    return moved


def fit_linear_ray_track(samples: torch.Tensor, times: torch.Tensor) -> LinearRayTrack:
    if samples.ndim != 2 or samples.shape[1] != 6:
        raise ValueError(f"samples must have shape [T,6], got {tuple(samples.shape)}")
    if times.ndim != 1 or times.shape[0] != samples.shape[0]:
        raise ValueError(f"times must have shape [T], got {tuple(times.shape)} for samples {tuple(samples.shape)}")
    samples = samples.to(dtype=torch.float64)
    times = times.to(dtype=torch.float64)
    if samples.shape[0] == 1:
        slope = torch.zeros(6, dtype=torch.float64)
        base = samples[0]
    else:
        t_centered = times - times.mean()
        denom = torch.sum(t_centered * t_centered)
        if float(denom.item()) <= 0.0:
            slope = torch.zeros(6, dtype=torch.float64)
        else:
            slope = torch.sum(t_centered[:, None] * (samples - samples.mean(dim=0, keepdim=True)), dim=0) / denom
        base = samples.mean(dim=0) - slope * times.mean()
    predicted = base[None, :] + times[:, None] * slope[None, :]
    residual = (predicted - samples).abs()
    return LinearRayTrack(
        origin_base=_track_tuple(base[:3]),
        origin_slope=_track_tuple(slope[:3]),
        direction_base=_track_tuple(base[3:]),
        direction_slope=_track_tuple(slope[3:]),
        max_origin_residual=float(residual[:, :3].max().item()),
        max_direction_residual=float(residual[:, 3:].max().item()),
    )


def track_ray_at(track: LinearRayTrack, time_value: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    origin = tuple(track.origin_base[i] + track.origin_slope[i] * time_value for i in range(3))
    direction = tuple(track.direction_base[i] + track.direction_slope[i] * time_value for i in range(3))
    return origin, direction


def boundary_depth_coefficients(boundary: Boundary4D, track: LinearRayTrack) -> tuple[float, float, float, float]:
    denom_base = (
        boundary.nx * track.direction_base[0]
        + boundary.ny * track.direction_base[1]
        + boundary.nz * track.direction_base[2]
    )
    denom_slope = (
        boundary.nx * track.direction_slope[0]
        + boundary.ny * track.direction_slope[1]
        + boundary.nz * track.direction_slope[2]
    )
    numer_base = -(
        boundary.nx * track.origin_base[0]
        + boundary.ny * track.origin_base[1]
        + boundary.nz * track.origin_base[2]
        + boundary.b
    )
    numer_slope = -(
        boundary.nx * track.origin_slope[0]
        + boundary.ny * track.origin_slope[1]
        + boundary.nz * track.origin_slope[2]
        + boundary.nt
    )
    return numer_base, numer_slope, denom_base, denom_slope


def compiled_slab_event_set_for_track(
    *,
    boundaries: tuple[Boundary4D, ...],
    track: LinearRayTrack,
    t0: float,
    t1: float,
    near: float,
    far: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
) -> tuple[set[int], int]:
    events: set[int] = set()
    conservative_denominator_events = 0
    for boundary_id, boundary in enumerate(boundaries):
        numer_base, numer_slope, denom_base, denom_slope = boundary_depth_coefficients(boundary, track)
        denom0 = denom_base + denom_slope * t0
        denom1 = denom_base + denom_slope * t1
        if abs(denom0) < invalid_epsilon or abs(denom1) < invalid_epsilon or denom0 * denom1 < 0.0:
            events.add(boundary_id)
            conservative_denominator_events += 1
            continue
        depth0 = (numer_base + numer_slope * t0) / denom0
        depth1 = (numer_base + numer_slope * t1) / denom1
        lo = min(depth0, depth1) - residual_depth_padding
        hi = max(depth0, depth1) + residual_depth_padding
        if max(lo, near) <= min(hi, far):
            events.add(boundary_id)
    return events, conservative_denominator_events


def event_set_for_ray(
    *,
    boundaries: tuple[Boundary4D, ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
) -> tuple[set[int], int]:
    events: set[int] = set()
    invalid = 0
    for boundary_id, boundary in enumerate(boundaries):
        depth = crossing_depth_4d(
            boundary,
            origin=origin,
            direction=direction,
            t=t,
            invalid_epsilon=invalid_epsilon,
        )
        if depth is None:
            invalid += 1
            continue
        if near <= depth <= far:
            events.add(boundary_id)
    return events, invalid


def ray_time_delta(rays: torch.Tensor, *, view_count: int, frame_count: int) -> dict[str, float]:
    max_origin = 0.0
    max_direction = 0.0
    for view in range(view_count):
        base = rays[view * frame_count]
        for frame in range(1, frame_count):
            current = rays[view * frame_count + frame]
            max_origin = max(max_origin, float((base[..., :3] - current[..., :3]).abs().max().item()))
            max_direction = max(max_direction, float((base[..., 3:] - current[..., 3:]).abs().max().item()))
    return {
        "max_origin_delta_within_view_over_time": max_origin,
        "max_direction_delta_within_view_over_time": max_direction,
    }


def _time_tensor_for_view(frame_indices: torch.Tensor, *, view: int, frame_count: int) -> torch.Tensor:
    return torch.tensor(
        [_frame_time(int(frame_indices[view * frame_count + frame].item()), frame_count) for frame in range(frame_count)],
        dtype=torch.float64,
    )


def profile_frame_count(
    *,
    frame_count: int,
    config_path: Path,
    render_size: int,
    site_count: int,
    time_slabs: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    site_initialization: str = SITE_INITIALIZATION_LEGACY_SPARSE,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    load_start = total_start
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    loaded_frame_count = int(data["frame_count"])
    if loaded_frame_count != frame_count:
        raise ValueError(f"requested {frame_count} frames but loader returned {loaded_frame_count}")
    rays = apply_synthetic_ray_motion(
        rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    load_elapsed_s = time.perf_counter() - load_start
    train_views = list(data["train_views"])
    view_count = len(train_views)
    if view_count <= 0:
        raise ValueError("Gate 4 moving-ray slab compiler requires train views")
    if rays.shape[0] != view_count * frame_count:
        raise ValueError(f"Expected view-major rays [V*T,H,W,6], got {tuple(rays.shape)} for V={view_count}, T={frame_count}")
    site_start = time.perf_counter()
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
        initialization=site_initialization,
    )
    boundaries = make_boundaries_4d(sites)
    site_elapsed_s = time.perf_counter() - site_start
    _sample_count, height, width, _payload = rays.shape
    slabs = slab_ranges(time_slabs)

    compile_start = time.perf_counter()
    compiled_cache: dict[tuple[int, int, int, int], set[int]] = {}
    max_origin_residual = 0.0
    max_direction_residual = 0.0
    compiled_candidate_events = 0
    conservative_denominator_events = 0
    for view in range(view_count):
        times = _time_tensor_for_view(frame_indices, view=view, frame_count=frame_count)
        for y in range(height):
            for x in range(width):
                samples = rays[view * frame_count : (view + 1) * frame_count, y, x, :]
                track = fit_linear_ray_track(samples, times)
                max_origin_residual = max(max_origin_residual, track.max_origin_residual)
                max_direction_residual = max(max_direction_residual, track.max_direction_residual)
                for slab_id, (t0, t1) in enumerate(slabs):
                    events, conservative = compiled_slab_event_set_for_track(
                        boundaries=boundaries,
                        track=track,
                        t0=t0,
                        t1=t1,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                        residual_depth_padding=residual_depth_padding,
                    )
                    compiled_cache[(view, y, x, slab_id)] = events
                    compiled_candidate_events += len(events)
                    conservative_denominator_events += conservative
    compile_elapsed_s = time.perf_counter() - compile_start

    replay_start = time.perf_counter()
    per_frame_events = 0
    missing_events = 0
    extra_candidate_events = 0
    invalid_per_frame = 0
    compiled_candidate_replay_iterations = 0
    for view in range(view_count):
        for frame in range(frame_count):
            sample_index = view * frame_count + frame
            t = _frame_time(int(frame_indices[sample_index].item()), frame_count)
            slab_id = min(int(math.floor(t * time_slabs)), time_slabs - 1)
            for y in range(height):
                for x in range(width):
                    origin, direction = _ray_tuple(rays[sample_index, y, x])
                    sample_events, invalid = event_set_for_ray(
                        boundaries=boundaries,
                        origin=origin,
                        direction=direction,
                        t=t,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                    )
                    candidates = compiled_cache[(view, y, x, slab_id)]
                    per_frame_events += len(sample_events)
                    compiled_candidate_replay_iterations += len(candidates)
                    missing_events += len(sample_events - candidates)
                    extra_candidate_events += len(candidates - sample_events)
                    invalid_per_frame += invalid
    replay_elapsed_s = time.perf_counter() - replay_start

    pixel_tracks = int(view_count * height * width)
    pixel_rays = int(view_count * frame_count * height * width)
    boundary_count = len(boundaries)
    direct_boundary_tests = int(pixel_rays * boundary_count)
    compiled_boundary_tests = int(pixel_tracks * time_slabs * boundary_count)
    return {
        "frames": frame_count,
        "render_size": render_size,
        "train_views": train_views,
        "pixel_tracks": pixel_tracks,
        "pixel_rays": pixel_rays,
        "time_slabs": time_slabs,
        "site_count": len(sites),
        "boundary_count": boundary_count,
        "per_frame_event_sum": int(per_frame_events),
        "compiled_slab_candidate_event_sum": int(compiled_candidate_events),
        "compiled_candidate_replay_iterations": int(compiled_candidate_replay_iterations),
        "compiled_event_sharing_ratio": float(compiled_candidate_events) / float(max(per_frame_events, 1)),
        "compiled_replay_iteration_ratio": float(compiled_candidate_replay_iterations) / float(max(per_frame_events, 1)),
        "missing_sample_events": int(missing_events),
        "extra_candidate_events": int(extra_candidate_events),
        "invalid_per_frame_denominator_count": int(invalid_per_frame),
        "conservative_denominator_candidate_events": int(conservative_denominator_events),
        "direct_forward_boundary_tests": direct_boundary_tests,
        "compiled_forward_boundary_tests": compiled_boundary_tests,
        "compiled_forward_boundary_test_ratio": float(compiled_boundary_tests) / float(max(direct_boundary_tests, 1)),
        "linear_fit": {
            "max_origin_residual": max_origin_residual,
            "max_direction_residual": max_direction_residual,
            "residual_depth_padding": residual_depth_padding,
        },
        "ray_time_delta": ray_time_delta(rays, view_count=view_count, frame_count=frame_count),
        "timing_s": {
            "data_load_and_motion": load_elapsed_s,
            "site_boundary_init": site_elapsed_s,
            "compile_candidate_tape": compile_elapsed_s,
            "validate_replay_accounting": replay_elapsed_s,
            "total": time.perf_counter() - total_start,
        },
    }


def run_benchmark(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
    site_count: int,
    time_slabs: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    site_initialization: str = SITE_INITIALIZATION_LEGACY_SPARSE,
) -> dict[str, Any]:
    rows = [
        profile_frame_count(
            frame_count=frame_count,
            config_path=config_path,
            render_size=render_size,
            site_count=site_count,
            time_slabs=time_slabs,
            near=near,
            far=far,
            density=density,
            invalid_epsilon=invalid_epsilon,
            residual_depth_padding=residual_depth_padding,
            synthetic_motion=synthetic_motion,
            site_initialization=site_initialization,
        )
        for frame_count in frame_counts
    ]
    first = rows[0]
    last = rows[-1]
    direct_growth = float(last["direct_forward_boundary_tests"]) / float(max(int(first["direct_forward_boundary_tests"]), 1))
    compiled_growth = float(last["compiled_forward_boundary_tests"]) / float(
        max(int(first["compiled_forward_boundary_tests"]), 1)
    )
    replay_growth = float(last["compiled_candidate_replay_iterations"]) / float(
        max(int(first["compiled_candidate_replay_iterations"]), 1)
    )
    per_frame_event_growth = float(last["per_frame_event_sum"]) / float(max(int(first["per_frame_event_sum"]), 1))
    acceptance = {
        "all_rows_zero_missing": all(int(row["missing_sample_events"]) == 0 for row in rows),
        "compiled_boundary_test_growth_sublinear": compiled_growth < direct_growth,
        "compiled_boundary_test_ratio_sublinear": all(float(row["compiled_forward_boundary_test_ratio"]) <= 1.0 for row in rows),
        "moving_ray_tracks_present": any(
            float(row["ray_time_delta"]["max_origin_delta_within_view_over_time"]) > 0.0
            or float(row["ray_time_delta"]["max_direction_delta_within_view_over_time"]) > 0.0
            for row in rows
        ),
        "affine_track_fit_exact": all(
            float(row["linear_fit"]["max_origin_residual"]) <= 1.0e-6
            and float(row["linear_fit"]["max_direction_residual"]) <= 1.0e-6
            for row in rows
        ),
    }
    return {
        "benchmark": "world_foam_lane2_gate4_moving_ray_slab_compiler",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "4_moving_ray_affine_slab_compiler_cpu",
        "device": "cpu",
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "time_slabs": time_slabs,
        "site_count": site_count,
        "near": near,
        "far": far,
        "density": density,
        "site_initialization": site_initialization,
        "synthetic_motion": synthetic_motion.to_dict(),
        "comparison_unit": "compiled_affine_ray_time_slab_4d_power_boundary_candidates",
        "sharing_scope": "cpu_compiler_for_moving_ray_tracks_no_metal_dispatch",
        "gradient_scope": "none_event_count_only_no_backward",
        "quality_claim": False,
        "rows": rows,
        "growth": {
            "from_frames": int(first["frames"]),
            "to_frames": int(last["frames"]),
            "direct_boundary_test_growth": direct_growth,
            "compiled_boundary_test_growth": compiled_growth,
            "compiled_replay_iteration_growth": replay_growth,
            "per_frame_event_growth": per_frame_event_growth,
            "compiled_boundary_tests_sublinear": compiled_growth < direct_growth,
            "replay_still_frame_scaled": replay_growth >= compiled_growth,
        },
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Gate 4 moving-ray slab compiler benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8")
    parser.add_argument("--render-size", type=int, default=16)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument(
        "--site-initialization",
        choices=SITE_INITIALIZATION_CHOICES,
        default=SITE_INITIALIZATION_LEGACY_SPARSE,
    )
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--residual-depth-padding", type=float, default=0.0)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        config_path=args.config,
        frame_counts=parse_int_list(args.frame_counts),
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        density=args.density,
        site_initialization=args.site_initialization,
        invalid_epsilon=args.invalid_epsilon,
        residual_depth_padding=args.residual_depth_padding,
        synthetic_motion=SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
