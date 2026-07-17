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
    Site4D,
    _frame_time,
    _load_config,
    crossing_depth_4d,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)


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


def slab_event_set_for_ray(
    *,
    boundaries: tuple[Boundary4D, ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t0: float,
    t1: float,
    near: float,
    far: float,
    invalid_epsilon: float,
) -> tuple[set[int], int]:
    events: set[int] = set()
    invalid = 0
    for boundary_id, boundary in enumerate(boundaries):
        depth0 = crossing_depth_4d(
            boundary,
            origin=origin,
            direction=direction,
            t=t0,
            invalid_epsilon=invalid_epsilon,
        )
        depth1 = crossing_depth_4d(
            boundary,
            origin=origin,
            direction=direction,
            t=t1,
            invalid_epsilon=invalid_epsilon,
        )
        if depth0 is None or depth1 is None:
            invalid += 1
            continue
        if max(min(depth0, depth1), near) <= min(max(depth0, depth1), far):
            events.add(boundary_id)
    return events, invalid


def _ray_tuple(ray: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
        (float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
    )


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
) -> dict[str, Any]:
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    loaded_frame_count = int(data["frame_count"])
    if loaded_frame_count != frame_count:
        raise ValueError(f"requested {frame_count} frames but loader returned {loaded_frame_count}")
    train_views = list(data["train_views"])
    view_count = len(train_views)
    if view_count <= 0:
        raise ValueError("Gate 2 real-ray sharing requires train views")
    if rays.shape[0] != view_count * frame_count:
        raise ValueError(f"Expected view-major rays [V*T,H,W,6], got {tuple(rays.shape)} for V={view_count}, T={frame_count}")
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
    _sample_count, height, width, _payload = rays.shape
    slabs = slab_ranges(time_slabs)
    per_frame_events = 0
    shared_candidate_events = 0
    missing_events = 0
    extra_candidate_events = 0
    invalid_per_frame = 0
    invalid_shared = 0

    slab_cache: dict[tuple[int, int, int, int], set[int]] = {}
    for view in range(view_count):
        base_rays = rays[view * frame_count]
        for y in range(height):
            for x in range(width):
                origin, direction = _ray_tuple(base_rays[y, x])
                for slab_id, (t0, t1) in enumerate(slabs):
                    events, invalid = slab_event_set_for_ray(
                        boundaries=boundaries,
                        origin=origin,
                        direction=direction,
                        t0=t0,
                        t1=t1,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                    )
                    slab_cache[(view, y, x, slab_id)] = events
                    shared_candidate_events += len(events)
                    invalid_shared += invalid

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
                    candidates = slab_cache[(view, y, x, slab_id)]
                    per_frame_events += len(sample_events)
                    missing_events += len(sample_events - candidates)
                    extra_candidate_events += len(candidates - sample_events)
                    invalid_per_frame += invalid

    pixel_tracks = int(view_count * height * width)
    pixel_rays = int(view_count * frame_count * height * width)
    return {
        "frames": frame_count,
        "render_size": render_size,
        "train_views": train_views,
        "pixel_tracks": pixel_tracks,
        "pixel_rays": pixel_rays,
        "time_slabs": time_slabs,
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "per_frame_event_sum": int(per_frame_events),
        "shared_slab_event_sum": int(shared_candidate_events),
        "event_sharing_ratio": float(shared_candidate_events) / float(max(per_frame_events, 1)),
        "missing_sample_events": int(missing_events),
        "extra_candidate_events": int(extra_candidate_events),
        "invalid_per_frame_denominator_count": int(invalid_per_frame),
        "invalid_shared_denominator_count": int(invalid_shared),
        "direct_forward_boundary_scans": int(pixel_rays * len(boundaries)),
        "shared_forward_boundary_scans": int(pixel_tracks * time_slabs * len(boundaries)),
        "shared_forward_boundary_scan_ratio": float(pixel_tracks * time_slabs * len(boundaries))
        / float(max(pixel_rays * len(boundaries), 1)),
        "ray_time_delta": ray_time_delta(rays, view_count=view_count, frame_count=frame_count),
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
        )
        for frame_count in frame_counts
    ]
    first = rows[0]
    last = rows[-1]
    per_frame_growth = float(last["per_frame_event_sum"]) / float(max(int(first["per_frame_event_sum"]), 1))
    shared_growth = float(last["shared_slab_event_sum"]) / float(max(int(first["shared_slab_event_sum"]), 1))
    acceptance = {
        "all_rows_zero_missing": all(int(row["missing_sample_events"]) == 0 for row in rows),
        "shared_event_growth_sublinear": shared_growth < per_frame_growth,
        "shared_scan_ratio_sublinear": all(float(row["shared_forward_boundary_scan_ratio"]) <= 1.0 for row in rows),
        "real_rays_static_within_view_over_time": all(
            float(row["ray_time_delta"]["max_origin_delta_within_view_over_time"]) == 0.0
            and float(row["ray_time_delta"]["max_direction_delta_within_view_over_time"]) == 0.0
            for row in rows
        ),
    }
    return {
        "benchmark": "world_foam_lane2_gate2_realray_event_sharing",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "2_realray_cpu_temporal_candidate_sharing",
        "device": "cpu",
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "time_slabs": time_slabs,
        "site_count": site_count,
        "near": near,
        "far": far,
        "density": density,
        "comparison_unit": "real_camera_ray_4d_power_boundary_events",
        "sharing_scope": "cpu_real_camera_ray_time_slab_candidate_events_only",
        "gradient_scope": "none_event_count_only_no_backward",
        "quality_claim": False,
        "rows": rows,
        "growth": {
            "from_frames": int(first["frames"]),
            "to_frames": int(last["frames"]),
            "per_frame_event_growth": per_frame_growth,
            "shared_event_growth": shared_growth,
            "sublinear_event_growth": shared_growth < per_frame_growth,
        },
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Gate 2 real-ray CPU event-sharing benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8")
    parser.add_argument("--render-size", type=int, default=16)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
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
        invalid_epsilon=args.invalid_epsilon,
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
