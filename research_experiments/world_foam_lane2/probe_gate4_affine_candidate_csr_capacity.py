#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from gate4_affine_slab_tape import build_gate4_affine_slab_tape
from gate4_moving_ray_slab_compiler import (
    DEFAULT_CONFIG,
    SITE_INITIALIZATION_CHOICES,
    SITE_INITIALIZATION_LEGACY_SPARSE,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    parse_int_list,
)


DYNAWORLD = Path(__file__).resolve().parents[2]
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
TAPE_MODE = "gate4-affine-candidate-num32-den16-fused-mse"


def _ratio_last_first(last: float, first: float) -> float:
    if abs(float(first)) <= 1.0e-12:
        return 0.0 if abs(float(last)) <= 1.0e-12 else float("inf")
    return float(last) / float(first)


def _tensor_bytes(numel: int, element_size: int) -> int:
    return int(numel) * int(element_size)


def candidate_csr_storage_breakdown(*, tape: Any, site_count: int) -> dict[str, int]:
    """MPS-resident-equivalent storage for the candidate CSR train/eval path."""
    by_key = {
        "affine_row_index_i32": _tensor_bytes(int(tape.row_index.numel()), 4),
        "affine_candidate_row_offsets_i32": _tensor_bytes(int(tape.row_offsets.numel()), 4),
        "affine_candidate_depth_num_f32": _tensor_bytes(int(tape.candidate_count) * 2, 4),
        "affine_candidate_depth_den_f16": _tensor_bytes(int(tape.candidate_count) * 2, 2),
        "affine_sites_f32": _tensor_bytes(int(site_count) * 5, 4),
        "affine_ray_f32": _tensor_bytes(int(tape.ray_coeff.numel()), 4),
        "affine_frame_t_f32": _tensor_bytes(int(tape.frame_t.numel()), 4),
    }
    by_key["total_bytes"] = int(sum(by_key.values()))
    return by_key


def _candidate_row_distribution(tape: Any) -> dict[str, float | int]:
    offsets = tape.row_offsets.detach().cpu().to(dtype=torch.int64)
    counts = (offsets[1:] - offsets[:-1]).tolist()
    if not counts:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0,
            "mean_to_median": 0.0,
            "max_to_median": 0.0,
        }
    sorted_counts = sorted(int(value) for value in counts)
    median = float(statistics.median(sorted_counts))
    p90_index = min(len(sorted_counts) - 1, math.ceil(0.90 * len(sorted_counts)) - 1)
    max_value = int(sorted_counts[-1])
    mean_value = float(statistics.fmean(sorted_counts))
    return {
        "count": int(len(sorted_counts)),
        "mean": mean_value,
        "median": median,
        "p90": float(sorted_counts[p90_index]),
        "max": max_value,
        "mean_to_median": mean_value / max(median, 1.0e-12),
        "max_to_median": float(max_value) / max(median, 1.0e-12),
    }


def _repeat_view_major_frames(
    values: torch.Tensor,
    *,
    loaded_frame_count: int,
    requested_frame_count: int,
    name: str,
) -> torch.Tensor:
    if loaded_frame_count < 1 or requested_frame_count < 1:
        raise ValueError(f"{name} frame counts must be positive")
    sample_count = int(values.shape[0])
    if sample_count % loaded_frame_count != 0:
        raise ValueError(f"{name} sample count {sample_count} is not divisible by {loaded_frame_count}")
    view_count = sample_count // loaded_frame_count
    indices = torch.arange(requested_frame_count, dtype=torch.long, device=values.device) % loaded_frame_count
    return (
        values.reshape(view_count, loaded_frame_count, *values.shape[1:])
        .index_select(1, indices)
        .reshape(view_count * requested_frame_count, *values.shape[1:])
        .contiguous()
    )


def _sequential_view_major_frame_indices(*, view_count: int, requested_frame_count: int) -> torch.Tensor:
    return torch.arange(requested_frame_count, dtype=torch.long).repeat(view_count)


def _fit_loaded_frame_count(
    *,
    targets: torch.Tensor,
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    loaded_frame_count: int,
    requested_frame_count: int,
    allow_repeat_loaded_frames: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if loaded_frame_count == requested_frame_count:
        return targets, rays, frame_indices, False
    if loaded_frame_count > requested_frame_count:
        raise ValueError(
            f"loader returned {loaded_frame_count} frames for requested {requested_frame_count}; "
            "expected the config loader to crop to the requested count"
        )
    if not allow_repeat_loaded_frames:
        raise ValueError(
            f"loader returned only {loaded_frame_count} frames for requested {requested_frame_count}; "
            "pass --repeat-loaded-frames only for synthetic topology capacity probes"
        )
    targets = _repeat_view_major_frames(
        targets,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=requested_frame_count,
        name="targets",
    )
    rays = _repeat_view_major_frames(
        rays,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=requested_frame_count,
        name="rays",
    )
    view_count = int(targets.shape[0]) // requested_frame_count
    frame_indices = _sequential_view_major_frame_indices(
        view_count=view_count,
        requested_frame_count=requested_frame_count,
    )
    return targets, rays, frame_indices, True


def profile_frame_count(
    *,
    frame_count: int,
    config_path: Path,
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
    gate4_time_slabs: int,
    synthetic_motion: SyntheticRayMotion,
    sample_validation: str,
    allow_repeat_loaded_frames: bool,
    site_initialization: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    loaded_frame_count = int(data["frame_count"])
    targets, rays, frame_indices, repeated = _fit_loaded_frame_count(
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=frame_count,
        allow_repeat_loaded_frames=allow_repeat_loaded_frames,
    )
    rays = apply_synthetic_ray_motion(
        rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    train_views = list(data["train_views"])
    view_count = len(train_views)
    if view_count <= 0:
        raise ValueError("candidate CSR capacity probe requires at least one train view")
    if int(rays.shape[0]) != view_count * frame_count:
        raise ValueError(f"expected view-major rays [V*T,H,W,6], got {tuple(rays.shape)}")
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
    build_start = time.perf_counter()
    tape = build_gate4_affine_slab_tape(
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        time_slabs=gate4_time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        residual_depth_padding=residual_depth_padding,
        layout="per-track",
        tile_h=1,
        tile_w=1,
        candidate_order="slab-mid-depth",
        sample_validation=sample_validation,
    )
    build_elapsed_s = time.perf_counter() - build_start
    storage = candidate_csr_storage_breakdown(tape=tape, site_count=len(sites))
    return {
        "status": "ok",
        "frame_count": int(frame_count),
        "loaded_frame_count": int(loaded_frame_count),
        "repeat_loaded_frames": bool(repeated),
        "render_size": int(render_size),
        "site_count": int(len(sites)),
        "site_initialization": site_initialization,
        "train_views": train_views,
        "view_count": int(view_count),
        "height": int(tape.height),
        "width": int(tape.width),
        "track_count": int(tape.track_count),
        "row_count": int(tape.row_count),
        "time_slabs": int(tape.time_slab_count),
        "candidate_count": int(tape.candidate_count),
        "candidate_replay_iterations": int(tape.candidate_replay_iterations),
        "direct_boundary_iterations": int(tape.direct_boundary_iterations),
        "compiled_boundary_tests": int(tape.compiled_boundary_tests),
        "max_candidates_per_row": int(tape.max_candidates_per_row),
        "avg_candidates_per_row": float(tape.avg_candidates_per_row),
        "empty_row_count": int(tape.empty_row_count),
        "candidate_row_distribution": _candidate_row_distribution(tape),
        "storage_bytes": int(storage["total_bytes"]),
        "storage_by_key": {key: int(value) for key, value in storage.items() if key != "total_bytes"},
        "max_origin_residual": float(tape.max_origin_residual),
        "max_direction_residual": float(tape.max_direction_residual),
        "sample_validation": sample_validation,
        "missing_sample_events": int(tape.missing_sample_events),
        "missing_sample_events_authoritative": bool(
            tape.candidate_depth_order.get("missing_sample_events_authoritative", False)
        ),
        "extra_candidate_events": int(tape.extra_candidate_events),
        "timing_s": {
            "cpu_build_gate4_affine_candidate_tape": float(build_elapsed_s),
            "total_cpu_probe": float(time.perf_counter() - start),
        },
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    max_storage_scale: float,
    max_candidate_scale: float,
    max_candidates_per_row: int,
    max_fit_residual: float,
) -> dict[str, Any]:
    if not rows:
        return {"status": "failed", "failures": ["no rows"], "acceptance": {}}
    first = rows[0]
    last = rows[-1]
    frame_scale = _ratio_last_first(float(last["frame_count"]), float(first["frame_count"]))
    storage_scale = _ratio_last_first(float(last["storage_bytes"]), float(first["storage_bytes"]))
    candidate_count_scale = _ratio_last_first(float(last["candidate_count"]), float(first["candidate_count"]))
    replay_iteration_scale = _ratio_last_first(
        float(last["candidate_replay_iterations"]),
        float(first["candidate_replay_iterations"]),
    )
    direct_boundary_iteration_scale = _ratio_last_first(
        float(last["direct_boundary_iterations"]),
        float(first["direct_boundary_iterations"]),
    )
    compiled_boundary_test_scale = _ratio_last_first(
        float(last["compiled_boundary_tests"]),
        float(first["compiled_boundary_tests"]),
    )
    scale_gate_required = len(rows) > 1 and int(last["frame_count"]) > int(first["frame_count"])
    acceptance = {
        "all_rows_ok": all(row.get("status") == "ok" for row in rows),
        "candidate_rows_under_cap": all(
            int(row["max_candidates_per_row"]) <= int(max_candidates_per_row) for row in rows
        ),
        "candidate_count_scale_within_limit": (not scale_gate_required)
        or candidate_count_scale <= float(max_candidate_scale),
        "storage_scale_within_limit": (not scale_gate_required) or storage_scale <= float(max_storage_scale),
        "candidate_count_sublinear_vs_frame_count": (not scale_gate_required) or candidate_count_scale < frame_scale,
        "storage_sublinear_vs_frame_count": (not scale_gate_required) or storage_scale < frame_scale,
        "compiled_boundary_tests_sublinear_vs_direct": (not scale_gate_required)
        or compiled_boundary_test_scale < direct_boundary_iteration_scale,
        "affine_fit_within_tolerance": all(
            float(row["max_origin_residual"]) <= float(max_fit_residual)
            and float(row["max_direction_residual"]) <= float(max_fit_residual)
            for row in rows
        ),
    }
    failures = [key for key, passed in acceptance.items() if not passed]
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "acceptance": acceptance,
        "scale_gate_required": bool(scale_gate_required),
        "frame_scale_first_to_last": frame_scale,
        "candidate_count_scale_first_to_last": candidate_count_scale,
        "storage_scale_first_to_last": storage_scale,
        "candidate_replay_iteration_scale_first_to_last": replay_iteration_scale,
        "direct_boundary_iteration_scale_first_to_last": direct_boundary_iteration_scale,
        "compiled_boundary_test_scale_first_to_last": compiled_boundary_test_scale,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = parse_int_list(args.frame_counts)
    rows = [
        profile_frame_count(
            frame_count=frame_count,
            config_path=args.config,
            render_size=args.render_size,
            site_count=args.site_count,
            near=args.near,
            far=args.far,
            density=args.density,
            invalid_epsilon=args.invalid_epsilon,
            residual_depth_padding=args.gate4_residual_depth_padding,
            gate4_time_slabs=args.gate4_time_slabs,
            synthetic_motion=SyntheticRayMotion(
                origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
                direction_velocity=(
                    args.direction_velocity_x,
                    args.direction_velocity_y,
                    args.direction_velocity_z,
                ),
            ),
            sample_validation=args.sample_validation,
            allow_repeat_loaded_frames=args.repeat_loaded_frames,
            site_initialization=args.site_initialization,
        )
        for frame_count in frame_counts
    ]
    summary = summarize_rows(
        rows,
        max_storage_scale=args.max_storage_scale,
        max_candidate_scale=args.max_candidate_scale,
        max_candidates_per_row=args.max_candidates_per_row,
        max_fit_residual=args.max_fit_residual,
    )
    return {
        "benchmark": "world_foam_lane2_gate4_affine_candidate_csr_capacity",
        "status": summary["status"],
        "gate": "gate4_affine_candidate_csr_topology_capacity_cpu",
        "device": "cpu",
        "tape_mode": TAPE_MODE,
        "config_path": str(args.config),
        "frame_counts": list(frame_counts),
        "render_size": int(args.render_size),
        "site_count": int(args.site_count),
        "site_initialization": str(args.site_initialization),
        "gate4_time_slabs": int(args.gate4_time_slabs),
        "gate4_residual_depth_padding": float(args.gate4_residual_depth_padding),
        "synthetic_motion": SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ).to_dict(),
        "gradient_scope": "none_topology_capacity_only_no_shader_dispatch_no_backward",
        "timing_scope": "cpu_build_timings_are_diagnostic_only_not_speed_claims",
        "quality_claim": False,
        "speed_claim": False,
        **summary,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Gate4 affine candidate CSR topology capacity on CPU.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument(
        "--site-initialization",
        choices=SITE_INITIALIZATION_CHOICES,
        default=SITE_INITIALIZATION_LEGACY_SPARSE,
    )
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--gate4-time-slabs", type=int, default=1)
    parser.add_argument("--gate4-residual-depth-padding", type=float, default=0.001)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--sample-validation", choices=("skip", "full"), default="skip")
    parser.add_argument("--repeat-loaded-frames", action="store_true")
    parser.add_argument("--max-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-candidate-scale", type=float, default=1.10)
    parser.add_argument("--max-candidates-per-row", type=int, default=256)
    parser.add_argument("--max-fit-residual", type=float, default=1.0e-5)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_probe(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
