#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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

from gate1_realray_per_sample_reference import crossing_depth_4d  # noqa: E402
from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _frame_time,
    _load_config,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402


def _ray_tuple(ray: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
        (float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
    )


def _edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        curr = [i]
        for j, right_value in enumerate(right, start=1):
            replace_cost = 0 if left_value == right_value else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + replace_cost))
        prev = curr
    return int(prev[-1])


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return int(ordered[max(index, 0)])


def _build_boundary_sequences(
    *,
    boundaries: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
) -> list[list[tuple[int, ...]]]:
    rays = rays.detach().cpu().to(dtype=torch.float32)
    frame_indices = frame_indices.detach().cpu().to(dtype=torch.long)
    sample_count, height, width, payload = rays.shape
    if payload != 6:
        raise ValueError(f"rays must have payload dimension 6, got {payload}")
    if sample_count % frame_count != 0:
        raise ValueError("sample count must be view_count * frame_count")
    view_count = int(sample_count // frame_count)
    sequences: list[list[tuple[int, ...]]] = []
    for view in range(view_count):
        for y in range(height):
            for x in range(width):
                frames: list[tuple[int, ...]] = []
                for frame in range(frame_count):
                    sample_index = view * frame_count + frame
                    t = _frame_time(int(frame_indices[sample_index].item()), frame_count)
                    origin, direction = _ray_tuple(rays[sample_index, y, x])
                    depths: list[tuple[float, int]] = []
                    for boundary_id, boundary in enumerate(boundaries):
                        depth = crossing_depth_4d(
                            boundary,
                            origin=origin,
                            direction=direction,
                            t=t,
                            invalid_epsilon=invalid_epsilon,
                        )
                        if depth is None or not math.isfinite(float(depth)):
                            continue
                        if near <= float(depth) <= far:
                            depths.append((float(depth), boundary_id))
                    depths.sort()
                    frames.append(tuple(boundary_id for _depth, boundary_id in depths))
                sequences.append(frames)
    return sequences


def _profile_sequences(sequences: list[list[tuple[int, ...]]], *, frame_count: int) -> dict[str, Any]:
    track_count = len(sequences)
    sample_count = int(track_count * frame_count)
    total_boundaries = sum(len(row) for frames in sequences for row in frames)
    total_segments = total_boundaries + sample_count
    first_frame_boundaries = sum(len(frames[0]) for frames in sequences) if frame_count else 0
    transition_slots = int(track_count * max(frame_count - 1, 0))
    change_events = 0
    unchanged_events = 0
    edit_ops_total = 0
    edit_ops_nonzero: list[int] = []
    changed_row_boundaries = 0
    same_boundary_order_tracks = 0
    unique_rows_per_track: list[int] = []
    global_unique_rows: set[tuple[int, ...]] = set()
    histogram: Counter[int] = Counter()

    for frames in sequences:
        unique_rows = set(frames)
        unique_rows_per_track.append(len(unique_rows))
        global_unique_rows.update(unique_rows)
        if len(unique_rows) == 1:
            same_boundary_order_tracks += 1
        for left, right in zip(frames[:-1], frames[1:], strict=True):
            edit_ops = _edit_distance(left, right)
            histogram[edit_ops] += 1
            if edit_ops == 0:
                unchanged_events += 1
                continue
            change_events += 1
            edit_ops_total += edit_ops
            edit_ops_nonzero.append(edit_ops)
            changed_row_boundaries += len(right)

    track_unique_rows = int(sum(unique_rows_per_track))
    full_segment_csr_bytes = int(total_segments * 12 + (sample_count + 1) * 4)
    full_boundary_order_csr_bytes = int(total_boundaries * 4 + (sample_count + 1) * 4)
    delta_replace_boundary_bytes = int(
        (first_frame_boundaries * 4) + ((track_count + 1) * 4) + (change_events * 12) + (changed_row_boundaries * 4)
    )
    delta_edit_op_stream_bytes = int(
        (first_frame_boundaries * 4) + ((track_count + 1) * 4) + (change_events * 8) + (edit_ops_total * 12)
    )
    return {
        "track_count": track_count,
        "sample_count": sample_count,
        "frame_count": frame_count,
        "total_boundaries": int(total_boundaries),
        "total_segments_implied": int(total_segments),
        "first_frame_boundaries": int(first_frame_boundaries),
        "avg_boundaries_per_sample": float(total_boundaries) / float(max(sample_count, 1)),
        "transition_slots": transition_slots,
        "change_events": int(change_events),
        "unchanged_events": int(unchanged_events),
        "change_event_rate": float(change_events) / float(max(transition_slots, 1)),
        "edit_ops_total": int(edit_ops_total),
        "edit_ops_per_transition": float(edit_ops_total) / float(max(transition_slots, 1)),
        "edit_ops_per_boundary": float(edit_ops_total) / float(max(total_boundaries, 1)),
        "nonzero_edit_ops_avg": float(sum(edit_ops_nonzero)) / float(max(len(edit_ops_nonzero), 1)),
        "nonzero_edit_ops_p50": _quantile(edit_ops_nonzero, 0.50),
        "nonzero_edit_ops_p95": _quantile(edit_ops_nonzero, 0.95),
        "nonzero_edit_ops_max": max(edit_ops_nonzero) if edit_ops_nonzero else 0,
        "same_boundary_order_all_frames_tracks": int(same_boundary_order_tracks),
        "same_boundary_order_all_frames_ratio": float(same_boundary_order_tracks) / float(max(track_count, 1)),
        "track_unique_boundary_rows": track_unique_rows,
        "track_unique_boundary_rows_vs_samples": float(track_unique_rows) / float(max(sample_count, 1)),
        "global_unique_boundary_rows": len(global_unique_rows),
        "global_unique_boundary_rows_vs_samples": float(len(global_unique_rows)) / float(max(sample_count, 1)),
        "edit_ops_histogram": {str(key): int(value) for key, value in sorted(histogram.items())},
        "storage_estimates": {
            "full_segment_csr_owner_length_mid_bytes": full_segment_csr_bytes,
            "full_boundary_order_csr_i32_bytes": full_boundary_order_csr_bytes,
            "delta_replace_boundary_order_bytes": delta_replace_boundary_bytes,
            "delta_edit_op_stream_boundary_only_bytes": delta_edit_op_stream_bytes,
            "delta_replace_boundary_order_vs_full_segment_csr": float(delta_replace_boundary_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "delta_edit_op_stream_boundary_only_vs_full_segment_csr": float(delta_edit_op_stream_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "note": (
                "Boundary order plus existing rational boundary-depth coefficients can recover segment "
                "length/mid values exactly; owner assignment still needs either owner rows or a cheap owner "
                "update/midpoint rule."
            ),
        },
    }


def _profile_frame_count(
    *,
    frame_count: int,
    config_path: Path,
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
    loaded_frame_count = int(data["frame_count"])
    if loaded_frame_count != frame_count:
        raise ValueError(f"requested {frame_count} frames but loader returned {loaded_frame_count}")
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
    sequences = _build_boundary_sequences(
        boundaries=make_boundaries_4d(sites),
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
    )
    return {
        "frames": frame_count,
        "render_size": render_size,
        "site_count": len(sites),
        "boundary_delta_tape": _profile_sequences(sequences, frame_count=frame_count),
    }


def _scale(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    def value(row: dict[str, Any]) -> Any:
        current: Any = row
        for key in path:
            current = current[key]
        return current

    return float(value(rows[-1])) / float(max(float(value(rows[0])), 1.0e-9))


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
            frame_count=frame_count,
            config_path=config_path,
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
    frame_scale = float(rows[-1]["frames"]) / float(max(rows[0]["frames"], 1))
    boundary_scale = _scale(rows, ("boundary_delta_tape", "total_boundaries"))
    edit_scale = _scale(rows, ("boundary_delta_tape", "edit_ops_total"))
    replace_storage_scale = _scale(
        rows,
        ("boundary_delta_tape", "storage_estimates", "delta_replace_boundary_order_bytes"),
    )
    acceptance = {
        "boundary_count_scales_about_with_frames": boundary_scale >= 0.75 * frame_scale,
        "boundary_edit_ops_scale_sublinear_vs_frames": edit_scale < frame_scale,
        "delta_replace_boundary_storage_scale_sublinear_vs_frames": replace_storage_scale < frame_scale,
        "last_delta_replace_boundary_storage_below_full_segment_csr": rows[-1]["boundary_delta_tape"][
            "storage_estimates"
        ]["delta_replace_boundary_order_vs_full_segment_csr"]
        < 1.0,
    }
    return {
        "benchmark": "world_foam_lane2_segment_boundary_delta_tape_probe",
        "status": "informational",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "site_count": site_count,
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "full_boundary_count_scale_first_to_last": boundary_scale,
        "boundary_edit_op_scale_first_to_last": edit_scale,
        "delta_replace_boundary_storage_scale_first_to_last": replace_storage_scale,
        "structural_read": {
            "boundary_order_can_recover_length_mid_from_coefficients": True,
            "owner_assignment_still_needs_exact_encoding_or_update_rule": True,
            "interpretation": (
                "Boundary-order deltas are a closer exact-replay candidate than owner-only deltas because "
                "stored boundary ids plus rational depth coefficients recover length/mid values."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe boundary-order delta tape structure for exact length/mid replay.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_segment_boundary_delta_tape_probe_render32_2_4_8_16.json",
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
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
