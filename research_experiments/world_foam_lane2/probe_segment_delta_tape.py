#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from probe_fused_slab_segment_tape import build_segment_tape  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402


def _owner_sequences(tape: Any) -> list[list[tuple[int, ...]]]:
    owners = tape.owners_i32.detach().cpu()
    counts = tape.counts_i32.detach().cpu()
    out: list[list[tuple[int, ...]]] = []
    for track_id in range(tape.track_count):
        frames: list[tuple[int, ...]] = []
        for frame_id in range(tape.frame_count):
            count = int(counts[track_id, frame_id].item())
            frames.append(tuple(int(value.item()) for value in owners[track_id, frame_id, :count]))
        out.append(frames)
    return out


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
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + replace_cost,
                )
            )
        prev = curr
    return int(prev[-1])


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return int(ordered[max(index, 0)])


def _delta_profile(tape: Any) -> dict[str, Any]:
    sequences = _owner_sequences(tape)
    sample_count = int(tape.track_count * tape.frame_count)
    total_segments = int(tape.counts_i32.to(dtype=torch.int64).sum().item())
    first_frame_segments = int(tape.counts_i32[:, 0].to(dtype=torch.int64).sum().item()) if tape.frame_count else 0
    transition_slots = int(tape.track_count * max(tape.frame_count - 1, 0))
    change_events = 0
    unchanged_events = 0
    edit_ops_total = 0
    edit_ops_nonzero: list[int] = []
    changed_row_segments = 0
    inserted_segments = 0
    deleted_segments = 0
    same_topology_tracks = 0
    unique_rows_per_track: list[int] = []
    global_unique_rows: set[tuple[int, ...]] = set()
    transition_histogram: Counter[int] = Counter()

    for frames in sequences:
        unique_rows = set(frames)
        unique_rows_per_track.append(len(unique_rows))
        global_unique_rows.update(unique_rows)
        if len(unique_rows) == 1:
            same_topology_tracks += 1
        for left, right in zip(frames[:-1], frames[1:], strict=True):
            edit_ops = _edit_distance(left, right)
            transition_histogram[edit_ops] += 1
            if edit_ops == 0:
                unchanged_events += 1
                continue
            change_events += 1
            edit_ops_total += edit_ops
            edit_ops_nonzero.append(edit_ops)
            changed_row_segments += len(right)
            if len(right) > len(left):
                inserted_segments += len(right) - len(left)
            elif len(left) > len(right):
                deleted_segments += len(left) - len(right)

    track_unique_rows = int(sum(unique_rows_per_track))
    full_compact_csr_bytes = int(total_segments * 12 + (sample_count + 1) * 4)
    owner_sequence_csr_bytes = int(total_segments * 4 + (sample_count + 1) * 4)
    delta_replace_owner_bytes = int((first_frame_segments * 4) + ((tape.track_count + 1) * 4) + (change_events * 12) + (changed_row_segments * 4))
    delta_replace_geometry_bytes = int(
        (first_frame_segments * 12) + ((tape.track_count + 1) * 4) + (change_events * 12) + (changed_row_segments * 12)
    )
    edit_op_stream_bytes = int(
        (first_frame_segments * 4)
        + ((tape.track_count + 1) * 4)
        + (change_events * 8)
        + (edit_ops_total * 12)
    )

    return {
        "sample_count": sample_count,
        "track_count": int(tape.track_count),
        "frame_count": int(tape.frame_count),
        "total_segments": total_segments,
        "first_frame_segments": first_frame_segments,
        "avg_segments_per_sample": float(total_segments) / float(max(sample_count, 1)),
        "transition_slots": transition_slots,
        "change_events": change_events,
        "unchanged_events": unchanged_events,
        "change_event_rate": float(change_events) / float(max(transition_slots, 1)),
        "edit_ops_total": edit_ops_total,
        "edit_ops_per_transition": float(edit_ops_total) / float(max(transition_slots, 1)),
        "edit_ops_per_full_segment": float(edit_ops_total) / float(max(total_segments, 1)),
        "nonzero_edit_ops_avg": float(sum(edit_ops_nonzero)) / float(max(len(edit_ops_nonzero), 1)),
        "nonzero_edit_ops_p50": _quantile(edit_ops_nonzero, 0.50),
        "nonzero_edit_ops_p95": _quantile(edit_ops_nonzero, 0.95),
        "nonzero_edit_ops_max": max(edit_ops_nonzero) if edit_ops_nonzero else 0,
        "inserted_segments_upper_bound": inserted_segments,
        "deleted_segments_upper_bound": deleted_segments,
        "changed_row_segments": changed_row_segments,
        "changed_row_segments_vs_full_segments": float(changed_row_segments) / float(max(total_segments, 1)),
        "same_topology_all_frames_tracks": same_topology_tracks,
        "same_topology_all_frames_ratio": float(same_topology_tracks) / float(max(tape.track_count, 1)),
        "track_unique_topology_rows": track_unique_rows,
        "track_unique_topology_rows_vs_samples": float(track_unique_rows) / float(max(sample_count, 1)),
        "global_unique_owner_sequences": len(global_unique_rows),
        "global_unique_owner_sequences_vs_samples": float(len(global_unique_rows)) / float(max(sample_count, 1)),
        "edit_ops_histogram": {str(key): int(value) for key, value in sorted(transition_histogram.items())},
        "storage_estimates": {
            "full_compact_csr_owner_length_mid_i32_f32_f32_bytes": full_compact_csr_bytes,
            "owner_sequence_csr_i32_only_bytes": owner_sequence_csr_bytes,
            "delta_replace_owner_sequence_bytes": delta_replace_owner_bytes,
            "delta_replace_geometry_rows_bytes": delta_replace_geometry_bytes,
            "delta_edit_op_stream_owner_only_bytes": edit_op_stream_bytes,
            "delta_replace_owner_sequence_vs_full_compact_csr": float(delta_replace_owner_bytes)
            / float(max(full_compact_csr_bytes, 1)),
            "delta_replace_geometry_rows_vs_full_compact_csr": float(delta_replace_geometry_bytes)
            / float(max(full_compact_csr_bytes, 1)),
            "delta_edit_op_stream_owner_only_vs_full_compact_csr": float(edit_op_stream_bytes)
            / float(max(full_compact_csr_bytes, 1)),
            "notes": (
                "Delta estimates are topology-focused. The geometry-row variant stores base and changed "
                "owner/length/mid rows, but unchanged topology still needs a separate model or stream for "
                "frame-varying length/mid if exact replay is required."
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
    transmittance_threshold: float,
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
    tape = build_segment_tape(
        sites=sites,
        boundaries=make_boundaries_4d(sites),
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    return {
        "frames": frame_count,
        "render_size": render_size,
        "site_count": len(sites),
        "delta_tape": _delta_profile(tape),
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
    transmittance_threshold: float,
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
            transmittance_threshold=transmittance_threshold,
            synthetic_motion=synthetic_motion,
        )
        for frame_count in frame_counts
    ]
    frame_scale = float(rows[-1]["frames"]) / float(max(rows[0]["frames"], 1))
    segment_scale = float(rows[-1]["delta_tape"]["total_segments"]) / float(
        max(int(rows[0]["delta_tape"]["total_segments"]), 1)
    )
    change_event_scale = float(rows[-1]["delta_tape"]["change_events"]) / float(
        max(int(rows[0]["delta_tape"]["change_events"]), 1)
    )
    edit_op_scale = float(rows[-1]["delta_tape"]["edit_ops_total"]) / float(
        max(int(rows[0]["delta_tape"]["edit_ops_total"]), 1)
    )
    delta_storage_scale = float(
        rows[-1]["delta_tape"]["storage_estimates"]["delta_replace_owner_sequence_bytes"]
    ) / float(max(int(rows[0]["delta_tape"]["storage_estimates"]["delta_replace_owner_sequence_bytes"]), 1))
    full_csr_storage_scale = float(
        rows[-1]["delta_tape"]["storage_estimates"]["full_compact_csr_owner_length_mid_i32_f32_f32_bytes"]
    ) / float(max(int(rows[0]["delta_tape"]["storage_estimates"]["full_compact_csr_owner_length_mid_i32_f32_f32_bytes"]), 1))
    acceptance = {
        "change_events_scale_sublinear_vs_frames": change_event_scale < frame_scale,
        "edit_ops_scale_sublinear_vs_frames": edit_op_scale < frame_scale,
        "delta_owner_storage_scale_sublinear_vs_frames": delta_storage_scale < frame_scale,
        "last_delta_owner_storage_below_full_compact_csr": rows[-1]["delta_tape"]["storage_estimates"][
            "delta_replace_owner_sequence_vs_full_compact_csr"
        ]
        < 1.0,
    }
    return {
        "benchmark": "world_foam_lane2_segment_delta_tape_probe",
        "status": "ok" if all(acceptance.values()) else "informational",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "site_count": site_count,
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "full_segment_scale_first_to_last": segment_scale,
        "change_event_scale_first_to_last": change_event_scale,
        "edit_op_scale_first_to_last": edit_op_scale,
        "delta_owner_storage_scale_first_to_last": delta_storage_scale,
        "full_compact_csr_storage_scale_first_to_last": full_csr_storage_scale,
        "structural_read": {
            "delta_tape_topology_is_sublinear": bool(
                acceptance["change_events_scale_sublinear_vs_frames"]
                and acceptance["edit_ops_scale_sublinear_vs_frames"]
            ),
            "owner_topology_delta_is_not_exact_geometry_replay": True,
            "interpretation": (
                "Frame-to-frame owner topology changes can be much smaller than full per-sample rows; "
                "exact segment replay still needs a compact representation for per-frame length/mid values."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe frame-to-frame segment owner delta tape structure.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_segment_delta_tape_probe_render32_2_4_8_16.json",
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
        transmittance_threshold=args.transmittance_threshold,
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
