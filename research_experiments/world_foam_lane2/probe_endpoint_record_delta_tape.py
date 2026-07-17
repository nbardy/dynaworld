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
from probe_endpoint_run_tape import compress_same_owner_endpoint_runs  # noqa: E402
from probe_fused_slab_segment_tape import build_segment_tape  # noqa: E402
from probe_owner_run_boundary_tape import _build_owner_run_sequences, _verify_against_owner_run_tape  # noqa: E402
from probe_segment_owner_run_tape import compress_same_owner_runs  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402


EndpointRecord = tuple[int, int, int]


def _edit_distance(left: tuple[EndpointRecord, ...], right: tuple[EndpointRecord, ...]) -> int:
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


def _endpoint_record_rows(sequences: list[list[tuple[Any, ...]]]) -> list[list[tuple[EndpointRecord, ...]]]:
    rows: list[list[tuple[EndpointRecord, ...]]] = []
    for frames in sequences:
        frame_rows: list[tuple[EndpointRecord, ...]] = []
        for records in frames:
            frame_rows.append(
                tuple((int(record.owner), int(record.left_cut_id), int(record.right_cut_id)) for record in records)
            )
        rows.append(frame_rows)
    return rows


def _verify_against_endpoint_run_tape(sequences: list[list[tuple[Any, ...]]], endpoint_tape: Any) -> dict[str, Any]:
    offsets = endpoint_tape.offsets_i32.detach().cpu()
    owners = endpoint_tape.owners_i32.detach().cpu()
    count_mismatches = 0
    owner_mismatches = 0
    max_count_delta = 0
    sample_id = 0
    for frames in sequences:
        for records in frames:
            start = int(offsets[sample_id].item())
            end = int(offsets[sample_id + 1].item())
            expected_count = end - start
            if expected_count != len(records):
                count_mismatches += 1
                max_count_delta = max(max_count_delta, abs(expected_count - len(records)))
            compare_count = min(expected_count, len(records))
            for local_id in range(compare_count):
                if int(owners[start + local_id].item()) != int(records[local_id].owner):
                    owner_mismatches += 1
            sample_id += 1
    return {
        "count_mismatches": int(count_mismatches),
        "owner_mismatches": int(owner_mismatches),
        "max_count_delta": int(max_count_delta),
        "matches_endpoint_run_counts_and_owners": count_mismatches == 0 and owner_mismatches == 0,
    }


def _profile_endpoint_records(
    sequences: list[list[tuple[Any, ...]]],
    *,
    frame_count: int,
    full_segment_count: int,
) -> dict[str, Any]:
    rows = _endpoint_record_rows(sequences)
    track_count = len(rows)
    sample_count = int(track_count * frame_count)
    total_records = sum(len(row) for frames in rows for row in frames)
    first_frame_records = sum(len(frames[0]) for frames in rows) if frame_count else 0
    transition_slots = int(track_count * max(frame_count - 1, 0))
    change_events = 0
    unchanged_events = 0
    edit_ops_total = 0
    edit_ops_nonzero: list[int] = []
    changed_row_records = 0
    same_record_tracks = 0
    unique_rows_per_track: list[int] = []
    global_unique_rows: set[tuple[EndpointRecord, ...]] = set()
    histogram: Counter[int] = Counter()

    for frames in rows:
        unique_rows = set(frames)
        unique_rows_per_track.append(len(unique_rows))
        global_unique_rows.update(unique_rows)
        if len(unique_rows) == 1:
            same_record_tracks += 1
        for left, right in zip(frames[:-1], frames[1:], strict=True):
            edit_ops = _edit_distance(left, right)
            histogram[edit_ops] += 1
            if edit_ops == 0:
                unchanged_events += 1
                continue
            change_events += 1
            edit_ops_total += edit_ops
            edit_ops_nonzero.append(edit_ops)
            changed_row_records += len(right)

    track_unique_rows = int(sum(unique_rows_per_track))
    full_segment_csr_bytes = int(full_segment_count * 12 + (sample_count + 1) * 4)
    full_endpoint_record_csr_bytes = int(total_records * 12 + (sample_count + 1) * 4)
    delta_replace_endpoint_record_bytes = int(
        (first_frame_records * 12) + ((track_count + 1) * 4) + (change_events * 12) + (changed_row_records * 12)
    )
    delta_edit_op_endpoint_record_bytes = int(
        (first_frame_records * 12) + ((track_count + 1) * 4) + (change_events * 8) + (edit_ops_total * 20)
    )
    return {
        "track_count": int(track_count),
        "sample_count": int(sample_count),
        "frame_count": int(frame_count),
        "total_endpoint_records": int(total_records),
        "first_frame_endpoint_records": int(first_frame_records),
        "avg_endpoint_records_per_sample": float(total_records) / float(max(sample_count, 1)),
        "transition_slots": int(transition_slots),
        "change_events": int(change_events),
        "unchanged_events": int(unchanged_events),
        "change_event_rate": float(change_events) / float(max(transition_slots, 1)),
        "edit_ops_total": int(edit_ops_total),
        "edit_ops_per_transition": float(edit_ops_total) / float(max(transition_slots, 1)),
        "edit_ops_per_endpoint_record": float(edit_ops_total) / float(max(total_records, 1)),
        "nonzero_edit_ops_avg": float(sum(edit_ops_nonzero)) / float(max(len(edit_ops_nonzero), 1)),
        "nonzero_edit_ops_p50": _quantile(edit_ops_nonzero, 0.50),
        "nonzero_edit_ops_p95": _quantile(edit_ops_nonzero, 0.95),
        "nonzero_edit_ops_max": max(edit_ops_nonzero) if edit_ops_nonzero else 0,
        "same_endpoint_records_all_frames_tracks": int(same_record_tracks),
        "same_endpoint_records_all_frames_ratio": float(same_record_tracks) / float(max(track_count, 1)),
        "track_unique_endpoint_record_rows": int(track_unique_rows),
        "track_unique_endpoint_record_rows_vs_samples": float(track_unique_rows) / float(max(sample_count, 1)),
        "global_unique_endpoint_record_rows": int(len(global_unique_rows)),
        "global_unique_endpoint_record_rows_vs_samples": float(len(global_unique_rows)) / float(max(sample_count, 1)),
        "edit_ops_histogram": {str(key): int(value) for key, value in sorted(histogram.items())},
        "storage_estimates": {
            "full_segment_csr_owner_length_mid_bytes": int(full_segment_csr_bytes),
            "full_endpoint_record_csr_owner_left_right_i32_bytes": int(full_endpoint_record_csr_bytes),
            "delta_replace_endpoint_record_bytes": int(delta_replace_endpoint_record_bytes),
            "delta_edit_op_endpoint_record_stream_bytes": int(delta_edit_op_endpoint_record_bytes),
            "full_endpoint_record_csr_vs_full_segment_csr": float(full_endpoint_record_csr_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "delta_replace_endpoint_record_vs_full_segment_csr": float(delta_replace_endpoint_record_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "delta_edit_op_endpoint_record_stream_vs_full_segment_csr": float(delta_edit_op_endpoint_record_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "delta_replace_endpoint_record_vs_full_endpoint_record_csr": float(delta_replace_endpoint_record_bytes)
            / float(max(full_endpoint_record_csr_bytes, 1)),
            "delta_edit_op_endpoint_record_stream_vs_full_endpoint_record_csr": float(
                delta_edit_op_endpoint_record_bytes
            )
            / float(max(full_endpoint_record_csr_bytes, 1)),
            "note": (
                "Endpoint records store owner plus left/right boundary cut ids for continuous-absorption "
                "same-owner runs. Boundary ids can recover endpoint depths; this estimates only discrete "
                "record-stream compactness, not a shipped delta replay shader."
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
    boundaries = make_boundaries_4d(sites)
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    active_sequences, _sample_meta = _build_owner_run_sequences(
        sites=sites,
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        site_rgba=site_rgba,
    )
    all_sequences, _sample_meta = _build_owner_run_sequences(
        sites=sites,
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=0.0,
        site_rgba=site_rgba,
    )
    tape = build_segment_tape(
        sites=sites,
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    endpoint_run_tape = compress_same_owner_endpoint_runs(tape)
    active_owner_run_tape = compress_same_owner_runs(
        tape=tape,
        site_rgba_f32=site_rgba,
        transmittance_threshold=transmittance_threshold,
    )
    full_segment_count = int(tape.counts_i32.detach().cpu().to(dtype=torch.int64).sum().item())
    return {
        "frames": int(frame_count),
        "render_size": int(render_size),
        "site_count": int(len(sites)),
        "active_owner_run_verification": _verify_against_owner_run_tape(
            active_sequences,
            owner_run_tape=active_owner_run_tape,
        ),
        "endpoint_run_verification": _verify_against_endpoint_run_tape(all_sequences, endpoint_run_tape),
        "active_endpoint_record_delta_tape": _profile_endpoint_records(
            active_sequences,
            frame_count=frame_count,
            full_segment_count=full_segment_count,
        ),
        "endpoint_record_delta_tape": _profile_endpoint_records(
            all_sequences,
            frame_count=frame_count,
            full_segment_count=full_segment_count,
        ),
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
    record_scale = _scale(rows, ("endpoint_record_delta_tape", "total_endpoint_records"))
    edit_scale = _scale(rows, ("endpoint_record_delta_tape", "edit_ops_total"))
    replace_storage_scale = _scale(
        rows,
        ("endpoint_record_delta_tape", "storage_estimates", "delta_replace_endpoint_record_bytes"),
    )
    edit_storage_scale = _scale(
        rows,
        ("endpoint_record_delta_tape", "storage_estimates", "delta_edit_op_endpoint_record_stream_bytes"),
    )
    last_storage = rows[-1]["endpoint_record_delta_tape"]["storage_estimates"]
    acceptance = {
        "endpoint_records_match_endpoint_run_counts_and_owners": all(
            bool(row["endpoint_run_verification"]["matches_endpoint_run_counts_and_owners"]) for row in rows
        ),
        "endpoint_record_count_sublinear_vs_frames": record_scale < frame_scale,
        "endpoint_record_edit_ops_sublinear_vs_frames": edit_scale < frame_scale,
        "delta_replace_endpoint_record_storage_sublinear_vs_frames": replace_storage_scale < frame_scale,
        "delta_edit_op_endpoint_record_storage_sublinear_vs_frames": edit_storage_scale < frame_scale,
        "last_full_endpoint_record_storage_below_full_segment_csr": last_storage[
            "full_endpoint_record_csr_vs_full_segment_csr"
        ]
        < 0.20,
        "last_delta_edit_op_endpoint_record_storage_below_full_segment_csr": last_storage[
            "delta_edit_op_endpoint_record_stream_vs_full_segment_csr"
        ]
        < 0.20,
    }
    return {
        "benchmark": "world_foam_lane2_endpoint_record_delta_tape_probe",
        "status": "informational",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": int(render_size),
        "site_count": int(site_count),
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "endpoint_record_count_scale_first_to_last": record_scale,
        "endpoint_record_edit_op_scale_first_to_last": edit_scale,
        "delta_replace_endpoint_record_storage_scale_first_to_last": replace_storage_scale,
        "delta_edit_op_endpoint_record_storage_scale_first_to_last": edit_storage_scale,
        "structural_read": {
            "records_are_continuous_endpoint_semantic": True,
            "records_match_endpoint_run_counts_and_owners": acceptance[
                "endpoint_records_match_endpoint_run_counts_and_owners"
            ],
            "not_a_shipped_delta_shader": True,
            "interpretation": (
                "Endpoint owner+boundary records are the discrete version of the continuous-absorption "
                "endpoint shader tape. This probe checks whether those already-compact records have another "
                "STAR-like delta layer across frames."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe endpoint-run owner+boundary record deltas.")
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
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_endpoint_record_delta_tape_probe_render32_2_4_8_16.json",
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
