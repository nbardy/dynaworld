#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
VARIANT_TOOLS = VARIANT_ROOT / "tools"
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
NEAR_CUT_ID = -1
FAR_CUT_ID = -2
EPS = 1.0e-8

for path in (VARIANT_ROOT, VARIANT_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gate1_realray_per_sample_reference import crossing_depth_4d, owner_at_4d  # noqa: E402
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
from probe_fused_slab_segment_tape import build_segment_tape  # noqa: E402
from probe_segment_owner_run_tape import compress_same_owner_runs  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402


@dataclass(frozen=True)
class SegmentRecord:
    owner: int
    left_cut_id: int
    right_cut_id: int
    length: float
    mid: float


@dataclass(frozen=True)
class OwnerRunRecord:
    owner: int
    left_cut_id: int
    right_cut_id: int
    length: float
    segment_count: int


def _ray_tuple(ray: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
        (float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
    )


def _dedupe_depth_ids(depth_ids: list[tuple[float, int]], *, epsilon: float = 1.0e-6) -> list[tuple[float, int]]:
    depth_ids.sort()
    unique: list[tuple[float, int]] = []
    for depth, boundary_id in depth_ids:
        if not unique or abs(depth - unique[-1][0]) > epsilon:
            unique.append((float(depth), int(boundary_id)))
    return unique


def _cut_depth(
    cut_id: int,
    *,
    boundaries: tuple[Any, ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
) -> float:
    if cut_id == NEAR_CUT_ID:
        return near
    if cut_id == FAR_CUT_ID:
        return far
    depth = crossing_depth_4d(
        boundaries[int(cut_id)],
        origin=origin,
        direction=direction,
        t=t,
        invalid_epsilon=invalid_epsilon,
    )
    if depth is None or not math.isfinite(float(depth)):
        raise ValueError(f"cut id {cut_id} did not recover a finite depth")
    return float(depth)


def _continuous_absorption_mid(left_depth: float, right_depth: float, density: float) -> float:
    length = float(right_depth - left_depth)
    if length <= EPS:
        return 0.5 * float(left_depth + right_depth)
    if density <= EPS:
        return 0.5 * float(left_depth + right_depth)
    optical_depth = float(density * length)
    if optical_depth < 1.0e-4:
        return 0.5 * float(left_depth + right_depth)
    if optical_depth > 80.0:
        return float(left_depth + 1.0 / density)
    return float(left_depth + 1.0 / density - length / math.expm1(optical_depth))


def _segment_records_for_sample(
    *,
    sites: tuple[Any, ...],
    boundaries: tuple[Any, ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
) -> tuple[SegmentRecord, ...]:
    depth_ids: list[tuple[float, int]] = []
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
            depth_ids.append((float(depth), int(boundary_id)))
    cut_depth_ids = _dedupe_depth_ids(depth_ids)
    cut_depths = [near, *(depth for depth, _boundary_id in cut_depth_ids), far]
    cut_ids = [NEAR_CUT_ID, *(boundary_id for _depth, boundary_id in cut_depth_ids), FAR_CUT_ID]
    ox, oy, oz = origin
    dx, dy, dz = direction
    records: list[SegmentRecord] = []
    for segment_id, (depth0, depth1) in enumerate(zip(cut_depths[:-1], cut_depths[1:], strict=True)):
        length = float(depth1 - depth0)
        if length <= EPS:
            continue
        mid = 0.5 * float(depth0 + depth1)
        owner = owner_at_4d(
            sites,
            x=ox + dx * mid,
            y=oy + dy * mid,
            z=oz + dz * mid,
            t=t,
        )
        records.append(
            SegmentRecord(
                owner=int(owner),
                left_cut_id=int(cut_ids[segment_id]),
                right_cut_id=int(cut_ids[segment_id + 1]),
                length=length,
                mid=mid,
            )
        )
    return tuple(records)


def _compress_owner_runs(
    records: tuple[SegmentRecord, ...],
    *,
    site_rgba: torch.Tensor,
    transmittance_threshold: float,
) -> tuple[OwnerRunRecord, ...]:
    runs: list[OwnerRunRecord] = []
    transmittance = 1.0
    current_owner: int | None = None
    left_cut_id = 0
    right_cut_id = 0
    length_sum = 0.0
    segment_count = 0

    def flush() -> None:
        nonlocal current_owner, left_cut_id, right_cut_id, length_sum, segment_count
        if current_owner is None or segment_count == 0:
            return
        runs.append(
            OwnerRunRecord(
                owner=int(current_owner),
                left_cut_id=int(left_cut_id),
                right_cut_id=int(right_cut_id),
                length=float(length_sum),
                segment_count=int(segment_count),
            )
        )
        current_owner = None
        length_sum = 0.0
        segment_count = 0

    for record in records:
        if transmittance <= transmittance_threshold:
            break
        if current_owner is not None and record.owner != current_owner:
            flush()
        if current_owner is None:
            current_owner = int(record.owner)
            left_cut_id = int(record.left_cut_id)
        right_cut_id = int(record.right_cut_id)
        length_sum += float(record.length)
        segment_count += 1
        density = max(float(site_rgba[record.owner, 3].item()), 0.0)
        transmittance *= math.exp(-density * float(record.length))
    flush()
    return tuple(runs)


def _build_owner_run_sequences(
    *,
    sites: tuple[Any, ...],
    boundaries: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    site_rgba: torch.Tensor,
    include_sample_meta: bool = True,
) -> tuple[list[list[tuple[OwnerRunRecord, ...]]], list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]]]:
    rays = rays.detach().cpu().to(dtype=torch.float32)
    frame_indices = frame_indices.detach().cpu().to(dtype=torch.long)
    sample_count, height, width, payload = rays.shape
    if payload != 6:
        raise ValueError(f"rays must have payload dimension 6, got {payload}")
    if sample_count % frame_count != 0:
        raise ValueError("sample count must be view_count * frame_count")
    view_count = int(sample_count // frame_count)
    sequences: list[list[tuple[OwnerRunRecord, ...]]] = []
    sample_meta: list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]] = []
    for view in range(view_count):
        for y in range(height):
            for x in range(width):
                frames: list[tuple[OwnerRunRecord, ...]] = []
                metas: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = []
                for frame in range(frame_count):
                    sample_index = view * frame_count + frame
                    origin, direction = _ray_tuple(rays[sample_index, y, x])
                    t = _frame_time(int(frame_indices[sample_index].item()), frame_count)
                    records = _segment_records_for_sample(
                        sites=sites,
                        boundaries=boundaries,
                        origin=origin,
                        direction=direction,
                        t=t,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                    )
                    frames.append(
                        _compress_owner_runs(
                            records,
                            site_rgba=site_rgba,
                            transmittance_threshold=transmittance_threshold,
                        )
                    )
                    if include_sample_meta:
                        metas.append((origin, direction, t))
                sequences.append(frames)
                if include_sample_meta:
                    sample_meta.append(metas)
    return sequences, sample_meta


def _verify_against_owner_run_tape(
    sequences: list[list[tuple[OwnerRunRecord, ...]]],
    *,
    owner_run_tape: Any,
) -> dict[str, Any]:
    offsets = owner_run_tape.offsets_i32.detach().cpu()
    owners = owner_run_tape.owners_i32.detach().cpu()
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
            for local_id, record in enumerate(records[:expected_count]):
                if int(owners[start + local_id].item()) != int(record.owner):
                    owner_mismatches += 1
            sample_id += 1
    return {
        "count_mismatches": int(count_mismatches),
        "owner_mismatches": int(owner_mismatches),
        "max_count_delta": int(max_count_delta),
        "matches_current_owner_run_counts_and_owners": count_mismatches == 0 and owner_mismatches == 0,
    }


def _owner_run_row_signature(records: tuple[OwnerRunRecord, ...]) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(record.owner), int(record.left_cut_id), int(record.right_cut_id)) for record in records)


def _profile_packed_delta_owner_run_storage(
    sequences: list[list[tuple[OwnerRunRecord, ...]]],
    *,
    frame_count: int,
) -> dict[str, Any]:
    track_count = len(sequences)
    base_record_count = 0
    change_event_count = 0
    change_record_count = 0
    unchanged_frame_rows = 0
    changed_frame_rows = 0

    for frames in sequences:
        if len(frames) != frame_count:
            raise ValueError("every owner-run sequence must have frame_count rows")
        if frame_count == 0:
            continue
        previous = _owner_run_row_signature(frames[0])
        base_record_count += len(previous)
        unchanged_frame_rows += 1
        for records in frames[1:]:
            current = _owner_run_row_signature(records)
            if current == previous:
                unchanged_frame_rows += 1
                continue
            change_event_count += 1
            changed_frame_rows += 1
            change_record_count += len(current)
            previous = current

    base_offsets_bytes = int((track_count + 1) * 4)
    track_change_offsets_bytes = int((track_count + 1) * 4)
    change_frame_bytes = int(change_event_count * 4)
    change_offsets_bytes = int((change_event_count + 1) * 4)
    separate_record_bytes = int((base_record_count + change_record_count) * 3 * 4)
    packed_record_bytes = int((base_record_count + change_record_count) * 4)
    separate_i32_bytes = int(
        base_offsets_bytes
        + track_change_offsets_bytes
        + change_frame_bytes
        + change_offsets_bytes
        + separate_record_bytes
    )
    packed_i32_bytes = int(
        base_offsets_bytes
        + track_change_offsets_bytes
        + change_frame_bytes
        + change_offsets_bytes
        + packed_record_bytes
    )
    sample_count = int(track_count * frame_count)
    materialized_boundary_csr_bytes = int(
        sum(len(row) for frames in sequences for row in frames) * 12 + (sample_count + 1) * 4
    )
    return {
        "track_count": int(track_count),
        "frame_count": int(frame_count),
        "base_record_count": int(base_record_count),
        "change_event_count": int(change_event_count),
        "change_record_count": int(change_record_count),
        "unchanged_frame_rows": int(unchanged_frame_rows),
        "changed_frame_rows": int(changed_frame_rows),
        "bytes_by_key": {
            "base_offsets_i32": base_offsets_bytes,
            "track_change_offsets_i32": track_change_offsets_bytes,
            "change_frame_i32": change_frame_bytes,
            "change_offsets_i32": change_offsets_bytes,
            "separate_owner_left_right_i32": separate_record_bytes,
            "packed_owner_left_right_i32": packed_record_bytes,
        },
        "separate_i32_storage_bytes": separate_i32_bytes,
        "packed_i32_storage_bytes": packed_i32_bytes,
        "packed_i32_vs_materialized_boundary_csr": float(packed_i32_bytes)
        / float(max(materialized_boundary_csr_bytes, 1)),
    }


def _profile_endpoint_density_replay(
    sequences: list[list[tuple[OwnerRunRecord, ...]]],
    sample_meta: list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]],
    *,
    owner_run_tape: Any,
    boundaries: tuple[Any, ...],
    site_rgba: torch.Tensor,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
) -> dict[str, Any]:
    offsets = owner_run_tape.offsets_i32.detach().cpu()
    current_lengths = owner_run_tape.lengths_f32.detach().cpu()
    current_mids = owner_run_tape.mids_f32.detach().cpu()
    site_rgba_cpu = site_rgba.detach().cpu().to(dtype=torch.float32)
    max_length_error = 0.0
    max_mid_error = 0.0
    sum_mid_error = 0.0
    mid_count = 0
    max_alpha_error = 0.0
    max_depth_error = 0.0
    sum_depth_error = 0.0
    sample_count = 0
    sample_id = 0
    for frames, metas in zip(sequences, sample_meta, strict=True):
        for records, (origin, direction, t) in zip(frames, metas, strict=True):
            start = int(offsets[sample_id].item())
            endpoint_transmittance = 1.0
            current_transmittance = 1.0
            endpoint_alpha = 0.0
            current_alpha = 0.0
            endpoint_depth_num = 0.0
            current_depth_num = 0.0
            for local_id, record in enumerate(records):
                left_depth = _cut_depth(
                    record.left_cut_id,
                    boundaries=boundaries,
                    origin=origin,
                    direction=direction,
                    t=t,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                )
                right_depth = _cut_depth(
                    record.right_cut_id,
                    boundaries=boundaries,
                    origin=origin,
                    direction=direction,
                    t=t,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                )
                endpoint_length = max(float(right_depth - left_depth), 0.0)
                current_length = max(float(current_lengths[start + local_id].item()), 0.0)
                density = max(float(site_rgba_cpu[record.owner, 3].item()), 0.0)
                endpoint_mid = _continuous_absorption_mid(left_depth, right_depth, density)
                current_mid = float(current_mids[start + local_id].item())
                mid_error = abs(endpoint_mid - current_mid)
                max_length_error = max(max_length_error, abs(endpoint_length - current_length))
                max_mid_error = max(max_mid_error, mid_error)
                sum_mid_error += mid_error
                mid_count += 1

                endpoint_segment_alpha = 1.0 - math.exp(-density * endpoint_length)
                current_segment_alpha = 1.0 - math.exp(-density * current_length)
                endpoint_weight = endpoint_transmittance * endpoint_segment_alpha
                current_weight = current_transmittance * current_segment_alpha
                endpoint_alpha += endpoint_weight
                current_alpha += current_weight
                endpoint_depth_num += endpoint_weight * endpoint_mid
                current_depth_num += current_weight * current_mid
                endpoint_transmittance *= math.exp(-density * endpoint_length)
                current_transmittance *= math.exp(-density * current_length)
                if endpoint_transmittance <= transmittance_threshold and current_transmittance <= transmittance_threshold:
                    break

            endpoint_depth = endpoint_depth_num / endpoint_alpha if endpoint_alpha > EPS else far
            current_depth = current_depth_num / current_alpha if current_alpha > EPS else far
            depth_error = abs(endpoint_depth - current_depth)
            max_alpha_error = max(max_alpha_error, abs(endpoint_alpha - current_alpha))
            max_depth_error = max(max_depth_error, depth_error)
            sum_depth_error += depth_error
            sample_count += 1
            sample_id += 1
    return {
        "endpoint_continuous_density_mid_formula": "left + 1 / density - length / expm1(density * length)",
        "max_endpoint_length_abs_error_vs_current_owner_run": float(max_length_error),
        "max_endpoint_density_mid_abs_error_vs_current_owner_run": float(max_mid_error),
        "mean_endpoint_density_mid_abs_error_vs_current_owner_run": float(sum_mid_error) / float(max(mid_count, 1)),
        "max_endpoint_alpha_abs_error_vs_current_owner_run": float(max_alpha_error),
        "max_endpoint_density_depth_abs_error_vs_current_owner_run": float(max_depth_error),
        "mean_endpoint_density_depth_abs_error_vs_current_owner_run": float(sum_depth_error)
        / float(max(sample_count, 1)),
        "endpoint_only_depth_matches_current_segment_mid_depth": max_depth_error <= 5.0e-5,
        "interpretation": (
            "Endpoint-only continuous absorption depth is well-defined and keeps RGB/alpha geometry intact, "
            "but it does not match the current segment-mid depth tape after same-owner internal cuts are "
            "discarded. Exact current-depth replay needs internal moments/cuts or an explicit depth semantic "
            "change."
        ),
    }


def _profile_owner_run_boundary_tape(
    sequences: list[list[tuple[OwnerRunRecord, ...]]],
    sample_meta: list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]],
    *,
    boundaries: tuple[Any, ...],
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    full_segment_count: int,
) -> dict[str, Any]:
    track_count = len(sequences)
    sample_count = int(track_count * frame_count)
    total_runs = sum(len(row) for frames in sequences for row in frames)
    merged_segments = sum(record.segment_count for frames in sequences for row in frames for record in row)
    multi_segment_runs = sum(1 for frames in sequences for row in frames for record in row if record.segment_count > 1)
    max_segments_per_run = max((record.segment_count for frames in sequences for row in frames for record in row), default=0)
    max_length_error = 0.0
    for frames, metas in zip(sequences, sample_meta, strict=True):
        for records, (origin, direction, t) in zip(frames, metas, strict=True):
            for record in records:
                left_depth = _cut_depth(
                    record.left_cut_id,
                    boundaries=boundaries,
                    origin=origin,
                    direction=direction,
                    t=t,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                )
                right_depth = _cut_depth(
                    record.right_cut_id,
                    boundaries=boundaries,
                    origin=origin,
                    direction=direction,
                    t=t,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                )
                max_length_error = max(max_length_error, abs(float(right_depth - left_depth) - float(record.length)))
    active_segment_csr_bytes = int(merged_segments * 12 + (sample_count + 1) * 4)
    full_segment_csr_bytes = int(full_segment_count * 12 + (sample_count + 1) * 4)
    owner_run_owner_length_csr_bytes = int(total_runs * 8 + (sample_count + 1) * 4)
    owner_run_length_mid_csr_bytes = int(total_runs * 12 + (sample_count + 1) * 4)
    owner_run_boundary_id_csr_bytes = int(total_runs * 12 + (sample_count + 1) * 4)
    packed_delta_storage = _profile_packed_delta_owner_run_storage(sequences, frame_count=frame_count)
    packed_delta_bytes = int(packed_delta_storage["packed_i32_storage_bytes"])
    separate_delta_bytes = int(packed_delta_storage["separate_i32_storage_bytes"])
    return {
        "track_count": int(track_count),
        "sample_count": int(sample_count),
        "frame_count": int(frame_count),
        "total_runs": int(total_runs),
        "full_original_segments": int(full_segment_count),
        "merged_original_segments": int(merged_segments),
        "runs_vs_original_segments": float(total_runs) / float(max(merged_segments, 1)),
        "runs_vs_full_original_segments": float(total_runs) / float(max(full_segment_count, 1)),
        "avg_runs_per_sample": float(total_runs) / float(max(sample_count, 1)),
        "multi_segment_runs": int(multi_segment_runs),
        "multi_segment_run_ratio": float(multi_segment_runs) / float(max(total_runs, 1)),
        "max_segments_per_run": int(max_segments_per_run),
        "max_endpoint_length_abs_error": float(max_length_error),
        "storage_estimates": {
            "full_segment_csr_owner_length_mid_bytes": full_segment_csr_bytes,
            "active_segment_csr_owner_length_mid_bytes": active_segment_csr_bytes,
            "owner_run_owner_length_csr_bytes": owner_run_owner_length_csr_bytes,
            "owner_run_length_mid_csr_bytes": owner_run_length_mid_csr_bytes,
            "owner_run_boundary_id_csr_bytes": owner_run_boundary_id_csr_bytes,
            "owner_run_boundary_delta_separate_i32_bytes": separate_delta_bytes,
            "owner_run_boundary_delta_packed_i32_bytes": packed_delta_bytes,
            "owner_run_boundary_id_vs_full_segment_csr": float(owner_run_boundary_id_csr_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "owner_run_boundary_id_vs_active_segment_csr": float(owner_run_boundary_id_csr_bytes)
            / float(max(active_segment_csr_bytes, 1)),
            "owner_run_boundary_id_vs_current_owner_run_length_mid_csr": float(owner_run_boundary_id_csr_bytes)
            / float(max(owner_run_length_mid_csr_bytes, 1)),
            "owner_run_boundary_delta_packed_i32_vs_current_owner_run_owner_length_csr": float(packed_delta_bytes)
            / float(max(owner_run_owner_length_csr_bytes, 1)),
            "owner_run_boundary_delta_packed_i32_vs_full_segment_csr": float(packed_delta_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "owner_run_boundary_delta_packed_i32_vs_active_segment_csr": float(packed_delta_bytes)
            / float(max(active_segment_csr_bytes, 1)),
            "owner_run_boundary_delta_packed_i32_vs_current_owner_run_length_mid_csr": float(packed_delta_bytes)
            / float(max(owner_run_length_mid_csr_bytes, 1)),
            "owner_run_boundary_delta_packed_i32_vs_materialized_boundary_id_csr": float(packed_delta_bytes)
            / float(max(owner_run_boundary_id_csr_bytes, 1)),
            "packed_delta": packed_delta_storage,
            "note": (
                "Boundary endpoint ids replace per-run length/mid floats with left/right cut ids. Existing "
                "boundary-depth coefficients plus ray coefficients recover run length exactly. The packed-delta "
                "estimate stores one base row per track plus changed frame rows with packed owner/left/right i32 "
                "records; endpoint-only continuous density depth is tested separately because it does not match "
                "the current midpoint tape."
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
    sequences, sample_meta = _build_owner_run_sequences(
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
    current_owner_run = compress_same_owner_runs(
        tape=tape,
        site_rgba_f32=site_rgba,
        transmittance_threshold=transmittance_threshold,
    )
    full_segment_count = int(tape.counts_i32.detach().cpu().to(dtype=torch.int64).sum().item())
    return {
        "frames": frame_count,
        "render_size": render_size,
        "site_count": len(sites),
        "verification": _verify_against_owner_run_tape(sequences, owner_run_tape=current_owner_run),
        "endpoint_density_replay": _profile_endpoint_density_replay(
            sequences,
            sample_meta,
            owner_run_tape=current_owner_run,
            boundaries=boundaries,
            site_rgba=site_rgba,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
        ),
        "owner_run_boundary_tape": _profile_owner_run_boundary_tape(
            sequences,
            sample_meta,
            boundaries=boundaries,
            frame_count=frame_count,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            full_segment_count=full_segment_count,
        ),
    }


def _scale(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    def value(row: dict[str, Any]) -> Any:
        current: Any = row
        for key in path:
            current = current[key]
        return current

    return float(value(rows[-1])) / float(max(value(rows[0]), 1))


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
    run_scale = _scale(rows, ("owner_run_boundary_tape", "total_runs"))
    storage_scale = _scale(rows, ("owner_run_boundary_tape", "storage_estimates", "owner_run_boundary_id_csr_bytes"))
    packed_delta_storage_scale = _scale(
        rows,
        ("owner_run_boundary_tape", "storage_estimates", "owner_run_boundary_delta_packed_i32_bytes"),
    )
    max_length_error = max(float(row["owner_run_boundary_tape"]["max_endpoint_length_abs_error"]) for row in rows)
    max_endpoint_depth_error = max(
        float(row["endpoint_density_replay"]["max_endpoint_density_depth_abs_error_vs_current_owner_run"])
        for row in rows
    )
    acceptance = {
        "matches_current_owner_run_counts_and_owners": all(
            bool(row["verification"]["matches_current_owner_run_counts_and_owners"]) for row in rows
        ),
        "endpoint_ids_recover_run_lengths": max_length_error <= 5.0e-5,
        "endpoint_continuous_density_depth_matches_current_segment_mid_depth": max_endpoint_depth_error <= 5.0e-5,
        "owner_run_boundary_storage_below_full_at_max_frame": rows[-1]["owner_run_boundary_tape"][
            "storage_estimates"
        ]["owner_run_boundary_id_vs_full_segment_csr"]
        < 0.25,
        "owner_run_boundary_run_count_sublinear_vs_frames": frame_scale <= 1.0 or run_scale < frame_scale,
        "owner_run_boundary_packed_delta_storage_sublinear_vs_frames": frame_scale <= 1.0
        or packed_delta_storage_scale < frame_scale,
        "owner_run_boundary_packed_delta_storage_below_materialized_csr_at_max_frame": rows[-1][
            "owner_run_boundary_tape"
        ]["storage_estimates"]["owner_run_boundary_delta_packed_i32_vs_materialized_boundary_id_csr"]
        < 1.0,
        "owner_run_boundary_packed_delta_storage_below_current_nomid_csr_at_max_frame": rows[-1][
            "owner_run_boundary_tape"
        ]["storage_estimates"]["owner_run_boundary_delta_packed_i32_vs_current_owner_run_owner_length_csr"]
        < 1.0,
    }
    return {
        "benchmark": "world_foam_lane2_owner_run_boundary_tape_probe",
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
        "owner_run_boundary_run_scale_first_to_last": run_scale,
        "owner_run_boundary_storage_scale_first_to_last": storage_scale,
        "owner_run_boundary_packed_delta_storage_scale_first_to_last": packed_delta_storage_scale,
        "max_endpoint_length_abs_error": max_length_error,
        "max_endpoint_density_depth_abs_error_vs_current_owner_run": max_endpoint_depth_error,
        "structural_read": {
            "owner_runs_match_current_length_mid_tape_counts_and_owners": acceptance[
                "matches_current_owner_run_counts_and_owners"
            ],
            "boundary_endpoint_ids_replace_length_mid_storage": True,
            "length_recovered_from_boundary_coefficients": acceptance["endpoint_ids_recover_run_lengths"],
            "packed_delta_reuses_unchanged_track_rows": acceptance[
                "owner_run_boundary_packed_delta_storage_below_materialized_csr_at_max_frame"
            ],
            "packed_delta_below_current_nomid_owner_run_storage": acceptance[
                "owner_run_boundary_packed_delta_storage_below_current_nomid_csr_at_max_frame"
            ],
            "packed_delta_storage_sublinear_vs_frames": acceptance[
                "owner_run_boundary_packed_delta_storage_sublinear_vs_frames"
            ],
            "endpoint_continuous_density_mid_tested": True,
            "endpoint_continuous_density_depth_matches_current_segment_mid_depth": acceptance[
                "endpoint_continuous_density_depth_matches_current_segment_mid_depth"
            ],
            "exact_current_depth_needs_internal_moments_or_semantic_change": not acceptance[
                "endpoint_continuous_density_depth_matches_current_segment_mid_depth"
            ],
            "interpretation": (
                "Owner-run boundary endpoint records preserve the compressed owner sequence and recover run "
                "length from cut ids plus boundary/ray coefficients. Endpoint-only continuous absorption depth "
                "is a plausible replay semantic, but it does not reproduce the current segment-mid depth after "
                "same-owner internal cuts are discarded."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe owner-run boundary endpoint tapes.")
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
        default=RESULTS_DIR / "2026-05-15_owner_run_boundary_tape_probe_render32_2_4_8_16.json",
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
