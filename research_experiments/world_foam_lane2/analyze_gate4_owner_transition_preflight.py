#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gate4_affine_slab_tape import (
    _boundary_other_by_owner,
    _cut_arrays_from_ordered_depth_ids,
    _owner_indices_for_points,
    build_gate4_affine_slab_tape,
)
from gate4_moving_ray_slab_compiler import (
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    parse_int_list,
)
from probe_gate4_affine_candidate_csr_capacity import (
    _fit_loaded_frame_count,
    _ratio_last_first,
    candidate_csr_storage_breakdown,
)


DYNAWORLD = Path(__file__).resolve().parents[2]
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"


@dataclass(frozen=True)
class OwnerTransitionStats:
    active_segments: int = 0
    active_samples: int = 0
    transition_owner_scans: int = 0
    boundary_checks: int = 0
    unrelated_boundary_crossings: int = 0
    owner_switches: int = 0
    fallback_resets: int = 0
    ambiguous_boundary_groups: int = 0
    owner_runs: int = 0
    mismatches: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "active_segments": int(self.active_segments),
            "active_samples": int(self.active_samples),
            "transition_owner_scans": int(self.transition_owner_scans),
            "boundary_checks": int(self.boundary_checks),
            "unrelated_boundary_crossings": int(self.unrelated_boundary_crossings),
            "owner_switches": int(self.owner_switches),
            "fallback_resets": int(self.fallback_resets),
            "ambiguous_boundary_groups": int(self.ambiguous_boundary_groups),
            "owner_runs": int(self.owner_runs),
            "mismatches": int(self.mismatches),
        }

    def __add__(self, other: "OwnerTransitionStats") -> "OwnerTransitionStats":
        return OwnerTransitionStats(
            active_segments=self.active_segments + other.active_segments,
            active_samples=self.active_samples + other.active_samples,
            transition_owner_scans=self.transition_owner_scans + other.transition_owner_scans,
            boundary_checks=self.boundary_checks + other.boundary_checks,
            unrelated_boundary_crossings=self.unrelated_boundary_crossings
            + other.unrelated_boundary_crossings,
            owner_switches=self.owner_switches + other.owner_switches,
            fallback_resets=self.fallback_resets + other.fallback_resets,
            ambiguous_boundary_groups=self.ambiguous_boundary_groups
            + other.ambiguous_boundary_groups,
            owner_runs=self.owner_runs + other.owner_runs,
            mismatches=self.mismatches + other.mismatches,
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1.0e-12:
        return 0.0 if abs(float(numerator)) <= 1.0e-12 else float("inf")
    return float(numerator) / float(denominator)


def analyze_cut_owner_transition(
    *,
    cut_depths: np.ndarray,
    cut_ids: np.ndarray,
    full_owners_by_nonempty_segment: np.ndarray,
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    boundary_other_by_owner: np.ndarray,
    site_density: np.ndarray,
    transmittance_threshold: float,
    unrelated_policy: str,
    epsilon: float = 1.0e-8,
) -> OwnerTransitionStats:
    """Measure owner-transition replay against authoritative per-segment owners.

    `unrelated_policy="keep"` is the ideal scan-free update: crossing a boundary
    that does not involve the current owner leaves the owner unchanged.
    `unrelated_policy="fallback"` mirrors the earlier ownerupdate shader, which
    invalidates the cached owner and rescans the next segment.
    """
    if unrelated_policy not in {"keep", "fallback"}:
        raise ValueError("unrelated_policy must be 'keep' or 'fallback'")
    if int(cut_depths.shape[0]) != int(cut_ids.shape[0]):
        raise ValueError("cut_depths and cut_ids must have the same length")

    segment_count = int(cut_depths.shape[0]) - 1
    current_owner = -1
    previous_full_owner = -1
    full_owner_cursor = 0
    transmittance = 1.0
    stats = OwnerTransitionStats()

    for segment_id in range(segment_count):
        length = float(cut_depths[segment_id + 1] - cut_depths[segment_id])
        if length <= float(epsilon):
            continue
        if transmittance <= float(transmittance_threshold):
            break
        if full_owner_cursor >= int(full_owners_by_nonempty_segment.shape[0]):
            raise ValueError("full owner sequence is shorter than the nonempty cut segments")
        full_owner = int(full_owners_by_nonempty_segment[full_owner_cursor])
        full_owner_cursor += 1

        active_samples = 1 if stats.active_segments == 0 else 0
        owner_scans = 0
        mismatches = 0
        if current_owner < 0:
            current_owner = full_owner
            owner_scans = 1
        elif current_owner != full_owner:
            mismatches = 1

        owner_runs = 1 if previous_full_owner < 0 or previous_full_owner != full_owner else 0
        previous_full_owner = full_owner
        density = max(float(site_density[full_owner]), 0.0)
        transmittance *= math.exp(-density * length)

        boundary_checks = 0
        unrelated = 0
        switches = 0
        fallback_resets = 0
        right_cut_id = int(cut_ids[segment_id + 1])
        if right_cut_id >= 0:
            boundary_checks = 1
            other = -1
            if (
                0 <= current_owner < int(boundary_other_by_owner.shape[0])
                and 0 <= right_cut_id < int(boundary_other_by_owner.shape[1])
            ):
                other = int(boundary_other_by_owner[current_owner, right_cut_id])
            if other >= 0:
                left = int(boundary_left[right_cut_id])
                right = int(boundary_right[right_cut_id])
                if current_owner == left or current_owner == right:
                    current_owner = other
                    switches = 1
                else:
                    mismatches += 1
            else:
                unrelated = 1
                if unrelated_policy == "fallback":
                    current_owner = -1
                    fallback_resets = 1

        stats += OwnerTransitionStats(
            active_segments=1,
            active_samples=active_samples,
            transition_owner_scans=owner_scans,
            boundary_checks=boundary_checks,
            unrelated_boundary_crossings=unrelated,
            owner_switches=switches,
            fallback_resets=fallback_resets,
            owner_runs=owner_runs,
            mismatches=mismatches,
        )

    return stats


def _cut_groups_from_ordered_depth_ids(
    *,
    depths: np.ndarray,
    boundary_ids: np.ndarray,
    near: float,
    far: float,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    if depths.size != boundary_ids.size:
        raise ValueError("depths and boundary_ids must have the same size")
    if depths.size == 0:
        return np.array([near, far], dtype=np.float64), (tuple(), tuple())
    kept_depths: list[float] = []
    groups: list[list[int]] = []
    for depth, boundary_id in zip(depths, boundary_ids, strict=True):
        depth_value = float(depth)
        if not kept_depths or abs(depth_value - kept_depths[-1]) > float(epsilon):
            kept_depths.append(depth_value)
            groups.append([int(boundary_id)])
        else:
            groups[-1].append(int(boundary_id))
    cut_depths = np.empty(len(kept_depths) + 2, dtype=np.float64)
    cut_depths[0] = float(near)
    cut_depths[-1] = float(far)
    cut_depths[1:-1] = np.asarray(kept_depths, dtype=np.float64)
    return cut_depths, (tuple(), *(tuple(group) for group in groups), tuple())


def analyze_cut_owner_transition_groups(
    *,
    cut_depths: np.ndarray,
    cut_boundary_groups: tuple[tuple[int, ...], ...],
    full_owners_by_nonempty_segment: np.ndarray,
    boundary_left: np.ndarray,
    boundary_right: np.ndarray,
    boundary_other_by_owner: np.ndarray,
    site_density: np.ndarray,
    transmittance_threshold: float,
    epsilon: float = 1.0e-8,
) -> OwnerTransitionStats:
    """Tie-aware owner transition replay using all boundary ids at a cut depth."""
    if int(cut_depths.shape[0]) != len(cut_boundary_groups):
        raise ValueError("cut_depths and cut_boundary_groups must have the same length")

    segment_count = int(cut_depths.shape[0]) - 1
    current_owner = -1
    previous_full_owner = -1
    full_owner_cursor = 0
    transmittance = 1.0
    stats = OwnerTransitionStats()

    for segment_id in range(segment_count):
        length = float(cut_depths[segment_id + 1] - cut_depths[segment_id])
        if length <= float(epsilon):
            continue
        if transmittance <= float(transmittance_threshold):
            break
        if full_owner_cursor >= int(full_owners_by_nonempty_segment.shape[0]):
            raise ValueError("full owner sequence is shorter than the nonempty cut segments")
        full_owner = int(full_owners_by_nonempty_segment[full_owner_cursor])
        full_owner_cursor += 1

        active_samples = 1 if stats.active_segments == 0 else 0
        owner_scans = 0
        mismatches = 0
        if current_owner < 0:
            current_owner = full_owner
            owner_scans = 1
        elif current_owner != full_owner:
            mismatches = 1

        owner_runs = 1 if previous_full_owner < 0 or previous_full_owner != full_owner else 0
        previous_full_owner = full_owner
        density = max(float(site_density[full_owner]), 0.0)
        transmittance *= math.exp(-density * length)

        boundary_checks = 0
        unrelated = 0
        switches = 0
        fallback_resets = 0
        ambiguous_groups = 0
        right_boundary_ids = cut_boundary_groups[segment_id + 1]
        if right_boundary_ids:
            matching_others: list[int] = []
            for boundary_id in right_boundary_ids:
                if boundary_id < 0:
                    continue
                boundary_checks += 1
                other = -1
                if (
                    0 <= current_owner < int(boundary_other_by_owner.shape[0])
                    and 0 <= int(boundary_id) < int(boundary_other_by_owner.shape[1])
                ):
                    other = int(boundary_other_by_owner[current_owner, int(boundary_id)])
                if other >= 0:
                    matching_others.append(other)
                else:
                    unrelated += 1
            if len(matching_others) == 1:
                current_owner = matching_others[0]
                switches = 1
            elif len(matching_others) > 1:
                current_owner = -1
                fallback_resets = 1
                ambiguous_groups = 1

        stats += OwnerTransitionStats(
            active_segments=1,
            active_samples=active_samples,
            transition_owner_scans=owner_scans,
            boundary_checks=boundary_checks,
            unrelated_boundary_crossings=unrelated,
            owner_switches=switches,
            fallback_resets=fallback_resets,
            ambiguous_boundary_groups=ambiguous_groups,
            owner_runs=owner_runs,
            mismatches=mismatches,
        )

    return stats


def _site_arrays(sites: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([(float(site.x), float(site.y), float(site.z)) for site in sites], dtype=np.float64),
        np.array([float(site.t) for site in sites], dtype=np.float64),
        np.array([float(site.weight) for site in sites], dtype=np.float64),
        np.array([float(site.rgba[3]) for site in sites], dtype=np.float64),
    )


def _full_owners_for_cut_segments(
    *,
    cut_depths: np.ndarray,
    ray_coeff: np.ndarray,
    t: float,
    site_xyz: np.ndarray,
    site_t: np.ndarray,
    site_weight: np.ndarray,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    points: list[tuple[float, float, float, float]] = []
    ox = float(ray_coeff[0] + ray_coeff[3] * t)
    oy = float(ray_coeff[1] + ray_coeff[4] * t)
    oz = float(ray_coeff[2] + ray_coeff[5] * t)
    dx = float(ray_coeff[6] + ray_coeff[9] * t)
    dy = float(ray_coeff[7] + ray_coeff[10] * t)
    dz = float(ray_coeff[8] + ray_coeff[11] * t)
    for segment_id in range(int(cut_depths.shape[0]) - 1):
        length = float(cut_depths[segment_id + 1] - cut_depths[segment_id])
        if length <= float(epsilon):
            continue
        mid_depth = 0.5 * float(cut_depths[segment_id] + cut_depths[segment_id + 1])
        points.append((ox + dx * mid_depth, oy + dy * mid_depth, oz + dz * mid_depth, float(t)))
    if not points:
        return np.empty((0,), dtype=np.int64)
    return _owner_indices_for_points(
        points=np.array(points, dtype=np.float64),
        site_xyz=site_xyz,
        site_t=site_t,
        site_weight=site_weight,
    )


def _analyze_tape_owner_transitions(
    *,
    tape: Any,
    sites: tuple[Any, ...],
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
) -> dict[str, Any]:
    row_offsets = tape.row_offsets.detach().cpu().to(dtype=torch.long).numpy()
    candidate_ids = tape.candidate_ids.detach().cpu().to(dtype=torch.long).numpy()
    coeffs = tape.candidate_depth_coeffs.detach().cpu().to(dtype=torch.float64).numpy()
    frame_t = tape.frame_t.detach().cpu().to(dtype=torch.float64).numpy()
    row_index = tape.row_index.detach().cpu().to(dtype=torch.long).numpy()
    ray_coeff = tape.ray_coeff.detach().cpu().to(dtype=torch.float64).numpy()
    boundaries = make_boundaries_4d(sites)
    boundary_left = np.array([int(boundary.left) for boundary in boundaries], dtype=np.int64)
    boundary_right = np.array([int(boundary.right) for boundary in boundaries], dtype=np.int64)
    boundary_other = _boundary_other_by_owner(
        boundary_left=boundary_left,
        boundary_right=boundary_right,
        site_count=len(sites),
    )
    site_xyz, site_t, site_weight, site_density = _site_arrays(sites)

    keep_stats = OwnerTransitionStats()
    group_stats = OwnerTransitionStats()
    fallback_stats = OwnerTransitionStats()
    candidate_depth_evaluations = 0
    valid_depth_values = 0
    sample_count = 0
    active_sample_count = 0

    for track_id in range(int(tape.track_count)):
        row = int(row_index[track_id])
        if row < 0 or row >= int(tape.row_count):
            continue
        track_ray_coeff = ray_coeff[track_id]
        for frame_id in range(int(tape.frame_count)):
            sample_count += 1
            t = float(frame_t[frame_id])
            slab_id = min(int(math.floor(t * int(tape.time_slab_count))), int(tape.time_slab_count) - 1)
            csr_row = row * int(tape.time_slab_count) + slab_id
            begin = int(row_offsets[csr_row])
            end = int(row_offsets[csr_row + 1])
            candidate_depth_evaluations += max(0, end - begin)
            if end > begin:
                local_coeffs = coeffs[begin:end]
                denom = local_coeffs[:, 2] + local_coeffs[:, 3] * t
                numer = local_coeffs[:, 0] + local_coeffs[:, 1] * t
                with np.errstate(divide="ignore", invalid="ignore"):
                    depths = numer / denom
                valid = (
                    (np.abs(denom) >= float(invalid_epsilon))
                    & np.isfinite(depths)
                    & (depths >= float(near))
                    & (depths <= float(far))
                )
                valid_indices = np.flatnonzero(valid)
                valid_depth_values += int(valid_indices.size)
                if valid_indices.size:
                    valid_depths = depths[valid_indices]
                    order = np.argsort(valid_depths, kind="mergesort")
                    ordered_indices = valid_indices[order]
                    sorted_depths = depths[ordered_indices]
                    sorted_ids = candidate_ids[begin:end][ordered_indices]
                else:
                    sorted_depths = np.empty((0,), dtype=np.float64)
                    sorted_ids = np.empty((0,), dtype=np.int64)
            else:
                sorted_depths = np.empty((0,), dtype=np.float64)
                sorted_ids = np.empty((0,), dtype=np.int64)

            cut_depths, cut_ids = _cut_arrays_from_ordered_depth_ids(
                depths=sorted_depths,
                boundary_ids=sorted_ids,
                near=float(near),
                far=float(far),
            )
            group_cut_depths, cut_boundary_groups = _cut_groups_from_ordered_depth_ids(
                depths=sorted_depths,
                boundary_ids=sorted_ids,
                near=float(near),
                far=float(far),
            )
            full_owners = _full_owners_for_cut_segments(
                cut_depths=group_cut_depths,
                ray_coeff=track_ray_coeff,
                t=t,
                site_xyz=site_xyz,
                site_t=site_t,
                site_weight=site_weight,
            )
            if full_owners.size == 0:
                continue
            keep_sample = analyze_cut_owner_transition(
                cut_depths=cut_depths,
                cut_ids=cut_ids,
                full_owners_by_nonempty_segment=full_owners,
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_other_by_owner=boundary_other,
                site_density=site_density,
                transmittance_threshold=transmittance_threshold,
                unrelated_policy="keep",
            )
            group_sample = analyze_cut_owner_transition_groups(
                cut_depths=group_cut_depths,
                cut_boundary_groups=cut_boundary_groups,
                full_owners_by_nonempty_segment=full_owners,
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_other_by_owner=boundary_other,
                site_density=site_density,
                transmittance_threshold=transmittance_threshold,
            )
            fallback_sample = analyze_cut_owner_transition(
                cut_depths=cut_depths,
                cut_ids=cut_ids,
                full_owners_by_nonempty_segment=full_owners,
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                boundary_other_by_owner=boundary_other,
                site_density=site_density,
                transmittance_threshold=transmittance_threshold,
                unrelated_policy="fallback",
            )
            if keep_sample.active_segments > 0:
                active_sample_count += 1
            keep_stats += keep_sample
            group_stats += group_sample
            fallback_stats += fallback_sample

    baseline_owner_scans = int(keep_stats.active_segments)
    keep_scans = int(keep_stats.transition_owner_scans)
    group_scans = int(group_stats.transition_owner_scans)
    fallback_scans = int(fallback_stats.transition_owner_scans)
    return {
        "sample_count": int(sample_count),
        "active_sample_count": int(active_sample_count),
        "candidate_depth_evaluations": int(candidate_depth_evaluations),
        "valid_depth_values": int(valid_depth_values),
        "baseline_owner_scans": int(baseline_owner_scans),
        "ownerkeep": {
            **keep_stats.as_dict(),
            "exact": bool(keep_stats.mismatches == 0),
            "owner_scan_ratio_vs_baseline": _safe_ratio(keep_scans, baseline_owner_scans),
            "owner_scan_reduction_vs_baseline": _safe_ratio(baseline_owner_scans, keep_scans),
            "unrelated_boundary_fraction": _safe_ratio(
                keep_stats.unrelated_boundary_crossings,
                keep_stats.boundary_checks,
            ),
        },
        "ownergroup_keep": {
            **group_stats.as_dict(),
            "exact": bool(group_stats.mismatches == 0),
            "owner_scan_ratio_vs_baseline": _safe_ratio(group_scans, baseline_owner_scans),
            "owner_scan_reduction_vs_baseline": _safe_ratio(baseline_owner_scans, group_scans),
            "unrelated_boundary_fraction": _safe_ratio(
                group_stats.unrelated_boundary_crossings,
                group_stats.boundary_checks,
            ),
            "ambiguous_group_fraction": _safe_ratio(
                group_stats.ambiguous_boundary_groups,
                group_stats.boundary_checks,
            ),
        },
        "ownerupdate_fallback": {
            **fallback_stats.as_dict(),
            "exact": bool(fallback_stats.mismatches == 0),
            "owner_scan_ratio_vs_baseline": _safe_ratio(fallback_scans, baseline_owner_scans),
            "owner_scan_reduction_vs_baseline": _safe_ratio(baseline_owner_scans, fallback_scans),
            "fallback_reset_fraction": _safe_ratio(fallback_stats.fallback_resets, fallback_stats.boundary_checks),
        },
    }


def _storage_estimates(*, tape: Any, site_count: int, active_sample_count: int, owner_runs: int) -> dict[str, int]:
    coeff16_candidate = {
        "row_index_i32": int(tape.row_index.numel()) * 4,
        "row_offsets_i32": int(tape.row_offsets.numel()) * 4,
        "candidate_depth_coeff_f16": int(tape.candidate_count) * 4 * 2,
        "sites_f32": int(site_count) * 5 * 4,
        "ray_f32": int(tape.ray_coeff.numel()) * 4,
        "frame_t_f32": int(tape.frame_t.numel()) * 4,
    }
    coeff16_total = int(sum(coeff16_candidate.values()))
    boundary_count = int(site_count) * max(int(site_count) - 1, 0) // 2
    ownerkeep_i32_extra = int(tape.candidate_count) * 4 + boundary_count * 2 * 4
    ownerkeep_i16_extra = int(tape.candidate_count) * 2 + boundary_count * 2 * 2
    owner_run_i16 = int(active_sample_count + 1) * 4 + int(owner_runs) * 3 * 2
    owner_run_i32 = int(active_sample_count + 1) * 4 + int(owner_runs) * 3 * 4
    return {
        "coeff16_candidate_core_bytes": coeff16_total,
        "ownerkeep_i32_extra_bytes": int(ownerkeep_i32_extra),
        "ownerkeep_i16_extra_bytes": int(ownerkeep_i16_extra),
        "owner_run_offsets_plus_i16_records_bytes": int(owner_run_i16),
        "owner_run_offsets_plus_i32_records_bytes": int(owner_run_i32),
    }


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
    transmittance_threshold: float,
    residual_depth_padding: float,
    gate4_time_slabs: int,
    synthetic_motion: SyntheticRayMotion,
    sample_validation: str,
    allow_repeat_loaded_frames: bool,
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
    analyze_start = time.perf_counter()
    transition = _analyze_tape_owner_transitions(
        tape=tape,
        sites=sites,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    analyze_elapsed_s = time.perf_counter() - analyze_start
    candidate_storage = candidate_csr_storage_breakdown(tape=tape, site_count=len(sites))
    storage_estimates = _storage_estimates(
        tape=tape,
        site_count=len(sites),
        active_sample_count=int(transition["active_sample_count"]),
        owner_runs=int(transition["ownerkeep"]["owner_runs"]),
    )
    return {
        "status": "ok",
        "frame_count": int(frame_count),
        "loaded_frame_count": int(loaded_frame_count),
        "repeat_loaded_frames": bool(repeated),
        "render_size": int(render_size),
        "site_count": int(len(sites)),
        "track_count": int(tape.track_count),
        "sample_count": int(transition["sample_count"]),
        "active_sample_count": int(transition["active_sample_count"]),
        "candidate_count": int(tape.candidate_count),
        "candidate_replay_iterations": int(tape.candidate_replay_iterations),
        "max_candidates_per_row": int(tape.max_candidates_per_row),
        "avg_candidates_per_row": float(tape.avg_candidates_per_row),
        "missing_sample_events": int(tape.missing_sample_events),
        "extra_candidate_events": int(tape.extra_candidate_events),
        "missing_sample_events_authoritative": bool(
            tape.candidate_depth_order.get("missing_sample_events_authoritative", False)
        ),
        "candidate_storage_bytes": int(candidate_storage["total_bytes"]),
        "transition_storage_estimates": storage_estimates,
        "transition": transition,
        "timing_s": {
            "cpu_build_gate4_affine_candidate_tape": float(build_elapsed_s),
            "cpu_analyze_owner_transitions": float(analyze_elapsed_s),
            "total_cpu_probe": float(time.perf_counter() - start),
        },
    }


def summarize_rows(rows: list[dict[str, Any]], *, min_owner_scan_reduction: float) -> dict[str, Any]:
    if not rows:
        return {"status": "failed", "failures": ["no rows"], "acceptance": {}}
    first = rows[0]
    last = rows[-1]
    frame_scale = _ratio_last_first(float(last["frame_count"]), float(first["frame_count"]))
    candidate_scale = _ratio_last_first(float(last["candidate_count"]), float(first["candidate_count"]))
    baseline_scan_scale = _ratio_last_first(
        float(last["transition"]["baseline_owner_scans"]),
        float(first["transition"]["baseline_owner_scans"]),
    )
    ownerkeep_scan_scale = _ratio_last_first(
        float(last["transition"]["ownerkeep"]["transition_owner_scans"]),
        float(first["transition"]["ownerkeep"]["transition_owner_scans"]),
    )
    ownergroup_scan_scale = _ratio_last_first(
        float(last["transition"]["ownergroup_keep"]["transition_owner_scans"]),
        float(first["transition"]["ownergroup_keep"]["transition_owner_scans"]),
    )
    ownerrun_storage_scale = _ratio_last_first(
        float(last["transition_storage_estimates"]["owner_run_offsets_plus_i16_records_bytes"]),
        float(first["transition_storage_estimates"]["owner_run_offsets_plus_i16_records_bytes"]),
    )
    acceptance = {
        "all_rows_ok": all(row.get("status") == "ok" for row in rows),
        "all_ownergroup_keep_exact": all(row["transition"]["ownergroup_keep"]["exact"] for row in rows),
        "all_ownerupdate_fallback_exact": all(row["transition"]["ownerupdate_fallback"]["exact"] for row in rows),
        "ownergroup_keep_reduces_owner_scans": all(
            float(row["transition"]["ownergroup_keep"]["owner_scan_reduction_vs_baseline"])
            >= float(min_owner_scan_reduction)
            for row in rows
        ),
    }
    failures = [key for key, passed in acceptance.items() if not passed]
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "acceptance": acceptance,
        "frame_scale_first_to_last": float(frame_scale),
        "candidate_count_scale_first_to_last": float(candidate_scale),
        "baseline_owner_scan_scale_first_to_last": float(baseline_scan_scale),
        "ownerkeep_owner_scan_scale_first_to_last": float(ownerkeep_scan_scale),
        "ownergroup_keep_owner_scan_scale_first_to_last": float(ownergroup_scan_scale),
        "ownergroup_keep_scan_count_sublinear_vs_frames": bool(ownergroup_scan_scale < frame_scale),
        "owner_run_i16_storage_scale_first_to_last": float(ownerrun_storage_scale),
        "owner_run_i16_storage_sublinear_vs_frames": bool(ownerrun_storage_scale < frame_scale),
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
            transmittance_threshold=args.transmittance_threshold,
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
        )
        for frame_count in frame_counts
    ]
    summary = summarize_rows(rows, min_owner_scan_reduction=args.min_owner_scan_reduction)
    return {
        "benchmark": "world_foam_lane2_gate4_owner_transition_preflight",
        "status": summary["status"],
        "gate": "gate4_scan_free_owner_transition_cpu_preflight",
        "device": "cpu",
        "config_path": str(args.config),
        "frame_counts": list(frame_counts),
        "render_size": int(args.render_size),
        "site_count": int(args.site_count),
        "gate4_time_slabs": int(args.gate4_time_slabs),
        "gate4_residual_depth_padding": float(args.gate4_residual_depth_padding),
        "transmittance_threshold": float(args.transmittance_threshold),
        "synthetic_motion": SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ).to_dict(),
        "gradient_scope": "none_cpu_owner_transition_preflight_no_shader_dispatch_no_backward",
        "speed_claim": False,
        "quality_claim": False,
        **summary,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exact Gate4 owner-transition potential on CPU.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=16)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--gate4-time-slabs", type=int, default=1)
    parser.add_argument("--gate4-residual-depth-padding", type=float, default=0.001)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--sample-validation", choices=("skip", "full"), default="full")
    parser.add_argument("--repeat-loaded-frames", action="store_true")
    parser.add_argument("--min-owner-scan-reduction", type=float, default=4.0)
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
