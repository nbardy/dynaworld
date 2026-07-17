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

from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from probe_fused_slab_segment_tape import (  # noqa: E402
    build_segment_tape,
)
from smoke_fused_slab_affine_realray_mps import (  # noqa: E402
    _build_affine_csr_bundle,
    _parse_int_list,
)


def _linear_fit_max_residual(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    xs = torch.linspace(0.0, 1.0, len(values), dtype=torch.float64)
    ys = torch.tensor(values, dtype=torch.float64)
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = torch.sum((xs - x_mean) ** 2)
    if float(denom.item()) <= 0.0:
        pred = torch.full_like(ys, float(y_mean.item()))
    else:
        slope = torch.sum((xs - x_mean) * (ys - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        pred = intercept + slope * xs
    return float((ys - pred).abs().max().item())


def _track_sequences(tape: Any) -> list[list[tuple[int, ...]]]:
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


def _same_topology_fit_summary(tape: Any, same_tracks: list[int]) -> dict[str, Any]:
    if not same_tracks:
        return {
            "tracks": 0,
            "max_length_linear_fit_residual": None,
            "max_mid_linear_fit_residual": None,
            "p95_length_linear_fit_residual": None,
            "p95_mid_linear_fit_residual": None,
        }
    lengths = tape.lengths_f32.detach().cpu()
    mids = tape.mids_f32.detach().cpu()
    counts = tape.counts_i32.detach().cpu()
    length_residuals: list[float] = []
    mid_residuals: list[float] = []
    for track_id in same_tracks:
        segment_count = int(counts[track_id, 0].item())
        for segment_id in range(segment_count):
            length_residuals.append(
                _linear_fit_max_residual(
                    [float(lengths[track_id, frame_id, segment_id].item()) for frame_id in range(tape.frame_count)]
                )
            )
            mid_residuals.append(
                _linear_fit_max_residual(
                    [float(mids[track_id, frame_id, segment_id].item()) for frame_id in range(tape.frame_count)]
                )
            )

    def quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(math.ceil(q * len(ordered))) - 1, len(ordered) - 1)
        return float(ordered[max(index, 0)])

    return {
        "tracks": len(same_tracks),
        "segment_count": len(length_residuals),
        "max_length_linear_fit_residual": max(length_residuals) if length_residuals else 0.0,
        "max_mid_linear_fit_residual": max(mid_residuals) if mid_residuals else 0.0,
        "p95_length_linear_fit_residual": quantile(length_residuals, 0.95),
        "p95_mid_linear_fit_residual": quantile(mid_residuals, 0.95),
    }


def _profile_frame_count(
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
    transmittance_threshold: float,
    residual_depth_padding: float,
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
    bundle = _build_affine_csr_bundle(
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        residual_depth_padding=residual_depth_padding,
        layout="per-track",
        tile_h=8,
        tile_w=8,
        candidate_order="boundary-id",
    )
    sequences = _track_sequences(tape)
    same_tracks: list[int] = []
    unique_per_track: list[int] = []
    unique_global: set[tuple[int, ...]] = set()
    transitions = 0
    for track_id, frames in enumerate(sequences):
        unique = set(frames)
        unique_global.update(unique)
        unique_per_track.append(len(unique))
        if len(unique) == 1:
            same_tracks.append(track_id)
        transitions += sum(1 for left, right in zip(frames[:-1], frames[1:], strict=True) if left != right)
    total_segments = int(tape.counts_i32.to(dtype=torch.int64).sum().item())
    topology_rows_if_track_unique = int(sum(unique_per_track))
    sample_count = int(tape.track_count * tape.frame_count)
    transition_slots = int(tape.track_count * max(tape.frame_count - 1, 0))
    unique_counter = Counter(tuple(frame_seq for frame_seq in frames) for frames in sequences)
    return {
        "frames": frame_count,
        "render_size": render_size,
        "track_count": int(tape.track_count),
        "sample_count": sample_count,
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "total_segments": total_segments,
        "avg_segments_per_sample": float(total_segments) / float(max(sample_count, 1)),
        "max_segments_per_sample": int(tape.counts_i32.max().item()) if tape.counts_i32.numel() else 0,
        "topology": {
            "same_topology_all_frames_tracks": len(same_tracks),
            "same_topology_all_frames_ratio": float(len(same_tracks)) / float(max(tape.track_count, 1)),
            "avg_unique_topologies_per_track": float(sum(unique_per_track)) / float(max(tape.track_count, 1)),
            "max_unique_topologies_per_track": max(unique_per_track) if unique_per_track else 0,
            "track_unique_topology_rows": topology_rows_if_track_unique,
            "track_unique_topology_rows_vs_samples": float(topology_rows_if_track_unique) / float(max(sample_count, 1)),
            "global_unique_owner_sequences": len(unique_global),
            "global_unique_owner_sequences_vs_samples": float(len(unique_global)) / float(max(sample_count, 1)),
            "frame_to_frame_topology_transitions": transitions,
            "frame_to_frame_topology_transition_rate": float(transitions) / float(max(transition_slots, 1)),
            "repeated_full_track_topology_count": sum(count for count in unique_counter.values() if count > 1),
            "unique_full_track_topology_count": len(unique_counter),
        },
        "same_topology_linear_fit": _same_topology_fit_summary(tape, same_tracks),
        "current_csr": {
            "candidate_count": int(bundle["candidate_count"]),
            "candidate_replay_iterations": int(bundle["candidate_replay_iterations"]),
            "candidate_replay_iterations_vs_segments": float(bundle["candidate_replay_iterations"])
            / float(max(total_segments, 1)),
            "max_candidates_per_row": int(bundle["max_candidates_per_row"]),
            "missing_sample_events": int(bundle["missing_sample_events"]),
            "candidate_depth_order": bundle["candidate_depth_order"],
        },
    }


def run_probe(
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
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
) -> dict[str, Any]:
    rows = [
        _profile_frame_count(
            frame_count=frame_count,
            config_path=config_path,
            render_size=render_size,
            site_count=site_count,
            time_slabs=time_slabs,
            near=near,
            far=far,
            density=density,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            residual_depth_padding=residual_depth_padding,
            synthetic_motion=synthetic_motion,
        )
        for frame_count in frame_counts
    ]
    frame_scale = float(rows[-1]["frames"]) / float(max(rows[0]["frames"], 1))
    segment_scale = float(rows[-1]["total_segments"]) / float(max(rows[0]["total_segments"], 1))
    topology_row_scale = float(rows[-1]["topology"]["track_unique_topology_rows"]) / float(
        max(rows[0]["topology"]["track_unique_topology_rows"], 1)
    )
    avg_transition_rate = float(rows[-1]["topology"]["frame_to_frame_topology_transition_rate"])
    acceptance = {
        "zero_missing_sample_events": all(row["current_csr"]["missing_sample_events"] == 0 for row in rows),
        "topology_rows_scale_sublinear_vs_frames": topology_row_scale < frame_scale,
        "global_owner_sequences_compress_samples": rows[-1]["topology"]["global_unique_owner_sequences_vs_samples"] < 0.25,
    }
    return {
        "benchmark": "world_foam_lane2_segment_topology_sharing_probe",
        "status": "ok" if all(acceptance.values()) else "informational",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "site_count": site_count,
        "time_slabs": time_slabs,
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "segment_scale_first_to_last": segment_scale,
        "track_unique_topology_row_scale_first_to_last": topology_row_scale,
        "last_frame_transition_rate": avg_transition_rate,
        "structural_read": {
            "owner_topology_can_be_shared": acceptance["topology_rows_scale_sublinear_vs_frames"],
            "naive_segments_still_per_frame": True,
            "next_shader_hypothesis": (
                "Store per-track unique owner topology rows and per-frame numeric length/mid streams, "
                "or move to an evented boundary tape; plain per-sample segment materialization is not STAR-like."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe segment owner-topology sharing across World Foam frames.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--residual-depth-padding", type=float, default=1.0e-4)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_segment_topology_sharing_probe_render32_pertrack_2_4_8_16.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_probe(
        config_path=args.config,
        frame_counts=_parse_int_list(args.frame_counts),
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        residual_depth_padding=args.residual_depth_padding,
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
