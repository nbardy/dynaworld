#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
TRAIN_SRC = DYNAWORLD / "src" / "train"
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"

for path in (THIS_DIR, TRAIN_SRC, VARIANT_ROOT):
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
from probe_delta_framegroup_variant_timing import _synthetic_sequences  # noqa: E402
from probe_endpoint_record_delta_replay import (  # noqa: E402
    EndpointRecordDeltaReplaceTape,
    pack_endpoint_record_delta_replace_tape,
)
from probe_owner_run_boundary_tape import _build_owner_run_sequences  # noqa: E402


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one integer")
    if any(item <= 0 for item in out):
        raise ValueError("integer list values must be positive")
    return out


def _percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(ordered)),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": int(ordered[-1]),
    }


def _repeat_view_major_frames(
    values: torch.Tensor,
    *,
    loaded_frame_count: int,
    requested_frame_count: int,
    name: str,
) -> torch.Tensor:
    if int(values.shape[0]) % loaded_frame_count != 0:
        raise ValueError(f"{name} is not view-major by loaded_frame_count={loaded_frame_count}")
    view_count = int(values.shape[0]) // loaded_frame_count
    source_frame_indices = torch.arange(requested_frame_count, dtype=torch.long) % loaded_frame_count
    return (
        values.reshape(view_count, loaded_frame_count, *values.shape[1:])
        .index_select(1, source_frame_indices)
        .reshape(view_count * requested_frame_count, *values.shape[1:])
        .contiguous()
    )


def _fit_loaded_frame_count(
    *,
    split_name: str,
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
            f"{split_name} loader returned {loaded_frame_count} frames for requested "
            f"{requested_frame_count}; expected the data loader to crop to the requested count"
        )
    if not allow_repeat_loaded_frames:
        raise ValueError(
            f"{split_name} loader returned only {loaded_frame_count} frames for requested "
            f"{requested_frame_count}; pass --repeat-loaded-frames for a repeated-fixture topology probe"
        )
    targets = _repeat_view_major_frames(
        targets,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=requested_frame_count,
        name=f"{split_name}.targets",
    )
    rays = _repeat_view_major_frames(
        rays,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=requested_frame_count,
        name=f"{split_name}.rays",
    )
    view_count = int(targets.shape[0]) // requested_frame_count
    frame_indices = torch.arange(requested_frame_count, dtype=torch.long, device=frame_indices.device).repeat(view_count)
    return targets, rays, frame_indices, True


def _row_owners(
    *,
    source: int,
    begin: int,
    end: int,
    delta: EndpointRecordDeltaReplaceTape,
    site_count: int,
) -> list[int]:
    if begin < 0 or end < begin:
        raise ValueError(f"invalid row bounds begin={begin} end={end}")
    if source == 0:
        owners = delta.base_owner_i32
    elif source == 1:
        owners = delta.change_owner_i32
    else:
        raise ValueError(f"invalid source {source}")
    if end > int(owners.numel()):
        raise ValueError(f"row end {end} exceeds owner tensor length {int(owners.numel())}")
    return [owner for owner in (int(value) for value in owners[begin:end].tolist()) if 0 <= owner < site_count]


def summarize_owner_reduce_potential(
    delta: EndpointRecordDeltaReplaceTape,
    *,
    frame_count: int,
    site_count: int,
    chunk_size: int = 32,
    owner_cap: int = 16,
) -> dict[str, Any]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if site_count <= 0:
        raise ValueError("site_count must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if owner_cap <= 0:
        raise ValueError("owner_cap must be positive")

    chunk_count = (frame_count + chunk_size - 1) // chunk_size
    track_change_offsets = [int(value) for value in delta.track_change_offsets_i32.tolist()]
    base_offsets = [int(value) for value in delta.base_offsets_i32.tolist()]
    change_frames = [int(value) for value in delta.change_frame_i32.tolist()]
    change_offsets = [int(value) for value in delta.change_offsets_i32.tolist()]
    track_count = len(base_offsets) - 1
    if len(track_change_offsets) != track_count + 1:
        raise ValueError("track_change_offsets and base_offsets disagree on track count")

    current_atomic_calls = 0
    ownerreduce_atomic_calls = 0
    ideal_atomic_calls = 0
    segment_atomic_calls = 0
    fallback_chunks = 0
    empty_chunks = 0
    total_chunks = 0
    unique_counts: list[int] = []
    segment_counts: list[int] = []
    row_counts: list[int] = []
    repeated_owner_segments = 0
    cap_histogram = {str(index): 0 for index in range(owner_cap + 1)}
    cap_histogram[f">{owner_cap}"] = 0

    for track_id in range(track_count):
        change_begin = track_change_offsets[track_id]
        change_end = track_change_offsets[track_id + 1]
        for chunk_id in range(chunk_count):
            frame_start = chunk_id * chunk_size
            frames_in_chunk = max(0, min(chunk_size, frame_count - frame_start))
            if frames_in_chunk == 0:
                continue

            selected_change = -1
            change_cursor = change_begin
            while change_cursor < change_end and change_frames[change_cursor] < frame_start:
                if change_frames[change_cursor] >= 0:
                    selected_change = change_cursor
                change_cursor += 1

            chunk_owners: set[int] = set()
            chunk_segment_count = 0
            chunk_row_count = 0
            for local_frame in range(frames_in_chunk):
                global_frame = frame_start + local_frame
                while change_cursor < change_end:
                    changed_frame = change_frames[change_cursor]
                    if changed_frame < 0:
                        change_cursor += 1
                        continue
                    if changed_frame > global_frame:
                        break
                    selected_change = change_cursor
                    change_cursor += 1

                if selected_change >= 0:
                    source = 1
                    begin = change_offsets[selected_change]
                    end = change_offsets[selected_change + 1]
                else:
                    source = 0
                    begin = base_offsets[track_id]
                    end = base_offsets[track_id + 1]
                owners = _row_owners(source=source, begin=begin, end=end, delta=delta, site_count=site_count)
                chunk_segment_count += len(owners)
                chunk_row_count += end - begin
                chunk_owners.update(owners)

            unique_count = len(chunk_owners)
            total_chunks += 1
            unique_counts.append(unique_count)
            segment_counts.append(chunk_segment_count)
            row_counts.append(chunk_row_count)
            segment_atomic_calls += chunk_segment_count
            ideal_atomic_calls += unique_count
            repeated_owner_segments += max(0, chunk_segment_count - unique_count)
            if unique_count == 0:
                empty_chunks += 1

            if site_count <= owner_cap:
                current_atomic_calls += site_count
            else:
                current_atomic_calls += chunk_segment_count

            if unique_count <= owner_cap:
                ownerreduce_atomic_calls += unique_count
            else:
                ownerreduce_atomic_calls += chunk_segment_count
                fallback_chunks += 1

            cap_histogram[str(unique_count) if unique_count <= owner_cap else f">{owner_cap}"] += 1

    ratio = ownerreduce_atomic_calls / max(current_atomic_calls, 1)
    ideal_ratio = ideal_atomic_calls / max(current_atomic_calls, 1)
    return {
        "track_count": track_count,
        "frame_count": int(frame_count),
        "site_count": int(site_count),
        "chunk_size": int(chunk_size),
        "chunk_count": int(chunk_count),
        "owner_cap": int(owner_cap),
        "total_track_chunks": int(total_chunks),
        "empty_track_chunks": int(empty_chunks),
        "fallback_track_chunks": int(fallback_chunks),
        "fallback_fraction": float(fallback_chunks / max(total_chunks, 1)),
        "current_atomic_calls_est": int(current_atomic_calls),
        "ownerreduce_atomic_calls_est": int(ownerreduce_atomic_calls),
        "ideal_unique_owner_atomic_calls_est": int(ideal_atomic_calls),
        "segment_atomic_calls_est": int(segment_atomic_calls),
        "ownerreduce_to_current_atomic_ratio": float(ratio),
        "ideal_unique_to_current_atomic_ratio": float(ideal_ratio),
        "atomic_reduction_factor": float(1.0 / max(ratio, 1.0e-12)),
        "repeated_owner_segments": int(repeated_owner_segments),
        "unique_owner_count_per_track_chunk": _distribution(unique_counts),
        "valid_segment_count_per_track_chunk": _distribution(segment_counts),
        "record_row_count_per_track_chunk": _distribution(row_counts),
        "unique_owner_cap_histogram": cap_histogram,
        "tape": {
            "storage_bytes": int(delta.storage_bytes),
            "base_records": int(delta.base_owner_i32.numel()),
            "changes": int(delta.change_frame_i32.numel()),
            "change_records": int(delta.change_owner_i32.numel()),
        },
    }


def _synthetic_row(
    *,
    frame_count: int,
    site_count: int,
    track_repeats: int,
    chunk_size: int,
    owner_cap: int,
) -> dict[str, Any]:
    sequences = _synthetic_sequences(frame_count, track_repeats=track_repeats, site_count=site_count)
    delta = pack_endpoint_record_delta_replace_tape(sequences, frame_count=frame_count)
    summary = summarize_owner_reduce_potential(
        delta,
        frame_count=frame_count,
        site_count=site_count,
        chunk_size=chunk_size,
        owner_cap=owner_cap,
    )
    summary["mode"] = "synthetic"
    summary["track_repeats"] = int(track_repeats)
    return summary


def _real_rows_for_frame_count(
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
    repeat_loaded_frames: bool,
    chunk_size: int,
    owner_cap: int,
    splits: tuple[str, ...],
    sequence_spatial_stride: int,
    progress: bool,
) -> list[dict[str, Any]]:
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    loaded_frame_count = int(data["frame_count"])
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    train_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    train_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    targets, train_rays, train_frame_indices, train_repeated = _fit_loaded_frame_count(
        split_name="train",
        targets=targets,
        rays=train_rays,
        frame_indices=train_frame_indices,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=frame_count,
        allow_repeat_loaded_frames=repeat_loaded_frames,
    )
    train_rays = apply_synthetic_ray_motion(
        train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    boundaries = make_boundaries_4d(sites)
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32)

    split_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "train": (train_rays, train_frame_indices),
    }
    heldout_repeated = False
    if "heldout" in splits:
        heldout_targets = data["heldout_targets"]
        heldout_rays = data["heldout_rays"]
        heldout_frame_indices = data["heldout_frame_indices"]
        if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
            raise ValueError("heldout split requested but fixture has no heldout tensors")
        heldout_targets, heldout_rays, heldout_frame_indices, heldout_repeated = _fit_loaded_frame_count(
            split_name="heldout",
            targets=heldout_targets.detach().cpu().to(dtype=torch.float32),
            rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
            frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
            loaded_frame_count=loaded_frame_count,
            requested_frame_count=frame_count,
            allow_repeat_loaded_frames=repeat_loaded_frames,
        )
        del heldout_targets
        split_tensors["heldout"] = (
            apply_synthetic_ray_motion(
                heldout_rays,
                frame_indices=heldout_frame_indices,
                frame_count=frame_count,
                motion=synthetic_motion,
            ),
            heldout_frame_indices,
        )

    rows: list[dict[str, Any]] = []
    for split in splits:
        rays, frame_indices = split_tensors[split]
        if sequence_spatial_stride > 1:
            rays = rays[:, ::sequence_spatial_stride, ::sequence_spatial_stride, :].contiguous()
        if progress:
            sample_count, height, width, _payload = rays.shape
            print(
                f"[ownerreduce-potential] frame_count={frame_count} split={split} "
                f"tracks={(sample_count // frame_count) * height * width} "
                f"site_count={site_count} stride={sequence_spatial_stride}",
                file=sys.stderr,
                flush=True,
            )
        sequences, _sample_meta = _build_owner_run_sequences(
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
        delta = pack_endpoint_record_delta_replace_tape(sequences, frame_count=frame_count)
        summary = summarize_owner_reduce_potential(
            delta,
            frame_count=frame_count,
            site_count=site_count,
            chunk_size=chunk_size,
            owner_cap=owner_cap,
        )
        summary.update(
            {
                "mode": "real",
                "split": split,
                "loaded_frame_count": loaded_frame_count,
                "repeated_loaded_frames": bool(train_repeated or heldout_repeated),
                "render_size": int(render_size),
                "motion": synthetic_motion.to_dict(),
                "sequence_spatial_stride": int(sequence_spatial_stride),
            }
        )
        rows.append(summary)
    return rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = _parse_int_list(args.frame_counts)
    if args.mode == "synthetic":
        rows = [
            _synthetic_row(
                frame_count=frame_count,
                site_count=args.site_count,
                track_repeats=args.track_repeats,
                chunk_size=args.chunk_size,
                owner_cap=args.owner_cap,
            )
            for frame_count in frame_counts
        ]
    else:
        splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
        if not splits:
            raise ValueError("--splits must name at least one split")
        allowed_splits = {"train", "heldout"}
        unknown = sorted(set(splits) - allowed_splits)
        if unknown:
            raise ValueError(f"unknown splits: {unknown}")
        motion = SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        )
        rows = []
        for frame_count in frame_counts:
            rows.extend(
                _real_rows_for_frame_count(
                    frame_count=frame_count,
                    config_path=args.config,
                    render_size=args.render_size,
                    site_count=args.site_count,
                    near=args.near,
                    far=args.far,
                    density=args.density,
                    invalid_epsilon=args.invalid_epsilon,
                    transmittance_threshold=args.transmittance_threshold,
                    synthetic_motion=motion,
                    repeat_loaded_frames=args.repeat_loaded_frames,
                    chunk_size=args.chunk_size,
                    owner_cap=args.owner_cap,
                    splits=splits,
                    sequence_spatial_stride=args.sequence_spatial_stride,
                    progress=args.progress,
                )
            )

    return {
        "probe": "delta_framegroup_owner_reduce_potential",
        "scope": (
            "producer-side topology estimate only; current atomics model matches the live "
            "framegroup16 small-site reduction and direct-atomic fallback"
        ),
        "frame_counts": list(frame_counts),
        "mode": args.mode,
        "site_count": int(args.site_count),
        "chunk_size": int(args.chunk_size),
        "owner_cap": int(args.owner_cap),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate whether framegroup delta tapes can profit from owner-reduce.")
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--frame-counts", default="16,32,64,128")
    parser.add_argument("--site-count", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--owner-cap", type=int, default=16)
    parser.add_argument("--track-repeats", type=int, default=64)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--splits", default="train,heldout")
    parser.add_argument(
        "--sequence-spatial-stride",
        type=int,
        default=1,
        help="Subsample rays spatially before building owner-run sequences for real topology probes.",
    )
    parser.add_argument("--progress", action="store_true")
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
    parser.add_argument("--repeat-loaded-frames", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.site_count <= 0:
        raise ValueError("--site-count must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.owner_cap <= 0:
        raise ValueError("--owner-cap must be positive")
    if args.track_repeats <= 0:
        raise ValueError("--track-repeats must be positive")
    if args.sequence_spatial_stride <= 0:
        raise ValueError("--sequence-spatial-stride must be positive")

    payload = run_probe(args)
    out_path = args.out_json
    if out_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "delta_framegroup_owner_reduce_potential.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
