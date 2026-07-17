#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
RESULTS_DIR = THIS_DIR / "results"

for path in (THIS_DIR, VARIANT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_endpoint_record_delta_replay import (  # noqa: E402
    build_delta_replace_chunk_change_offsets,
    build_delta_replace_frame_row_descriptors,
    pack_endpoint_record_delta_replace_tape,
)
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only,
)


TensorOp = Callable[[], tuple[torch.Tensor, torch.Tensor]]


def _parse_frame_counts(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    if any(frame_count <= 0 for frame_count in out):
        raise ValueError("frame counts must be positive")
    return out


def _record(owner: int, left: int, right: int) -> SimpleNamespace:
    return SimpleNamespace(owner=int(owner), left_cut_id=int(left), right_cut_id=int(right))


def _stage_row(stage: tuple[tuple[int, int, int], ...]) -> tuple[SimpleNamespace, ...]:
    return tuple(_record(owner, left, right) for owner, left, right in stage)


def _piecewise_track(
    *,
    frame_count: int,
    first_change: int,
    second_change: int,
    stages: tuple[tuple[tuple[int, int, int], ...], ...],
) -> list[tuple[SimpleNamespace, ...]]:
    if len(stages) != 3:
        raise ValueError("expected exactly three stages")
    first_change = max(1, min(first_change, frame_count - 1))
    second_change = max(first_change + 1, min(second_change, frame_count))
    rows: list[tuple[SimpleNamespace, ...]] = []
    for frame_id in range(frame_count):
        if frame_id < first_change:
            rows.append(_stage_row(stages[0]))
        elif frame_id < second_change:
            rows.append(_stage_row(stages[1]))
        else:
            rows.append(_stage_row(stages[2]))
    return rows


def _offset_row(
    row: tuple[SimpleNamespace, ...],
    *,
    owner_offset: int,
    site_count: int,
) -> tuple[SimpleNamespace, ...]:
    return tuple(
        _record((int(record.owner) + owner_offset) % site_count, int(record.left_cut_id), int(record.right_cut_id))
        for record in row
    )


def _synthetic_sequences(frame_count: int, *, track_repeats: int, site_count: int) -> list[list[tuple[SimpleNamespace, ...]]]:
    if track_repeats <= 0:
        raise ValueError("track_repeats must be positive")
    track0_stages = (
        ((0, -1, 0), (1, 0, 1), (2, 1, -2)),
        ((0, -1, 2), (3, 2, 3), (2, 3, -2)),
        ((4, -1, 4), (5, 4, 5), (6, 5, -2), (7, 2, 6)),
    )
    track1_stages = (
        ((8, -1, 1), (9, 1, 2), (10, 2, -2)),
        ((8, -1, 3), (11, 3, 4), (10, 4, -2)),
        ((1, -1, 5), (9, 5, 6), (3, 6, -2), (11, 0, 7)),
    )
    base_tracks = [
        _piecewise_track(
            frame_count=frame_count,
            first_change=max(1, frame_count // 8),
            second_change=max(2, (5 * frame_count) // 8),
            stages=track0_stages,
        ),
        _piecewise_track(
            frame_count=frame_count,
            first_change=max(1, frame_count // 4),
            second_change=max(2, (3 * frame_count) // 4),
            stages=track1_stages,
        ),
    ]
    sequences: list[list[tuple[SimpleNamespace, ...]]] = []
    for repeat_id in range(track_repeats):
        owner_offset = (repeat_id * 3) % site_count
        for track in base_tracks:
            sequences.append([_offset_row(row, owner_offset=owner_offset, site_count=site_count) for row in track])
    return sequences


def _coefficients(*, track_count: int, boundary_count: int, device: torch.device) -> torch.Tensor:
    rows = []
    for track_id in range(track_count):
        for boundary_id in range(boundary_count):
            base = 0.25 + 0.08 * boundary_id + 0.025 * track_id
            slope = 0.03 * ((boundary_id % 3) - 1) + 0.005 * track_id
            curve = 0.004 * ((boundary_id % 5) - 2)
            cubic = 0.001 * ((boundary_id % 7) - 3)
            rows.append((base, slope, curve, cubic))
    return torch.tensor(rows, device=device, dtype=torch.float16)


def _site_rgba(*, site_count: int, device: torch.device) -> torch.Tensor:
    rows = []
    for site_id in range(site_count):
        rows.append(
            (
                0.11 + 0.031 * site_id,
                0.72 - 0.027 * site_id,
                0.19 + 0.041 * (site_id % 5),
                0.25 + 0.035 * (site_id % 4),
            )
        )
    return torch.tensor(rows, device=device, dtype=torch.float32).clamp(0.0, 1.0)


def _target(*, track_count: int, frame_count: int, device: torch.device) -> torch.Tensor:
    values = torch.linspace(
        0.05,
        0.95,
        track_count * frame_count * 3,
        device=device,
        dtype=torch.float32,
    )
    return values.reshape(track_count, frame_count, 3)


def _tape_i16x3(owner_i32: torch.Tensor, left_i32: torch.Tensor, right_i32: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            owner_i32.to(dtype=torch.int16),
            left_i32.to(dtype=torch.int16),
            right_i32.to(dtype=torch.int16),
        ),
        dim=1,
    ).reshape(-1)


def _pack_cut_ids_i32(cut_i32: torch.Tensor) -> torch.Tensor:
    cut = cut_i32.detach().cpu().to(dtype=torch.int64)
    if bool((cut < -2).any().item()):
        raise ValueError("packed delta records only support -1, -2, or nonnegative cuts")
    code = torch.where(cut == -1, torch.zeros_like(cut), torch.where(cut == -2, torch.ones_like(cut), cut + 2))
    if code.numel() and int(code.max().item()) > 4095:
        raise ValueError("packed delta records support cut codes up to 4095")
    return code


def _tape_packed_i32(owner_i32: torch.Tensor, left_i32: torch.Tensor, right_i32: torch.Tensor) -> torch.Tensor:
    owner = owner_i32.detach().cpu().to(dtype=torch.int64)
    if owner.shape != left_i32.shape or owner.shape != right_i32.shape:
        raise ValueError("packed delta record tensors must have matching shapes")
    if owner.numel() and (int(owner.min().item()) < -1 or int(owner.max().item()) > 255):
        raise ValueError("packed delta records support owner ids in [-1, 255]")
    owner = torch.where(owner < 0, torch.zeros_like(owner), owner)
    packed = owner | (_pack_cut_ids_i32(left_i32) << 8) | (_pack_cut_ids_i32(right_i32) << 20)
    if packed.numel() and int(packed.max().item()) > 2_147_483_647:
        raise ValueError("packed delta record exceeded signed int32 range")
    return packed.to(dtype=torch.int32)


def _tape_i16x4(owner_i32: torch.Tensor, left_i32: torch.Tensor, right_i32: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            owner_i32.to(dtype=torch.int16),
            left_i32.to(dtype=torch.int16),
            right_i32.to(dtype=torch.int16),
            torch.zeros_like(owner_i32, dtype=torch.int16),
        ),
        dim=1,
    ).reshape(-1)


def _stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("cannot summarize empty samples")
    ordered = sorted(samples_ms)
    trim = len(ordered) // 10
    trimmed = ordered[trim : len(ordered) - trim] if trim > 0 and len(ordered) - trim > trim else ordered
    return {
        "mean_ms": float(statistics.fmean(samples_ms)),
        "trimmed_mean_ms": float(statistics.fmean(trimmed)),
        "median_ms": float(statistics.median(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "max_to_median": float(max(samples_ms) / max(statistics.median(samples_ms), 1.0e-12)),
    }


def _time_op(op: TensorOp, *, warmup: int, steps: int) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    last_loss: torch.Tensor | None = None
    last_grad: torch.Tensor | None = None
    for _ in range(warmup):
        last_loss, last_grad = op()
    torch.mps.synchronize()

    samples: list[float] = []
    for _ in range(steps):
        torch.mps.synchronize()
        started = time.perf_counter()
        last_loss, last_grad = op()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if last_loss is None or last_grad is None:
        raise RuntimeError("op did not run")
    return _stats(samples), last_loss.detach(), last_grad.detach()


def _time_ops_interleaved(
    ops: dict[str, TensorOp],
    *,
    warmup: int,
    steps: int,
) -> dict[str, tuple[dict[str, float], torch.Tensor, torch.Tensor]]:
    names = tuple(ops.keys())
    last: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for _ in range(warmup):
        for name in names:
            last[name] = ops[name]()
    torch.mps.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in names}
    for step_id in range(steps):
        ordered = names[step_id % len(names) :] + names[: step_id % len(names)]
        for name in ordered:
            torch.mps.synchronize()
            started = time.perf_counter()
            last[name] = ops[name]()
            torch.mps.synchronize()
            samples[name].append((time.perf_counter() - started) * 1000.0)
    return {name: (_stats(samples[name]), last[name][0].detach(), last[name][1].detach()) for name in names}


def _packed_launch_only_op(
    op_name: str,
    *,
    coeff: torch.Tensor,
    frame_t: torch.Tensor,
    base_offsets: torch.Tensor,
    base_packed: torch.Tensor,
    track_change_offsets: torch.Tensor,
    chunk_offsets: torch.Tensor,
    change_frame: torch.Tensor,
    change_offsets: torch.Tensor,
    change_packed: torch.Tensor,
    site_rgba: torch.Tensor,
    target_rgb: torch.Tensor,
    config: RealRayReplayConfig,
    boundary_count: int,
    track_count: int,
    frame_count: int,
    unchecked: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    config_i32 = torch.tensor(
        [
            int(boundary_count),
            int(track_count),
            int(frame_count),
            int(site_rgba.shape[0]),
            int(base_packed.numel()),
            int(change_frame.numel()),
            int(change_packed.numel()),
        ],
        device=coeff.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    common_args = (
        coeff,
        frame_t,
        base_offsets,
        base_packed,
        track_change_offsets,
        chunk_offsets,
        change_frame,
        change_offsets,
        change_packed,
        site_rgba,
        target_rgb,
        config_i32,
        config_f32,
    )
    if unchecked:
        return getattr(ops, op_name)(
            *common_args,
            int(track_count),
            int(frame_count),
            int(site_rgba.shape[0]),
        )
    return getattr(ops, op_name)(
        *common_args,
        int(boundary_count),
        int(track_count),
        int(frame_count),
        int(site_rgba.shape[0]),
        int(base_packed.numel()),
        int(change_frame.numel()),
        int(change_packed.numel()),
    )


def _packed_rowdesc_launch_only_op(
    *,
    coeff: torch.Tensor,
    frame_t: torch.Tensor,
    row_begin: torch.Tensor,
    row_len_source: torch.Tensor,
    base_packed: torch.Tensor,
    change_packed: torch.Tensor,
    site_rgba: torch.Tensor,
    target_rgb: torch.Tensor,
    config: RealRayReplayConfig,
    boundary_count: int,
    track_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    config_i32 = torch.tensor(
        [
            int(boundary_count),
            int(track_count),
            int(frame_count),
            int(site_rgba.shape[0]),
            int(base_packed.numel()),
            0,
            int(change_packed.numel()),
        ],
        device=coeff.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff.device,
        dtype=torch.float32,
    )
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only"
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff,
        frame_t,
        row_begin,
        row_len_source,
        base_packed,
        change_packed,
        site_rgba,
        target_rgb,
        config_i32,
        config_f32,
        int(boundary_count),
        int(track_count),
        int(frame_count),
        int(site_rgba.shape[0]),
        int(base_packed.numel()),
        int(change_packed.numel()),
    )


def _frame_case(
    frame_count: int,
    *,
    warmup: int,
    steps: int,
    track_repeats: int,
    interleave_variants: bool,
    include_diagnostic_packed_variants: bool,
    include_launch_only_variants: bool,
) -> dict[str, Any]:
    device = torch.device("mps")
    boundary_count = 8
    site_count = 64
    sequences = _synthetic_sequences(frame_count, track_repeats=track_repeats, site_count=site_count)
    track_count = len(sequences)
    delta = pack_endpoint_record_delta_replace_tape(sequences, frame_count=frame_count)

    coeff = _coefficients(track_count=track_count, boundary_count=boundary_count, device=device)
    frame_t = torch.linspace(0.0, 1.0, frame_count, device=device, dtype=torch.float32)
    site_rgba = _site_rgba(site_count=site_count, device=device)
    target_rgb = _target(track_count=track_count, frame_count=frame_count, device=device)
    config = RealRayReplayConfig(near=0.0, far=3.5, invalid_epsilon=1.0e-7, transmittance_threshold=1.0e-4)

    base_offsets = delta.base_offsets_i32.to(device=device).contiguous()
    track_change_offsets = delta.track_change_offsets_i32.to(device=device).contiguous()
    change_frame = delta.change_frame_i32.to(device=device).contiguous()
    change_offsets = delta.change_offsets_i32.to(device=device).contiguous()
    base_i16x3 = _tape_i16x3(delta.base_owner_i32, delta.base_left_i32, delta.base_right_i32).to(device=device)
    change_i16x3 = _tape_i16x3(delta.change_owner_i32, delta.change_left_i32, delta.change_right_i32).to(
        device=device
    )
    base_packed = _tape_packed_i32(delta.base_owner_i32, delta.base_left_i32, delta.base_right_i32).to(device=device)
    change_packed = _tape_packed_i32(delta.change_owner_i32, delta.change_left_i32, delta.change_right_i32).to(
        device=device
    )
    base_i16x4 = _tape_i16x4(delta.base_owner_i32, delta.base_left_i32, delta.base_right_i32).to(device=device)
    change_i16x4 = _tape_i16x4(delta.change_owner_i32, delta.change_left_i32, delta.change_right_i32).to(
        device=device
    )
    chunk32 = build_delta_replace_chunk_change_offsets(delta, frame_count=frame_count).to(
        device=device,
        dtype=torch.int16,
    )
    row_begin32, row_len_source32 = build_delta_replace_frame_row_descriptors(delta, frame_count=frame_count)
    row_begin32 = row_begin32.to(device=device)
    row_len_source32 = row_len_source32.to(device=device)
    chunk16 = build_delta_replace_chunk_change_offsets(delta, frame_count=frame_count, chunk_size=16).to(
        device=device,
        dtype=torch.int16,
    )
    record_count = int(delta.base_owner_i32.numel() + delta.change_owner_i32.numel())
    shared32_storage_bytes = int(
        delta.base_offsets_i32.numel() * 4
        + delta.track_change_offsets_i32.numel() * 4
        + delta.change_frame_i32.numel() * 4
        + delta.change_offsets_i32.numel() * 4
        + chunk32.numel() * chunk32.element_size()
    )
    shared16_storage_bytes = int(
        delta.base_offsets_i32.numel() * 4
        + delta.track_change_offsets_i32.numel() * 4
        + delta.change_frame_i32.numel() * 4
        + delta.change_offsets_i32.numel() * 4
        + chunk16.numel() * chunk16.element_size()
    )
    variant_storage_bytes = {
        "i16x3_framegroup32_lossreduce": shared32_storage_bytes + record_count * 6,
        "packed_framegroup32": shared32_storage_bytes + record_count * 4,
        "i16x3_materialized_framegroup16": shared16_storage_bytes + record_count * 6,
        "i16x4_framegroup32": shared32_storage_bytes + record_count * 8,
    }
    if include_diagnostic_packed_variants:
        variant_storage_bytes.update(
            {
                "packed_framegroup32_recompute": shared32_storage_bytes + record_count * 4,
                "packed_framegroup32_smallrun16": shared32_storage_bytes + record_count * 4,
                "packed_materialized_framegroup16": shared16_storage_bytes + record_count * 4,
            }
        )
    if include_launch_only_variants:
        variant_storage_bytes.update(
            {
                "packed_framegroup32_launch_only": shared32_storage_bytes + record_count * 4,
                "packed_framegroup32_unchecked_launch_only": shared32_storage_bytes + record_count * 4,
                "packed_framegroup32_rowdesc_launch_only": int(
                    row_begin32.numel() * row_begin32.element_size()
                    + row_len_source32.numel() * row_len_source32.element_size()
                    + record_count * 4
                ),
                "packed_framegroup32_recompute_launch_only": shared32_storage_bytes + record_count * 4,
                "packed_framegroup32_smallrun16_launch_only": shared32_storage_bytes + record_count * 4,
                "packed_materialized_framegroup16_launch_only": shared16_storage_bytes + record_count * 4,
            }
        )

    ops: dict[str, TensorOp] = {
        "i16x3_framegroup32_lossreduce": lambda: endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff,
            frame_t,
            base_offsets,
            base_i16x3,
            track_change_offsets,
            chunk32,
            change_frame,
            change_offsets,
            change_i16x3,
            site_rgba,
            target_rgb,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        ),
        "packed_framegroup32": lambda: endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff,
            frame_t,
            base_offsets,
            base_packed,
            track_change_offsets,
            chunk32,
            change_frame,
            change_offsets,
            change_packed,
            site_rgba,
            target_rgb,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        ),
        "i16x3_materialized_framegroup16": lambda: endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
            coeff,
            frame_t,
            base_offsets,
            base_i16x3,
            track_change_offsets,
            chunk16,
            change_frame,
            change_offsets,
            change_i16x3,
            site_rgba,
            target_rgb,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        ),
        "i16x4_framegroup32": lambda: endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff,
            frame_t,
            base_offsets,
            base_i16x4,
            track_change_offsets,
            chunk32,
            change_frame,
            change_offsets,
            change_i16x4,
            site_rgba,
            target_rgb,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        ),
    }
    if include_diagnostic_packed_variants:
        ops.update(
            {
                "packed_framegroup32_recompute": lambda: endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
                    coeff,
                    frame_t,
                    base_offsets,
                    base_packed,
                    track_change_offsets,
                    chunk32,
                    change_frame,
                    change_offsets,
                    change_packed,
                    site_rgba,
                    target_rgb,
                    config,
                    track_count=track_count,
                    frame_count=frame_count,
                    boundary_count=boundary_count,
                ),
                "packed_framegroup32_smallrun16": lambda: endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only(
                    coeff,
                    frame_t,
                    base_offsets,
                    base_packed,
                    track_change_offsets,
                    chunk32,
                    change_frame,
                    change_offsets,
                    change_packed,
                    site_rgba,
                    target_rgb,
                    config,
                    track_count=track_count,
                    frame_count=frame_count,
                    boundary_count=boundary_count,
                ),
                "packed_materialized_framegroup16": lambda: endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
                    coeff,
                    frame_t,
                    base_offsets,
                    base_packed,
                    track_change_offsets,
                    chunk16,
                    change_frame,
                    change_offsets,
                    change_packed,
                    site_rgba,
                    target_rgb,
                    config,
                    track_count=track_count,
                    frame_count=frame_count,
                    boundary_count=boundary_count,
                ),
            }
        )
    if include_launch_only_variants:
        ops.update(
            {
                "packed_framegroup32_launch_only": lambda: _packed_launch_only_op(
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only",
                    coeff=coeff,
                    frame_t=frame_t,
                    base_offsets=base_offsets,
                    base_packed=base_packed,
                    track_change_offsets=track_change_offsets,
                    chunk_offsets=chunk32,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_packed=change_packed,
                    site_rgba=site_rgba,
                    target_rgb=target_rgb,
                    config=config,
                    boundary_count=boundary_count,
                    track_count=track_count,
                    frame_count=frame_count,
                ),
                "packed_framegroup32_unchecked_launch_only": lambda: _packed_launch_only_op(
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
                    coeff=coeff,
                    frame_t=frame_t,
                    base_offsets=base_offsets,
                    base_packed=base_packed,
                    track_change_offsets=track_change_offsets,
                    chunk_offsets=chunk32,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_packed=change_packed,
                    site_rgba=site_rgba,
                    target_rgb=target_rgb,
                    config=config,
                    boundary_count=boundary_count,
                    track_count=track_count,
                    frame_count=frame_count,
                    unchecked=True,
                ),
                "packed_framegroup32_rowdesc_launch_only": lambda: _packed_rowdesc_launch_only_op(
                    coeff=coeff,
                    frame_t=frame_t,
                    row_begin=row_begin32,
                    row_len_source=row_len_source32,
                    base_packed=base_packed,
                    change_packed=change_packed,
                    site_rgba=site_rgba,
                    target_rgb=target_rgb,
                    config=config,
                    boundary_count=boundary_count,
                    track_count=track_count,
                    frame_count=frame_count,
                ),
                "packed_framegroup32_recompute_launch_only": lambda: _packed_launch_only_op(
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only",
                    coeff=coeff,
                    frame_t=frame_t,
                    base_offsets=base_offsets,
                    base_packed=base_packed,
                    track_change_offsets=track_change_offsets,
                    chunk_offsets=chunk32,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_packed=change_packed,
                    site_rgba=site_rgba,
                    target_rgb=target_rgb,
                    config=config,
                    boundary_count=boundary_count,
                    track_count=track_count,
                    frame_count=frame_count,
                ),
                "packed_framegroup32_smallrun16_launch_only": lambda: _packed_launch_only_op(
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only",
                    coeff=coeff,
                    frame_t=frame_t,
                    base_offsets=base_offsets,
                    base_packed=base_packed,
                    track_change_offsets=track_change_offsets,
                    chunk_offsets=chunk32,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_packed=change_packed,
                    site_rgba=site_rgba,
                    target_rgb=target_rgb,
                    config=config,
                    boundary_count=boundary_count,
                    track_count=track_count,
                    frame_count=frame_count,
                ),
                "packed_materialized_framegroup16_launch_only": lambda: _packed_launch_only_op(
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only",
                    coeff=coeff,
                    frame_t=frame_t,
                    base_offsets=base_offsets,
                    base_packed=base_packed,
                    track_change_offsets=track_change_offsets,
                    chunk_offsets=chunk16,
                    change_frame=change_frame,
                    change_offsets=change_offsets,
                    change_packed=change_packed,
                    site_rgba=site_rgba,
                    target_rgb=target_rgb,
                    config=config,
                    boundary_count=boundary_count,
                    track_count=track_count,
                    frame_count=frame_count,
                ),
            }
        )

    rows: dict[str, Any] = {}
    reference_loss: torch.Tensor | None = None
    reference_grad: torch.Tensor | None = None
    timed_ops = (
        _time_ops_interleaved(ops, warmup=warmup, steps=steps)
        if interleave_variants
        else {name: _time_op(op, warmup=warmup, steps=steps) for name, op in ops.items()}
    )
    for name, (stats, loss, grad) in timed_ops.items():
        if reference_loss is None or reference_grad is None:
            reference_loss = loss
            reference_grad = grad
        loss_abs_diff = float((loss - reference_loss).abs().detach().cpu().item())
        grad_abs_diff = float((grad - reference_grad).abs().max().detach().cpu().item())
        rows[name] = {
            **stats,
            "loss": float(loss.detach().cpu().item()),
            "loss_abs_diff_vs_i16x3_framegroup32": loss_abs_diff,
            "grad_max_abs_diff_vs_i16x3_framegroup32": grad_abs_diff,
            "storage_bytes": int(variant_storage_bytes[name]),
            "storage_ratio_vs_i16x3_framegroup32": float(
                variant_storage_bytes[name] / max(variant_storage_bytes["i16x3_framegroup32_lossreduce"], 1)
            ),
        }

    return {
        "track_count": track_count,
        "track_repeats": track_repeats,
        "interleave_variants": bool(interleave_variants),
        "include_diagnostic_packed_variants": bool(include_diagnostic_packed_variants),
        "include_launch_only_variants": bool(include_launch_only_variants),
        "boundary_count": boundary_count,
        "site_count": site_count,
        "change_count": int(delta.change_frame_i32.numel()),
        "base_record_count": int(delta.base_owner_i32.numel()),
        "change_record_count": int(delta.change_owner_i32.numel()),
        "delta_storage_bytes": delta.storage_bytes,
        "variant_storage_bytes": variant_storage_bytes,
        "variants": rows,
    }


def _variant_scales(rows_by_frame: dict[str, Any], frame_counts: tuple[int, ...]) -> dict[str, dict[str, float | bool]]:
    if len(frame_counts) < 2:
        return {}
    first = str(frame_counts[0])
    last = str(frame_counts[-1])
    frame_scale = frame_counts[-1] / frame_counts[0]
    variants = rows_by_frame[first]["variants"].keys()
    out: dict[str, dict[str, float | bool]] = {}
    for variant in variants:
        first_row = rows_by_frame[first]["variants"][variant]
        last_row = rows_by_frame[last]["variants"][variant]
        mean_scale = float(last_row["mean_ms"] / max(first_row["mean_ms"], 1.0e-12))
        trimmed_mean_scale = float(last_row["trimmed_mean_ms"] / max(first_row["trimmed_mean_ms"], 1.0e-12))
        median_scale = float(last_row["median_ms"] / max(first_row["median_ms"], 1.0e-12))
        out[variant] = {
            "mean_first_to_last": mean_scale,
            "trimmed_mean_first_to_last": trimmed_mean_scale,
            "median_first_to_last": median_scale,
            "frame_scale": float(frame_scale),
            "mean_is_sublinear_vs_frame_count": bool(mean_scale < frame_scale),
            "trimmed_mean_is_sublinear_vs_frame_count": bool(trimmed_mean_scale < frame_scale),
            "median_is_sublinear_vs_frame_count": bool(median_scale < frame_scale),
        }
    return out


def run_probe(
    *,
    frame_counts: tuple[int, ...],
    warmup: int,
    steps: int,
    track_repeats: int,
    prewarm_sweep: bool,
    interleave_variants: bool,
    include_diagnostic_packed_variants: bool,
    include_launch_only_variants: bool,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for world foam Metal op timing")
    rows_by_frame: dict[str, Any] = {}
    failures: list[str] = []
    if prewarm_sweep:
        for frame_count in frame_counts:
            try:
                _frame_case(
                    frame_count,
                    warmup=1,
                    steps=1,
                    track_repeats=track_repeats,
                    interleave_variants=interleave_variants,
                    include_diagnostic_packed_variants=include_diagnostic_packed_variants,
                    include_launch_only_variants=include_launch_only_variants,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"prewarm {frame_count}f: {type(exc).__name__}: {exc}")
        if failures:
            return {
                "benchmark": "delta_framegroup_variant_timing_probe",
                "scope": "single-process synthetic MPS op timing; not a train/eval promotion artifact",
                "frame_counts": list(frame_counts),
                "track_repeats": int(track_repeats),
                "interleave_variants": bool(interleave_variants),
                "include_diagnostic_packed_variants": bool(include_diagnostic_packed_variants),
                "include_launch_only_variants": bool(include_launch_only_variants),
                "prewarm_sweep": bool(prewarm_sweep),
                "warmup": int(warmup),
                "steps": int(steps),
                "status": "failed",
                "rows_by_frame": rows_by_frame,
                "scales": {},
                "failures": failures,
            }
    for frame_count in frame_counts:
        try:
            rows_by_frame[str(frame_count)] = _frame_case(
                frame_count,
                warmup=warmup,
                steps=steps,
                track_repeats=track_repeats,
                interleave_variants=interleave_variants,
                include_diagnostic_packed_variants=include_diagnostic_packed_variants,
                include_launch_only_variants=include_launch_only_variants,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{frame_count}f: {type(exc).__name__}: {exc}")
    return {
        "benchmark": "delta_framegroup_variant_timing_probe",
        "scope": "single-process synthetic MPS op timing; not a train/eval promotion artifact",
        "frame_counts": list(frame_counts),
        "track_repeats": int(track_repeats),
        "interleave_variants": bool(interleave_variants),
        "include_diagnostic_packed_variants": bool(include_diagnostic_packed_variants),
        "include_launch_only_variants": bool(include_launch_only_variants),
        "prewarm_sweep": bool(prewarm_sweep),
        "warmup": int(warmup),
        "steps": int(steps),
        "status": "ok" if not failures else "failed",
        "rows_by_frame": rows_by_frame,
        "scales": _variant_scales(rows_by_frame, frame_counts) if not failures else {},
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-counts", default="16,32,64,128")
    parser.add_argument("--track-repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--interleave-variants", action="store_true")
    parser.add_argument(
        "--include-diagnostic-packed-variants",
        action="store_true",
        help="Also time the recompute, smallrun16, and materialized packed framegroup diagnostic shaders.",
    )
    parser.add_argument(
        "--include-launch-only-variants",
        action="store_true",
        help="Also time the launch-only packed framegroup ops that skip per-launch CPU metadata validation.",
    )
    parser.add_argument("--no-prewarm-sweep", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.track_repeats <= 0:
        raise ValueError("--track-repeats must be positive")
    frame_counts = _parse_frame_counts(args.frame_counts)
    payload = run_probe(
        frame_counts=frame_counts,
        warmup=args.warmup,
        steps=args.steps,
        track_repeats=args.track_repeats,
        prewarm_sweep=not args.no_prewarm_sweep,
        interleave_variants=bool(args.interleave_variants),
        include_diagnostic_packed_variants=bool(args.include_diagnostic_packed_variants),
        include_launch_only_variants=bool(args.include_launch_only_variants),
    )

    out_path = args.out_json
    if out_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "delta_framegroup_variant_timing_probe.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
