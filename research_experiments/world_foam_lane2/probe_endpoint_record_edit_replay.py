#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
VARIANT_TOOLS = VARIANT_ROOT / "tools"
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
EPS = 1.0e-8
OP_INSERT = 0
OP_DELETE = 1
OP_REPLACE = 2

for path in (VARIANT_ROOT, VARIANT_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    boundary_depth_coefficients,
    fit_linear_ray_track,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from probe_endpoint_record_delta_replay import (  # noqa: E402
    EndpointRecord,
    _boundary_tensor,
    _record_row,
    _tensor_error,
    _track_frame_rays,
)
from probe_endpoint_run_tape import CompactEndpointRunTape, compress_same_owner_endpoint_runs  # noqa: E402
from probe_fused_slab_segment_tape import build_segment_tape, compact_segment_tape  # noqa: E402
from probe_owner_run_boundary_tape import _build_owner_run_sequences  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list, _timed_mps_call  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff_rgba_depth_replay,
    endpoint_record_edit_block_coeff_rgb_replay,
    endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block4_rgba_depth_replay,
    endpoint_record_edit_block4_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_rgba_depth_replay,
    endpoint_record_edit_rgba_depth_replay_framegroup16,
    endpoint_record_edit_rgba_depth_replay_trackloop,
    endpoint_record_edit_vjp_direct_atomic_grad_only,
    endpoint_record_edit_vjp_direct_atomic_rgb_only,
    endpoint_run_rgba_depth_replay,
    endpoint_run_vjp_direct_atomic_grad_only,
)


EditOp = tuple[int, int, EndpointRecord | None]


@dataclass(frozen=True)
class EndpointRecordEditTape:
    base_offsets_i32: torch.Tensor
    base_owner_i32: torch.Tensor
    base_left_i32: torch.Tensor
    base_right_i32: torch.Tensor
    track_change_offsets_i32: torch.Tensor
    change_frame_i32: torch.Tensor
    op_offsets_i32: torch.Tensor
    op_type_i32: torch.Tensor
    op_pos_i32: torch.Tensor
    op_owner_i32: torch.Tensor
    op_left_i32: torch.Tensor
    op_right_i32: torch.Tensor
    changed_records: int

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.base_offsets_i32,
            self.base_owner_i32,
            self.base_left_i32,
            self.base_right_i32,
            self.track_change_offsets_i32,
            self.change_frame_i32,
            self.op_offsets_i32,
            self.op_type_i32,
            self.op_pos_i32,
            self.op_owner_i32,
            self.op_left_i32,
            self.op_right_i32,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))


@dataclass(frozen=True)
class EndpointRecordBlockEditTape:
    anchor_offsets_i32: torch.Tensor
    anchor_owner_i32: torch.Tensor
    anchor_left_i32: torch.Tensor
    anchor_right_i32: torch.Tensor
    track_block_change_offsets_i32: torch.Tensor
    change_frame_i32: torch.Tensor
    op_offsets_i32: torch.Tensor
    op_type_i32: torch.Tensor
    op_pos_i32: torch.Tensor
    op_owner_i32: torch.Tensor
    op_left_i32: torch.Tensor
    op_right_i32: torch.Tensor
    changed_records: int
    block_size: int
    block_count: int

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.anchor_offsets_i32,
            self.anchor_owner_i32,
            self.anchor_left_i32,
            self.anchor_right_i32,
            self.track_block_change_offsets_i32,
            self.change_frame_i32,
            self.op_offsets_i32,
            self.op_type_i32,
            self.op_pos_i32,
            self.op_owner_i32,
            self.op_left_i32,
            self.op_right_i32,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))


def _edit_script(left: tuple[EndpointRecord, ...], right: tuple[EndpointRecord, ...]) -> list[EditOp]:
    if left == right:
        return []
    n = len(left)
    m = len(right)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        dp[i][m] = n - i
    for j in range(m - 1, -1, -1):
        dp[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if left[i] == right[j]:
                dp[i][j] = dp[i + 1][j + 1]
            else:
                dp[i][j] = 1 + min(dp[i + 1][j + 1], dp[i + 1][j], dp[i][j + 1])

    ops: list[EditOp] = []
    i = 0
    j = 0
    pos = 0
    while i < n or j < m:
        if i < n and j < m and left[i] == right[j]:
            i += 1
            j += 1
            pos += 1
            continue
        replace_cost = 1 + dp[i + 1][j + 1] if i < n and j < m else 10**9
        delete_cost = 1 + dp[i + 1][j] if i < n else 10**9
        insert_cost = 1 + dp[i][j + 1] if j < m else 10**9
        best = min(replace_cost, delete_cost, insert_cost)
        if replace_cost == best:
            ops.append((OP_REPLACE, pos, right[j]))
            i += 1
            j += 1
            pos += 1
        elif delete_cost == best:
            ops.append((OP_DELETE, pos, None))
            i += 1
        else:
            ops.append((OP_INSERT, pos, right[j]))
            j += 1
            pos += 1
    return ops


def _apply_ops(row: tuple[EndpointRecord, ...], ops: list[EditOp]) -> tuple[EndpointRecord, ...]:
    out = list(row)
    for op_type, pos, payload in ops:
        if op_type == OP_INSERT:
            if payload is None:
                raise ValueError("insert op requires payload")
            out.insert(pos, payload)
        elif op_type == OP_DELETE:
            del out[pos]
        elif op_type == OP_REPLACE:
            if payload is None:
                raise ValueError("replace op requires payload")
            out[pos] = payload
        else:
            raise ValueError(f"unknown op type {op_type}")
    return tuple(out)


def pack_endpoint_record_edit_tape(
    sequences: list[list[tuple[Any, ...]]],
    *,
    frame_count: int,
) -> EndpointRecordEditTape:
    base_offsets = [0]
    base_owner: list[int] = []
    base_left: list[int] = []
    base_right: list[int] = []
    track_change_offsets = [0]
    change_frame: list[int] = []
    op_offsets = [0]
    op_type: list[int] = []
    op_pos: list[int] = []
    op_owner: list[int] = []
    op_left: list[int] = []
    op_right: list[int] = []
    changed_records = 0

    for frames in sequences:
        if len(frames) != frame_count:
            raise ValueError("every track sequence must have frame_count rows")
        previous = _record_row(frames[0])
        for owner, left, right in previous:
            base_owner.append(owner)
            base_left.append(left)
            base_right.append(right)
        base_offsets.append(len(base_owner))

        for frame_id, records in enumerate(frames[1:], start=1):
            current = _record_row(records)
            if current == previous:
                continue
            ops = _edit_script(previous, current)
            if _apply_ops(previous, ops) != current:
                raise AssertionError("edit script failed to reconstruct target row")
            change_frame.append(frame_id)
            changed_records += len(current)
            for op_kind, pos, payload in ops:
                op_type.append(op_kind)
                op_pos.append(pos)
                if payload is None:
                    op_owner.append(-1)
                    op_left.append(-1)
                    op_right.append(-1)
                else:
                    owner, left, right = payload
                    op_owner.append(owner)
                    op_left.append(left)
                    op_right.append(right)
            op_offsets.append(len(op_type))
            previous = current
        track_change_offsets.append(len(change_frame))

    return EndpointRecordEditTape(
        base_offsets_i32=torch.tensor(base_offsets, dtype=torch.int32),
        base_owner_i32=torch.tensor(base_owner, dtype=torch.int32),
        base_left_i32=torch.tensor(base_left, dtype=torch.int32),
        base_right_i32=torch.tensor(base_right, dtype=torch.int32),
        track_change_offsets_i32=torch.tensor(track_change_offsets, dtype=torch.int32),
        change_frame_i32=torch.tensor(change_frame, dtype=torch.int32),
        op_offsets_i32=torch.tensor(op_offsets, dtype=torch.int32),
        op_type_i32=torch.tensor(op_type, dtype=torch.int32),
        op_pos_i32=torch.tensor(op_pos, dtype=torch.int32),
        op_owner_i32=torch.tensor(op_owner, dtype=torch.int32),
        op_left_i32=torch.tensor(op_left, dtype=torch.int32),
        op_right_i32=torch.tensor(op_right, dtype=torch.int32),
        changed_records=int(changed_records),
    )


def pack_endpoint_record_block_edit_tape(
    sequences: list[list[tuple[Any, ...]]],
    *,
    frame_count: int,
    block_size: int = 4,
) -> EndpointRecordBlockEditTape:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    block_count = (frame_count + block_size - 1) // block_size
    anchor_offsets = [0]
    anchor_owner: list[int] = []
    anchor_left: list[int] = []
    anchor_right: list[int] = []
    track_block_change_offsets: list[int] = []
    change_frame: list[int] = []
    op_offsets = [0]
    op_type: list[int] = []
    op_pos: list[int] = []
    op_owner: list[int] = []
    op_left: list[int] = []
    op_right: list[int] = []
    changed_records = 0

    for frames in sequences:
        if len(frames) != frame_count:
            raise ValueError("every track sequence must have frame_count rows")
        for block_id in range(block_count):
            anchor_frame = block_id * block_size
            block_end = min(anchor_frame + block_size, frame_count)
            previous = _record_row(frames[anchor_frame])
            for owner, left, right in previous:
                anchor_owner.append(owner)
                anchor_left.append(left)
                anchor_right.append(right)
            anchor_offsets.append(len(anchor_owner))
            track_block_change_offsets.append(len(change_frame))

            for frame_id in range(anchor_frame + 1, block_end):
                current = _record_row(frames[frame_id])
                if current == previous:
                    continue
                ops = _edit_script(previous, current)
                if _apply_ops(previous, ops) != current:
                    raise AssertionError("block edit script failed to reconstruct target row")
                change_frame.append(frame_id)
                changed_records += len(current)
                for op_kind, pos, payload in ops:
                    op_type.append(op_kind)
                    op_pos.append(pos)
                    if payload is None:
                        op_owner.append(-1)
                        op_left.append(-1)
                        op_right.append(-1)
                    else:
                        owner, left, right = payload
                        op_owner.append(owner)
                        op_left.append(left)
                        op_right.append(right)
                op_offsets.append(len(op_type))
                previous = current
        track_block_change_offsets.append(len(change_frame))

    return EndpointRecordBlockEditTape(
        anchor_offsets_i32=torch.tensor(anchor_offsets, dtype=torch.int32),
        anchor_owner_i32=torch.tensor(anchor_owner, dtype=torch.int32),
        anchor_left_i32=torch.tensor(anchor_left, dtype=torch.int32),
        anchor_right_i32=torch.tensor(anchor_right, dtype=torch.int32),
        track_block_change_offsets_i32=torch.tensor(track_block_change_offsets, dtype=torch.int32),
        change_frame_i32=torch.tensor(change_frame, dtype=torch.int32),
        op_offsets_i32=torch.tensor(op_offsets, dtype=torch.int32),
        op_type_i32=torch.tensor(op_type, dtype=torch.int32),
        op_pos_i32=torch.tensor(op_pos, dtype=torch.int32),
        op_owner_i32=torch.tensor(op_owner, dtype=torch.int32),
        op_left_i32=torch.tensor(op_left, dtype=torch.int32),
        op_right_i32=torch.tensor(op_right, dtype=torch.int32),
        changed_records=int(changed_records),
        block_size=int(block_size),
        block_count=int(block_count),
    )


def _track_boundary_coefficients(
    *,
    boundaries: tuple[Any, ...],
    track_rays: torch.Tensor,
    frame_t: torch.Tensor,
) -> torch.Tensor:
    if track_rays.ndim != 3 or track_rays.shape[2] != 6:
        raise ValueError("track_rays must have shape [track_count, frame_count, 6]")
    track_count = int(track_rays.shape[0])
    times = frame_t.to(dtype=torch.float64).cpu()
    coeffs: list[tuple[float, float, float, float]] = []
    for track_id in range(track_count):
        track = fit_linear_ray_track(track_rays[track_id].cpu(), times)
        for boundary in boundaries:
            coeffs.append(boundary_depth_coefficients(boundary, track))
    return torch.tensor(coeffs, dtype=torch.float32)


def _mps_compare(
    *,
    endpoint: CompactEndpointRunTape,
    edit: EndpointRecordEditTape,
    block_edit: EndpointRecordBlockEditTape,
    boundary_f32: torch.Tensor,
    coeff_f32: torch.Tensor,
    rays_f32: torch.Tensor,
    frame_t_f32: torch.Tensor,
    site_rgba: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    timing_iters: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    endpoint_offsets = endpoint.offsets_i32.to(device=device).contiguous()
    endpoint_owner = endpoint.owners_i32.to(device=device).contiguous()
    endpoint_start = endpoint.starts_f32.to(device=device).contiguous()
    endpoint_end = endpoint.ends_f32.to(device=device).contiguous()
    boundary_mps = boundary_f32.to(device=device).contiguous()
    coeff_mps = coeff_f32.to(device=device).contiguous()
    rays_mps = rays_f32.to(device=device).contiguous()
    frame_t_mps = frame_t_f32.to(device=device).contiguous()
    site_rgba_mps = site_rgba.to(device=device).contiguous()
    base_offsets = edit.base_offsets_i32.to(device=device).contiguous()
    base_owner = edit.base_owner_i32.to(device=device).contiguous()
    base_left = edit.base_left_i32.to(device=device).contiguous()
    base_right = edit.base_right_i32.to(device=device).contiguous()
    track_change_offsets = edit.track_change_offsets_i32.to(device=device).contiguous()
    change_frame = edit.change_frame_i32.to(device=device).contiguous()
    op_offsets = edit.op_offsets_i32.to(device=device).contiguous()
    op_type = edit.op_type_i32.to(device=device).contiguous()
    op_pos = edit.op_pos_i32.to(device=device).contiguous()
    op_owner = edit.op_owner_i32.to(device=device).contiguous()
    op_left = edit.op_left_i32.to(device=device).contiguous()
    op_right = edit.op_right_i32.to(device=device).contiguous()
    anchor_offsets = block_edit.anchor_offsets_i32.to(device=device).contiguous()
    anchor_owner = block_edit.anchor_owner_i32.to(device=device).contiguous()
    anchor_left = block_edit.anchor_left_i32.to(device=device).contiguous()
    anchor_right = block_edit.anchor_right_i32.to(device=device).contiguous()
    track_block_change_offsets = block_edit.track_block_change_offsets_i32.to(device=device).contiguous()
    block_change_frame = block_edit.change_frame_i32.to(device=device).contiguous()
    block_op_offsets = block_edit.op_offsets_i32.to(device=device).contiguous()
    block_op_type = block_edit.op_type_i32.to(device=device).contiguous()
    block_op_pos = block_edit.op_pos_i32.to(device=device).contiguous()
    block_op_owner = block_edit.op_owner_i32.to(device=device).contiguous()
    block_op_left = block_edit.op_left_i32.to(device=device).contiguous()
    block_op_right = block_edit.op_right_i32.to(device=device).contiguous()

    endpoint_forward, endpoint_forward_ms = _timed_mps_call(
        lambda: endpoint_run_rgba_depth_replay(
            endpoint_offsets,
            endpoint_owner,
            endpoint_start,
            endpoint_end,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        ),
        timing_iters=timing_iters,
    )
    edit_forward, edit_forward_ms = _timed_mps_call(
        lambda: endpoint_record_edit_rgba_depth_replay(
            boundary_mps,
            rays_mps,
            frame_t_mps,
            base_offsets,
            base_owner,
            base_left,
            base_right,
            track_change_offsets,
            change_frame,
            op_offsets,
            op_type,
            op_pos,
            op_owner,
            op_left,
            op_right,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        ),
        timing_iters=timing_iters,
    )
    edit_block4_forward, edit_block4_forward_ms = _timed_mps_call(
        lambda: endpoint_record_edit_block4_rgba_depth_replay(
            boundary_mps,
            rays_mps,
            frame_t_mps,
            anchor_offsets,
            anchor_owner,
            anchor_left,
            anchor_right,
            track_block_change_offsets,
            block_change_frame,
            block_op_offsets,
            block_op_type,
            block_op_pos,
            block_op_owner,
            block_op_left,
            block_op_right,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            block_size=block_edit.block_size,
        ),
        timing_iters=timing_iters,
    )
    edit_block_coeff_forward, edit_block_coeff_forward_ms = _timed_mps_call(
        lambda: endpoint_record_edit_block_coeff_rgba_depth_replay(
            coeff_mps,
            frame_t_mps,
            anchor_offsets,
            anchor_owner,
            anchor_left,
            anchor_right,
            track_block_change_offsets,
            block_change_frame,
            block_op_offsets,
            block_op_type,
            block_op_pos,
            block_op_owner,
            block_op_left,
            block_op_right,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_f32.shape[0],
            block_size=block_edit.block_size,
        ),
        timing_iters=timing_iters,
    )
    edit_block_coeff_rgb_forward, edit_block_coeff_rgb_forward_ms = _timed_mps_call(
        lambda: endpoint_record_edit_block_coeff_rgb_replay(
            coeff_mps,
            frame_t_mps,
            anchor_offsets,
            anchor_owner,
            anchor_left,
            anchor_right,
            track_block_change_offsets,
            block_change_frame,
            block_op_offsets,
            block_op_type,
            block_op_pos,
            block_op_owner,
            block_op_left,
            block_op_right,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_f32.shape[0],
            block_size=block_edit.block_size,
        ),
        timing_iters=timing_iters,
    )
    edit_trackloop_forward, edit_trackloop_forward_ms = _timed_mps_call(
        lambda: endpoint_record_edit_rgba_depth_replay_trackloop(
            boundary_mps,
            rays_mps,
            frame_t_mps,
            base_offsets,
            base_owner,
            base_left,
            base_right,
            track_change_offsets,
            change_frame,
            op_offsets,
            op_type,
            op_pos,
            op_owner,
            op_left,
            op_right,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        ),
        timing_iters=timing_iters,
    )
    edit_framegroup16_forward, edit_framegroup16_forward_ms = _timed_mps_call(
        lambda: endpoint_record_edit_rgba_depth_replay_framegroup16(
            boundary_mps,
            rays_mps,
            frame_t_mps,
            base_offsets,
            base_owner,
            base_left,
            base_right,
            track_change_offsets,
            change_frame,
            op_offsets,
            op_type,
            op_pos,
            op_owner,
            op_left,
            op_right,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        ),
        timing_iters=timing_iters,
    )

    grad_rgb = torch.linspace(-0.25, 0.75, track_count * frame_count * 3, dtype=torch.float32, device=device).reshape(
        track_count,
        frame_count,
        3,
    )
    grad_alpha = torch.linspace(-0.5, 0.5, track_count * frame_count, dtype=torch.float32, device=device).reshape(
        track_count,
        frame_count,
    )
    grad_depth = torch.linspace(0.1, 0.6, track_count * frame_count, dtype=torch.float32, device=device).reshape(
        track_count,
        frame_count,
    )
    endpoint_grad, endpoint_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_run_vjp_direct_atomic_grad_only(
                endpoint_offsets,
                endpoint_owner,
                endpoint_start,
                endpoint_end,
                site_rgba_mps,
                grad_rgb,
                grad_alpha,
                grad_depth,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
            ),
        ),
        timing_iters=timing_iters,
    )
    edit_grad, edit_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_mps,
                rays_mps,
                frame_t_mps,
                base_offsets,
                base_owner,
                base_left,
                base_right,
                track_change_offsets,
                change_frame,
                op_offsets,
                op_type,
                op_pos,
                op_owner,
                op_left,
                op_right,
                site_rgba_mps,
                grad_rgb,
                grad_alpha,
                grad_depth,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
            ),
        ),
        timing_iters=timing_iters,
    )
    zero_alpha = torch.zeros_like(grad_alpha)
    zero_depth = torch.zeros_like(grad_depth)
    edit_rgb_grad_full, edit_rgb_full_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_mps,
                rays_mps,
                frame_t_mps,
                base_offsets,
                base_owner,
                base_left,
                base_right,
                track_change_offsets,
                change_frame,
                op_offsets,
                op_type,
                op_pos,
                op_owner,
                op_left,
                op_right,
                site_rgba_mps,
                grad_rgb,
                zero_alpha,
                zero_depth,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
            ),
        ),
        timing_iters=timing_iters,
    )
    edit_rgb_grad, edit_rgb_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_record_edit_vjp_direct_atomic_rgb_only(
                boundary_mps,
                rays_mps,
                frame_t_mps,
                base_offsets,
                base_owner,
                base_left,
                base_right,
                track_change_offsets,
                change_frame,
                op_offsets,
                op_type,
                op_pos,
                op_owner,
                op_left,
                op_right,
                site_rgba_mps,
                grad_rgb,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
            ),
        ),
        timing_iters=timing_iters,
    )
    edit_block4_rgb_grad, edit_block4_rgb_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
                boundary_mps,
                rays_mps,
                frame_t_mps,
                anchor_offsets,
                anchor_owner,
                anchor_left,
                anchor_right,
                track_block_change_offsets,
                block_change_frame,
                block_op_offsets,
                block_op_type,
                block_op_pos,
                block_op_owner,
                block_op_left,
                block_op_right,
                site_rgba_mps,
                grad_rgb,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                block_size=block_edit.block_size,
            ),
        ),
        timing_iters=timing_iters,
    )
    edit_block_coeff_rgb_grad, edit_block_coeff_rgb_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
                coeff_mps,
                frame_t_mps,
                anchor_offsets,
                anchor_owner,
                anchor_left,
                anchor_right,
                track_block_change_offsets,
                block_change_frame,
                block_op_offsets,
                block_op_type,
                block_op_pos,
                block_op_owner,
                block_op_left,
                block_op_right,
                site_rgba_mps,
                grad_rgb,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_f32.shape[0],
                block_size=block_edit.block_size,
            ),
        ),
        timing_iters=timing_iters,
    )
    mse_pattern = torch.linspace(
        -0.03,
        0.03,
        track_count * frame_count * 3,
        dtype=torch.float32,
        device=device,
    ).reshape(track_count, frame_count, 3)
    mse_target = (edit_block_coeff_rgb_forward.detach() + mse_pattern).contiguous()
    manual_mse_loss = (edit_block_coeff_rgb_forward - mse_target).square().mean().detach()
    mse_grad_rgb = (2.0 / float(edit_block_coeff_rgb_forward.numel())) * (edit_block_coeff_rgb_forward - mse_target)
    edit_block_coeff_mse_manual_grad, edit_block_coeff_mse_manual_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
                coeff_mps,
                frame_t_mps,
                anchor_offsets,
                anchor_owner,
                anchor_left,
                anchor_right,
                track_block_change_offsets,
                block_change_frame,
                block_op_offsets,
                block_op_type,
                block_op_pos,
                block_op_owner,
                block_op_left,
                block_op_right,
                site_rgba_mps,
                mse_grad_rgb.contiguous(),
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_f32.shape[0],
                block_size=block_edit.block_size,
            ),
        ),
        timing_iters=timing_iters,
    )
    edit_block_coeff_mse_fused, edit_block_coeff_mse_fused_ms = _timed_mps_call(
        lambda: endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
            coeff_mps,
            frame_t_mps,
            anchor_offsets,
            anchor_owner,
            anchor_left,
            anchor_right,
            track_block_change_offsets,
            block_change_frame,
            block_op_offsets,
            block_op_type,
            block_op_pos,
            block_op_owner,
            block_op_left,
            block_op_right,
            site_rgba_mps,
            mse_target,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_f32.shape[0],
            block_size=block_edit.block_size,
        ),
        timing_iters=timing_iters,
    )
    torch.mps.synchronize()
    return {
        "forward_errors_vs_endpoint_run": {
            "rgb": _tensor_error(edit_forward[0], endpoint_forward[0]),
            "alpha": _tensor_error(edit_forward[1], endpoint_forward[1]),
            "depth": _tensor_error(edit_forward[2], endpoint_forward[2]),
        },
        "trackloop_forward_errors_vs_endpoint_run": {
            "rgb": _tensor_error(edit_trackloop_forward[0], endpoint_forward[0]),
            "alpha": _tensor_error(edit_trackloop_forward[1], endpoint_forward[1]),
            "depth": _tensor_error(edit_trackloop_forward[2], endpoint_forward[2]),
        },
        "block4_forward_errors_vs_endpoint_run": {
            "rgb": _tensor_error(edit_block4_forward[0], endpoint_forward[0]),
            "alpha": _tensor_error(edit_block4_forward[1], endpoint_forward[1]),
            "depth": _tensor_error(edit_block4_forward[2], endpoint_forward[2]),
        },
        "block_coeff_forward_errors_vs_endpoint_run": {
            "rgb": _tensor_error(edit_block_coeff_forward[0], endpoint_forward[0]),
            "alpha": _tensor_error(edit_block_coeff_forward[1], endpoint_forward[1]),
            "depth": _tensor_error(edit_block_coeff_forward[2], endpoint_forward[2]),
        },
        "block_coeff_rgb_forward_errors": {
            "vs_endpoint_run_rgb": _tensor_error(edit_block_coeff_rgb_forward, endpoint_forward[0]),
            "vs_block_coeff_rgb": _tensor_error(edit_block_coeff_rgb_forward, edit_block_coeff_forward[0]),
        },
        "framegroup16_forward_errors_vs_endpoint_run": {
            "rgb": _tensor_error(edit_framegroup16_forward[0], endpoint_forward[0]),
            "alpha": _tensor_error(edit_framegroup16_forward[1], endpoint_forward[1]),
            "depth": _tensor_error(edit_framegroup16_forward[2], endpoint_forward[2]),
        },
        "vjp_error_vs_endpoint_run": _tensor_error(edit_grad[0], endpoint_grad[0]),
        "rgb_only_vjp_error_vs_full_zero_alpha_depth": _tensor_error(edit_rgb_grad[0], edit_rgb_grad_full[0]),
        "block4_rgb_only_vjp_error_vs_full_zero_alpha_depth": _tensor_error(
            edit_block4_rgb_grad[0],
            edit_rgb_grad_full[0],
        ),
        "block4_rgb_only_vjp_error_vs_edit_rgb_only": _tensor_error(edit_block4_rgb_grad[0], edit_rgb_grad[0]),
        "block_coeff_rgb_only_vjp_error_vs_full_zero_alpha_depth": _tensor_error(
            edit_block_coeff_rgb_grad[0],
            edit_rgb_grad_full[0],
        ),
        "block_coeff_rgb_only_vjp_error_vs_edit_rgb_only": _tensor_error(
            edit_block_coeff_rgb_grad[0],
            edit_rgb_grad[0],
        ),
        "block_coeff_rgb_only_vjp_error_vs_block4_rgb_only": _tensor_error(
            edit_block_coeff_rgb_grad[0],
            edit_block4_rgb_grad[0],
        ),
        "block_coeff_mse_fused_errors": {
            "loss": _tensor_error(edit_block_coeff_mse_fused[0], manual_mse_loss.reshape(1)),
            "grad": _tensor_error(edit_block_coeff_mse_fused[1], edit_block_coeff_mse_manual_grad[0]),
        },
        "timing_ms": {
            "endpoint_forward": float(endpoint_forward_ms),
            "edit_forward": float(edit_forward_ms),
            "edit_block4_forward": float(edit_block4_forward_ms),
            "edit_block_coeff_forward": float(edit_block_coeff_forward_ms),
            "edit_block_coeff_rgb_forward": float(edit_block_coeff_rgb_forward_ms),
            "edit_trackloop_forward": float(edit_trackloop_forward_ms),
            "edit_framegroup16_forward": float(edit_framegroup16_forward_ms),
            "endpoint_vjp": float(endpoint_vjp_ms),
            "edit_vjp": float(edit_vjp_ms),
            "edit_rgb_full_vjp": float(edit_rgb_full_vjp_ms),
            "edit_rgb_only_vjp": float(edit_rgb_vjp_ms),
            "edit_block4_rgb_only_vjp": float(edit_block4_rgb_vjp_ms),
            "edit_block_coeff_rgb_only_vjp": float(edit_block_coeff_rgb_vjp_ms),
            "edit_block_coeff_mse_manual_vjp": float(edit_block_coeff_mse_manual_vjp_ms),
            "edit_block_coeff_mse_fused_loss_vjp": float(edit_block_coeff_mse_fused_ms),
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
    timing_iters: int,
    edit_block_size: int,
) -> dict[str, Any]:
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
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
    full = compact_segment_tape(tape)
    endpoint = compress_same_owner_endpoint_runs(tape)
    edit = pack_endpoint_record_edit_tape(all_sequences, frame_count=frame_count)
    block_edit = pack_endpoint_record_block_edit_tape(
        all_sequences,
        frame_count=frame_count,
        block_size=edit_block_size,
    )
    track_rays, frame_t = _track_frame_rays(rays, frame_indices, frame_count=frame_count)
    coeff_f32 = _track_boundary_coefficients(
        boundaries=boundaries,
        track_rays=track_rays,
        frame_t=frame_t,
    )
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    mps = _mps_compare(
        endpoint=endpoint,
        edit=edit,
        block_edit=block_edit,
        boundary_f32=_boundary_tensor(boundaries),
        coeff_f32=coeff_f32,
        rays_f32=track_rays,
        frame_t_f32=frame_t,
        site_rgba=site_rgba,
        op_config=op_config,
        track_count=tape.track_count,
        frame_count=frame_count,
        timing_iters=timing_iters,
    )
    return {
        "frames": int(frame_count),
        "edit_block_size": int(edit_block_size),
        "render_size": int(render_size),
        "track_count": int(tape.track_count),
        "sample_count": int(tape.sample_count),
        "site_count": int(len(sites)),
        "boundary_count": int(len(boundaries)),
        "full_segments": int(full.owners_i32.numel()),
        "endpoint_runs": int(endpoint.owners_i32.numel()),
        "change_events": int(edit.change_frame_i32.numel()),
        "edit_ops": int(edit.op_type_i32.numel()),
        "block4_edit_ops": int(block_edit.op_type_i32.numel()),
        "changed_records": int(edit.changed_records),
        "block4_changed_records": int(block_edit.changed_records),
        "endpoint_storage_bytes": int(endpoint.storage_bytes),
        "edit_storage_bytes": int(edit.storage_bytes),
        "block4_storage_bytes": int(block_edit.storage_bytes),
        "block_coeff_storage_bytes": int(block_edit.storage_bytes + coeff_f32.numel() * coeff_f32.element_size()),
        "edit_storage_vs_endpoint_csr": float(edit.storage_bytes) / float(max(endpoint.storage_bytes, 1)),
        "edit_storage_vs_full_segment_csr": float(edit.storage_bytes) / float(max(full.storage_bytes, 1)),
        "block4_storage_vs_endpoint_csr": float(block_edit.storage_bytes) / float(max(endpoint.storage_bytes, 1)),
        "block4_storage_vs_full_segment_csr": float(block_edit.storage_bytes) / float(max(full.storage_bytes, 1)),
        "block_coeff_storage_vs_endpoint_csr": float(block_edit.storage_bytes + coeff_f32.numel() * coeff_f32.element_size())
        / float(max(endpoint.storage_bytes, 1)),
        "block_coeff_storage_vs_full_segment_csr": float(block_edit.storage_bytes + coeff_f32.numel() * coeff_f32.element_size())
        / float(max(full.storage_bytes, 1)),
        "mps": mps,
    }


def _scale(rows: list[dict[str, Any]], key: str) -> float:
    return float(rows[-1][key]) / float(max(float(rows[0][key]), 1.0e-9))


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
    timing_iters: int,
    edit_block_size: int = 4,
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
            timing_iters=timing_iters,
            edit_block_size=edit_block_size,
        )
        for frame_count in frame_counts
    ]
    frame_scale = float(rows[-1]["frames"]) / float(max(rows[0]["frames"], 1))
    storage_scale = _scale(rows, "edit_storage_bytes")
    block_storage_scale = _scale(rows, "block4_storage_bytes")
    endpoint_scale = _scale(rows, "endpoint_runs")
    op_scale = _scale(rows, "edit_ops")
    scaling_gate_applicable = len(rows) >= 2 and frame_scale > 1.0
    max_forward_error = max(
        float(row["mps"]["forward_errors_vs_endpoint_run"][name]["max_abs"])
        for row in rows
        for name in ("rgb", "alpha", "depth")
    )
    max_trackloop_forward_error = max(
        float(row["mps"]["trackloop_forward_errors_vs_endpoint_run"][name]["max_abs"])
        for row in rows
        for name in ("rgb", "alpha", "depth")
    )
    max_block4_forward_error = max(
        float(row["mps"]["block4_forward_errors_vs_endpoint_run"][name]["max_abs"])
        for row in rows
        for name in ("rgb", "alpha", "depth")
    )
    max_block_coeff_forward_error = max(
        float(row["mps"]["block_coeff_forward_errors_vs_endpoint_run"][name]["max_abs"])
        for row in rows
        for name in ("rgb", "alpha", "depth")
    )
    max_block_coeff_rgb_forward_error = max(
        float(row["mps"]["block_coeff_rgb_forward_errors"][name]["max_abs"])
        for row in rows
        for name in ("vs_endpoint_run_rgb", "vs_block_coeff_rgb")
    )
    max_framegroup16_forward_error = max(
        float(row["mps"]["framegroup16_forward_errors_vs_endpoint_run"][name]["max_abs"])
        for row in rows
        for name in ("rgb", "alpha", "depth")
    )
    max_vjp_rel_error = max(float(row["mps"]["vjp_error_vs_endpoint_run"]["rel_to_rhs_abs_max"]) for row in rows)
    max_rgb_only_vjp_rel_error = max(
        float(row["mps"]["rgb_only_vjp_error_vs_full_zero_alpha_depth"]["rel_to_rhs_abs_max"]) for row in rows
    )
    max_block4_rgb_only_vjp_rel_error = max(
        float(row["mps"]["block4_rgb_only_vjp_error_vs_full_zero_alpha_depth"]["rel_to_rhs_abs_max"])
        for row in rows
    )
    max_block4_vs_edit_rgb_only_vjp_rel_error = max(
        float(row["mps"]["block4_rgb_only_vjp_error_vs_edit_rgb_only"]["rel_to_rhs_abs_max"]) for row in rows
    )
    max_block_coeff_rgb_only_vjp_rel_error = max(
        float(row["mps"]["block_coeff_rgb_only_vjp_error_vs_full_zero_alpha_depth"]["rel_to_rhs_abs_max"])
        for row in rows
    )
    max_block_coeff_vs_edit_rgb_only_vjp_rel_error = max(
        float(row["mps"]["block_coeff_rgb_only_vjp_error_vs_edit_rgb_only"]["rel_to_rhs_abs_max"]) for row in rows
    )
    max_block_coeff_vs_block4_rgb_only_vjp_rel_error = max(
        float(row["mps"]["block_coeff_rgb_only_vjp_error_vs_block4_rgb_only"]["rel_to_rhs_abs_max"])
        for row in rows
    )
    max_block_coeff_mse_fused_loss_error = max(
        float(row["mps"]["block_coeff_mse_fused_errors"]["loss"]["max_abs"]) for row in rows
    )
    max_block_coeff_mse_fused_grad_rel_error = max(
        float(row["mps"]["block_coeff_mse_fused_errors"]["grad"]["rel_to_rhs_abs_max"]) for row in rows
    )
    acceptance = {
        "metal_forward_matches_endpoint_run": max_forward_error < 1.0e-4,
        "metal_trackloop_forward_matches_endpoint_run": max_trackloop_forward_error < 1.0e-4,
        "metal_block4_forward_matches_endpoint_run": max_block4_forward_error < 1.0e-4,
        "metal_block_coeff_forward_matches_endpoint_run": max_block_coeff_forward_error < 1.0e-4,
        "metal_block_coeff_rgb_forward_matches": max_block_coeff_rgb_forward_error < 1.0e-4,
        "metal_framegroup16_forward_matches_endpoint_run": max_framegroup16_forward_error < 1.0e-4,
        "metal_vjp_matches_endpoint_run": max_vjp_rel_error < 1.0e-4,
        "metal_rgb_only_vjp_matches_full_zero_alpha_depth": max_rgb_only_vjp_rel_error < 1.0e-4,
        "metal_block4_rgb_only_vjp_matches_full_zero_alpha_depth": max_block4_rgb_only_vjp_rel_error < 1.0e-4,
        "metal_block4_rgb_only_vjp_matches_edit_rgb_only": max_block4_vs_edit_rgb_only_vjp_rel_error < 1.0e-4,
        "metal_block_coeff_rgb_only_vjp_matches_full_zero_alpha_depth": max_block_coeff_rgb_only_vjp_rel_error < 1.0e-4,
        "metal_block_coeff_rgb_only_vjp_matches_edit_rgb_only": max_block_coeff_vs_edit_rgb_only_vjp_rel_error < 1.0e-4,
        "metal_block_coeff_rgb_only_vjp_matches_block4_rgb_only": max_block_coeff_vs_block4_rgb_only_vjp_rel_error < 1.0e-4,
        "metal_block_coeff_mse_fused_matches_manual_loss": max_block_coeff_mse_fused_loss_error < 1.0e-6,
        "metal_block_coeff_mse_fused_matches_manual_grad": max_block_coeff_mse_fused_grad_rel_error < 1.0e-4,
        "edit_ops_sublinear_vs_frames": True if not scaling_gate_applicable else op_scale < frame_scale,
        "edit_storage_sublinear_vs_frames": True if not scaling_gate_applicable else storage_scale < frame_scale,
        "last_edit_storage_below_endpoint_csr": rows[-1]["edit_storage_vs_endpoint_csr"] < 0.50,
    }
    return {
        "benchmark": "world_foam_lane2_endpoint_record_edit_replay",
        "status": "ok" if all(acceptance.values()) else "negative",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "edit_block_size": int(edit_block_size),
        "render_size": int(render_size),
        "site_count": int(site_count),
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "scaling_gate_applicable": scaling_gate_applicable,
        "endpoint_run_scale_first_to_last": endpoint_scale,
        "edit_op_scale_first_to_last": op_scale,
        "edit_storage_scale_first_to_last": storage_scale,
        "block_edit_storage_scale_first_to_last": block_storage_scale,
        "max_forward_abs_error_vs_endpoint_run": max_forward_error,
        "max_trackloop_forward_abs_error_vs_endpoint_run": max_trackloop_forward_error,
        "max_block4_forward_abs_error_vs_endpoint_run": max_block4_forward_error,
        "max_block_coeff_forward_abs_error_vs_endpoint_run": max_block_coeff_forward_error,
        "max_block_coeff_rgb_forward_abs_error": max_block_coeff_rgb_forward_error,
        "max_framegroup16_forward_abs_error_vs_endpoint_run": max_framegroup16_forward_error,
        "max_vjp_rel_error_vs_endpoint_run": max_vjp_rel_error,
        "max_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": max_rgb_only_vjp_rel_error,
        "max_block4_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": max_block4_rgb_only_vjp_rel_error,
        "max_block4_rgb_only_vjp_rel_error_vs_edit_rgb_only": max_block4_vs_edit_rgb_only_vjp_rel_error,
        "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": max_block_coeff_rgb_only_vjp_rel_error,
        "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only": max_block_coeff_vs_edit_rgb_only_vjp_rel_error,
        "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only": max_block_coeff_vs_block4_rgb_only_vjp_rel_error,
        "max_block_coeff_mse_fused_loss_abs_error": max_block_coeff_mse_fused_loss_error,
        "max_block_coeff_mse_fused_grad_rel_error": max_block_coeff_mse_fused_grad_rel_error,
        "structural_read": {
            "shader_replays_owner_cut_id_edit_stream": True,
            "depths_recovered_from_boundary_ids_and_rays": True,
            "not_main_trainer_integration": True,
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe endpoint owner+cut-id edit-stream Metal replay.")
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
    parser.add_argument("--edit-block-size", type=int, default=4)
    parser.add_argument("--timing-iters", type=int, default=5)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_endpoint_record_edit_replay_render32_2_4_8_16.json",
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
        timing_iters=max(int(args.timing_iters), 1),
        edit_block_size=int(args.edit_block_size),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
