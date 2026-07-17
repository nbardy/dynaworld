#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
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

from gate1_realray_per_sample_reference import (  # noqa: E402
    candidate_depths_4d,
    owner_at_4d,
)
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
from smoke_fused_slab_affine_realray_mps import (  # noqa: E402
    _build_affine_csr_bundle,
    _make_vjp_seed_tensors,
    _parse_int_list,
    _storage_bytes,
    _timed_mps_call,
)
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    fused_slab_affine_num32_den16_realray_rgba_depth_replay,
    fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only,
    fused_slab_affine_num32_den16_vjp_reduce,
    segment_tape_rgba_depth_replay,
    segment_tape_vjp_direct_atomic_grad_only,
    segment_tape_vjp_direct_atomic_track,
)


EPS = 1.0e-8


@dataclass(frozen=True)
class SegmentTape:
    owners_i32: torch.Tensor
    lengths_f32: torch.Tensor
    mids_f32: torch.Tensor
    counts_i32: torch.Tensor
    active_counts_i32: torch.Tensor
    frame_t_f32: torch.Tensor
    track_count: int
    frame_count: int
    max_segments: int

    @property
    def sample_count(self) -> int:
        return int(self.track_count * self.frame_count)

    @property
    def storage_bytes(self) -> int:
        return _storage_bytes(self.owners_i32, self.lengths_f32, self.mids_f32, self.counts_i32)


@dataclass(frozen=True)
class CompactSegmentTape:
    offsets_i32: torch.Tensor
    owners_i32: torch.Tensor
    lengths_f32: torch.Tensor
    mids_f32: torch.Tensor

    @property
    def storage_bytes(self) -> int:
        return _storage_bytes(self.offsets_i32, self.owners_i32, self.lengths_f32, self.mids_f32)


def _track_time_values(frame_indices: torch.Tensor, *, view: int, frame_count: int) -> torch.Tensor:
    return torch.tensor(
        [_frame_time(int(frame_indices[view * frame_count + frame].item()), frame_count) for frame in range(frame_count)],
        dtype=torch.float32,
    )


def _ray_tuple(ray: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
        (float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
    )


def _build_segment_lists(
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
) -> tuple[list[list[tuple[int, float, float]]], list[int], torch.Tensor]:
    rays = rays.detach().cpu().to(dtype=torch.float32)
    frame_indices = frame_indices.detach().cpu().to(dtype=torch.long)
    sample_count, height, width, payload = rays.shape
    if payload != 6:
        raise ValueError(f"rays must have payload dimension 6, got {payload}")
    if sample_count % frame_count != 0:
        raise ValueError("sample count must be view_count * frame_count")
    view_count = int(sample_count // frame_count)
    frame_t = _track_time_values(frame_indices, view=0, frame_count=frame_count)
    for view in range(1, view_count):
        view_t = _track_time_values(frame_indices, view=view, frame_count=frame_count)
        if not torch.allclose(frame_t, view_t):
            raise ValueError("segment tape expects all train views to use the same frame times")

    segment_lists: list[list[tuple[int, float, float]]] = []
    active_counts: list[int] = []
    for view in range(view_count):
        for y in range(height):
            for x in range(width):
                for frame in range(frame_count):
                    sample_index = view * frame_count + frame
                    t = float(frame_t[frame].item())
                    origin, direction = _ray_tuple(rays[sample_index, y, x])
                    depths, _invalid = candidate_depths_4d(
                        boundaries,
                        origin=origin,
                        direction=direction,
                        t=t,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                    )
                    ox, oy, oz = origin
                    dx, dy, dz = direction
                    sample_segments: list[tuple[int, float, float]] = []
                    transmittance = 1.0
                    active_count = 0
                    cuts = [near, *depths, far]
                    for depth0, depth1 in zip(cuts[:-1], cuts[1:], strict=True):
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
                        sample_segments.append((int(owner), length, mid))
                        if transmittance > transmittance_threshold:
                            active_count += 1
                        density = max(float(sites[owner].rgba[3]), 0.0)
                        transmittance *= math.exp(-density * length)
                    segment_lists.append(sample_segments)
                    active_counts.append(active_count)
    return segment_lists, active_counts, frame_t.contiguous()


def build_segment_tape(
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
) -> SegmentTape:
    segment_lists, active_counts, frame_t_f32 = _build_segment_lists(
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
    sample_count = len(segment_lists)
    if sample_count % frame_count != 0:
        raise ValueError("segment sample count must be track_count * frame_count")
    track_count = int(sample_count // frame_count)
    max_segments = max((len(segments) for segments in segment_lists), default=0)
    owners = torch.full((sample_count, max_segments), -1, dtype=torch.int32)
    lengths = torch.zeros((sample_count, max_segments), dtype=torch.float32)
    mids = torch.zeros((sample_count, max_segments), dtype=torch.float32)
    counts = torch.empty((sample_count,), dtype=torch.int32)
    active_counts_tensor = torch.empty((sample_count,), dtype=torch.int32)
    for sample_id, segments in enumerate(segment_lists):
        counts[sample_id] = len(segments)
        active_counts_tensor[sample_id] = active_counts[sample_id]
        for segment_id, (owner, length, mid) in enumerate(segments):
            owners[sample_id, segment_id] = int(owner)
            lengths[sample_id, segment_id] = float(length)
            mids[sample_id, segment_id] = float(mid)
    return SegmentTape(
        owners_i32=owners.reshape(track_count, frame_count, max_segments).contiguous(),
        lengths_f32=lengths.reshape(track_count, frame_count, max_segments).contiguous(),
        mids_f32=mids.reshape(track_count, frame_count, max_segments).contiguous(),
        counts_i32=counts.reshape(track_count, frame_count).contiguous(),
        active_counts_i32=active_counts_tensor.reshape(track_count, frame_count).contiguous(),
        frame_t_f32=frame_t_f32,
        track_count=track_count,
        frame_count=frame_count,
        max_segments=max_segments,
    )


def replay_segment_tape(
    *,
    tape: SegmentTape,
    site_rgba_f32: torch.Tensor,
    far: float,
    transmittance_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    owners = tape.owners_i32.to(device=device, dtype=torch.long).reshape(tape.sample_count, tape.max_segments)
    lengths = tape.lengths_f32.to(device=device).reshape(tape.sample_count, tape.max_segments)
    mids = tape.mids_f32.to(device=device).reshape(tape.sample_count, tape.max_segments)
    counts = tape.counts_i32.to(device=device, dtype=torch.long).reshape(tape.sample_count)
    if tape.max_segments == 0:
        rgb = torch.zeros((tape.track_count, tape.frame_count, 3), device=device, dtype=site_rgba_f32.dtype)
        alpha = torch.zeros((tape.track_count, tape.frame_count), device=device, dtype=site_rgba_f32.dtype)
        depth = torch.full((tape.track_count, tape.frame_count), far, device=device, dtype=site_rgba_f32.dtype)
        return rgb, alpha, depth

    segment_ids = torch.arange(tape.max_segments, device=device)[None, :]
    valid = segment_ids < counts[:, None]
    safe_owners = owners.clamp_min(0)
    segment_rgba = site_rgba_f32[safe_owners]
    density = segment_rgba[..., 3].clamp_min(0.0)
    optical = torch.where(valid, density * lengths, torch.zeros_like(lengths))
    cumulative_before = torch.cumsum(optical, dim=1) - optical
    trans_before = torch.exp(-cumulative_before)
    segment_alpha = 1.0 - torch.exp(-optical)
    active = valid & (trans_before > transmittance_threshold)
    weight = torch.where(active, trans_before * segment_alpha, torch.zeros_like(segment_alpha))
    rgb = torch.sum(weight[..., None] * segment_rgba[..., :3], dim=1)
    alpha = torch.sum(weight, dim=1)
    depth_num = torch.sum(weight * mids, dim=1)
    depth = torch.where(alpha > EPS, depth_num / torch.clamp(alpha, min=EPS), torch.full_like(alpha, far))
    return (
        rgb.reshape(tape.track_count, tape.frame_count, 3).contiguous(),
        alpha.reshape(tape.track_count, tape.frame_count).contiguous(),
        depth.reshape(tape.track_count, tape.frame_count).contiguous(),
    )


def compact_segment_tape(tape: SegmentTape) -> CompactSegmentTape:
    counts = tape.counts_i32.reshape(-1).to(dtype=torch.int32).contiguous()
    offsets = torch.empty((counts.numel() + 1,), dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    if tape.max_segments == 0:
        return CompactSegmentTape(
            offsets_i32=offsets,
            owners_i32=torch.empty((0,), dtype=torch.int32),
            lengths_f32=torch.empty((0,), dtype=torch.float32),
            mids_f32=torch.empty((0,), dtype=torch.float32),
        )
    segment_ids = torch.arange(tape.max_segments, dtype=torch.int64)[None, :]
    mask = segment_ids < counts.to(dtype=torch.int64)[:, None]
    return CompactSegmentTape(
        offsets_i32=offsets,
        owners_i32=tape.owners_i32.reshape(-1, tape.max_segments)[mask].to(dtype=torch.int32).contiguous(),
        lengths_f32=tape.lengths_f32.reshape(-1, tape.max_segments)[mask].to(dtype=torch.float32).contiguous(),
        mids_f32=tape.mids_f32.reshape(-1, tape.max_segments)[mask].to(dtype=torch.float32).contiguous(),
    )


def _current_fused_outputs_and_grad(
    *,
    bundle: dict[str, Any],
    sites: tuple[Any, ...],
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    time_slabs: int,
    vjp_seed_mode: str,
    vjp_reduce_chunk_size: int,
    timing_iters: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    sites_f32 = torch.tensor(
        [[site.x, site.y, site.z, site.t, site.weight] for site in sites],
        dtype=torch.float32,
        device=device,
    )
    site_rgba_f32 = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    row_index_i32 = bundle["row_index"].to(device)
    row_offsets_i32 = bundle["row_offsets"].to(device)
    candidate_depth_num_f32 = bundle["candidate_depth_coeffs"][:, :2].contiguous().to(device)
    candidate_depth_den_f16 = bundle["candidate_depth_coeffs"][:, 2:].contiguous().to(device=device, dtype=torch.float16)
    ray_coeff_f32 = bundle["ray_coeff"].to(device)
    frame_t_f32 = bundle["frame_t"].to(device)

    fused, fused_ms = _timed_mps_call(
        lambda: fused_slab_affine_num32_den16_realray_rgba_depth_replay(
            row_index_i32,
            row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            op_config,
            time_slab_count=time_slabs,
            row_count=int(bundle["row_count"]),
        ),
        timing_iters=timing_iters,
    )
    grad_rgb_f32, grad_alpha_f32, grad_depth_f32, seed_summary = _make_vjp_seed_tensors(
        mode=vjp_seed_mode,
        track_count=int(bundle["track_count"]),
        frame_count=frame_count,
        device=device,
    )
    vjp_reduce, vjp_reduce_ms = _timed_mps_call(
        lambda: fused_slab_affine_num32_den16_vjp_reduce(
            row_index_i32,
            row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            grad_rgb_f32,
            grad_alpha_f32,
            grad_depth_f32,
            op_config,
            time_slab_count=time_slabs,
            row_count=int(bundle["row_count"]),
            reduce_chunk_size=vjp_reduce_chunk_size,
        ),
        timing_iters=timing_iters,
    )
    grad_only_out, grad_only_ms = _timed_mps_call(
        lambda: (
            fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
                row_index_i32,
                row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb_f32,
                grad_alpha_f32,
                grad_depth_f32,
                op_config,
                time_slab_count=time_slabs,
                row_count=int(bundle["row_count"]),
            ),
        ),
        timing_iters=timing_iters,
    )
    fused_rgb, fused_alpha, fused_depth = (tensor.detach().cpu() for tensor in fused)
    vjp_rgb, vjp_alpha, vjp_depth, vjp_grad = (tensor.detach().cpu() for tensor in vjp_reduce)
    grad_only = grad_only_out[0].detach().cpu()
    track_count = int(bundle["track_count"])
    return {
        "rgb": fused_rgb.reshape(track_count, frame_count, 3),
        "alpha": fused_alpha.reshape(track_count, frame_count),
        "depth": fused_depth.reshape(track_count, frame_count),
        "vjp_rgb": vjp_rgb.reshape(track_count, frame_count, 3),
        "vjp_alpha": vjp_alpha.reshape(track_count, frame_count),
        "vjp_depth": vjp_depth.reshape(track_count, frame_count),
        "vjp_grad": vjp_grad,
        "grad_only": grad_only,
        "grad_rgb": grad_rgb_f32.detach().cpu(),
        "grad_alpha": grad_alpha_f32.detach().cpu(),
        "grad_depth": grad_depth_f32.detach().cpu(),
        "vjp_seed_summary": seed_summary,
        "timing_ms": {
            "current_fused_forward": float(fused_ms),
            "current_vjp_reduce": float(vjp_reduce_ms),
            "current_vjp_direct_atomic_grad_only": float(grad_only_ms),
        },
    }


def _tape_grad_and_timing(
    *,
    tape: SegmentTape,
    sites: tuple[Any, ...],
    far: float,
    transmittance_threshold: float,
    grad_rgb: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_depth: torch.Tensor,
    device: torch.device,
    timing_iters: int,
) -> dict[str, Any]:
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    replay, forward_ms = _timed_torch_call(
        lambda: replay_segment_tape(
            tape=tape,
            site_rgba_f32=site_rgba,
            far=far,
            transmittance_threshold=transmittance_threshold,
            device=device,
        ),
        device=device,
        timing_iters=timing_iters,
    )
    grad_rgb_device = grad_rgb.to(device=device).contiguous()
    grad_alpha_device = grad_alpha.to(device=device).contiguous()
    grad_depth_device = grad_depth.to(device=device).contiguous()

    def grad_call() -> tuple[torch.Tensor]:
        site_rgba_leaf = site_rgba.detach().clone().requires_grad_(True)
        rgb, alpha, depth = replay_segment_tape(
            tape=tape,
            site_rgba_f32=site_rgba_leaf,
            far=far,
            transmittance_threshold=transmittance_threshold,
            device=device,
        )
        loss = (rgb * grad_rgb_device).sum() + (alpha * grad_alpha_device).sum() + (depth * grad_depth_device).sum()
        loss.backward()
        grad = site_rgba_leaf.grad
        if grad is None:
            raise RuntimeError("segment tape replay did not produce site_rgba grad")
        return (grad.detach(),)

    grad_tuple, backward_ms = _timed_torch_call(grad_call, device=device, timing_iters=timing_iters)
    rgb, alpha, depth = (tensor.detach().cpu() for tensor in replay)
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "grad": grad_tuple[0].detach().cpu(),
        "timing_ms": {
            "torch_segment_tape_forward": float(forward_ms),
            "torch_segment_tape_forward_backward": float(backward_ms),
        },
    }


def _metal_tape_grad_and_timing(
    *,
    tape: SegmentTape,
    compact: CompactSegmentTape,
    sites: tuple[Any, ...],
    far: float,
    transmittance_threshold: float,
    grad_rgb: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_depth: torch.Tensor,
    timing_iters: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    op_config = RealRayReplayConfig(
        near=0.0,
        far=far,
        invalid_epsilon=1.0e-7,
        transmittance_threshold=transmittance_threshold,
    )
    offsets = compact.offsets_i32.to(device=device).contiguous()
    owners = compact.owners_i32.to(device=device).contiguous()
    lengths = compact.lengths_f32.to(device=device).contiguous()
    mids = compact.mids_f32.to(device=device).contiguous()
    forward, forward_ms = _timed_mps_call(
        lambda: segment_tape_rgba_depth_replay(
            offsets,
            owners,
            lengths,
            mids,
            site_rgba,
            op_config,
            track_count=tape.track_count,
            frame_count=tape.frame_count,
        ),
        timing_iters=timing_iters,
    )
    grad_rgb_device = grad_rgb.to(device=device).contiguous()
    grad_alpha_device = grad_alpha.to(device=device).contiguous()
    grad_depth_device = grad_depth.to(device=device).contiguous()
    grad_only, grad_ms = _timed_mps_call(
        lambda: (
            segment_tape_vjp_direct_atomic_grad_only(
                offsets,
                owners,
                lengths,
                mids,
                site_rgba,
                grad_rgb_device,
                grad_alpha_device,
                grad_depth_device,
                op_config,
                track_count=tape.track_count,
                frame_count=tape.frame_count,
            ),
        ),
        timing_iters=timing_iters,
    )
    track_out, track_ms = _timed_mps_call(
        lambda: (
            segment_tape_vjp_direct_atomic_track(
                offsets,
                owners,
                lengths,
                mids,
                site_rgba,
                grad_rgb_device,
                grad_alpha_device,
                grad_depth_device,
                op_config,
                track_count=tape.track_count,
                frame_count=tape.frame_count,
            ),
        ),
        timing_iters=timing_iters,
    )
    rgb, alpha, depth = (tensor.detach().cpu() for tensor in forward)
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "grad": grad_only[0].detach().cpu(),
        "track_grad": track_out[0].detach().cpu(),
        "timing_ms": {
            "metal_segment_tape_forward": float(forward_ms),
            "metal_segment_tape_grad_only": float(grad_ms),
            "metal_segment_tape_track_grad_only": float(track_ms),
        },
    }


def _timed_torch_call(
    fn: Any,
    *,
    device: torch.device,
    timing_iters: int,
) -> tuple[tuple[torch.Tensor, ...], float]:
    out = fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(timing_iters):
        out = fn()
    _sync(device)
    return out, (time.perf_counter() - start) * 1000.0 / float(timing_iters)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _tensor_error(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    diff = (lhs - rhs).abs()
    denom = max(float(rhs.abs().max().item()), 1.0e-9)
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "rel_to_rhs_abs_max": float(diff.max().item() / denom) if diff.numel() else 0.0,
    }


def _profile_frame_count(
    *,
    frame_count: int,
    config_path: Path,
    render_size: int,
    site_count: int,
    time_slabs: int,
    layout: str,
    tile_h: int,
    tile_w: int,
    candidate_order: str,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    vjp_seed_mode: str,
    vjp_reduce_chunk_size: int,
    timing_iters: int,
    tape_device: torch.device,
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
    compact = compact_segment_tape(tape)
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
        layout=layout,
        tile_h=tile_h,
        tile_w=tile_w,
        candidate_order=candidate_order,
    )
    current = _current_fused_outputs_and_grad(
        bundle=bundle,
        sites=sites,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        time_slabs=time_slabs,
        vjp_seed_mode=vjp_seed_mode,
        vjp_reduce_chunk_size=vjp_reduce_chunk_size,
        timing_iters=timing_iters,
    )
    tape_replay = _tape_grad_and_timing(
        tape=tape,
        sites=sites,
        far=far,
        transmittance_threshold=transmittance_threshold,
        grad_rgb=current["grad_rgb"],
        grad_alpha=current["grad_alpha"],
        grad_depth=current["grad_depth"],
        device=tape_device,
        timing_iters=timing_iters,
    )
    metal_tape = _metal_tape_grad_and_timing(
        tape=tape,
        compact=compact,
        sites=sites,
        far=far,
        transmittance_threshold=transmittance_threshold,
        grad_rgb=current["grad_rgb"],
        grad_alpha=current["grad_alpha"],
        grad_depth=current["grad_depth"],
        timing_iters=timing_iters,
    )
    explicit_ray_storage_bytes = _storage_bytes(bundle["explicit_rays"])
    mixed_candidate_storage_bytes = _storage_bytes(
        bundle["row_index"],
        bundle["row_offsets"],
        bundle["candidate_depth_coeffs"][:, :2].contiguous(),
        bundle["candidate_depth_coeffs"][:, 2:].contiguous().to(dtype=torch.float16),
    )
    affine_ray_storage_bytes = _storage_bytes(bundle["ray_coeff"])
    total_tape_segments = int(tape.counts_i32.to(dtype=torch.int64).sum().item())
    active_tape_segments = int(tape.active_counts_i32.to(dtype=torch.int64).sum().item())
    compact_tape_storage_bytes = int(total_tape_segments * 12 + (tape.sample_count + 1) * 4)
    current_density_active_storage_bytes = int(active_tape_segments * 12 + (tape.sample_count + 1) * 4)
    forward_errors = {
        "rgb_vs_current_mixed": _tensor_error(tape_replay["rgb"], current["rgb"]),
        "alpha_vs_current_mixed": _tensor_error(tape_replay["alpha"], current["alpha"]),
        "depth_vs_current_mixed": _tensor_error(tape_replay["depth"], current["depth"]),
        "rgb_vs_current_vjp_forward": _tensor_error(tape_replay["rgb"], current["vjp_rgb"]),
        "alpha_vs_current_vjp_forward": _tensor_error(tape_replay["alpha"], current["vjp_alpha"]),
        "depth_vs_current_vjp_forward": _tensor_error(tape_replay["depth"], current["vjp_depth"]),
    }
    grad_errors = {
        "grad_vs_current_reduce": _tensor_error(tape_replay["grad"], current["vjp_grad"]),
        "grad_vs_current_direct_atomic_grad_only": _tensor_error(tape_replay["grad"], current["grad_only"]),
        "current_direct_atomic_grad_only_vs_reduce": _tensor_error(current["grad_only"], current["vjp_grad"]),
    }
    metal_forward_errors = {
        "rgb_vs_current_mixed": _tensor_error(metal_tape["rgb"], current["rgb"]),
        "alpha_vs_current_mixed": _tensor_error(metal_tape["alpha"], current["alpha"]),
        "depth_vs_current_mixed": _tensor_error(metal_tape["depth"], current["depth"]),
        "rgb_vs_torch_tape": _tensor_error(metal_tape["rgb"], tape_replay["rgb"]),
        "alpha_vs_torch_tape": _tensor_error(metal_tape["alpha"], tape_replay["alpha"]),
        "depth_vs_torch_tape": _tensor_error(metal_tape["depth"], tape_replay["depth"]),
    }
    metal_grad_errors = {
        "grad_vs_current_reduce": _tensor_error(metal_tape["grad"], current["vjp_grad"]),
        "grad_vs_current_direct_atomic_grad_only": _tensor_error(metal_tape["grad"], current["grad_only"]),
        "grad_vs_torch_tape": _tensor_error(metal_tape["grad"], tape_replay["grad"]),
        "track_grad_vs_current_reduce": _tensor_error(metal_tape["track_grad"], current["vjp_grad"]),
        "track_grad_vs_current_direct_atomic_grad_only": _tensor_error(
            metal_tape["track_grad"],
            current["grad_only"],
        ),
        "track_grad_vs_metal_grad_only": _tensor_error(metal_tape["track_grad"], metal_tape["grad"]),
        "track_grad_vs_torch_tape": _tensor_error(metal_tape["track_grad"], tape_replay["grad"]),
    }
    return {
        "frames": frame_count,
        "render_size": render_size,
        "train_views": list(data["train_views"]),
        "track_count": int(tape.track_count),
        "pixel_rays": int(tape.sample_count),
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "time_slabs": time_slabs,
        "layout": layout,
        "candidate_order": candidate_order,
        "row_count": int(bundle["row_count"]),
        "candidate_count": int(bundle["candidate_count"]),
        "candidate_replay_iterations": int(bundle["candidate_replay_iterations"]),
        "candidate_depth_order": bundle["candidate_depth_order"],
        "max_candidates_per_row": int(bundle["max_candidates_per_row"]),
        "missing_sample_events": int(bundle["missing_sample_events"]),
        "segment_tape": {
            "format": "geometry_only_owner_i32_length_f32_mid_f32_padded_dense",
            "compact_csr_format": "sample_offsets_i32_owner_i32_length_f32_mid_f32",
            "reusable_across_site_rgba_density": True,
            "track_major_shape": [int(tape.track_count), int(tape.frame_count), int(tape.max_segments)],
            "max_segments_per_sample": int(tape.max_segments),
            "total_segments": total_tape_segments,
            "active_segments_for_current_density": active_tape_segments,
            "avg_segments_per_sample": float(total_tape_segments) / float(max(tape.sample_count, 1)),
            "avg_active_segments_per_sample": float(active_tape_segments) / float(max(tape.sample_count, 1)),
            "storage_bytes": int(tape.storage_bytes),
            "compact_csr_storage_bytes": int(compact.storage_bytes),
            "current_density_active_csr_storage_bytes": current_density_active_storage_bytes,
            "storage_vs_explicit_rays": float(tape.storage_bytes) / float(max(explicit_ray_storage_bytes, 1)),
            "compact_csr_storage_vs_explicit_rays": float(compact.storage_bytes)
            / float(max(explicit_ray_storage_bytes, 1)),
            "storage_vs_current_mixed_csr_plus_affine_ray": float(tape.storage_bytes)
            / float(max(mixed_candidate_storage_bytes + affine_ray_storage_bytes, 1)),
            "compact_csr_storage_vs_current_mixed_csr_plus_affine_ray": float(compact.storage_bytes)
            / float(max(mixed_candidate_storage_bytes + affine_ray_storage_bytes, 1)),
        },
        "storage_bytes": {
            "explicit_rays": int(explicit_ray_storage_bytes),
            "current_mixed_csr_candidates": int(mixed_candidate_storage_bytes),
            "current_affine_rays": int(affine_ray_storage_bytes),
            "current_mixed_total": int(mixed_candidate_storage_bytes + affine_ray_storage_bytes),
            "segment_tape": int(tape.storage_bytes),
            "segment_tape_compact_csr": int(compact.storage_bytes),
            "segment_tape_current_density_active_csr": current_density_active_storage_bytes,
        },
        "forward_errors": forward_errors,
        "grad_errors": grad_errors,
        "metal_forward_errors": metal_forward_errors,
        "metal_grad_errors": metal_grad_errors,
        "timing_ms": {
            **current["timing_ms"],
            **tape_replay["timing_ms"],
            **metal_tape["timing_ms"],
        },
        "vjp_seed_summary": current["vjp_seed_summary"],
        "outputs_are_finite": bool(
            torch.isfinite(tape_replay["rgb"]).all().item()
            and torch.isfinite(tape_replay["alpha"]).all().item()
            and torch.isfinite(tape_replay["depth"]).all().item()
            and torch.isfinite(tape_replay["grad"]).all().item()
            and torch.isfinite(metal_tape["rgb"]).all().item()
            and torch.isfinite(metal_tape["alpha"]).all().item()
            and torch.isfinite(metal_tape["depth"]).all().item()
            and torch.isfinite(metal_tape["grad"]).all().item()
            and torch.isfinite(metal_tape["track_grad"]).all().item()
        ),
    }


def run_probe(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
    site_count: int,
    time_slabs: int,
    layout: str,
    tile_h: int,
    tile_w: int,
    candidate_order: str,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    vjp_seed_mode: str,
    vjp_reduce_chunk_size: int,
    timing_iters: int,
    tape_device: torch.device,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required to compare the segment tape probe against the current fused shader path")
    rows = [
        _profile_frame_count(
            frame_count=frame_count,
            config_path=config_path,
            render_size=render_size,
            site_count=site_count,
            time_slabs=time_slabs,
            layout=layout,
            tile_h=tile_h,
            tile_w=tile_w,
            candidate_order=candidate_order,
            near=near,
            far=far,
            density=density,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            residual_depth_padding=residual_depth_padding,
            synthetic_motion=synthetic_motion,
            vjp_seed_mode=vjp_seed_mode,
            vjp_reduce_chunk_size=vjp_reduce_chunk_size,
            timing_iters=timing_iters,
            tape_device=tape_device,
        )
        for frame_count in frame_counts
    ]

    def _max_error(section: str, key: str, metric: str = "max_abs") -> float:
        return max(float(row[section][key][metric]) for row in rows)

    max_forward_error = max(
        _max_error("forward_errors", "rgb_vs_current_mixed"),
        _max_error("forward_errors", "alpha_vs_current_mixed"),
        _max_error("forward_errors", "depth_vs_current_mixed"),
    )
    max_grad_rel_error = _max_error("grad_errors", "grad_vs_current_reduce", "rel_to_rhs_abs_max")
    max_grad_only_rel_error = _max_error(
        "grad_errors",
        "grad_vs_current_direct_atomic_grad_only",
        "rel_to_rhs_abs_max",
    )
    max_metal_forward_error = max(
        _max_error("metal_forward_errors", "rgb_vs_current_mixed"),
        _max_error("metal_forward_errors", "alpha_vs_current_mixed"),
        _max_error("metal_forward_errors", "depth_vs_current_mixed"),
    )
    max_metal_grad_only_rel_error = _max_error(
        "metal_grad_errors",
        "grad_vs_current_direct_atomic_grad_only",
        "rel_to_rhs_abs_max",
    )
    max_metal_grad_rel_error = _max_error("metal_grad_errors", "grad_vs_current_reduce", "rel_to_rhs_abs_max")
    max_metal_track_grad_only_rel_error = _max_error(
        "metal_grad_errors",
        "track_grad_vs_current_direct_atomic_grad_only",
        "rel_to_rhs_abs_max",
    )
    max_metal_track_vs_grad_only_rel_error = _max_error(
        "metal_grad_errors",
        "track_grad_vs_metal_grad_only",
        "rel_to_rhs_abs_max",
    )
    max_current_grad_rel_error = _max_error(
        "grad_errors",
        "current_direct_atomic_grad_only_vs_reduce",
        "rel_to_rhs_abs_max",
    )
    segment_scale = float(rows[-1]["segment_tape"]["total_segments"]) / float(
        max(int(rows[0]["segment_tape"]["total_segments"]), 1)
    )
    active_segment_scale = float(rows[-1]["segment_tape"]["active_segments_for_current_density"]) / float(
        max(int(rows[0]["segment_tape"]["active_segments_for_current_density"]), 1)
    )
    frame_scale = float(rows[-1]["frames"]) / float(max(int(rows[0]["frames"]), 1))
    forward_scale = float(rows[-1]["timing_ms"]["torch_segment_tape_forward"]) / float(
        max(float(rows[0]["timing_ms"]["torch_segment_tape_forward"]), 1.0e-9)
    )
    acceptance = {
        "zero_missing_sample_events": all(int(row["missing_sample_events"]) == 0 for row in rows),
        "outputs_are_finite": all(bool(row["outputs_are_finite"]) for row in rows),
        "matches_current_mixed_forward": max_forward_error <= 5.0e-4,
        "tape_vjp_matches_current_reduce": max_grad_rel_error <= 2.0e-5,
        "tape_vjp_matches_current_winner_grad_only": max_grad_only_rel_error <= 2.0e-5,
        "metal_tape_matches_current_mixed_forward": max_metal_forward_error <= 5.0e-4,
        "metal_tape_vjp_matches_current_winner_grad_only": max_metal_grad_only_rel_error <= 2.0e-5,
        "metal_tape_track_vjp_matches_current_winner_grad_only": max_metal_track_grad_only_rel_error <= 2.0e-5,
        "metal_tape_track_vjp_matches_sample_atomic": max_metal_track_vs_grad_only_rel_error <= 2.0e-5,
        "current_direct_atomic_grad_only_still_matches_reduce": max_current_grad_rel_error <= 1.0e-5,
        "tape_storage_is_geometry_only": all(
            bool(row["segment_tape"]["reusable_across_site_rgba_density"]) for row in rows
        ),
    }
    return {
        "benchmark": "world_foam_lane2_fused_slab_segment_tape_probe",
        "status": "ok" if all(acceptance.values()) else "failed",
        "claim": "geometry_only_segment_tape_matches_current_fused_forward_and_site_rgba_vjp",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "site_count": site_count,
        "time_slabs": time_slabs,
        "layout": layout,
        "candidate_order": candidate_order,
        "tape_device": tape_device.type,
        "vjp_seed_mode": vjp_seed_mode,
        "timing_iters": timing_iters,
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "forward_abs_tolerance": 5.0e-4,
        "vjp_reduce_relative_tolerance": 2.0e-5,
        "vjp_winner_relative_tolerance": 2.0e-5,
        "max_forward_error_vs_current_mixed": max_forward_error,
        "max_grad_rel_error_vs_current_reduce": max_grad_rel_error,
        "max_grad_rel_error_vs_current_winner_grad_only": max_grad_only_rel_error,
        "max_metal_forward_error_vs_current_mixed": max_metal_forward_error,
        "max_metal_grad_rel_error_vs_current_reduce": max_metal_grad_rel_error,
        "max_metal_grad_rel_error_vs_current_winner_grad_only": max_metal_grad_only_rel_error,
        "max_metal_track_grad_rel_error_vs_current_winner_grad_only": max_metal_track_grad_only_rel_error,
        "max_metal_track_grad_rel_error_vs_sample_atomic": max_metal_track_vs_grad_only_rel_error,
        "frame_scale_first_to_last": frame_scale,
        "segment_scale_first_to_last": segment_scale,
        "active_segment_scale_first_to_last": active_segment_scale,
        "torch_tape_forward_scale_first_to_last": forward_scale,
        "structural_read": {
            "segment_tape_is_reusable_for_fixed_geometry": True,
            "segment_tape_removes_per_step_depth_sort_and_owner_lookup": True,
            "segment_tape_still_replays_segments_per_frame": True,
            "metal_kernel_implemented": True,
            "reason_not_completion": "The compact-tape shader is implemented, but the naive per-sample tape still scales roughly with frame count and has not been trained in the full loop.",
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a geometry-only segment tape for World Foam fused slab replay.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--layout", choices=("tiled", "per-track"), default="per-track")
    parser.add_argument("--candidate-order", choices=("boundary-id", "slab-mid-depth"), default="boundary-id")
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
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
    parser.add_argument("--vjp-seed-mode", choices=("rgb", "rgba-depth"), default="rgba-depth")
    parser.add_argument("--vjp-reduce-chunk-size", type=int, default=4)
    parser.add_argument("--timing-iters", type=int, default=5)
    parser.add_argument("--tape-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json",
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
        layout=args.layout,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        candidate_order=args.candidate_order,
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
        vjp_seed_mode=str(args.vjp_seed_mode),
        vjp_reduce_chunk_size=args.vjp_reduce_chunk_size,
        timing_iters=args.timing_iters,
        tape_device=torch.device(str(args.tape_device)),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
