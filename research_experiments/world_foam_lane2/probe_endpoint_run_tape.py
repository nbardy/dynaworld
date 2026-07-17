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
EPS = 1.0e-8

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
from probe_fused_slab_segment_tape import SegmentTape, build_segment_tape, compact_segment_tape  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list, _timed_mps_call  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    endpoint_run_rgba_depth_replay,
    endpoint_run_vjp_direct_atomic_grad_only,
)


@dataclass(frozen=True)
class CompactEndpointRunTape:
    offsets_i32: torch.Tensor
    owners_i32: torch.Tensor
    starts_f32: torch.Tensor
    ends_f32: torch.Tensor
    total_original_segments: int

    @property
    def storage_bytes(self) -> int:
        return int(
            self.offsets_i32.numel() * self.offsets_i32.element_size()
            + self.owners_i32.numel() * self.owners_i32.element_size()
            + self.starts_f32.numel() * self.starts_f32.element_size()
            + self.ends_f32.numel() * self.ends_f32.element_size()
        )


def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.detach().cpu()
    right = right.detach().cpu()
    diff = (left - right).abs()
    rhs_abs = right.abs().max().item() if right.numel() else 0.0
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "rel_to_rhs_abs_max": float(diff.max().item() / max(rhs_abs, EPS)) if diff.numel() else 0.0,
    }


def compress_same_owner_endpoint_runs(tape: SegmentTape) -> CompactEndpointRunTape:
    owners = tape.owners_i32.detach().cpu().reshape(tape.sample_count, tape.max_segments)
    lengths = tape.lengths_f32.detach().cpu().reshape(tape.sample_count, tape.max_segments)
    mids = tape.mids_f32.detach().cpu().reshape(tape.sample_count, tape.max_segments)
    counts = tape.counts_i32.detach().cpu().reshape(tape.sample_count)
    offsets = [0]
    out_owners: list[int] = []
    out_starts: list[float] = []
    out_ends: list[float] = []

    def flush(owner: int | None, start: float | None, end: float | None) -> None:
        if owner is None or start is None or end is None or end <= start:
            return
        out_owners.append(int(owner))
        out_starts.append(float(start))
        out_ends.append(float(end))

    for sample_id in range(tape.sample_count):
        current_owner: int | None = None
        current_start: float | None = None
        current_end: float | None = None
        for segment_id in range(int(counts[sample_id].item())):
            owner = int(owners[sample_id, segment_id].item())
            length = float(lengths[sample_id, segment_id].item())
            if owner < 0 or length <= EPS:
                continue
            mid = float(mids[sample_id, segment_id].item())
            start = mid - 0.5 * length
            end = mid + 0.5 * length
            if current_owner is not None and owner != current_owner:
                flush(current_owner, current_start, current_end)
                current_start = None
                current_end = None
            current_owner = owner
            current_start = start if current_start is None else min(current_start, start)
            current_end = end if current_end is None else max(current_end, end)
        flush(current_owner, current_start, current_end)
        offsets.append(len(out_owners))

    return CompactEndpointRunTape(
        offsets_i32=torch.tensor(offsets, dtype=torch.int32),
        owners_i32=torch.tensor(out_owners, dtype=torch.int32),
        starts_f32=torch.tensor(out_starts, dtype=torch.float32),
        ends_f32=torch.tensor(out_ends, dtype=torch.float32),
        total_original_segments=int(counts.to(dtype=torch.int64).sum().item()),
    )


def replay_endpoint_run_tape_torch(
    *,
    tape: CompactEndpointRunTape,
    site_rgba_f32: torch.Tensor,
    track_count: int,
    frame_count: int,
    far: float,
    transmittance_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = site_rgba_f32.device
    dtype = site_rgba_f32.dtype
    rgb_rows: list[torch.Tensor] = []
    alpha_rows: list[torch.Tensor] = []
    depth_rows: list[torch.Tensor] = []
    zero = torch.zeros((), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    far_value = torch.tensor(float(far), dtype=dtype, device=device)
    for sample_id in range(track_count * frame_count):
        rgb = torch.zeros((3,), dtype=dtype, device=device)
        alpha = zero
        depth_weighted = zero
        transmittance = one
        begin = int(tape.offsets_i32[sample_id].item())
        end = int(tape.offsets_i32[sample_id + 1].item())
        for run_id in range(begin, end):
            if float(transmittance.detach().cpu().item()) <= transmittance_threshold:
                break
            owner = int(tape.owners_i32[run_id].item())
            start = float(tape.starts_f32[run_id].item())
            stop = float(tape.ends_f32[run_id].item())
            length = stop - start
            if owner < 0 or owner >= site_rgba_f32.shape[0] or length <= EPS:
                continue
            rgba = site_rgba_f32[owner]
            density = rgba[3].clamp_min(0.0)
            length_t = torch.tensor(length, dtype=dtype, device=device)
            start_t = torch.tensor(start, dtype=dtype, device=device)
            segment_transmittance = torch.exp(-density * length_t)
            segment_alpha = 1.0 - segment_transmittance
            small_density_mass = density * (start_t * length_t + 0.5 * length_t * length_t)
            safe_density = density.clamp_min(1.0e-6)
            regular_density_mass = (
                start_t * segment_alpha + segment_alpha / safe_density - length_t * segment_transmittance
            )
            depth_mass = torch.where(density > 1.0e-6, regular_density_mass, small_density_mass)
            weight = transmittance * segment_alpha
            rgb = rgb + weight * rgba[:3]
            alpha = alpha + weight
            depth_weighted = depth_weighted + transmittance * depth_mass
            transmittance = transmittance * segment_transmittance
        depth = torch.where(alpha > EPS, depth_weighted / alpha.clamp_min(EPS), far_value)
        rgb_rows.append(rgb)
        alpha_rows.append(alpha)
        depth_rows.append(depth)
    return (
        torch.stack(rgb_rows).reshape(track_count, frame_count, 3),
        torch.stack(alpha_rows).reshape(track_count, frame_count),
        torch.stack(depth_rows).reshape(track_count, frame_count),
    )


def _mps_forward_and_vjp(
    *,
    tape: CompactEndpointRunTape,
    site_rgba: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    timing_iters: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    offsets = tape.offsets_i32.to(device=device).contiguous()
    owners = tape.owners_i32.to(device=device).contiguous()
    starts = tape.starts_f32.to(device=device).contiguous()
    ends = tape.ends_f32.to(device=device).contiguous()
    site_rgba_mps = site_rgba.to(device=device).contiguous()
    forward, forward_ms = _timed_mps_call(
        lambda: endpoint_run_rgba_depth_replay(
            offsets,
            owners,
            starts,
            ends,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        ),
        timing_iters=timing_iters,
    )
    grad_rgb = torch.linspace(
        -0.25,
        0.75,
        track_count * frame_count * 3,
        dtype=torch.float32,
        device=device,
    ).reshape(track_count, frame_count, 3)
    grad_alpha = torch.linspace(-0.5, 0.5, track_count * frame_count, dtype=torch.float32, device=device).reshape(
        track_count,
        frame_count,
    )
    grad_depth = torch.linspace(0.1, 0.6, track_count * frame_count, dtype=torch.float32, device=device).reshape(
        track_count,
        frame_count,
    )
    grad_out, grad_ms = _timed_mps_call(
        lambda: (
            endpoint_run_vjp_direct_atomic_grad_only(
                offsets,
                owners,
                starts,
                ends,
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
    torch_site = site_rgba.to(device=device).contiguous().detach().clone().requires_grad_(True)
    torch_forward = replay_endpoint_run_tape_torch(
        tape=tape,
        site_rgba_f32=torch_site,
        track_count=track_count,
        frame_count=frame_count,
        far=op_config.far,
        transmittance_threshold=op_config.transmittance_threshold,
    )
    torch_loss = (
        (torch_forward[0] * grad_rgb).sum()
        + (torch_forward[1] * grad_alpha).sum()
        + (torch_forward[2] * grad_depth).sum()
    )
    torch_loss.backward()
    torch.mps.synchronize()
    rgb, alpha, depth = (tensor.detach().cpu() for tensor in forward)
    torch_rgb, torch_alpha, torch_depth = (tensor.detach().cpu() for tensor in torch_forward)
    return {
        "rgb": rgb.reshape(track_count, frame_count, 3),
        "alpha": alpha.reshape(track_count, frame_count),
        "depth": depth.reshape(track_count, frame_count),
        "torch_rgb": torch_rgb,
        "torch_alpha": torch_alpha,
        "torch_depth": torch_depth,
        "site_rgba_grad": grad_out[0].detach().cpu(),
        "torch_site_rgba_grad": torch_site.grad.detach().cpu(),
        "timing_ms": {
            "endpoint_forward": float(forward_ms),
            "endpoint_vjp": float(grad_ms),
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
    full = compact_segment_tape(tape)
    endpoint = compress_same_owner_endpoint_runs(tape)
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    mps = _mps_forward_and_vjp(
        tape=endpoint,
        site_rgba=site_rgba,
        op_config=op_config,
        track_count=tape.track_count,
        frame_count=frame_count,
        timing_iters=timing_iters,
    )
    full_storage_bytes = int(full.storage_bytes)
    run_count = int(endpoint.owners_i32.numel())
    max_runs = int((endpoint.offsets_i32[1:] - endpoint.offsets_i32[:-1]).max().item())
    return {
        "frames": frame_count,
        "render_size": render_size,
        "track_count": int(tape.track_count),
        "sample_count": int(tape.sample_count),
        "site_count": len(sites),
        "full_segments": int(full.owners_i32.numel()),
        "endpoint_runs": run_count,
        "endpoint_runs_vs_full_segments": float(run_count) / float(max(int(full.owners_i32.numel()), 1)),
        "endpoint_storage_bytes": int(endpoint.storage_bytes),
        "full_storage_bytes": full_storage_bytes,
        "endpoint_storage_vs_full_segment_csr": float(endpoint.storage_bytes) / float(max(full_storage_bytes, 1)),
        "max_endpoint_runs_per_sample": max_runs,
        "forward_errors_vs_torch": {
            "rgb": _tensor_error(mps["rgb"], mps["torch_rgb"]),
            "alpha": _tensor_error(mps["alpha"], mps["torch_alpha"]),
            "depth": _tensor_error(mps["depth"], mps["torch_depth"]),
        },
        "vjp_errors_vs_torch_autograd": {
            "site_rgba_grad": _tensor_error(mps["site_rgba_grad"], mps["torch_site_rgba_grad"]),
        },
        "timing_ms": mps["timing_ms"],
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
    timing_iters: int,
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
        )
        for frame_count in frame_counts
    ]
    frame_scale = float(rows[-1]["frames"]) / float(max(rows[0]["frames"], 1))
    endpoint_run_scale = float(rows[-1]["endpoint_runs"]) / float(max(rows[0]["endpoint_runs"], 1))
    has_scale = len(rows) > 1
    max_forward_error = max(
        float(row["forward_errors_vs_torch"][key]["max_abs"])
        for row in rows
        for key in ("rgb", "alpha", "depth")
    )
    max_vjp_rel_error = max(
        float(row["vjp_errors_vs_torch_autograd"]["site_rgba_grad"]["rel_to_rhs_abs_max"]) for row in rows
    )
    acceptance = {
        "metal_forward_matches_torch_continuous_endpoint_replay": max_forward_error <= 5.0e-5,
        "metal_vjp_matches_torch_autograd": max_vjp_rel_error <= 5.0e-4,
        "endpoint_storage_below_full_at_max_frame": rows[-1]["endpoint_storage_vs_full_segment_csr"] < 0.20,
        "endpoint_runs_under_vjp_cap": rows[-1]["max_endpoint_runs_per_sample"] <= 129,
        "endpoint_run_count_sublinear_vs_frames": (not has_scale) or endpoint_run_scale < frame_scale,
    }
    return {
        "benchmark": "world_foam_lane2_endpoint_run_tape_probe",
        "status": "ok" if all(acceptance.values()) else "informational",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "site_count": site_count,
        "synthetic_motion": synthetic_motion.to_dict(),
        "timing_iters": timing_iters,
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "endpoint_run_scale_first_to_last": endpoint_run_scale,
        "max_forward_abs_error_vs_torch": max_forward_error,
        "max_vjp_rel_error_vs_torch_autograd": max_vjp_rel_error,
        "structural_read": {
            "same_owner_endpoint_runs_are_density_independent_for_continuous_absorption_depth": True,
            "semantic_change_from_segment_mid_depth": True,
            "not_main_trainer_integrated": True,
            "interpretation": (
                "Endpoint runs provide the compact density-independent replay path if World Foam accepts "
                "continuous absorption depth inside a same-owner run. This is not the old segment-mid depth "
                "contract, and the run count still needs a matched STAR-style structural comparison."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe compact endpoint-run continuous-depth World Foam tape.")
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
    parser.add_argument("--timing-iters", type=int, default=3)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_endpoint_run_tape_probe_render32_2_4_8_16.json",
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
        timing_iters=args.timing_iters,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
