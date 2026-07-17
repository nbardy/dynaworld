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
from probe_fused_slab_segment_tape import build_segment_tape, compact_segment_tape  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list, _timed_mps_call  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    segment_tape_rgba_depth_replay,
    segment_tape_vjp_direct_atomic_grad_only,
)


@dataclass(frozen=True)
class CompactRunTape:
    offsets_i32: torch.Tensor
    owners_i32: torch.Tensor
    lengths_f32: torch.Tensor
    mids_f32: torch.Tensor
    active_original_segments: int
    total_original_segments: int

    @property
    def storage_bytes(self) -> int:
        return int(
            self.offsets_i32.numel() * self.offsets_i32.element_size()
            + self.owners_i32.numel() * self.owners_i32.element_size()
            + self.lengths_f32.numel() * self.lengths_f32.element_size()
            + self.mids_f32.numel() * self.mids_f32.element_size()
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


def _effective_mid_for_run(
    *,
    density: float,
    lengths: list[float],
    mids: list[float],
) -> float:
    total_length = float(sum(lengths))
    if total_length <= EPS:
        return float(mids[0]) if mids else 0.0
    if density <= 0.0:
        return float(sum(length * mid for length, mid in zip(lengths, mids, strict=True)) / total_length)
    transmittance = 1.0
    alpha = 0.0
    depth_num = 0.0
    for length, mid in zip(lengths, mids, strict=True):
        segment_transmittance = math.exp(-density * length)
        weight = transmittance * (1.0 - segment_transmittance)
        alpha += weight
        depth_num += weight * mid
        transmittance *= segment_transmittance
    if alpha <= EPS:
        return float(sum(length * mid for length, mid in zip(lengths, mids, strict=True)) / total_length)
    return float(depth_num / alpha)


def compress_same_owner_runs(
    *,
    tape: Any,
    site_rgba_f32: torch.Tensor,
    transmittance_threshold: float,
) -> CompactRunTape:
    site_rgba_cpu = site_rgba_f32.detach().cpu().to(dtype=torch.float32)
    owners = tape.owners_i32.detach().cpu().reshape(tape.sample_count, tape.max_segments)
    lengths = tape.lengths_f32.detach().cpu().reshape(tape.sample_count, tape.max_segments)
    mids = tape.mids_f32.detach().cpu().reshape(tape.sample_count, tape.max_segments)
    counts = tape.counts_i32.detach().cpu().reshape(tape.sample_count)
    offsets = [0]
    out_owners: list[int] = []
    out_lengths: list[float] = []
    out_mids: list[float] = []
    active_original_segments = 0
    total_original_segments = int(counts.to(dtype=torch.int64).sum().item())

    def flush(owner: int | None, run_lengths: list[float], run_mids: list[float]) -> None:
        if owner is None or not run_lengths:
            return
        density = max(float(site_rgba_cpu[owner, 3].item()), 0.0)
        out_owners.append(int(owner))
        out_lengths.append(float(sum(run_lengths)))
        out_mids.append(_effective_mid_for_run(density=density, lengths=run_lengths, mids=run_mids))

    for sample_id in range(tape.sample_count):
        transmittance = 1.0
        current_owner: int | None = None
        run_lengths: list[float] = []
        run_mids: list[float] = []
        for segment_id in range(int(counts[sample_id].item())):
            if transmittance <= transmittance_threshold:
                break
            owner = int(owners[sample_id, segment_id].item())
            if owner < 0 or owner >= site_rgba_cpu.shape[0]:
                continue
            length = float(lengths[sample_id, segment_id].item())
            if length <= EPS:
                continue
            if current_owner is not None and owner != current_owner:
                flush(current_owner, run_lengths, run_mids)
                run_lengths = []
                run_mids = []
            current_owner = owner
            run_lengths.append(length)
            run_mids.append(float(mids[sample_id, segment_id].item()))
            density = max(float(site_rgba_cpu[owner, 3].item()), 0.0)
            transmittance *= math.exp(-density * length)
            active_original_segments += 1
        flush(current_owner, run_lengths, run_mids)
        offsets.append(len(out_owners))

    return CompactRunTape(
        offsets_i32=torch.tensor(offsets, dtype=torch.int32),
        owners_i32=torch.tensor(out_owners, dtype=torch.int32),
        lengths_f32=torch.tensor(out_lengths, dtype=torch.float32),
        mids_f32=torch.tensor(out_mids, dtype=torch.float32),
        active_original_segments=active_original_segments,
        total_original_segments=total_original_segments,
    )


def _mps_forward_and_rgb_vjp(
    *,
    tape: Any,
    site_rgba: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    timing_iters: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    offsets = tape.offsets_i32.to(device=device)
    owners = tape.owners_i32.to(device=device)
    lengths = tape.lengths_f32.to(device=device)
    mids = tape.mids_f32.to(device=device)
    site_rgba_mps = site_rgba.to(device=device)
    forward, forward_ms = _timed_mps_call(
        lambda: segment_tape_rgba_depth_replay(
            offsets,
            owners,
            lengths,
            mids,
            site_rgba_mps,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        ),
        timing_iters=timing_iters,
    )
    grad_rgb = torch.ones((track_count, frame_count, 3), dtype=torch.float32, device=device)
    grad_alpha = torch.zeros((track_count, frame_count), dtype=torch.float32, device=device)
    grad_depth = torch.zeros((track_count, frame_count), dtype=torch.float32, device=device)
    grad_out, grad_ms = _timed_mps_call(
        lambda: (
            segment_tape_vjp_direct_atomic_grad_only(
                offsets,
                owners,
                lengths,
                mids,
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
    rgb, alpha, depth = (tensor.detach().cpu() for tensor in forward)
    return {
        "rgb": rgb.reshape(track_count, frame_count, 3),
        "alpha": alpha.reshape(track_count, frame_count),
        "depth": depth.reshape(track_count, frame_count),
        "rgb_only_grad": grad_out[0].detach().cpu(),
        "timing_ms": {
            "forward": float(forward_ms),
            "rgb_only_grad": float(grad_ms),
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
    full = compact_segment_tape(tape)
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    owner_run = compress_same_owner_runs(
        tape=tape,
        site_rgba_f32=site_rgba,
        transmittance_threshold=transmittance_threshold,
    )
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    full_mps = _mps_forward_and_rgb_vjp(
        tape=full,
        site_rgba=site_rgba,
        op_config=op_config,
        track_count=tape.track_count,
        frame_count=frame_count,
        timing_iters=timing_iters,
    )
    owner_mps = _mps_forward_and_rgb_vjp(
        tape=owner_run,
        site_rgba=site_rgba,
        op_config=op_config,
        track_count=tape.track_count,
        frame_count=frame_count,
        timing_iters=timing_iters,
    )
    full_segments = int(full.owners_i32.numel())
    run_segments = int(owner_run.owners_i32.numel())
    return {
        "frames": frame_count,
        "render_size": render_size,
        "track_count": int(tape.track_count),
        "sample_count": int(tape.sample_count),
        "site_count": len(sites),
        "full_segments": full_segments,
        "owner_run_segments": run_segments,
        "active_original_segments_for_current_density": int(owner_run.active_original_segments),
        "owner_run_segments_vs_full_segments": float(run_segments) / float(max(full_segments, 1)),
        "owner_run_storage_bytes": owner_run.storage_bytes,
        "full_storage_bytes": full.storage_bytes,
        "owner_run_storage_vs_full": float(owner_run.storage_bytes) / float(max(full.storage_bytes, 1)),
        "max_owner_run_segments_per_sample": int((owner_run.offsets_i32[1:] - owner_run.offsets_i32[:-1]).max().item()),
        "forward_errors": {
            "rgb": _tensor_error(owner_mps["rgb"], full_mps["rgb"]),
            "alpha": _tensor_error(owner_mps["alpha"], full_mps["alpha"]),
            "depth": _tensor_error(owner_mps["depth"], full_mps["depth"]),
        },
        "rgb_only_vjp_errors": {
            "site_rgba_grad": _tensor_error(owner_mps["rgb_only_grad"], full_mps["rgb_only_grad"]),
        },
        "timing_ms": {
            "full_forward": full_mps["timing_ms"]["forward"],
            "owner_run_forward": owner_mps["timing_ms"]["forward"],
            "full_rgb_only_grad": full_mps["timing_ms"]["rgb_only_grad"],
            "owner_run_rgb_only_grad": owner_mps["timing_ms"]["rgb_only_grad"],
        },
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
    full_segment_scale = float(rows[-1]["full_segments"]) / float(max(rows[0]["full_segments"], 1))
    owner_run_segment_scale = float(rows[-1]["owner_run_segments"]) / float(max(rows[0]["owner_run_segments"], 1))
    max_rgb_error = max(float(row["forward_errors"]["rgb"]["max_abs"]) for row in rows)
    max_alpha_error = max(float(row["forward_errors"]["alpha"]["max_abs"]) for row in rows)
    max_depth_error = max(float(row["forward_errors"]["depth"]["max_abs"]) for row in rows)
    max_rgb_vjp_rel_error = max(
        float(row["rgb_only_vjp_errors"]["site_rgba_grad"]["rel_to_rhs_abs_max"]) for row in rows
    )
    acceptance = {
        "owner_run_forward_rgb_matches_full": max_rgb_error <= 5.0e-6,
        "owner_run_forward_alpha_matches_full": max_alpha_error <= 5.0e-6,
        "owner_run_forward_depth_matches_current_density_full": max_depth_error <= 5.0e-5,
        "owner_run_rgb_only_vjp_matches_full": max_rgb_vjp_rel_error <= 2.0e-5,
        "owner_run_segments_below_full": rows[-1]["owner_run_segments_vs_full_segments"] < 1.0,
        "owner_run_segment_scale_sublinear_vs_frames": owner_run_segment_scale < frame_scale,
    }
    return {
        "benchmark": "world_foam_lane2_segment_owner_run_tape_probe",
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
        "full_segment_scale_first_to_last": full_segment_scale,
        "owner_run_segment_scale_first_to_last": owner_run_segment_scale,
        "max_forward_rgb_abs_error": max_rgb_error,
        "max_forward_alpha_abs_error": max_alpha_error,
        "max_forward_depth_abs_error": max_depth_error,
        "max_rgb_only_vjp_rel_error": max_rgb_vjp_rel_error,
        "structural_read": {
            "same_owner_runs_preserve_rgb_alpha": True,
            "depth_mid_is_current_density_effective_mid": True,
            "not_yet_geometry_only_for_depth_or_threshold": True,
            "interpretation": (
                "Same-owner run compression can reuse the existing segment-tape Metal kernels and preserves "
                "RGB/alpha plus RGB-only VJP. Depth uses a current-density effective mid, so this is a strong "
                "RGB-training candidate but not a final density-independent geometry tape."
            ),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe same-owner run compression for compact segment tapes.")
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
    parser.add_argument("--timing-iters", type=int, default=3)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_segment_owner_run_tape_probe_render32_2_4_8_16.json",
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
