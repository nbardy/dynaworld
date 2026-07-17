#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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
)
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    segment_tape_rgba_depth_autograd,
    segment_tape_vjp_direct_atomic_grad_only,
)
from train_eval_owner_run_tape import (  # noqa: E402
    _image_rgb_from_track_major,
    _prepare_owner_run_tapes,
    _track_major_grad_from_image,
)


EPS = 1.0e-8


def _error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    diff = (left_cpu - right_cpu).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    rhs_abs = float(right_cpu.abs().max().item()) if right_cpu.numel() else 0.0
    return {
        "max_abs": max_abs,
        "rel_to_manual_abs_max": max_abs / max(rhs_abs, EPS),
    }


def _run_mode(
    *,
    mode: str,
    tape_device: dict[str, torch.Tensor],
    site_rgba_initial: torch.Tensor,
    target: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    view_count: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    site_rgba = site_rgba_initial.detach().clone().requires_grad_(True)
    rgb, _alpha, _depth = segment_tape_rgba_depth_autograd(
        tape_device["offsets_i32"],
        tape_device["owners_i32"],
        tape_device["lengths_f32"],
        tape_device["mids_f32"],
        site_rgba,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
        vjp_mode=mode,
    )
    rendered = _image_rgb_from_track_major(
        rgb.reshape(track_count, frame_count, 3),
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    loss = F.mse_loss(rendered, target)
    loss.backward()
    torch.mps.synchronize()
    autograd_grad = site_rgba.grad
    if autograd_grad is None:
        raise RuntimeError("segment tape autograd did not produce site_rgba.grad")

    grad_rgb_image = (2.0 / float(rendered.numel())) * (rendered.detach() - target)
    grad_rgb = _track_major_grad_from_image(
        grad_rgb_image.contiguous(),
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    grad_alpha = torch.zeros((track_count, frame_count), dtype=torch.float32, device=site_rgba.device)
    grad_depth = torch.zeros_like(grad_alpha)
    manual_grad = segment_tape_vjp_direct_atomic_grad_only(
        tape_device["offsets_i32"],
        tape_device["owners_i32"],
        tape_device["lengths_f32"],
        tape_device["mids_f32"],
        site_rgba.detach(),
        grad_rgb,
        grad_alpha,
        grad_depth,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
    )
    torch.mps.synchronize()
    err = _error(autograd_grad, manual_grad)
    return {
        "mode": mode,
        "loss": float(loss.detach().cpu().item()),
        "first_grad_abs_sum": float(autograd_grad.detach().abs().sum().cpu().item()),
        "grad_error_vs_manual_direct_atomic_grad_only": err,
        "status": "ok" if math.isfinite(err["rel_to_manual_abs_max"]) and err["rel_to_manual_abs_max"] <= 2.0e-5 else "failed",
    }


def run_smoke(
    *,
    config_path: Path,
    frame_count: int,
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    synthetic_motion: SyntheticRayMotion,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
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
    site_rgba_cpu = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    tape = _prepare_owner_run_tapes(
        sites=sites,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        site_rgba=site_rgba_cpu,
    )
    sample_count, height, width, _payload = rays.shape
    view_count = int(sample_count // frame_count)
    track_count = int(tape["track_count"])
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    site_rgba_initial = site_rgba_cpu.to(device=torch.device("mps")).contiguous()
    target_device = targets.to(device=torch.device("mps"))
    rows = [
        _run_mode(
            mode=mode,
            tape_device=tape["owner_run_device"],
            site_rgba_initial=site_rgba_initial,
            target=target_device,
            op_config=op_config,
            track_count=track_count,
            frame_count=frame_count,
            view_count=view_count,
            height=int(height),
            width=int(width),
        )
        for mode in ("direct_atomic_grad_only", "direct_atomic_track")
    ]
    acceptance = {
        "all_modes_ok": all(row["status"] == "ok" for row in rows),
        "owner_run_segments_below_full": int(tape["owner_run_segments"]) < int(tape["full_segments"]),
        "owner_run_vjp_under_segment_cap": int(tape["max_owner_run_segments_per_sample"]) <= 129,
    }
    return {
        "benchmark": "world_foam_lane2_segment_tape_autograd_smoke_mps",
        "status": "ok" if all(acceptance.values()) else "failed",
        "config_path": str(config_path),
        "frame_count": frame_count,
        "render_size": render_size,
        "site_count": site_count,
        "synthetic_motion": synthetic_motion.to_dict(),
        "gradient_scope": "frozen_geometry_segment_tape_site_rgba_autograd",
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "density_independent_depth_claim": False,
        "acceptance": acceptance,
        "full_segments": int(tape["full_segments"]),
        "owner_run_segments": int(tape["owner_run_segments"]),
        "owner_run_segments_vs_full": float(tape["owner_run_segments"]) / float(tape["full_segments"]),
        "max_owner_run_segments_per_sample": int(tape["max_owner_run_segments_per_sample"]),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke segment-tape autograd wrapper against explicit VJP.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2")
    parser.add_argument("--render-size", type=int, default=16)
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
        default=RESULTS_DIR / "2026-05-15_segment_tape_autograd_smoke_render16_2f.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_counts = _parse_int_list(args.frame_counts)
    if len(frame_counts) != 1:
        raise ValueError("segment tape autograd smoke expects exactly one frame count")
    payload = run_smoke(
        config_path=args.config,
        frame_count=frame_counts[0],
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
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
