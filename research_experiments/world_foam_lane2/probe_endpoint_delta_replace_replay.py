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
from probe_endpoint_run_tape import CompactEndpointRunTape, compress_same_owner_endpoint_runs  # noqa: E402
from probe_fused_slab_segment_tape import build_segment_tape, compact_segment_tape  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list, _timed_mps_call  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    endpoint_delta_replace_rgba_depth_replay,
    endpoint_delta_replace_vjp_direct_atomic_grad_only,
    endpoint_run_rgba_depth_replay,
    endpoint_run_vjp_direct_atomic_grad_only,
)


@dataclass(frozen=True)
class EndpointDeltaReplaceTape:
    base_offsets_i32: torch.Tensor
    base_owner_i32: torch.Tensor
    base_start_f32: torch.Tensor
    base_end_f32: torch.Tensor
    track_change_offsets_i32: torch.Tensor
    change_frame_i32: torch.Tensor
    change_offsets_i32: torch.Tensor
    change_owner_i32: torch.Tensor
    change_start_f32: torch.Tensor
    change_end_f32: torch.Tensor
    endpoint_run_count: int

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.base_offsets_i32,
            self.base_owner_i32,
            self.base_start_f32,
            self.base_end_f32,
            self.track_change_offsets_i32,
            self.change_frame_i32,
            self.change_offsets_i32,
            self.change_owner_i32,
            self.change_start_f32,
            self.change_end_f32,
        )
        return int(sum(t.numel() * t.element_size() for t in tensors))


def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.detach().cpu()
    right = right.detach().cpu()
    diff = (left - right).abs()
    rhs_abs = right.abs().max().item() if right.numel() else 0.0
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "rel_to_rhs_abs_max": float(diff.max().item() / max(rhs_abs, EPS)) if diff.numel() else 0.0,
    }


def _endpoint_row(
    endpoint: CompactEndpointRunTape,
    *,
    sample_id: int,
) -> tuple[tuple[int, float, float], ...]:
    begin = int(endpoint.offsets_i32[sample_id].item())
    end = int(endpoint.offsets_i32[sample_id + 1].item())
    return tuple(
        (
            int(endpoint.owners_i32[idx].item()),
            float(endpoint.starts_f32[idx].item()),
            float(endpoint.ends_f32[idx].item()),
        )
        for idx in range(begin, end)
    )


def pack_endpoint_delta_replace_tape(
    endpoint: CompactEndpointRunTape,
    *,
    track_count: int,
    frame_count: int,
) -> EndpointDeltaReplaceTape:
    base_offsets = [0]
    base_owner: list[int] = []
    base_start: list[float] = []
    base_end: list[float] = []
    track_change_offsets = [0]
    change_frame: list[int] = []
    change_offsets = [0]
    change_owner: list[int] = []
    change_start: list[float] = []
    change_end: list[float] = []

    for track_id in range(track_count):
        previous = _endpoint_row(endpoint, sample_id=track_id * frame_count)
        for owner, start, end in previous:
            base_owner.append(owner)
            base_start.append(start)
            base_end.append(end)
        base_offsets.append(len(base_owner))

        for frame_id in range(1, frame_count):
            sample_id = track_id * frame_count + frame_id
            current = _endpoint_row(endpoint, sample_id=sample_id)
            if current == previous:
                continue
            change_frame.append(frame_id)
            for owner, start, end in current:
                change_owner.append(owner)
                change_start.append(start)
                change_end.append(end)
            change_offsets.append(len(change_owner))
            previous = current
        track_change_offsets.append(len(change_frame))

    return EndpointDeltaReplaceTape(
        base_offsets_i32=torch.tensor(base_offsets, dtype=torch.int32),
        base_owner_i32=torch.tensor(base_owner, dtype=torch.int32),
        base_start_f32=torch.tensor(base_start, dtype=torch.float32),
        base_end_f32=torch.tensor(base_end, dtype=torch.float32),
        track_change_offsets_i32=torch.tensor(track_change_offsets, dtype=torch.int32),
        change_frame_i32=torch.tensor(change_frame, dtype=torch.int32),
        change_offsets_i32=torch.tensor(change_offsets, dtype=torch.int32),
        change_owner_i32=torch.tensor(change_owner, dtype=torch.int32),
        change_start_f32=torch.tensor(change_start, dtype=torch.float32),
        change_end_f32=torch.tensor(change_end, dtype=torch.float32),
        endpoint_run_count=int(endpoint.owners_i32.numel()),
    )


def _mps_compare(
    *,
    endpoint: CompactEndpointRunTape,
    delta: EndpointDeltaReplaceTape,
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
    site_rgba_mps = site_rgba.to(device=device).contiguous()
    base_offsets = delta.base_offsets_i32.to(device=device).contiguous()
    base_owner = delta.base_owner_i32.to(device=device).contiguous()
    base_start = delta.base_start_f32.to(device=device).contiguous()
    base_end = delta.base_end_f32.to(device=device).contiguous()
    track_change_offsets = delta.track_change_offsets_i32.to(device=device).contiguous()
    change_frame = delta.change_frame_i32.to(device=device).contiguous()
    change_offsets = delta.change_offsets_i32.to(device=device).contiguous()
    change_owner = delta.change_owner_i32.to(device=device).contiguous()
    change_start = delta.change_start_f32.to(device=device).contiguous()
    change_end = delta.change_end_f32.to(device=device).contiguous()

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
    delta_forward, delta_forward_ms = _timed_mps_call(
        lambda: endpoint_delta_replace_rgba_depth_replay(
            base_offsets,
            base_owner,
            base_start,
            base_end,
            track_change_offsets,
            change_frame,
            change_offsets,
            change_owner,
            change_start,
            change_end,
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
    delta_grad, delta_vjp_ms = _timed_mps_call(
        lambda: (
            endpoint_delta_replace_vjp_direct_atomic_grad_only(
                base_offsets,
                base_owner,
                base_start,
                base_end,
                track_change_offsets,
                change_frame,
                change_offsets,
                change_owner,
                change_start,
                change_end,
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
    torch.mps.synchronize()
    return {
        "forward_errors_vs_endpoint_run": {
            "rgb": _tensor_error(delta_forward[0], endpoint_forward[0]),
            "alpha": _tensor_error(delta_forward[1], endpoint_forward[1]),
            "depth": _tensor_error(delta_forward[2], endpoint_forward[2]),
        },
        "vjp_error_vs_endpoint_run": _tensor_error(delta_grad[0], endpoint_grad[0]),
        "timing_ms": {
            "endpoint_forward": float(endpoint_forward_ms),
            "delta_replace_forward": float(delta_forward_ms),
            "endpoint_vjp": float(endpoint_vjp_ms),
            "delta_replace_vjp": float(delta_vjp_ms),
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
    delta = pack_endpoint_delta_replace_tape(endpoint, track_count=tape.track_count, frame_count=frame_count)
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    mps = _mps_compare(
        endpoint=endpoint,
        delta=delta,
        site_rgba=site_rgba,
        op_config=op_config,
        track_count=tape.track_count,
        frame_count=frame_count,
        timing_iters=timing_iters,
    )
    return {
        "frames": int(frame_count),
        "render_size": int(render_size),
        "track_count": int(tape.track_count),
        "sample_count": int(tape.sample_count),
        "site_count": int(len(sites)),
        "full_segments": int(full.owners_i32.numel()),
        "endpoint_runs": int(endpoint.owners_i32.numel()),
        "change_events": int(delta.change_frame_i32.numel()),
        "changed_records": int(delta.change_owner_i32.numel()),
        "endpoint_storage_bytes": int(endpoint.storage_bytes),
        "delta_replace_storage_bytes": int(delta.storage_bytes),
        "delta_replace_storage_vs_endpoint_csr": float(delta.storage_bytes) / float(max(endpoint.storage_bytes, 1)),
        "delta_replace_storage_vs_full_segment_csr": float(delta.storage_bytes) / float(max(full.storage_bytes, 1)),
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
    storage_scale = _scale(rows, "delta_replace_storage_bytes")
    endpoint_scale = _scale(rows, "endpoint_runs")
    max_forward_error = max(
        float(row["mps"]["forward_errors_vs_endpoint_run"][name]["max_abs"])
        for row in rows
        for name in ("rgb", "alpha", "depth")
    )
    max_vjp_rel_error = max(float(row["mps"]["vjp_error_vs_endpoint_run"]["rel_to_rhs_abs_max"]) for row in rows)
    acceptance = {
        "metal_forward_matches_endpoint_run": max_forward_error < 1.0e-5,
        "metal_vjp_matches_endpoint_run": max_vjp_rel_error < 1.0e-4,
        "depth_float_delta_storage_sublinear_vs_frames": storage_scale < frame_scale,
    }
    return {
        "benchmark": "world_foam_lane2_endpoint_delta_replace_replay",
        "status": "ok" if all(acceptance.values()) else "negative",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": int(render_size),
        "site_count": int(site_count),
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "endpoint_run_scale_first_to_last": endpoint_scale,
        "delta_replace_storage_scale_first_to_last": storage_scale,
        "max_forward_abs_error_vs_endpoint_run": max_forward_error,
        "max_vjp_rel_error_vs_endpoint_run": max_vjp_rel_error,
        "structural_read": {
            "shader_replays_delta_replacement_rows": True,
            "stores_depth_floats_not_cut_ids": True,
            "not_final_star_like_edit_stream": True,
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe endpoint delta replacement-row Metal replay.")
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
    parser.add_argument("--timing-iters", type=int, default=5)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_endpoint_delta_replace_replay_render32_2_4_8_16.json",
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
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
