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
from probe_fused_slab_segment_tape import SegmentTape, build_segment_tape, replay_segment_tape  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402


@dataclass(frozen=True)
class OwnerRunInternalStats:
    run_count: int
    internal_segment_count: int
    multi_segment_runs: int
    max_internal_segments_per_run: int
    nested_csr_bytes: int

    def as_dict(self, *, sample_count: int, full_segment_csr_bytes: int) -> dict[str, Any]:
        endpoint_run_csr_bytes = int((sample_count + 1) * 4 + self.run_count * 12)
        return {
            "run_count": int(self.run_count),
            "internal_segment_count": int(self.internal_segment_count),
            "multi_segment_runs": int(self.multi_segment_runs),
            "multi_segment_run_ratio": float(self.multi_segment_runs) / float(max(self.run_count, 1)),
            "max_internal_segments_per_run": int(self.max_internal_segments_per_run),
            "avg_runs_per_sample": float(self.run_count) / float(max(sample_count, 1)),
            "endpoint_run_csr_bytes": endpoint_run_csr_bytes,
            "endpoint_run_csr_vs_full_segment_csr": float(endpoint_run_csr_bytes)
            / float(max(full_segment_csr_bytes, 1)),
            "nested_csr_bytes": int(self.nested_csr_bytes),
            "nested_csr_vs_full_segment_csr": float(self.nested_csr_bytes) / float(max(full_segment_csr_bytes, 1)),
        }


def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    diff = (left.detach().cpu() - right.detach().cpu()).abs()
    rhs_abs = right.detach().cpu().abs().max().item() if right.numel() else 0.0
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "rel_to_rhs_abs_max": float(diff.max().item() / max(rhs_abs, EPS)) if diff.numel() else 0.0,
    }


def _scaled_density_rgba(site_rgba: torch.Tensor, density_scale: float) -> torch.Tensor:
    scaled = site_rgba.detach().clone().to(dtype=torch.float32)
    scaled[:, 3] = scaled[:, 3] * float(density_scale)
    return scaled


def _with_counts(tape: SegmentTape, counts_i32: torch.Tensor) -> SegmentTape:
    return SegmentTape(
        owners_i32=tape.owners_i32,
        lengths_f32=tape.lengths_f32,
        mids_f32=tape.mids_f32,
        counts_i32=counts_i32.to(dtype=torch.int32).reshape(tape.track_count, tape.frame_count).contiguous(),
        active_counts_i32=counts_i32.to(dtype=torch.int32).reshape(tape.track_count, tape.frame_count).contiguous(),
        frame_t_f32=tape.frame_t_f32,
        track_count=tape.track_count,
        frame_count=tape.frame_count,
        max_segments=tape.max_segments,
    )


def _owner_run_internal_stats(tape: SegmentTape, counts_i32: torch.Tensor) -> OwnerRunInternalStats:
    owners = tape.owners_i32.detach().cpu().reshape(tape.sample_count, tape.max_segments)
    counts = counts_i32.detach().cpu().reshape(tape.sample_count)
    run_count = 0
    internal_segment_count = 0
    multi_segment_runs = 0
    max_internal_segments_per_run = 0
    for sample_id in range(tape.sample_count):
        current_owner: int | None = None
        current_run_segments = 0
        for segment_id in range(int(counts[sample_id].item())):
            owner = int(owners[sample_id, segment_id].item())
            if owner < 0:
                continue
            if current_owner is not None and owner != current_owner:
                run_count += 1
                if current_run_segments > 1:
                    multi_segment_runs += 1
                max_internal_segments_per_run = max(max_internal_segments_per_run, current_run_segments)
                current_run_segments = 0
            current_owner = owner
            current_run_segments += 1
            internal_segment_count += 1
        if current_owner is not None and current_run_segments > 0:
            run_count += 1
            if current_run_segments > 1:
                multi_segment_runs += 1
            max_internal_segments_per_run = max(max_internal_segments_per_run, current_run_segments)

    # Nested CSR layout:
    # - sample_to_run_offsets: int32 [sample_count + 1]
    # - run_owner_i32: int32 [run_count]
    # - run_to_internal_offsets: int32 [run_count + 1]
    # - internal cut payload: two int32 endpoint ids or two f32 length/mid scalars per internal segment.
    nested_csr_bytes = int((tape.sample_count + 1) * 4 + run_count * 4 + (run_count + 1) * 4 + internal_segment_count * 8)
    return OwnerRunInternalStats(
        run_count=run_count,
        internal_segment_count=internal_segment_count,
        multi_segment_runs=multi_segment_runs,
        max_internal_segments_per_run=max_internal_segments_per_run,
        nested_csr_bytes=nested_csr_bytes,
    )


def _full_segment_csr_bytes(tape: SegmentTape, counts_i32: torch.Tensor) -> int:
    segment_count = int(counts_i32.detach().cpu().to(dtype=torch.int64).sum().item())
    return int((tape.sample_count + 1) * 4 + segment_count * 12)


def _replay_errors(
    *,
    candidate_tape: SegmentTape,
    full_tape: SegmentTape,
    site_rgba: torch.Tensor,
    far: float,
    transmittance_threshold: float,
) -> dict[str, Any]:
    full_rgb, full_alpha, full_depth = replay_segment_tape(
        tape=full_tape,
        site_rgba_f32=site_rgba,
        far=far,
        transmittance_threshold=transmittance_threshold,
        device=torch.device("cpu"),
    )
    candidate_rgb, candidate_alpha, candidate_depth = replay_segment_tape(
        tape=candidate_tape,
        site_rgba_f32=site_rgba,
        far=far,
        transmittance_threshold=transmittance_threshold,
        device=torch.device("cpu"),
    )
    return {
        "rgb": _tensor_error(candidate_rgb, full_rgb),
        "alpha": _tensor_error(candidate_alpha, full_alpha),
        "depth": _tensor_error(candidate_depth, full_depth),
    }


def _density_scale_errors(
    *,
    tape: SegmentTape,
    active_tape: SegmentTape,
    site_rgba: torch.Tensor,
    density_scales: tuple[float, ...],
    far: float,
    transmittance_threshold: float,
) -> dict[str, Any]:
    by_scale: dict[str, Any] = {}
    for density_scale in density_scales:
        scaled_rgba = _scaled_density_rgba(site_rgba, density_scale)
        by_scale[f"{density_scale:g}"] = {
            "active_internal_vs_full": _replay_errors(
                candidate_tape=active_tape,
                full_tape=tape,
                site_rgba=scaled_rgba,
                far=far,
                transmittance_threshold=transmittance_threshold,
            ),
            "all_internal_vs_full": {
                "rgb": {"max_abs": 0.0, "rel_to_rhs_abs_max": 0.0},
                "alpha": {"max_abs": 0.0, "rel_to_rhs_abs_max": 0.0},
                "depth": {"max_abs": 0.0, "rel_to_rhs_abs_max": 0.0},
                "note": "All-internal owner-run tape preserves every segment, so it is equivalent to full segment replay.",
            },
        }
    return by_scale


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
    density_scales: tuple[float, ...],
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
    site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    active_tape = _with_counts(tape, tape.active_counts_i32)
    full_bytes = _full_segment_csr_bytes(tape, tape.counts_i32)
    active_segment_csr_bytes = _full_segment_csr_bytes(tape, tape.active_counts_i32)
    all_internal = _owner_run_internal_stats(tape, tape.counts_i32)
    active_internal = _owner_run_internal_stats(tape, tape.active_counts_i32)
    density_errors = _density_scale_errors(
        tape=tape,
        active_tape=active_tape,
        site_rgba=site_rgba,
        density_scales=density_scales,
        far=far,
        transmittance_threshold=transmittance_threshold,
    )
    return {
        "frames": frame_count,
        "render_size": render_size,
        "track_count": int(tape.track_count),
        "sample_count": int(tape.sample_count),
        "site_count": len(sites),
        "full_segment_count": int(tape.counts_i32.detach().cpu().to(dtype=torch.int64).sum().item()),
        "active_segment_count": int(tape.active_counts_i32.detach().cpu().to(dtype=torch.int64).sum().item()),
        "full_segment_csr_bytes": full_bytes,
        "active_segment_csr_bytes": active_segment_csr_bytes,
        "active_segment_csr_vs_full_segment_csr": float(active_segment_csr_bytes) / float(max(full_bytes, 1)),
        "all_internal_owner_run_tape": all_internal.as_dict(
            sample_count=tape.sample_count,
            full_segment_csr_bytes=full_bytes,
        ),
        "active_internal_owner_run_tape": active_internal.as_dict(
            sample_count=tape.sample_count,
            full_segment_csr_bytes=full_bytes,
        ),
        "density_scale_errors": density_errors,
    }


def _scale(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    def value(row: dict[str, Any]) -> Any:
        current: Any = row
        for key in path:
            current = current[key]
        return current

    return float(value(rows[-1])) / float(max(value(rows[0]), 1))


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
    density_scales: tuple[float, ...],
    synthetic_motion: SyntheticRayMotion,
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
            density_scales=density_scales,
            synthetic_motion=synthetic_motion,
        )
        for frame_count in frame_counts
    ]
    frame_scale = float(rows[-1]["frames"]) / float(max(rows[0]["frames"], 1))
    all_internal_scale = _scale(rows, ("all_internal_owner_run_tape", "internal_segment_count"))
    active_internal_scale = _scale(rows, ("active_internal_owner_run_tape", "internal_segment_count"))
    max_current_density_error = max(
        float(row["density_scale_errors"]["1"]["active_internal_vs_full"]["depth"]["max_abs"])
        for row in rows
        if "1" in row["density_scale_errors"]
    )
    lower_density_errors = [
        float(scale_payload["active_internal_vs_full"]["alpha"]["max_abs"])
        + float(scale_payload["active_internal_vs_full"]["depth"]["max_abs"])
        for row in rows
        for scale_key, scale_payload in row["density_scale_errors"].items()
        if float(scale_key) < 1.0
    ]
    acceptance = {
        "active_internal_matches_current_density_depth": max_current_density_error <= 5.0e-5,
        "active_internal_not_density_independent_under_lower_density": bool(lower_density_errors)
        and max(lower_density_errors) > 5.0e-5,
        "all_internal_preserves_density_independent_replay_by_construction": True,
        "active_internal_storage_below_full_at_max_frame": rows[-1]["active_internal_owner_run_tape"][
            "nested_csr_vs_full_segment_csr"
        ]
        < 0.35,
        "all_internal_storage_not_star_like_at_max_frame": rows[-1]["all_internal_owner_run_tape"][
            "nested_csr_vs_full_segment_csr"
        ]
        > 0.50,
        "all_internal_segment_count_sublinear_vs_frames": all_internal_scale < frame_scale,
        "active_internal_segment_count_sublinear_vs_frames": active_internal_scale < frame_scale,
    }
    return {
        "benchmark": "world_foam_lane2_owner_run_internal_tape_probe",
        "status": "informational",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "site_count": site_count,
        "density_scales": list(density_scales),
        "synthetic_motion": synthetic_motion.to_dict(),
        "acceptance": acceptance,
        "frame_scale_first_to_last": frame_scale,
        "all_internal_segment_count_scale_first_to_last": all_internal_scale,
        "active_internal_segment_count_scale_first_to_last": active_internal_scale,
        "structural_read": {
            "endpoint_only_rgb_alpha_sufficient_but_depth_insufficient": True,
            "active_internal_cuts_match_current_density_depth": acceptance[
                "active_internal_matches_current_density_depth"
            ],
            "active_internal_cuts_are_density_dependent_due_to_threshold_truncation": acceptance[
                "active_internal_not_density_independent_under_lower_density"
            ],
            "all_internal_cuts_preserve_density_independent_replay": True,
            "all_internal_cuts_move_storage_back_toward_full_segment_tape": acceptance[
                "all_internal_storage_not_star_like_at_max_frame"
            ],
            "all_owner_run_endpoint_storage_is_compact_if_depth_semantic_changes": rows[-1][
                "all_internal_owner_run_tape"
            ]["endpoint_run_csr_vs_full_segment_csr"]
            < 0.20,
            "interpretation": (
                "Internal cuts close the current-depth replay gap, but the exact density-independent version "
                "must keep every segment and therefore gives back much of the STAR-style compactness. The "
                "active-only version is compact and exact at the reference density, but lower density can "
                "reactivate segments that were truncated by the original threshold. If depth semantics move to "
                "continuous absorption within a same-owner run, endpoint run storage is compact, but that is an "
                "explicit semantic change from the current segment-mid tape."
            ),
        },
        "rows": rows,
    }


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one density scale")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe owner-run internal-cut tapes for depth replay.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--density-scales", default="1,0.5,0.25,2")
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_owner_run_internal_tape_probe_render32_2_4_8_16.json",
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
        density_scales=_parse_float_list(args.density_scales),
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
