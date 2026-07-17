from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

try:
    from .report_artifacts import ROOT, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, write_report_json

from config_utils import load_config_file
from research_project.trainer_harness.data import load_video_target
from research_project.trainer_harness.model import ScreenTimeTubeModel
from torch_gsplat_bridge_star_uvt import (
    UVTRenderConfig,
    count_projective_trace_dense_per_frame_tile_pairs,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_fallback_stats,
    uvt_tubes_to_projective_trace_cell_atlas,
)


SCHEMA_VERSION = "projective_high_motion_trace_geometry_report_v1"
SOURCE_CONFIG = ROOT / "src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc"
HIGH_MOTION_SMOKE_VIDEO = ROOT / "data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4"


@dataclass(frozen=True)
class HighMotionTraceGeometryCase:
    name: str
    source: str
    trained_checkpoint: str | None
    source_config_path: str
    source_video_path: str
    source_video_exists: bool
    target_size: int
    frames: int
    tube_count: int
    sample_mode: str
    velocity_init: str
    velocity_search_radius: int
    velocity_patch_radius: int
    spatial_precision: float
    temporal_precision: float
    opacity: float
    tile_size: int
    support_uv_padding: float
    trace_count: int
    cell_count: int
    tile_active_set_groups: int
    max_cells_per_active_set_group: int
    interval_trace_entries: int
    dense_trace_samples: int
    interval_to_dense_trace_sample_ratio: float
    dense_per_frame_tile_pairs: int
    interval_to_dense_tile_pair_ratio: float
    fallback_cells: int
    fallback_fraction: float
    fallback_reasons: tuple[str, ...]
    velocity_nonzero_count: int
    velocity_mean_px_per_frame: float
    velocity_max_px_per_frame: float
    opacity_min: float
    opacity_max: float
    train_steps: int
    train_lr: float | None
    train_initial_loss: float | None
    train_final_loss: float | None
    train_loss_ratio: float | None
    trained_parameter_l1_delta: float | None
    trained_parameter_l1_deltas: dict[str, float] | None
    trained_moved_parameter_names: tuple[str, ...] | None


def _load_source_cfg() -> dict[str, Any]:
    return load_config_file(SOURCE_CONFIG)


def _support_padding(
    *,
    opacity: torch.Tensor,
    alpha_threshold: float,
    sigma_px: float,
    q_uvt: torch.Tensor | None = None,
) -> float:
    opacity_max = float(opacity.detach().max().cpu().item()) if int(opacity.numel()) else 0.0
    if opacity_max <= alpha_threshold:
        return 0.0
    radius2 = 2.0 * math.log(opacity_max / float(alpha_threshold))
    radius = math.sqrt(radius2 * float(sigma_px) * float(sigma_px))
    if q_uvt is not None and int(q_uvt.numel()) > 0:
        q = q_uvt.detach()
        det = (q[:, 0] * q[:, 3] - q[:, 1].square()).clamp_min(1.0e-12)
        inv00 = (q[:, 3] / det).clamp_min(0.0)
        inv11 = (q[:, 0] / det).clamp_min(0.0)
        anisotropic_radius = torch.sqrt((radius2 * torch.maximum(inv00, inv11)).clamp_min(0.0)).amax()
        radius = max(radius, float(anisotropic_radius.cpu().item()))
    return float(math.ceil(radius) + 1.0)


def _compile_trace_geometry_case(
    *,
    name: str,
    velocity_init: str,
    tube_count: int = 64,
    target_size: int = 64,
    frames: int = 16,
    tile_size: int = 8,
    train_steps: int = 0,
    train_lr: float = 0.03,
) -> HighMotionTraceGeometryCase:
    cfg = _load_source_cfg()
    uvt_cfg = cfg["uvt"]
    target = load_video_target(
        HIGH_MOTION_SMOKE_VIDEO,
        target_size=int(target_size),
        max_frames=int(frames),
        device="cpu",
        image_crop_mode=str(cfg["data"]["image_crop_mode"]),
    )
    render_config = UVTRenderConfig(
        height=int(target_size),
        width=int(target_size),
        frames=int(frames),
        tile_x=int(tile_size),
        tile_y=int(tile_size),
        tile_t=1,
        tile_capacity=int(uvt_cfg["tile_capacity"]),
    )
    model = ScreenTimeTubeModel.from_video_samples(
        target,
        render_config,
        tube_count=int(tube_count),
        seed=int(cfg["train"]["seed"]),
        spatial_precision=float(uvt_cfg["spatial_precision"]),
        temporal_precision=float(uvt_cfg["temporal_precision"]),
        opacity=float(uvt_cfg["opacity"]),
        sample_mode=str(uvt_cfg["sample_mode"]),
        velocity_init=str(velocity_init),
        velocity_search_radius=int(uvt_cfg["velocity_search_radius"]),
        velocity_patch_radius=int(uvt_cfg["velocity_patch_radius"]),
        velocity_min_improvement_ratio=float(uvt_cfg["velocity_min_improvement_ratio"]),
    )
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    losses: list[float] = []
    if train_steps > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(train_lr))
        for step in range(int(train_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model_render(model), target)
            losses.append(float(loss.detach().cpu().item()))
            if step == int(train_steps):
                break
            loss.backward()
            optimizer.step()
    ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
    times = (torch.arange(int(frames), dtype=torch.float32) - 0.5 * float(int(frames) - 1)).contiguous()
    sigma_px = math.sqrt(1.0 / float(uvt_cfg["spatial_precision"]))
    support_uv_padding = _support_padding(
        opacity=opacity,
        alpha_threshold=float(render_config.alpha_threshold),
        sigma_px=sigma_px,
        q_uvt=q_uvt,
    )
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma.detach(),
        q_uvt.detach(),
        depth0.detach(),
        depth_beta.detach(),
        opacity.detach(),
        color.detach(),
        times,
        sigma_px=float(sigma_px),
        image_width=int(render_config.width),
        image_height=int(render_config.height),
        tile_size=int(tile_size),
        uv_padding=float(support_uv_padding),
        alpha_threshold=float(render_config.alpha_threshold),
        temporal_mode="trace",
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
        stratify_visibility=True,
        mark_visibility_fallback=True,
    )
    complexity = projective_trace_cell_atlas_complexity_stats(atlas)
    fallback = projective_trace_cell_atlas_fallback_stats(atlas)
    dense_tile_pairs = count_projective_trace_dense_per_frame_tile_pairs(
        atlas.coeffs,
        times,
        image_width=int(render_config.width),
        image_height=int(render_config.height),
        tile_size=int(tile_size),
        uv_padding=float(support_uv_padding),
    )
    velocity_norm = model.velocity_uv.detach().norm(dim=1)
    interval_entries = int(complexity.interval_trace_entries)
    parameter_l1_delta = 0.0
    parameter_l1_deltas: dict[str, float] = {}
    if int(train_steps) > 0:
        for param_name, value in model.state_dict().items():
            delta = float((value.detach() - initial_state[param_name]).abs().sum().cpu().item())
            parameter_l1_deltas[param_name] = delta
            parameter_l1_delta += delta
    return HighMotionTraceGeometryCase(
        name=name,
        source="star_uvt_trainer_harness_video_samples",
        trained_checkpoint=None,
        source_config_path=str(SOURCE_CONFIG),
        source_video_path=str(HIGH_MOTION_SMOKE_VIDEO),
        source_video_exists=bool(HIGH_MOTION_SMOKE_VIDEO.exists()),
        target_size=int(target_size),
        frames=int(frames),
        tube_count=int(tube_count),
        sample_mode=str(uvt_cfg["sample_mode"]),
        velocity_init=str(velocity_init),
        velocity_search_radius=int(uvt_cfg["velocity_search_radius"]),
        velocity_patch_radius=int(uvt_cfg["velocity_patch_radius"]),
        spatial_precision=float(uvt_cfg["spatial_precision"]),
        temporal_precision=float(uvt_cfg["temporal_precision"]),
        opacity=float(uvt_cfg["opacity"]),
        tile_size=int(tile_size),
        support_uv_padding=float(support_uv_padding),
        trace_count=int(atlas.coeffs.shape[0]),
        cell_count=int(complexity.total_cells),
        tile_active_set_groups=int(complexity.tile_active_set_groups),
        max_cells_per_active_set_group=int(complexity.max_cells_per_active_set_group),
        interval_trace_entries=interval_entries,
        dense_trace_samples=int(complexity.dense_trace_samples),
        interval_to_dense_trace_sample_ratio=float(complexity.interval_to_dense_trace_sample_ratio),
        dense_per_frame_tile_pairs=int(dense_tile_pairs),
        interval_to_dense_tile_pair_ratio=float(interval_entries) / float(max(1, dense_tile_pairs)),
        fallback_cells=int(fallback.fallback_cells),
        fallback_fraction=float(fallback.fallback_fraction),
        fallback_reasons=tuple(str(reason) for reason in fallback.fallback_reasons),
        velocity_nonzero_count=int((velocity_norm > 0.0).sum().cpu().item()),
        velocity_mean_px_per_frame=float(velocity_norm.mean().cpu().item()),
        velocity_max_px_per_frame=float(velocity_norm.max().cpu().item()),
        opacity_min=float(opacity.detach().min().cpu().item()),
        opacity_max=float(opacity.detach().max().cpu().item()),
        train_steps=int(train_steps),
        train_lr=float(train_lr) if int(train_steps) > 0 else None,
        train_initial_loss=losses[0] if losses else None,
        train_final_loss=losses[-1] if losses else None,
        train_loss_ratio=(losses[-1] / max(losses[0], 1.0e-12)) if losses else None,
        trained_parameter_l1_delta=parameter_l1_delta if int(train_steps) > 0 else None,
        trained_parameter_l1_deltas=parameter_l1_deltas if int(train_steps) > 0 else None,
        trained_moved_parameter_names=tuple(
            param_name for param_name, delta in parameter_l1_deltas.items() if delta > 0.0
        )
        if int(train_steps) > 0
        else None,
    )


def model_render(model: ScreenTimeTubeModel) -> torch.Tensor:
    from research_project.trainer_harness.model import render_model

    return render_model(model)


def build_high_motion_trace_geometry_report() -> dict[str, Any]:
    cases = [
        _compile_trace_geometry_case(
            name="config_faithful_zero_velocity_init",
            velocity_init="zero",
        ),
        _compile_trace_geometry_case(
            name="block_match_motion_init",
            velocity_init="block_match_gated",
        ),
        _compile_trace_geometry_case(
            name="block_match_motion_trained_dense_3step",
            velocity_init="block_match_gated",
            train_steps=3,
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if all(case.fallback_fraction == 0.0 for case in cases) else "has_fallback",
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
        "summary": {
            "max_interval_to_dense_trace_sample_ratio": max(
                case.interval_to_dense_trace_sample_ratio for case in cases
            ),
            "max_interval_to_dense_tile_pair_ratio": max(case.interval_to_dense_tile_pair_ratio for case in cases),
            "max_fallback_fraction": max(case.fallback_fraction for case in cases),
            "max_velocity_px_per_frame": max(case.velocity_max_px_per_frame for case in cases),
            "trained_case_count": sum(1 for case in cases if case.train_steps > 0),
            "min_train_loss_ratio": min(
                (case.train_loss_ratio for case in cases if case.train_loss_ratio is not None),
                default=None,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write high-motion STAR UVT trace-geometry report JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/projective_high_motion_trace_geometry_report.json"),
    )
    args = parser.parse_args()
    output = write_report_json(args.output, build_high_motion_trace_geometry_report())
    print(output)


if __name__ == "__main__":
    main()
