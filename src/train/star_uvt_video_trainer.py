from __future__ import annotations

from pathlib import Path
from typing import Any

from config_utils import path_or_none
from star_uvt_video_overfit_config import resolve_config
from star_uvt_outputs import log_star_uvt_row_outputs, write_row_json_and_print
from star_uvt_runtime import ensure_star_uvt_on_path
from train_logging import finish_wandb_run, init_wandb_run


def _assert_loss_decreased(row: dict[str, Any]) -> None:
    for section in ("uvt", "per_frame"):
        metrics = row.get(section)
        if metrics is None:
            continue
        initial = float(metrics["initial_loss"])
        final = float(metrics["final_loss"])
        if final >= initial:
            raise AssertionError(f"{section} loss did not decrease: initial={initial}, final={final}")


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_config(config)
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from research_project.benchmarks.video_fit_comparison import run_video_fit_comparison

    run = init_wandb_run(cfg)
    try:
        row = run_video_fit_comparison(
            video_path=Path(cfg["data"]["video_path"]),
            start_seconds=cfg["data"]["start_seconds"],
            fps=cfg["data"]["fps"],
            duration_seconds=cfg["data"]["duration_seconds"],
            image_crop_mode=str(cfg["data"]["image_crop_mode"]),
            tube_count=int(cfg["uvt"]["tube_count"]),
            per_frame_splats=int(cfg["per_frame"]["splats"]),
            target_size=int(cfg["data"]["target_size"]),
            max_frames=int(cfg["data"]["max_frames"]),
            steps=int(cfg["train"]["steps"]),
            lr=float(cfg["train"]["lr"]),
            per_frame_lr=float(cfg["per_frame"]["lr"]),
            per_frame_init_mode=str(cfg["per_frame"]["init_mode"]),
            per_frame_render_backend=str(cfg["per_frame"]["render_backend"]),
            per_frame_fast_max_pairs=int(cfg["per_frame"]["fast_max_pairs"]),
            per_frame_spatial_precision=float(cfg["per_frame"]["spatial_precision"]),
            per_frame_opacity=float(cfg["per_frame"]["opacity"]),
            per_frame_sample_mode=str(cfg["per_frame"]["sample_mode"]),
            device=str(cfg["train"]["device"]),
            seed=int(cfg["train"]["seed"]),
            uvt_init_mode=str(cfg["uvt"]["init_mode"]),
            uvt_spatial_precision=float(cfg["uvt"]["spatial_precision"]),
            uvt_temporal_precision=float(cfg["uvt"]["temporal_precision"]),
            uvt_opacity=float(cfg["uvt"]["opacity"]),
            uvt_sample_mode=str(cfg["uvt"]["sample_mode"]),
            uvt_velocity_init=str(cfg["uvt"]["velocity_init"]),
            uvt_velocity_search_radius=int(cfg["uvt"]["velocity_search_radius"]),
            uvt_velocity_patch_radius=int(cfg["uvt"]["velocity_patch_radius"]),
            uvt_velocity_min_improvement_ratio=float(cfg["uvt"]["velocity_min_improvement_ratio"]),
            uvt_final_lr=cfg["uvt"]["final_lr"],
            uvt_final_lr_start_step=cfg["uvt"]["final_lr_start_step"],
            uvt_coarse_target_size=cfg["uvt"]["coarse_target_size"],
            uvt_coarse_steps=int(cfg["uvt"]["coarse_steps"]),
            uvt_coarse_lr=cfg["uvt"]["coarse_lr"],
            uvt_appearance_refine_steps=int(cfg["uvt"]["appearance_refine_steps"]),
            uvt_appearance_lr=float(cfg["uvt"]["appearance_lr"]),
            uvt_temporal_split_step=cfg["uvt"]["temporal_split_step"],
            uvt_temporal_split_offset=float(cfg["uvt"]["temporal_split_offset"]),
            uvt_temporal_split_precision_scale=float(cfg["uvt"]["temporal_split_precision_scale"]),
            uvt_temporal_split_opacity_scale=float(cfg["uvt"]["temporal_split_opacity_scale"]),
            uvt_temporal_split_depth_offset=float(cfg["uvt"]["temporal_split_depth_offset"]),
            uvt_temporal_split_lr=cfg["uvt"]["temporal_split_lr"],
            uvt_render_backend=str(cfg["uvt"]["render_backend"]),
            uvt_reduction_mode=str(cfg["uvt"]["reduction_mode"]),
            uvt_sample_emission_mode=str(cfg["uvt"]["sample_emission_mode"]),
            uvt_tile_t=int(cfg["uvt"]["tile_t"]),
            uvt_tile_capacity=int(cfg["uvt"]["tile_capacity"]),
            uvt_tile_load_reg_weight=float(cfg["uvt"]["tile_load_reg_weight"]),
            uvt_tile_load_target=float(cfg["uvt"]["tile_load_target"]),
            render_benchmark_repeats=int(cfg["train"]["render_benchmark_repeats"]),
            skip_uvt=bool(cfg["uvt"]["skip_uvt"]),
            skip_per_frame=bool(cfg["per_frame"]["skip_per_frame"]),
            contact_sheet=path_or_none(cfg["output"]["contact_sheet"]),
            contact_sheet_frames=int(cfg["output"]["contact_sheet_frames"]),
            contact_sheet_mode=str(cfg["output"]["contact_sheet_mode"]),
            side_by_side_video=path_or_none(cfg["output"]["side_by_side_video"]),
            side_by_side_fps=cfg["output"]["side_by_side_fps"],
        )
        if bool(cfg["train"]["require_loss_decrease"]):
            _assert_loss_decreased(row)
        if run is not None:
            log_star_uvt_row_outputs(row, cfg, metric_prefix="star_uvt")
        write_row_json_and_print(row, cfg["output"]["out_json"])
        return row
    finally:
        finish_wandb_run(run)


__all__ = ["run_training"]
