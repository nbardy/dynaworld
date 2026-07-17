from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pipeline.diagnostics import reconstruction_eval_metrics
from powerfoam_eval_color import (
    apply_eval_color_calibration,
    fit_eval_color_calibration,
    serialize_eval_color_calibration,
)
from powerfoam_eval_render import powerfoam_eval_batch_size, render_powerfoam_samples
from powerfoam_objectives import composite_fixed_background
from train_artifacts import write_json
from train_logging import log_wandb_run_payload_lazy, mapped_metric_payload, should_log_video
from wandb_media import build_rgb_alpha_eval_media_payload
from video_io import save_rgb_alpha_eval_media, video_fps_from_config


def log_powerfoam_artifacts(
    model: Any,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
    *,
    frame_indices: torch.Tensor | None = None,
    rays: torch.Tensor | None = None,
    heldout_targets: torch.Tensor | None = None,
    heldout_frame_indices: torch.Tensor | None = None,
    heldout_rays: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    if frame_indices is None:
        frame_indices = torch.arange(targets.size(0), device=device, dtype=torch.long)
    else:
        frame_indices = frame_indices.to(device=device, dtype=torch.long)
    renders, alphas = render_powerfoam_samples(
        model,
        frame_indices,
        batch_size=powerfoam_eval_batch_size(cfg),
        rays=rays,
    )
    renders = composite_fixed_background(renders, alphas, cfg["render"])
    targets_cpu = targets.detach().cpu()
    calibration = fit_eval_color_calibration(cfg["render"], renders, targets_cpu)
    raw_renders = renders
    renders = apply_eval_color_calibration(raw_renders, calibration)
    metrics = reconstruction_eval_metrics(renders, targets_cpu, cfg, prefix="eval")
    if calibration is not None:
        metrics.update(reconstruction_eval_metrics(raw_renders, targets_cpu, cfg, prefix="uncalibrated_eval"))
    metrics.update(model.aux_metrics(frame_indices, targets, rays=rays))
    heldout_renders = None
    heldout_alphas = None
    if heldout_targets is not None and heldout_frame_indices is not None:
        heldout_renders, heldout_alphas = render_powerfoam_samples(
            model,
            heldout_frame_indices,
            batch_size=powerfoam_eval_batch_size(cfg),
            rays=heldout_rays,
        )
        heldout_renders = composite_fixed_background(heldout_renders, heldout_alphas, cfg["render"])
        heldout_targets_cpu = heldout_targets.detach().cpu()
        raw_heldout_renders = heldout_renders
        heldout_renders = apply_eval_color_calibration(raw_heldout_renders, calibration)
        metrics.update(reconstruction_eval_metrics(heldout_renders, heldout_targets_cpu, cfg, prefix="heldout_eval"))
        if calibration is not None:
            metrics.update(
                reconstruction_eval_metrics(
                    raw_heldout_renders,
                    heldout_targets_cpu,
                    cfg,
                    prefix="uncalibrated_heldout_eval",
                )
            )
    metrics.update(model.parameter_drift_metrics())
    if calibration is not None:
        write_json(
            output_dir / f"eval_color_calibration_step_{step:04d}.json",
            serialize_eval_color_calibration(
                calibration,
                step=step,
                train_frame_indices=frame_indices,
                heldout_frame_indices=heldout_frame_indices,
            ),
        )
    save_rgb_alpha_eval_media(
        output_dir,
        step,
        renders,
        targets_cpu,
        alphas,
        fps=video_fps_from_config(cfg),
        save_videos=should_log_video(cfg, step),
        heldout_renders=heldout_renders,
        heldout_targets=heldout_targets,
        heldout_alphas=heldout_alphas,
    )

    def _wandb_payload() -> dict[str, Any]:
        fps = video_fps_from_config(cfg)
        payload: dict[str, Any] = mapped_metric_payload(
            metrics,
            (
                ("eval_l1", "Eval/L1"),
                ("eval_mse", "Eval/MSE"),
                ("eval_psnr", "Eval/PSNR"),
                ("eval_ssim", "Eval/SSIM"),
                ("state_mean_center_delta", "State/MeanCenterDelta"),
                ("state_p95_center_delta", "State/P95CenterDelta"),
                ("state_max_center_delta", "State/MaxCenterDelta"),
                ("state_mean_xy_delta", "State/MeanXYDelta"),
                ("state_mean_z_delta", "State/MeanZDelta"),
                ("state_mean_radius_delta", "State/MeanRadiusDelta"),
                ("state_mean_density_delta", "State/MeanDensityDelta"),
                ("state_mean_feature_delta", "State/MeanFeatureDelta"),
                ("state_cell_count", "State/CellCount"),
            ),
        )
        payload.update(
            mapped_metric_payload(
                metrics,
                (
                    ("heldout_eval_l1", "Heldout/EvalL1"),
                    ("heldout_eval_mse", "Heldout/EvalMSE"),
                    ("heldout_eval_psnr", "Heldout/EvalPSNR"),
                    ("heldout_eval_ssim", "Heldout/EvalSSIM"),
                    ("state_mean_normal_delta", "State/MeanNormalDelta"),
                    ("state_mean_normal_z", "State/MeanNormalZ"),
                    ("state_mean_texel_site_delta", "State/MeanTexelSiteDelta"),
                    ("state_mean_texel_height_delta", "State/MeanTexelHeightDelta"),
                    ("state_mean_texel_sv_axis_delta", "State/MeanTexelSvAxisDelta"),
                    ("state_mean_texel_sv_rgb_delta", "State/MeanTexelSvRgbDelta"),
                    ("state_mean_quaternion_delta", "State/MeanQuaternionDelta"),
                ),
                require=False,
            )
        )
        payload.update(
            build_rgb_alpha_eval_media_payload(
                renders,
                targets_cpu,
                alphas,
                step=step,
                fps=fps,
                include_videos=should_log_video(cfg, step),
            )
        )
        for key in (
            "aux_mean_contrib",
            "aux_max_contrib",
            "aux_mean_point_error",
            "aux_max_point_error",
            "aux_mean_contrib_ema",
            "aux_mean_point_error_ema",
            "aux_visible_fraction",
            "aux_mean_normal_distance",
            "aux_mean_normal_norm",
            "aux_median_depth_valid_fraction",
            "aux_mean_median_depth",
        ):
            if key in metrics:
                payload[f"Aux/{key.removeprefix('aux_')}"] = metrics[key]
        return payload

    log_wandb_run_payload_lazy(wandb_run, _wandb_payload, step=step)
    model.train()
    return metrics
