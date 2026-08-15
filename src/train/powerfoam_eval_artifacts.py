from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from paper_training_protocol import PaperRGBMetricAccumulator
from perceptual_metrics import video_lpips
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
from video_io import save_rgb_alpha_eval_media, video_fps_from_config
from wandb_media import build_rgb_alpha_eval_media_payload


@dataclass(frozen=True)
class StreamedEvalSplit:
    metrics: dict[str, float]
    renders: torch.Tensor
    targets: torch.Tensor
    alphas: torch.Tensor


def _media_positions(frame_count: int, max_frames: int) -> set[int]:
    count = min(int(frame_count), int(max_frames))
    return set(torch.linspace(0, frame_count - 1, steps=count).round().to(torch.long).tolist())


def _chunk_rays(
    rays: torch.Tensor | None,
    ray_provider: Any | None,
    start: int,
    stop: int,
) -> torch.Tensor | None:
    if rays is not None and ray_provider is not None:
        raise ValueError("provide materialized rays or a ray provider, not both")
    if ray_provider is not None:
        return ray_provider.select(torch.arange(start, stop, dtype=torch.long))
    return None if rays is None else rays[start:stop]


def _target_sample_count(
    targets: torch.Tensor | None,
    target_provider: Any | None,
    *,
    split: str,
) -> int:
    if targets is not None and target_provider is not None:
        raise ValueError(f"provide materialized {split} targets or a target provider, not both")
    if target_provider is not None:
        count = int(target_provider.sample_count)
    elif targets is not None:
        count = int(targets.size(0))
    else:
        raise ValueError(f"{split} targets require a materialized tensor or target provider")
    if count < 1:
        raise ValueError(f"{split} targets require at least one sample")
    return count


def _chunk_targets(
    targets: torch.Tensor | None,
    target_provider: Any | None,
    start: int,
    stop: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if target_provider is not None:
        if targets is not None:
            raise ValueError("provide materialized targets or a target provider, not both")
        return target_provider.select(
            torch.arange(start, stop, dtype=torch.long),
            device=device,
        )
    if targets is None:
        raise ValueError("target chunk requires a materialized tensor or target provider")
    return targets[start:stop].detach().to(device=device, dtype=torch.float32)


@torch.no_grad()
def _stream_aux_metrics(
    model: Any,
    targets: torch.Tensor | None,
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    ray_provider: Any | None,
    cfg: dict[str, Any],
    *,
    target_provider: Any | None = None,
) -> dict[str, float]:
    weighted_keys = (
        "aux_mean_contrib",
        "aux_mean_point_error",
        "aux_mean_visible_cells_per_frame",
        "aux_mean_normal_distance",
        "aux_mean_normal_norm",
        "aux_median_depth_valid_fraction",
    )
    maximum_keys = ("aux_max_contrib", "aux_max_point_error")
    totals = {key: 0.0 for key in weighted_keys}
    maxima = {key: -float("inf") for key in maximum_keys}
    visible_events = 0.0
    possible_events = 0.0
    median_depth_sum = 0.0
    median_depth_weight = 0.0
    sample_count = 0
    found = False
    batch_size = powerfoam_eval_batch_size(cfg)
    for start in range(0, int(frame_indices.numel()), batch_size):
        stop = min(start + batch_size, int(frame_indices.numel()))
        count = stop - start
        chunk = model.aux_metrics(
            frame_indices[start:stop],
            _chunk_targets(
                targets,
                target_provider,
                start,
                stop,
                device=frame_indices.device,
            ),
            rays=_chunk_rays(rays, ray_provider, start, stop),
        )
        if not chunk:
            continue
        found = True
        sample_count += count
        for key in weighted_keys:
            totals[key] += float(chunk[key]) * count
        for key in maximum_keys:
            maxima[key] = max(maxima[key], float(chunk[key]))
        visible_events += float(chunk["aux_visible_cell_frame_events"])
        possible = float(chunk["aux_possible_cell_frame_events"])
        possible_events += possible
        valid_weight = possible * float(chunk["aux_median_depth_valid_fraction"])
        if "aux_mean_median_depth" in chunk and valid_weight > 0.0:
            median_depth_sum += float(chunk["aux_mean_median_depth"]) * valid_weight
            median_depth_weight += valid_weight
    if not found:
        return {}
    metrics = {key: value / float(sample_count) for key, value in totals.items()}
    metrics.update(maxima)
    metrics["aux_visible_cell_frame_events"] = visible_events
    metrics["aux_possible_cell_frame_events"] = possible_events
    metrics["aux_visible_fraction"] = visible_events / possible_events if possible_events else 0.0
    ema_frames = frame_indices.to(device=model.contrib_ema.device, dtype=torch.long)
    metrics["aux_mean_contrib_ema"] = float(model.contrib_ema[ema_frames].mean().detach().cpu())
    metrics["aux_mean_point_error_ema"] = float(model.point_error_ema[ema_frames].mean().detach().cpu())
    if median_depth_weight > 0.0:
        metrics["aux_mean_median_depth"] = median_depth_sum / median_depth_weight
    return metrics


@torch.no_grad()
def _stream_eval_split(
    model: Any,
    targets: torch.Tensor | None,
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    ray_provider: Any | None,
    cfg: dict[str, Any],
    *,
    target_provider: Any | None = None,
    prefix: str,
    include_lpips: bool,
) -> StreamedEvalSplit:
    batch_size = powerfoam_eval_batch_size(cfg)
    media_limit = cfg["logging"]["eval_media_max_frames"]
    if media_limit is None:
        media_limit = int(frame_indices.numel())
    selected = _media_positions(int(frame_indices.numel()), int(media_limit))
    accumulator = PaperRGBMetricAccumulator(
        ssim_window_size=int(cfg["losses"]["ssim_window_size"]),
        ssim_c1=float(cfg["losses"]["ssim_c1"]),
        ssim_c2=float(cfg["losses"]["ssim_c2"]),
    )
    media_renders = []
    media_targets = []
    media_alphas = []
    lpips_sum = 0.0
    lpips_count = 0
    for start in range(0, int(frame_indices.numel()), batch_size):
        stop = min(start + batch_size, int(frame_indices.numel()))
        chunk_rays = _chunk_rays(rays, ray_provider, start, stop)
        renders, alphas = render_powerfoam_samples(
            model,
            frame_indices[start:stop],
            batch_size=batch_size,
            rays=chunk_rays,
        )
        renders = composite_fixed_background(renders, alphas, cfg["render"])
        target_chunk = _chunk_targets(
            targets,
            target_provider,
            start,
            stop,
            device=torch.device("cpu"),
        )
        accumulator.update(renders, target_chunk)
        if include_lpips:
            lpips_sum += video_lpips(renders, target_chunk) * float(stop - start)
            lpips_count += stop - start
        local_positions = [position - start for position in sorted(selected) if start <= position < stop]
        if local_positions:
            local = torch.tensor(local_positions, dtype=torch.long)
            media_renders.append(renders[local])
            media_targets.append(target_chunk[local])
            media_alphas.append(alphas[local])
    metrics = accumulator.metrics(prefix=prefix)
    if include_lpips:
        if lpips_count < 1:
            raise ValueError("heldout LPIPS requires at least one rendered frame")
        metrics[f"{prefix}_lpips"] = lpips_sum / float(lpips_count)
    return StreamedEvalSplit(
        metrics=metrics,
        renders=torch.cat(media_renders, dim=0),
        targets=torch.cat(media_targets, dim=0),
        alphas=torch.cat(media_alphas, dim=0),
    )


def log_powerfoam_artifacts(
    model: Any,
    targets: torch.Tensor | None,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
    *,
    frame_indices: torch.Tensor | None = None,
    rays: torch.Tensor | None = None,
    ray_provider: Any | None = None,
    target_provider: Any | None = None,
    heldout_targets: torch.Tensor | None = None,
    heldout_frame_indices: torch.Tensor | None = None,
    heldout_rays: torch.Tensor | None = None,
    heldout_ray_provider: Any | None = None,
    heldout_target_provider: Any | None = None,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    target_count = _target_sample_count(targets, target_provider, split="training")
    if frame_indices is None:
        frame_indices = torch.arange(target_count, device=device, dtype=torch.long)
    else:
        frame_indices = frame_indices.to(device=device, dtype=torch.long)
    if int(frame_indices.numel()) != target_count:
        raise ValueError(
            "training target/frame-index count mismatch: "
            f"{target_count} targets versus {int(frame_indices.numel())} frame indices"
        )
    stream_paper_eval = (
        bool(cfg["paper_protocol"]["enabled"])
        and str(cfg["render"]["eval_color_calibration"]) == "none"
    )
    if stream_paper_eval:
        train_eval = _stream_eval_split(
            model,
            targets,
            frame_indices,
            rays,
            ray_provider,
            cfg,
            target_provider=target_provider,
            prefix="eval",
            include_lpips=False,
        )
        renders, targets_cpu, alphas = train_eval.renders, train_eval.targets, train_eval.alphas
        metrics = dict(train_eval.metrics)
        calibration = None
    else:
        if target_provider is not None or targets is None:
            raise ValueError("target providers require streamed paper evaluation")
        renders, alphas = render_powerfoam_samples(
            model,
            frame_indices,
            batch_size=powerfoam_eval_batch_size(cfg),
            rays=(
                ray_provider.select(torch.arange(frame_indices.numel(), dtype=torch.long))
                if ray_provider is not None
                else rays
            ),
        )
        renders = composite_fixed_background(renders, alphas, cfg["render"])
        targets_cpu = targets.detach().cpu()
        calibration = fit_eval_color_calibration(cfg["render"], renders, targets_cpu)
        raw_renders = renders
        renders = apply_eval_color_calibration(raw_renders, calibration)
        metrics = reconstruction_eval_metrics(renders, targets_cpu, cfg, prefix="eval")
        if calibration is not None:
            metrics.update(reconstruction_eval_metrics(raw_renders, targets_cpu, cfg, prefix="uncalibrated_eval"))
    if stream_paper_eval:
        metrics.update(
            _stream_aux_metrics(
                model,
                targets,
                frame_indices,
                rays,
                ray_provider,
                cfg,
                target_provider=target_provider,
            )
        )
    else:
        metrics.update(model.aux_metrics(frame_indices, targets, rays=rays))
    heldout_renders = None
    heldout_alphas = None
    heldout_targets_cpu = None
    has_heldout_targets = heldout_targets is not None or heldout_target_provider is not None
    if has_heldout_targets != (heldout_frame_indices is not None):
        raise ValueError("heldout targets and heldout frame indices must be provided together")
    if has_heldout_targets and heldout_frame_indices is not None:
        heldout_count = _target_sample_count(
            heldout_targets,
            heldout_target_provider,
            split="heldout",
        )
        if int(heldout_frame_indices.numel()) != heldout_count:
            raise ValueError(
                "heldout target/frame-index count mismatch: "
                f"{heldout_count} targets versus {int(heldout_frame_indices.numel())} frame indices"
            )
        if stream_paper_eval:
            heldout_eval = _stream_eval_split(
                model,
                heldout_targets,
                heldout_frame_indices,
                heldout_rays,
                heldout_ray_provider,
                cfg,
                target_provider=heldout_target_provider,
                prefix="heldout_eval",
                include_lpips=int(step) == int(cfg["train"]["steps"]),
            )
            heldout_renders = heldout_eval.renders
            heldout_targets_cpu = heldout_eval.targets
            heldout_alphas = heldout_eval.alphas
            metrics.update(heldout_eval.metrics)
        else:
            if heldout_target_provider is not None or heldout_targets is None:
                raise ValueError("target providers require streamed paper evaluation")
            heldout_renders, heldout_alphas = render_powerfoam_samples(
                model,
                heldout_frame_indices,
                batch_size=powerfoam_eval_batch_size(cfg),
                rays=(
                    heldout_ray_provider.select(
                        torch.arange(heldout_frame_indices.numel(), dtype=torch.long)
                    )
                    if heldout_ray_provider is not None
                    else heldout_rays
                ),
            )
            heldout_renders = composite_fixed_background(heldout_renders, heldout_alphas, cfg["render"])
            heldout_targets_cpu = heldout_targets.detach().cpu()
            raw_heldout_renders = heldout_renders
            heldout_renders = apply_eval_color_calibration(raw_heldout_renders, calibration)
            metrics.update(reconstruction_eval_metrics(heldout_renders, heldout_targets_cpu, cfg, prefix="heldout_eval"))
            if int(step) == int(cfg["train"]["steps"]):
                metrics["heldout_eval_lpips"] = video_lpips(heldout_renders, heldout_targets_cpu)
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
        heldout_targets=heldout_targets_cpu,
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
            "aux_visible_cell_frame_events",
            "aux_possible_cell_frame_events",
            "aux_mean_visible_cells_per_frame",
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
