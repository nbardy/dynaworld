from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from renderers.common import build_pixel_grid, project_gaussians_2d_batch, transform_world_to_camera_batch
from renderers.overlap_metrics import (
    aggregate_stat_dicts,
    custom_rect_overlap_stats,
    exact_conic_overlap_stats,
    selected_overlap_summary,
)

DEFAULT_PIXEL_DIAG_MAX_WORK_ITEMS = 64_000_000
DEFAULT_TILE_DIAG_MAX_GAUSSIANS = 4096


@dataclass(frozen=True)
class MetricConfig:
    renderer: bool = False
    optimizer: bool = False
    every: int = 0
    print_summary: bool = False
    wandb: bool = True
    fail_fast: bool = True

    @property
    def enabled(self) -> bool:
        return self.renderer or self.optimizer

    def due(self, step: int) -> bool:
        return self.enabled and self.every > 0 and step % self.every == 0


def with_metrics(
    config: dict[str, Any],
    *,
    renderer: bool | None = None,
    optimizer: bool | None = None,
    every: int | None = None,
    print_summary: bool | None = None,
    wandb: bool | None = None,
    fail_fast: bool | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    logging_cfg = cfg.setdefault("logging", {})
    metrics_cfg = dict(logging_cfg.get("with_metrics") or {})
    updates = {
        "renderer": renderer,
        "optimizer": optimizer,
        "every": every,
        "print_summary": print_summary,
        "wandb": wandb,
        "fail_fast": fail_fast,
    }
    for key, value in updates.items():
        if value is not None:
            metrics_cfg[key] = value
    logging_cfg["with_metrics"] = metrics_cfg
    return cfg


def metric_config_from_logging(logging_cfg: dict[str, Any]) -> MetricConfig:
    raw_metrics = logging_cfg.get("with_metrics")
    if raw_metrics is True:
        raw_metrics = {}
    elif raw_metrics in (None, False):
        raw_metrics = {}
    elif not isinstance(raw_metrics, dict):
        raise ValueError("logging.with_metrics must be a boolean or object.")

    metric_sets = raw_metrics.get("sets", {})
    legacy_render_every = int(logging_cfg.get("debug_render_metrics_every", 0) or 0)
    every = int(raw_metrics.get("every", legacy_render_every) or 0)
    renderer = bool(raw_metrics.get("renderer", metric_sets.get("renderer", legacy_render_every > 0)))
    optimizer = bool(raw_metrics.get("optimizer", metric_sets.get("optimizer", False)))
    return MetricConfig(
        renderer=renderer,
        optimizer=optimizer,
        every=every,
        print_summary=bool(raw_metrics.get("print_summary", False)),
        wandb=bool(raw_metrics.get("wandb", True)),
        fail_fast=bool(raw_metrics.get("fail_fast", True)),
    )


def _camera_values(cameras, field_name: str, device) -> torch.Tensor:
    values = [getattr(camera, field_name) for camera in cameras]
    return torch.stack(
        [
            value.to(device=device, dtype=torch.float32)
            if torch.is_tensor(value)
            else torch.tensor(float(value), device=device, dtype=torch.float32)
            for value in values
        ],
        dim=0,
    ).reshape(-1)


def _safe_stat(tensor: torch.Tensor, reducer: str) -> float:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return float("nan")
    if reducer == "min":
        return float(finite.min().detach().cpu())
    if reducer == "max":
        return float(finite.max().detach().cpu())
    if reducer == "mean":
        return float(finite.float().mean().detach().cpu())
    if reducer == "sum":
        return float(finite.float().sum().detach().cpu())
    raise ValueError(f"Unknown reducer: {reducer}")


@torch.no_grad()
def dense_render_diagnostics(config, dense_grid, cameras, decoded, renders=None) -> dict[str, float]:
    model_cfg = config["model"]
    height = model_cfg["size"]
    width = model_cfg["size"]
    device = decoded.xyz.device

    xyz = decoded.xyz.detach().float()
    scales = decoded.scales.detach().float()
    quats = decoded.quats.detach().float()
    opacities = decoded.opacities.detach().float()
    rgbs = decoded.rgbs.detach().float()
    batch_size, gaussian_count, _channels = xyz.shape

    qr, qi, qj, qk = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    rotation = xyz.new_zeros((batch_size, gaussian_count, 3, 3))
    rotation[..., 0, 0] = 1.0 - 2 * (qj**2 + qk**2)
    rotation[..., 0, 1] = 2 * (qi * qj - qr * qk)
    rotation[..., 0, 2] = 2 * (qi * qk + qr * qj)
    rotation[..., 1, 0] = 2 * (qi * qj + qr * qk)
    rotation[..., 1, 1] = 1.0 - 2 * (qi**2 + qk**2)
    rotation[..., 1, 2] = 2 * (qj * qk - qr * qi)
    rotation[..., 2, 0] = 2 * (qi * qk - qr * qj)
    rotation[..., 2, 1] = 2 * (qj * qk + qr * qi)
    rotation[..., 2, 2] = 1.0 - 2 * (qi**2 + qj**2)

    scale_matrix = xyz.new_zeros((batch_size, gaussian_count, 3, 3))
    scale_matrix[..., 0, 0] = scales[..., 0]
    scale_matrix[..., 1, 1] = scales[..., 1]
    scale_matrix[..., 2, 2] = scales[..., 2]
    gaussian_basis = rotation @ scale_matrix
    cov3d = gaussian_basis @ gaussian_basis.transpose(-1, -2)

    camera_to_world = torch.stack(
        [camera.camera_to_world.to(device=device, dtype=torch.float32) for camera in cameras],
        dim=0,
    )
    camera_means, _camera_cov = transform_world_to_camera_batch(xyz, cov3d, camera_to_world)
    render_cfg = config.get("render", {})
    near_plane = float(render_cfg.get("near_plane", 1.0e-4))
    z = camera_means[..., 2]
    front_counts = (z > near_plane).sum(dim=1).float()

    fx = _camera_values(cameras, "fx", device)
    fy = _camera_values(cameras, "fy", device)
    cx = _camera_values(cameras, "cx", device)
    cy = _camera_values(cameras, "cy", device)
    means2d, inv_cov2d, cov2d, projected_opacities, _projected_rgbs = project_gaussians_2d_batch(
        xyz,
        scales,
        quats,
        opacities,
        rgbs,
        fx,
        fy,
        cx,
        cy,
        camera_to_world=camera_to_world,
        near_plane=near_plane,
    )

    raw_det = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
    pixel_work_items = int(batch_size) * int(gaussian_count) * int(height) * int(width)
    pixel_diag_max_work_items = int(
        render_cfg.get("render_diag_pixel_max_work_items", DEFAULT_PIXEL_DIAG_MAX_WORK_ITEMS)
    )

    metrics = {
        "RenderDiag/XYZMin": _safe_stat(xyz, "min"),
        "RenderDiag/XYZMax": _safe_stat(xyz, "max"),
        "RenderDiag/ScaleMin": _safe_stat(scales, "min"),
        "RenderDiag/ScaleMax": _safe_stat(scales, "max"),
        "RenderDiag/CameraZMin": _safe_stat(z, "min"),
        "RenderDiag/CameraZMax": _safe_stat(z, "max"),
        "RenderDiag/NearPlane": near_plane,
        "RenderDiag/FrontGaussiansMin": float(front_counts.min().detach().cpu()),
        "RenderDiag/FrontGaussiansMean": float(front_counts.mean().detach().cpu()),
        "RenderDiag/NearOrBehindGaussiansMean": float((gaussian_count - front_counts).mean().detach().cpu()),
        "RenderDiag/Means2DMin": _safe_stat(means2d, "min"),
        "RenderDiag/Means2DMax": _safe_stat(means2d, "max"),
        "RenderDiag/RawDetMin": _safe_stat(raw_det, "min"),
        "RenderDiag/RawDetMax": _safe_stat(raw_det, "max"),
        "RenderDiag/RawDetNegativeCount": float((raw_det < 0).sum().detach().cpu()),
        "RenderDiag/InvCovMin": _safe_stat(inv_cov2d, "min"),
        "RenderDiag/InvCovMax": _safe_stat(inv_cov2d, "max"),
        "RenderDiag/PixelWorkItems": float(pixel_work_items),
        "RenderDiag/PixelDiagSkipped": float(pixel_work_items > pixel_diag_max_work_items),
    }
    if renders is not None:
        metrics["RenderDiag/RenderNonfiniteCount"] = float((~torch.isfinite(renders.detach())).sum().detach().cpu())

    if pixel_work_items <= pixel_diag_max_work_items:
        grid = dense_grid if dense_grid is not None else build_pixel_grid(height, width, device)
        dx = grid.view(1, 1, height, width, 2) - means2d.view(batch_size, gaussian_count, 1, 1, 2)
        dx0 = dx[..., 0]
        dx1 = dx[..., 1]
        power = -0.5 * (
            inv_cov2d[..., 0, 0].view(batch_size, gaussian_count, 1, 1) * dx0.square()
            + (inv_cov2d[..., 0, 1] + inv_cov2d[..., 1, 0]).view(batch_size, gaussian_count, 1, 1) * dx0 * dx1
            + inv_cov2d[..., 1, 1].view(batch_size, gaussian_count, 1, 1) * dx1.square()
        )
        alpha_pre = projected_opacities.squeeze(-1).view(batch_size, gaussian_count, 1, 1) * torch.exp(power)
        metrics.update(
            {
                "RenderDiag/PowerMin": _safe_stat(power, "min"),
                "RenderDiag/PowerMax": _safe_stat(power, "max"),
                "RenderDiag/PowerGt80Count": float((power > 80).sum().detach().cpu()),
                "RenderDiag/PowerNonfiniteCount": float((~torch.isfinite(power)).sum().detach().cpu()),
                "RenderDiag/AlphaPreNonfiniteCount": float((~torch.isfinite(alpha_pre)).sum().detach().cpu()),
            }
        )
    tile_diag_max_gaussians = int(render_cfg.get("render_diag_tile_max_gaussians", DEFAULT_TILE_DIAG_MAX_GAUSSIANS))
    metrics["TileDiag/Skipped"] = float(gaussian_count > tile_diag_max_gaussians)
    if gaussian_count <= tile_diag_max_gaussians:
        metrics.update(
            tile_overlap_diagnostics(
                config,
                means2d=means2d,
                inv_cov2d=inv_cov2d,
                cov2d=cov2d,
                opacities=projected_opacities,
                image_size=(height, width),
            )
        )
    return metrics


@torch.no_grad()
def tile_overlap_diagnostics(
    config: dict[str, Any],
    *,
    means2d: torch.Tensor,
    inv_cov2d: torch.Tensor,
    cov2d: torch.Tensor,
    opacities: torch.Tensor,
    image_size: tuple[int, int],
) -> dict[str, float]:
    render_cfg = config.get("render", {})
    tile_size = int(render_cfg.get("tile_size", 16))
    alpha_threshold = float(render_cfg.get("alpha_threshold", 1.0 / 255.0))
    bound_scale = float(render_cfg.get("bound_scale", 3.0))
    large_threshold = int(render_cfg.get("large_splat_tile_threshold", 64))
    batch_size = int(render_cfg.get("overlap_metric_batch_size", 8192))
    max_candidate_pairs = int(render_cfg.get("overlap_metric_max_candidate_pairs_per_batch", 1_000_000))

    exact_stats = []
    custom_stats = []
    for frame_idx in range(int(means2d.shape[0])):
        conics = torch.stack(
            [
                inv_cov2d[frame_idx, :, 0, 0],
                inv_cov2d[frame_idx, :, 0, 1],
                inv_cov2d[frame_idx, :, 1, 1],
            ],
            dim=-1,
        )
        exact = exact_conic_overlap_stats(
            means2d[frame_idx],
            conics,
            opacities[frame_idx].squeeze(-1),
            image_size,
            tile_size=tile_size,
            alpha_threshold=alpha_threshold,
            large_splat_tile_threshold=large_threshold,
            batch_size=batch_size,
            max_candidate_pairs_per_batch=max_candidate_pairs,
        )
        custom = custom_rect_overlap_stats(
            means2d[frame_idx],
            cov2d[frame_idx],
            opacities[frame_idx],
            image_size,
            tile_size=tile_size,
            bound_scale=bound_scale,
            alpha_threshold=alpha_threshold,
            large_splat_tile_threshold=large_threshold,
            batch_size=batch_size,
            max_candidate_pairs_per_batch=max_candidate_pairs,
        )
        exact_stats.append(selected_overlap_summary(exact))
        custom_stats.append(selected_overlap_summary(custom))

    metrics = {}
    metrics.update(aggregate_stat_dicts(exact_stats, "TileDiag/ExactConic"))
    metrics.update(aggregate_stat_dicts(custom_stats, "TileDiag/CustomRect"))

    exact_total = sum(stats["total_overlap_keys"] for stats in exact_stats)
    custom_total = sum(stats["total_overlap_keys"] for stats in custom_stats)
    if exact_total > 0:
        metrics["TileDiag/CustomRectToExactKeyRatio"] = float(custom_total / exact_total)
    return metrics


@torch.no_grad()
def render_aux_diagnostics(aux: dict[str, torch.Tensor] | None, support_eps: float = 1e-6) -> dict[str, float]:
    if not aux:
        return {}

    alpha_max = aux.get("alpha_max")
    weight_sum = aux.get("weight_sum")
    if alpha_max is None or weight_sum is None:
        return {}

    alpha_max = alpha_max.detach().float()
    weight_sum = weight_sum.detach().float()
    if alpha_max.ndim == 1:
        alpha_max = alpha_max.unsqueeze(0)
    if weight_sum.ndim == 1:
        weight_sum = weight_sum.unsqueeze(0)

    alpha_finite = torch.isfinite(alpha_max)
    weight_finite = torch.isfinite(weight_sum)
    has_alpha_support = alpha_finite & (alpha_max > support_eps)
    has_contribution = weight_finite & (weight_sum > support_eps)
    alpha_support_counts = has_alpha_support.sum(dim=1).float()
    contribution_counts = has_contribution.sum(dim=1).float()
    no_alpha_slot_fraction = (~has_alpha_support).all(dim=0).float().mean()
    no_contribution_slot_fraction = (~has_contribution).all(dim=0).float().mean()

    return {
        "RenderAux/AlphaSupportGaussiansMin": float(alpha_support_counts.min().detach().cpu()),
        "RenderAux/AlphaSupportGaussiansMean": float(alpha_support_counts.mean().detach().cpu()),
        "RenderAux/ContributingGaussiansMin": float(contribution_counts.min().detach().cpu()),
        "RenderAux/ContributingGaussiansMean": float(contribution_counts.mean().detach().cpu()),
        "RenderAux/NoAlphaSupportSlotFraction": float(no_alpha_slot_fraction.detach().cpu()),
        "RenderAux/NoContributionSlotFraction": float(no_contribution_slot_fraction.detach().cpu()),
        "RenderAux/AlphaMaxMean": _safe_stat(alpha_max, "mean"),
        "RenderAux/AlphaMaxMax": _safe_stat(alpha_max, "max"),
        "RenderAux/AlphaMaxNonfiniteCount": float((~alpha_finite).sum().detach().cpu()),
        "RenderAux/WeightSumMean": _safe_stat(weight_sum, "mean"),
        "RenderAux/WeightSumMax": _safe_stat(weight_sum, "max"),
        "RenderAux/WeightSumNonfiniteCount": float((~weight_finite).sum().detach().cpu()),
    }


@torch.no_grad()
def optimizer_diagnostics(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    grad_sq_sum = 0.0
    param_sq_sum = 0.0
    grad_sq_sum_by_group: dict[str, float] = {}
    param_sq_sum_by_group: dict[str, float] = {}
    grad_abs_max = 0.0
    param_abs_max = 0.0
    grad_nonfinite_count = 0
    grad_nonfinite_param_count = 0
    param_nonfinite_count = 0
    param_count = 0
    grad_param_count = 0

    def group_name(parameter_name: str) -> str:
        if parameter_name == "tokens":
            return "tokens"
        parts = parameter_name.split(".")
        if len(parts) >= 2 and parts[0] == "gaussian_heads":
            return f"gaussian_heads_{parts[1]}"
        return parts[0]

    for name, parameter in model.named_parameters():
        group = group_name(name)
        detached = parameter.detach()
        param_count += detached.numel()
        param_nonfinite_count += int((~torch.isfinite(detached)).sum().detach().cpu())
        finite_param = detached[torch.isfinite(detached)]
        if finite_param.numel() > 0:
            param_sq = float(finite_param.float().square().sum().detach().cpu())
            param_sq_sum += param_sq
            param_sq_sum_by_group[group] = param_sq_sum_by_group.get(group, 0.0) + param_sq
            param_abs_max = max(param_abs_max, float(finite_param.abs().max().detach().cpu()))

        if parameter.grad is None:
            continue
        grad_param_count += 1
        grad = parameter.grad.detach()
        nonfinite_grad = int((~torch.isfinite(grad)).sum().detach().cpu())
        grad_nonfinite_count += nonfinite_grad
        if nonfinite_grad > 0:
            grad_nonfinite_param_count += 1
        finite_grad = grad[torch.isfinite(grad)]
        if finite_grad.numel() > 0:
            grad_sq = float(finite_grad.float().square().sum().detach().cpu())
            grad_sq_sum += grad_sq
            grad_sq_sum_by_group[group] = grad_sq_sum_by_group.get(group, 0.0) + grad_sq
            grad_abs_max = max(grad_abs_max, float(finite_grad.abs().max().detach().cpu()))

    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    payload = {
        "OptDiag/LR": learning_rates[0] if learning_rates else 0.0,
        "OptDiag/GradL2": grad_sq_sum**0.5,
        "OptDiag/GradAbsMax": grad_abs_max,
        "OptDiag/GradNonfiniteCount": float(grad_nonfinite_count),
        "OptDiag/GradNonfiniteParamCount": float(grad_nonfinite_param_count),
        "OptDiag/GradParamCount": float(grad_param_count),
        "OptDiag/ParamL2": param_sq_sum**0.5,
        "OptDiag/ParamAbsMax": param_abs_max,
        "OptDiag/ParamNonfiniteCount": float(param_nonfinite_count),
        "OptDiag/ParamCount": float(param_count),
    }
    for group, value in sorted(grad_sq_sum_by_group.items()):
        payload[f"OptDiag/GradL2ByGroup/{group}"] = value**0.5
    for group, value in sorted(param_sq_sum_by_group.items()):
        payload[f"OptDiag/ParamL2ByGroup/{group}"] = value**0.5
    return payload


def format_metric_summary(metrics: dict[str, float]) -> str:
    keys = (
        "RenderDiag/RenderNonfiniteCount",
        "RenderDiag/CameraZMin",
        "RenderDiag/CameraZMax",
        "RenderDiag/FrontGaussiansMin",
        "RenderDiag/NearOrBehindGaussiansMean",
        "RenderDiag/PixelWorkItems",
        "RenderDiag/PixelDiagSkipped",
        "RenderDiag/PowerMax",
        "RenderDiag/PowerGt80Count",
        "RenderDiag/AlphaPreNonfiniteCount",
        "RenderDiag/RawDetMin",
        "RenderDiag/RawDetNegativeCount",
        "RenderAux/AlphaSupportGaussiansMean",
        "RenderAux/ContributingGaussiansMean",
        "RenderAux/NoAlphaSupportSlotFraction",
        "RenderAux/NoContributionSlotFraction",
        "TileDiag/Skipped",
        "TileDiag/ExactConic_total_overlap_keys_mean",
        "TileDiag/ExactConic_duplication_factor_mean",
        "TileDiag/ExactConic_max_tiles_per_splat_max",
        "TileDiag/ExactConic_max_splats_per_tile_max",
        "TileDiag/CustomRect_total_overlap_keys_mean",
        "TileDiag/CustomRect_duplication_factor_mean",
        "TileDiag/CustomRectToExactKeyRatio",
        "OptDiag/LR",
        "OptDiag/GradL2",
        "OptDiag/GradAbsMax",
        "OptDiag/GradNonfiniteCount",
        "OptDiag/GradNonfiniteParamCount",
        "OptDiag/ParamL2",
        "OptDiag/ParamAbsMax",
        "OptDiag/ParamNonfiniteCount",
    )
    return ", ".join(f"{key}={metrics[key]:.4g}" for key in keys if key in metrics)


def print_metric_summary(
    step: int, metrics: dict[str, float], frame_indices=None, prefix: str = "DebugMetrics"
) -> None:
    frame_suffix = "" if frame_indices is None else f" frames={frame_indices}"
    print(f"{prefix} step={step}{frame_suffix}: {format_metric_summary(metrics)}")
