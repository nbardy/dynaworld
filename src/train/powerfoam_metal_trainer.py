from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from external_paths import ensure_third_party_path
from device_memory import DeviceMemorySampler
from powerfoam_adjacency import build_csr_adjacency, csr_adjacency_stats
from powerfoam_checkpoints import (
    maybe_save_best_powerfoam_checkpoint,
    save_powerfoam_checkpoint,
    select_best_metric,
)
from powerfoam_metal_config import (
    DATA_DEFAULTS,
    HEIGHT_TEXEL_SURFACE_MODES,
    LOGGING_DEFAULTS,
    LOSS_DEFAULTS,
    MODEL_DEFAULTS,
    ORIENTED_TEXEL_SURFACE_MODES,
    QUATERNION_TEXEL_SURFACE_MODES,
    RENDER_DEFAULTS,
    SV_TEXEL_SURFACE_MODES,
    TEXEL_SURFACE_MODES,
    TRAIN_DEFAULTS,
    resolve_config,
)
from powerfoam_raster_config import make_powerfoam_metal_raster_config as make_raster_config
from powerfoam_direct import (
    POWERFOAM_SOFTPLUS_BETA,
    camera_facing_quaternion,
    estimate_knn_radii,
    initialize_full_powerfoam_from_video,
    initialize_powerfoam_from_video,
    inverse_softplus,
    logit_clamped,
)
from powerfoam_eval_artifacts import log_powerfoam_artifacts as log_artifacts
from powerfoam_diagnostics import powerfoam_parameter_delta_metrics
from powerfoam_eval_render import render_powerfoam_samples as render_samples
from powerfoam_geometry import (
    make_pinhole_rays,
    orthonormal_surface_frame,
    powerfoam_rays_from_camera,
    stable_tangent_from_normals,
)
from powerfoam_point_cloud import (
    PointCloudInitialization,
    load_point_cloud_xyz_rgb,
    load_powerfoam_point_cloud_initialization,
)
from powerfoam_optim import (
    cosine_scheduled_lr,
    powerfoam_group_final_lr,
    powerfoam_group_initial_lr,
    powerfoam_group_lr_metadata,
    powerfoam_group_warmup_steps,
    update_powerfoam_learning_rates,
)
from powerfoam_objectives import (
    composite_powerfoam_background,
    fixed_background_tensor,
    normals_from_ray_depth,
    powerfoam_contribution_loss,
    powerfoam_normal_distance_loss,
    powerfoam_normal_map_loss,
    powerfoam_ssim_loss,
    scheduled_loss_weights,
    training_background_tensor,
)
from paper_training_protocol import (
    PaperCostTracker,
    PaperPhaseTimer,
    SpacetimeEpochSampler,
    normalize_image_size,
    normalize_paper_stages,
    paper_stage_for_step,
    resize_ray_grids,
    resize_video_frames,
)
from paper_training_types import MetalKernelSpec
from powerfoam_resampling import scheduled_resample_target_cells, should_resample_powerfoam_step
from powerfoam_training import powerfoam_train_batch_indices
from powerfoam_training_data import load_powerfoam_training_data
from train_artifacts import append_jsonl, write_json, write_resolved_config
from train_devices import resolve_torch_device
from train_logging import (
    log_wandb_run_payload,
    mapped_metric_payload,
    should_log_image,
    should_log_scalar,
    should_log_video,
    wandb_run_lifecycle,
)

POWERFOAM_METAL_ROOT = ensure_third_party_path("powerfoam-metal")

from torch_powerfoam_metal import (  # noqa: E402
    FoamRasterConfig,
    quaternion_frames,
    raytrace_power_foam_oriented_height_sv_texel_surface,
    rasterize_power_foam,
    rasterize_power_foam_linear,
    rasterize_power_foam_oriented_height_sv_texel_surface,
    rasterize_power_foam_oriented_height_sv_texel_surface_aux,
    rasterize_power_foam_oriented_height_texel_surface,
    rasterize_power_foam_oriented_surface_linear,
    rasterize_power_foam_oriented_texel_surface,
    rasterize_power_foam_quaternion_height_sv_texel_surface,
    rasterize_power_foam_quaternion_height_sv_texel_surface_aux,
    rasterize_power_foam_quaternion_height_texel_surface,
    rasterize_power_foam_quaternion_texel_surface,
    rasterize_power_foam_surface_linear,
)


@dataclass(frozen=True)
class PowerFoamAuxBatch:
    contrib: torch.Tensor
    point_error: torch.Tensor
    visible_mask: torch.Tensor
    normal_distance: torch.Tensor
    normal_norm: torch.Tensor
    median_depth: torch.Tensor


POWERFOAM_METAL_TRAIN_WANDB_KEYS = (
    ("loss", "Train/Loss"),
    ("l1", "Train/L1"),
    ("mse", "Train/MSE"),
    ("ssim_loss", "Train/SSIMLoss"),
    ("normal_loss", "Train/NormalLoss"),
    ("normal_weight", "Train/NormalWeight"),
    ("normal_map_loss", "Train/NormalMapLoss"),
    ("normal_map_weight", "Train/NormalMapWeight"),
    ("normal_map_valid_fraction", "Train/NormalMapValidFraction"),
    ("contribution_loss", "Train/ContributionLoss"),
    ("contribution_weight", "Train/ContributionWeight"),
    ("interpenetration_loss", "Train/InterpenetrationLoss"),
    ("interpenetration_weight", "Train/InterpenetrationWeight"),
    ("elapsed_s", "Timing/ElapsedSeconds"),
)


def rays_for_sample_batch(
    rays: torch.Tensor | None,
    *,
    sample_index: int,
    batch_size: int,
) -> torch.Tensor | None:
    if rays is None:
        return None
    if rays.ndim != 4:
        raise ValueError(f"Expected rays [B,H,W,6], got {tuple(rays.shape)}.")
    if rays.shape[-1] != 6:
        raise ValueError(f"Expected ray payload dimension 6, got {rays.shape[-1]}.")
    if rays.shape[0] == 1:
        return rays
    if rays.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} ray batches or one shared ray batch, got {rays.shape[0]}.")
    return rays[sample_index : sample_index + 1].contiguous()


class MetalPowerFoamVideo(nn.Module):
    def __init__(
        self,
        *,
        frame_count: int,
        cell_count: int,
        render_size: int,
        fov_degrees: float,
        neighbor_count: int,
        adjacency_mode: str,
        xy_extent: float,
        z_min: float,
        z_max: float,
        radius_init: float,
        radius_min: float,
        radius_scale: float,
        density_init: float,
        feature_mode: str,
        linear_coeff_init: float,
        linear_coeff_scale: float,
        normal_init_jitter: float,
        num_texel_sites: int,
        texel_site_scale: float,
        texel_height_scale: float,
        sv_dof: int,
        sv_axis_init: float,
        sv_axis_init_jitter: float,
        sv_rgb_init_jitter: float,
        color_init_mode: str,
        seed: int,
        init_frames: torch.Tensor | None,
        init_points: torch.Tensor | None,
        init_colors: torch.Tensor | None,
        image_init_depth: float | None,
        image_init_jitter: float,
        raster_config: FoamRasterConfig,
        use_raytrace: bool = False,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        texel_sites_init = None
        texel_colors_init = None
        texel_heights_init = None
        texel_sv_axis_init = None
        texel_sv_rgb_init = None
        quaternions_init = None
        texel_surface_modes = TEXEL_SURFACE_MODES
        height_texel_surface_modes = HEIGHT_TEXEL_SURFACE_MODES
        quaternion_texel_surface_modes = QUATERNION_TEXEL_SURFACE_MODES
        oriented_texel_surface_modes = ORIENTED_TEXEL_SURFACE_MODES
        sv_texel_surface_modes = SV_TEXEL_SURFACE_MODES
        if init_points is not None:
            if init_points.shape != (frame_count, cell_count, 3):
                raise ValueError(
                    f"init_points must have shape {(frame_count, cell_count, 3)}, got {tuple(init_points.shape)}."
                )
            init_points = init_points.to(dtype=torch.float32)
            if init_colors is None:
                init_colors = torch.full((frame_count, cell_count, 3), 0.5, dtype=init_points.dtype)
            if init_colors.shape != (frame_count, cell_count, 3):
                raise ValueError(
                    f"init_colors must have shape {(frame_count, cell_count, 3)}, got {tuple(init_colors.shape)}."
                )
            init_colors = init_colors.to(dtype=init_points.dtype).clamp(0.0, 1.0)
            init_radii = estimate_knn_radii(init_points, radius_scale=radius_scale, radius_min=radius_min)
            quaternions_init = camera_facing_quaternion(frame_count, cell_count)
        elif init_frames is not None and str(feature_mode) in texel_surface_modes:
            init = initialize_full_powerfoam_from_video(
                init_frames,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                fov_degrees=fov_degrees,
                image_init_depth=image_init_depth,
                radius_min=radius_min,
                radius_scale=radius_scale,
                num_texel_sites=int(num_texel_sites),
                sv_dof=int(sv_dof) if str(feature_mode) in sv_texel_surface_modes else 1,
                sv_axis_init=float(sv_axis_init),
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            quaternions_init = init.quaternions
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_sv_axis_init = init.texel_sv_axis
            texel_sv_rgb_init = init.texel_sv_rgb
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
            texel_heights_init = init.texel_height
            init_colors = texel_colors_init.mean(dim=2)
        elif init_frames is not None:
            init_points, init_colors = initialize_powerfoam_from_video(
                init_frames,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                fov_degrees=fov_degrees,
                image_init_depth=image_init_depth,
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_radii = estimate_knn_radii(init_points, radius_scale=radius_scale, radius_min=radius_min)
        else:
            xy = (torch.rand(frame_count, cell_count, 2, generator=generator) * 2.0 - 1.0) * float(xy_extent)
            z = torch.rand(frame_count, cell_count, 1, generator=generator) * (float(z_max) - float(z_min)) + float(z_min)
            init_points = torch.cat([xy, z], dim=-1)
            init_colors = torch.rand(frame_count, cell_count, 3, generator=generator)
            init_radii = torch.full((frame_count, cell_count), max(float(radius_init), float(radius_min)))
            quaternions_init = camera_facing_quaternion(frame_count, cell_count)
        if str(color_init_mode) == "random":
            init_colors = torch.rand(frame_count, cell_count, 3, generator=generator, dtype=init_points.dtype)
            if str(feature_mode) in sv_texel_surface_modes:
                texel_sv_axis_init = float(sv_axis_init) * F.normalize(
                    torch.randn(
                        frame_count,
                        cell_count,
                        int(num_texel_sites),
                        int(sv_dof),
                        3,
                        generator=generator,
                        dtype=init_points.dtype,
                    ),
                    dim=-1,
                    eps=1.0e-6,
                )
                texel_sv_rgb_init = (
                    torch.rand(
                        frame_count,
                        cell_count,
                        int(num_texel_sites),
                        int(sv_dof),
                        3,
                        generator=generator,
                        dtype=init_points.dtype,
                    )
                    - 0.5
                )
            elif str(feature_mode) in texel_surface_modes:
                texel_colors_init = torch.rand(
                    frame_count,
                    cell_count,
                    int(num_texel_sites),
                    3,
                    generator=generator,
                    dtype=init_points.dtype,
                )

        self.raw_xy = nn.Parameter(torch.atanh((init_points[..., :2] / float(xy_extent)).clamp(-0.9999, 0.9999)))
        self.raw_z = nn.Parameter(logit_clamped((init_points[..., 2:] - float(z_min)) / (float(z_max) - float(z_min))))
        self.raw_radii = nn.Parameter(
            inverse_softplus(
                (init_radii - float(radius_min)).clamp_min(1.0e-4),
                beta=POWERFOAM_SOFTPLUS_BETA,
            )
        )
        init_density = torch.full((frame_count, cell_count), max(float(density_init), 1.0e-4))
        self.raw_densities = nn.Parameter(inverse_softplus(init_density, beta=POWERFOAM_SOFTPLUS_BETA))
        self.feature_mode = str(feature_mode)
        self.linear_coeff_scale = float(linear_coeff_scale)
        self.texel_site_scale = float(texel_site_scale)
        self.texel_height_scale = float(texel_height_scale)
        if self.feature_mode in texel_surface_modes:
            if texel_sites_init is None:
                site_cols = math.ceil(math.sqrt(float(num_texel_sites)))
                site_rows = math.ceil(float(num_texel_sites) / float(site_cols))
                xs = (torch.arange(site_cols, dtype=init_colors.dtype) + 0.5) / float(site_cols) - 0.5
                ys = (torch.arange(site_rows, dtype=init_colors.dtype) + 0.5) / float(site_rows) - 0.5
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[: int(num_texel_sites)]
                texel_sites_init = grid.view(1, 1, int(num_texel_sites), 2).repeat(frame_count, cell_count, 1, 1)
                texel_colors_init = init_colors[:, :, None, :].repeat(1, 1, int(num_texel_sites), 1)
            if texel_heights_init is None:
                texel_heights_init = torch.zeros(
                    frame_count,
                    cell_count,
                    int(num_texel_sites),
                    dtype=init_colors.dtype,
                )
            self.raw_texel_sites = nn.Parameter(
                torch.atanh((texel_sites_init / self.texel_site_scale).clamp(-0.9999, 0.9999))
            )
            if self.feature_mode in sv_texel_surface_modes:
                if texel_sv_axis_init is None:
                    texel_sv_axis_init = float(sv_axis_init) * F.normalize(
                        torch.randn(
                            frame_count,
                            cell_count,
                            int(num_texel_sites),
                            int(sv_dof),
                            3,
                            generator=generator,
                            dtype=init_colors.dtype,
                        ),
                        dim=-1,
                        eps=1.0e-6,
                    )
                if texel_sv_rgb_init is None:
                    texel_sv_rgb_init = init_colors[:, :, None, None, :].repeat(
                        1,
                        1,
                        int(num_texel_sites),
                        int(sv_dof),
                        1,
                    ) - 0.5
                if float(sv_axis_init_jitter) != 0.0:
                    texel_sv_axis_init = texel_sv_axis_init + float(sv_axis_init_jitter) * torch.randn(
                        texel_sv_axis_init.shape,
                        generator=generator,
                        dtype=texel_sv_axis_init.dtype,
                    )
                if float(sv_rgb_init_jitter) != 0.0:
                    texel_sv_rgb_init = texel_sv_rgb_init + float(sv_rgb_init_jitter) * torch.randn(
                        texel_sv_rgb_init.shape,
                        generator=generator,
                        dtype=texel_sv_rgb_init.dtype,
                    )
                self.raw_texel_sv_axis = nn.Parameter(texel_sv_axis_init)
                self.raw_texel_sv_rgb = nn.Parameter(texel_sv_rgb_init)
                init_decoded_features = (texel_sv_rgb_init.mean(dim=3) + 0.5).clamp_min(0.0)
                self.raw_features = None
            else:
                init_decoded_features = texel_colors_init.clamp(0.0, 1.0)
                self.raw_features = nn.Parameter(logit_clamped(init_decoded_features))
                self.raw_texel_sv_axis = None
                self.raw_texel_sv_rgb = None
            if self.feature_mode in height_texel_surface_modes:
                self.raw_texel_heights = nn.Parameter(
                    torch.atanh((texel_heights_init / self.texel_height_scale).clamp(-0.9999, 0.9999))
                )
            else:
                self.raw_texel_heights = None
        elif self.feature_mode in {"linear", "surface_linear", "oriented_surface_linear"}:
            raw_features = torch.zeros(frame_count, cell_count, 3, 4, dtype=init_colors.dtype)
            raw_features[..., 0] = logit_clamped(init_colors.clamp(0.0, 1.0))
            if float(linear_coeff_init) != 0.0:
                raw_features[..., 1:] = float(linear_coeff_init) * torch.randn(
                    frame_count,
                    cell_count,
                    3,
                    3,
                    generator=generator,
                    dtype=init_colors.dtype,
                )
            init_decoded_features = torch.zeros_like(raw_features)
            init_decoded_features[..., 0] = init_colors.clamp(0.0, 1.0)
            self.raw_features = nn.Parameter(raw_features)
        else:
            init_decoded_features = init_colors.clamp(0.0, 1.0)
            self.raw_features = nn.Parameter(logit_clamped(init_decoded_features))
        if self.feature_mode not in texel_surface_modes:
            self.raw_texel_sites = None
            texel_sites_init = torch.zeros(frame_count, cell_count, int(num_texel_sites), 2, dtype=init_colors.dtype)
            texel_heights_init = torch.zeros(frame_count, cell_count, int(num_texel_sites), dtype=init_colors.dtype)
            self.raw_texel_heights = None
            self.raw_texel_sv_axis = None
            self.raw_texel_sv_rgb = None
        if self.feature_mode in quaternion_texel_surface_modes:
            if quaternions_init is None:
                quaternions_init = camera_facing_quaternion(frame_count, cell_count)
            self.raw_quaternions = nn.Parameter(quaternions_init)
            init_normals, init_tangents, _init_bitangents = quaternion_frames(quaternions_init, eps=1.0e-6)
            self.raw_normals = None
        else:
            self.raw_quaternions = None
        if self.feature_mode in {"oriented_surface_linear"} | oriented_texel_surface_modes:
            init_normals = torch.zeros(frame_count, cell_count, 3, dtype=init_colors.dtype)
            init_normals[..., 2] = -1.0
            if float(normal_init_jitter) != 0.0:
                init_normals = init_normals + float(normal_init_jitter) * torch.randn(
                    frame_count,
                    cell_count,
                    3,
                    generator=generator,
                    dtype=init_colors.dtype,
                )
            init_normals = F.normalize(init_normals, dim=-1, eps=1.0e-6)
            self.raw_normals = nn.Parameter(init_normals)
        else:
            if self.feature_mode not in quaternion_texel_surface_modes:
                init_normals = torch.zeros(frame_count, cell_count, 3, dtype=init_colors.dtype)
                self.raw_normals = None
        if self.feature_mode in oriented_texel_surface_modes:
            init_tangents = stable_tangent_from_normals(init_normals)
            self.raw_tangents = nn.Parameter(init_tangents)
        else:
            if self.feature_mode not in quaternion_texel_surface_modes:
                init_tangents = torch.zeros(frame_count, cell_count, 3, dtype=init_colors.dtype)
            self.raw_tangents = None
        self.register_buffer("initial_points", init_points.clone(), persistent=False)
        self.register_buffer("initial_radii", init_radii.clone(), persistent=False)
        self.register_buffer("initial_densities", init_density.clone(), persistent=False)
        self.register_buffer("initial_features", init_decoded_features.clone(), persistent=False)
        self.register_buffer("initial_normals", init_normals.clone(), persistent=False)
        self.register_buffer("initial_tangents", init_tangents.clone(), persistent=False)
        self.register_buffer("initial_texel_sites", texel_sites_init.clone(), persistent=False)
        self.register_buffer(
            "initial_texel_heights",
            (texel_heights_init * init_radii[:, :, None] * self.texel_height_scale).clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_texel_sv_axis",
            texel_sv_axis_init.clone()
            if texel_sv_axis_init is not None
            else torch.zeros(frame_count, cell_count, int(num_texel_sites), int(sv_dof), 3, dtype=init_colors.dtype),
            persistent=False,
        )
        self.register_buffer(
            "initial_texel_sv_rgb",
            texel_sv_rgb_init.clone()
            if texel_sv_rgb_init is not None
            else torch.zeros(frame_count, cell_count, int(num_texel_sites), int(sv_dof), 3, dtype=init_colors.dtype),
            persistent=False,
        )
        self.register_buffer(
            "initial_quaternions",
            quaternions_init.clone()
            if quaternions_init is not None
            else torch.zeros(frame_count, cell_count, 4, dtype=init_colors.dtype),
            persistent=False,
        )

        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.radius_min = float(radius_min)
        self.neighbor_count = int(neighbor_count)
        self.adjacency_mode = str(adjacency_mode)
        self.raster_config = raster_config
        self.use_raytrace = bool(use_raytrace)
        self.register_buffer("rays", make_pinhole_rays(render_size, render_size, fov_degrees, torch.device("cpu")), persistent=False)
        self.register_buffer("contrib_ema", torch.full((frame_count, cell_count), 1.0e-5), persistent=True)
        self.register_buffer("point_error_ema", torch.full((frame_count, cell_count), 1.0e-5), persistent=True)

    def decoded_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xy = torch.tanh(self.raw_xy) * self.xy_extent
        z = self.z_min + torch.sigmoid(self.raw_z) * (self.z_max - self.z_min)
        points = torch.cat([xy, z], dim=-1)
        radii = F.softplus(self.raw_radii, beta=POWERFOAM_SOFTPLUS_BETA) + self.radius_min
        densities = F.softplus(self.raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)
        if self.feature_mode in SV_TEXEL_SURFACE_MODES:
            if self.raw_texel_sv_rgb is None:
                raise RuntimeError("SV texel mode requires raw_texel_sv_rgb")
            features = (self.raw_texel_sv_rgb.mean(dim=3) + 0.5).clamp_min(0.0)
        elif self.feature_mode in TEXEL_SURFACE_MODES:
            features = torch.sigmoid(self.raw_features)
        elif self.feature_mode in {"linear", "surface_linear", "oriented_surface_linear"}:
            base = torch.sigmoid(self.raw_features[..., 0])
            coeffs = self.linear_coeff_scale * torch.tanh(self.raw_features[..., 1:])
            features = torch.cat([base[..., None], coeffs], dim=-1)
        else:
            features = torch.sigmoid(self.raw_features)
        normals = None
        if self.raw_normals is not None:
            normals = F.normalize(self.raw_normals, dim=-1, eps=1.0e-6)
        elif self.raw_quaternions is not None:
            normals, _tangents, _bitangents = quaternion_frames(self.raw_quaternions, eps=1.0e-6)
        return points, radii, densities, features, normals

    def decoded_surface_frame(self, normals: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if normals is None or self.raw_tangents is None:
            return None, None
        return orthonormal_surface_frame(normals, self.raw_tangents)

    def decoded_texel_sites(self) -> torch.Tensor | None:
        if self.raw_texel_sites is None:
            return None
        return self.texel_site_scale * torch.tanh(self.raw_texel_sites)

    def decoded_texel_heights(self, radii: torch.Tensor) -> torch.Tensor | None:
        if self.raw_texel_heights is None:
            return None
        return radii[..., None] * self.texel_height_scale * torch.tanh(self.raw_texel_heights)

    def decoded_texel_sv(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.raw_texel_sv_axis is None or self.raw_texel_sv_rgb is None:
            return None, None
        return self.raw_texel_sv_axis, self.raw_texel_sv_rgb

    def optimizer_param_groups(self, train_cfg: dict[str, Any]) -> list[dict[str, object]]:
        def group(params: list[nn.Parameter], name: str) -> dict[str, object]:
            return {"params": params, "name": name, **powerfoam_group_lr_metadata(train_cfg, name)}

        groups: list[dict[str, object]] = [
            group([self.raw_xy, self.raw_z], "points"),
            group([self.raw_radii], "radii"),
            group([self.raw_densities], "density"),
        ]
        if self.raw_features is not None:
            groups.append(group([self.raw_features], "features"))
        if self.raw_normals is not None:
            groups.append(group([self.raw_normals], "normals"))
        if self.raw_tangents is not None:
            groups.append(group([self.raw_tangents], "tangents"))
        if self.raw_quaternions is not None:
            groups.append(group([self.raw_quaternions], "quaternions"))
        if self.raw_texel_sites is not None:
            groups.append(group([self.raw_texel_sites], "texel_sites"))
        if self.raw_texel_heights is not None:
            groups.append(group([self.raw_texel_heights], "texel_height"))
        if self.raw_texel_sv_axis is not None:
            groups.append(group([self.raw_texel_sv_axis], "texel_sv_axis"))
        if self.raw_texel_sv_rgb is not None:
            groups.append(group([self.raw_texel_sv_rgb], "texel_sv_rgb"))
        return groups

    @torch.no_grad()
    def parameter_drift_metrics(self) -> dict[str, float]:
        points, radii, densities, features, normals = self.decoded_parameters()
        metrics = powerfoam_parameter_delta_metrics(
            points=points,
            initial_points=self.initial_points,
            radii=radii,
            initial_radii=self.initial_radii,
            densities=densities,
            initial_densities=self.initial_densities,
            features=features,
            initial_features=self.initial_features,
            normals=normals,
            initial_normals=self.initial_normals,
            include_cell_count=True,
        )
        if normals is not None:
            metrics["state_mean_normal_z"] = float(normals[..., 2].mean().cpu())
            tangents, _bitangents = self.decoded_surface_frame(normals)
            if tangents is not None:
                metrics["state_mean_tangent_delta"] = float(
                    (tangents - self.initial_tangents.to(tangents.device)).norm(dim=-1).mean().cpu()
                )
        if self.raw_quaternions is not None:
            metrics["state_mean_quaternion_delta"] = float(
                (self.raw_quaternions - self.initial_quaternions.to(self.raw_quaternions.device))
                .norm(dim=-1)
                .mean()
                .cpu()
            )
        texel_sites = self.decoded_texel_sites()
        if texel_sites is not None:
            metrics["state_mean_texel_site_delta"] = float(
                (texel_sites - self.initial_texel_sites.to(texel_sites.device)).norm(dim=-1).mean().cpu()
            )
        texel_heights = self.decoded_texel_heights(radii)
        if texel_heights is not None:
            metrics["state_mean_texel_height_delta"] = float(
                (texel_heights - self.initial_texel_heights.to(texel_heights.device)).abs().mean().cpu()
            )
        if self.raw_texel_sv_axis is not None:
            metrics["state_mean_texel_sv_axis_delta"] = float(
                (self.raw_texel_sv_axis - self.initial_texel_sv_axis.to(self.raw_texel_sv_axis.device))
                .norm(dim=-1)
                .mean()
                .cpu()
            )
        if self.raw_texel_sv_rgb is not None:
            metrics["state_mean_texel_sv_rgb_delta"] = float(
                (self.raw_texel_sv_rgb - self.initial_texel_sv_rgb.to(self.raw_texel_sv_rgb.device))
                .abs()
                .mean()
                .cpu()
            )
        return metrics

    def forward(
        self,
        frame_indices: torch.Tensor,
        rays: torch.Tensor | None = None,
        *,
        return_normal_distance: bool = False,
        return_rendered_normal: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if next(self.parameters()).device.type != "mps":
            raise RuntimeError("MetalPowerFoamVideo requires an MPS device")
        points, radii, densities, features, normals = self.decoded_parameters()
        frame_indices = frame_indices.to(device=points.device, dtype=torch.long)
        default_rays = self.rays.to(device=points.device, dtype=points.dtype)
        provided_rays = None if rays is None else rays.to(device=points.device, dtype=points.dtype)
        renders = []
        alphas = []
        normal_distances = []
        rendered_normals = []
        for sample_index, frame_index in enumerate(frame_indices):
            point = points[frame_index]
            radius = radii[frame_index]
            sample_rays = rays_for_sample_batch(
                provided_rays,
                sample_index=sample_index,
                batch_size=int(frame_indices.numel()),
            )
            if sample_rays is None:
                sample_rays = default_rays
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            normal_distance = None
            if self.feature_mode == "linear":
                out, alpha = rasterize_power_foam_linear(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "surface_linear":
                out, alpha = rasterize_power_foam_surface_linear(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "oriented_surface_linear":
                if normals is None:
                    raise RuntimeError("oriented_surface_linear requires decoded normals")
                out, alpha = rasterize_power_foam_oriented_surface_linear(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    normals[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "oriented_texel_surface":
                texel_sites = self.decoded_texel_sites()
                if normals is None or texel_sites is None:
                    raise RuntimeError("oriented_texel_surface requires decoded normals and texel sites")
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    raise RuntimeError("oriented_texel_surface requires decoded surface frame")
                out, alpha = rasterize_power_foam_oriented_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    features[frame_index],
                    normals[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                    tangents=tangents[frame_index],
                    bitangents=bitangents[frame_index],
                )
            elif self.feature_mode == "oriented_height_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                if normals is None or texel_sites is None or texel_heights is None:
                    raise RuntimeError("oriented_height_texel_surface requires decoded normals, texel sites, and heights")
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    raise RuntimeError("oriented_height_texel_surface requires decoded surface frame")
                out, alpha = rasterize_power_foam_oriented_height_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    texel_heights[frame_index],
                    features[frame_index],
                    normals[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                    tangents=tangents[frame_index],
                    bitangents=bitangents[frame_index],
                )
            elif self.feature_mode == "oriented_height_sv_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                texel_sv_axis, texel_sv_rgb = self.decoded_texel_sv()
                if normals is None or texel_sites is None or texel_heights is None:
                    raise RuntimeError("oriented_height_sv_texel_surface requires decoded normals, texel sites, and heights")
                if texel_sv_axis is None or texel_sv_rgb is None:
                    raise RuntimeError("oriented_height_sv_texel_surface requires decoded SV color parameters")
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    raise RuntimeError("oriented_height_sv_texel_surface requires decoded surface frame")
                if self.use_raytrace:
                    result = raytrace_power_foam_oriented_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        normals[frame_index],
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        tangents=tangents[frame_index],
                        bitangents=bitangents[frame_index],
                        return_normal_distance=return_normal_distance,
                        return_normal=return_rendered_normal,
                    )
                else:
                    if return_rendered_normal:
                        raise RuntimeError("return_rendered_normal is currently implemented for raytrace height+SV")
                    result = rasterize_power_foam_oriented_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        normals[frame_index],
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        tangents=tangents[frame_index],
                        bitangents=bitangents[frame_index],
                        return_normal_distance=return_normal_distance,
                    )
                if return_normal_distance:
                    out, alpha, normal_distance, *rest = result
                else:
                    out, alpha, *rest = result
                if return_rendered_normal:
                    rendered_normals.append(rest[0])
            elif self.feature_mode == "quaternion_texel_surface":
                texel_sites = self.decoded_texel_sites()
                if texel_sites is None or self.raw_quaternions is None:
                    raise RuntimeError("quaternion_texel_surface requires decoded texel sites and quaternions")
                out, alpha = rasterize_power_foam_quaternion_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    features[frame_index],
                    self.raw_quaternions[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "quaternion_height_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                if texel_sites is None or texel_heights is None or self.raw_quaternions is None:
                    raise RuntimeError("quaternion_height_texel_surface requires decoded texel sites, heights, and quaternions")
                out, alpha = rasterize_power_foam_quaternion_height_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    texel_heights[frame_index],
                    features[frame_index],
                    self.raw_quaternions[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "quaternion_height_sv_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                texel_sv_axis, texel_sv_rgb = self.decoded_texel_sv()
                if texel_sites is None or texel_heights is None or self.raw_quaternions is None:
                    raise RuntimeError("quaternion_height_sv_texel_surface requires decoded texel sites, heights, and quaternions")
                if texel_sv_axis is None or texel_sv_rgb is None:
                    raise RuntimeError("quaternion_height_sv_texel_surface requires decoded SV color parameters")
                if self.use_raytrace:
                    frame_normals, frame_tangents, frame_bitangents = quaternion_frames(
                        self.raw_quaternions[frame_index], eps=self.raster_config.eps
                    )
                    result = raytrace_power_foam_oriented_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        frame_normals,
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        tangents=frame_tangents,
                        bitangents=frame_bitangents,
                        return_normal_distance=return_normal_distance,
                        return_normal=return_rendered_normal,
                    )
                else:
                    if return_rendered_normal:
                        raise RuntimeError("return_rendered_normal is currently implemented for raytrace height+SV")
                    result = rasterize_power_foam_quaternion_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        self.raw_quaternions[frame_index],
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        return_normal_distance=return_normal_distance,
                    )
                if return_normal_distance:
                    out, alpha, normal_distance, *rest = result
                else:
                    out, alpha, *rest = result
                if return_rendered_normal:
                    rendered_normals.append(rest[0])
            else:
                out, alpha = rasterize_power_foam(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            renders.append(out.permute(0, 3, 1, 2))
            alphas.append(alpha)
            if return_normal_distance:
                if normal_distance is None:
                    raise RuntimeError("normal_distance output requires a height+SV PowerFoam feature mode")
                normal_distances.append(normal_distance)
        result = (torch.cat(renders, dim=0), torch.cat(alphas, dim=0))
        if return_normal_distance:
            result = (*result, torch.cat(normal_distances, dim=0))
        if return_rendered_normal:
            if len(rendered_normals) != int(frame_indices.numel()):
                raise RuntimeError("rendered normal output requires raytrace height+SV PowerFoam feature mode")
            result = (*result, torch.cat(rendered_normals, dim=0))
        return result

    @torch.no_grad()
    def adjacency_diagnostics(self, frame_index: int = 0) -> dict[str, float]:
        points, radii, _densities, _features, _normals = self.decoded_parameters()
        frame = int(frame_index) % int(points.shape[0])
        adjacency, offsets = build_csr_adjacency(
            points[frame],
            radii[frame],
            neighbor_count=self.neighbor_count,
            mode=self.adjacency_mode,
        )
        return csr_adjacency_stats(points[frame], radii[frame], adjacency, offsets)

    def interpenetration_loss(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        points, radii, _densities, _features, _normals = self.decoded_parameters()
        if frame_indices is None:
            frame_ids = list(range(int(points.shape[0])))
        else:
            frame_ids = sorted({int(v) for v in frame_indices.detach().cpu().flatten().tolist()})
        if not frame_ids:
            return points.sum() * 0.0 + radii.sum() * 0.0

        losses = []
        for frame in frame_ids:
            point = points[frame]
            radius = radii[frame]
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            if adjacency.numel() == 0:
                losses.append(point.sum() * 0.0 + radius.sum() * 0.0)
                continue
            degrees = (offsets[1:] - offsets[:-1]).to(dtype=torch.long)
            src = torch.repeat_interleave(torch.arange(point.shape[0], device=point.device), degrees)
            dst = adjacency.to(device=point.device, dtype=torch.long)
            distances = torch.linalg.vector_norm(point[src] - point[dst], dim=-1)
            overlap = (radius[src] + radius[dst] - distances).clamp_min(0.0)
            losses.append(overlap.square().sum())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def height_sv_aux_batch(
        self,
        frame_indices: torch.Tensor,
        targets: torch.Tensor | None = None,
        rays: torch.Tensor | None = None,
    ) -> PowerFoamAuxBatch | None:
        if self.feature_mode not in {"oriented_height_sv_texel_surface", "quaternion_height_sv_texel_surface"}:
            return None
        points, radii, densities, _features, normals = self.decoded_parameters()
        texel_sites = self.decoded_texel_sites()
        texel_heights = self.decoded_texel_heights(radii)
        texel_sv_axis, texel_sv_rgb = self.decoded_texel_sv()
        if texel_sites is None or texel_heights is None or texel_sv_axis is None or texel_sv_rgb is None:
            return None
        if normals is None and self.raw_quaternions is None:
            return None

        frame_indices = frame_indices.to(device=points.device, dtype=torch.long)
        default_rays = self.rays.to(device=points.device, dtype=points.dtype)
        provided_rays = None if rays is None else rays.to(device=points.device, dtype=points.dtype)
        targets = None if targets is None else targets.to(device=points.device, dtype=points.dtype)
        contribs = []
        point_errors = []
        visible = []
        normal_distance = []
        normal_norm = []
        median_depth = []
        for sample_index, frame_index in enumerate(frame_indices):
            frame = int(frame_index.detach().cpu())
            point = points[frame]
            radius = radii[frame]
            sample_rays = rays_for_sample_batch(
                provided_rays,
                sample_index=sample_index,
                batch_size=int(frame_indices.numel()),
            )
            if sample_rays is None:
                sample_rays = default_rays
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            target = None
            if targets is not None:
                target = targets[sample_index : sample_index + 1] if targets.shape[0] == int(frame_indices.numel()) else targets[frame : frame + 1]
            if self.feature_mode == "quaternion_height_sv_texel_surface":
                if self.raw_quaternions is None:
                    return None
                aux = rasterize_power_foam_quaternion_height_sv_texel_surface_aux(
                    point,
                    radius,
                    densities[frame],
                    texel_sites[frame],
                    texel_heights[frame],
                    texel_sv_axis[frame],
                    texel_sv_rgb[frame],
                    self.raw_quaternions[frame],
                    adjacency,
                    offsets,
                    sample_rays,
                    target,
                    self.raster_config,
                )
            else:
                if normals is None:
                    return None
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    return None
                aux = rasterize_power_foam_oriented_height_sv_texel_surface_aux(
                    point,
                    radius,
                    densities[frame],
                    texel_sites[frame],
                    texel_heights[frame],
                    texel_sv_axis[frame],
                    texel_sv_rgb[frame],
                    normals[frame],
                    adjacency,
                    offsets,
                    sample_rays,
                    target,
                    self.raster_config,
                    tangents=tangents[frame],
                    bitangents=bitangents[frame],
                )
            contribs.append(aux.contrib)
            point_errors.append(aux.point_error)
            visible.append(aux.visible_mask.to(dtype=points.dtype))
            normal_distance.append(aux.normal_distance)
            normal_norm.append(aux.normal.norm(dim=1))
            median_depth.append(aux.median_depth)

        return PowerFoamAuxBatch(
            contrib=torch.cat(contribs, dim=0),
            point_error=torch.cat(point_errors, dim=0),
            visible_mask=torch.cat(visible, dim=0),
            normal_distance=torch.cat(normal_distance, dim=0),
            normal_norm=torch.cat(normal_norm, dim=0),
            median_depth=torch.cat(median_depth, dim=0),
        )

    @torch.no_grad()
    def aux_metrics(
        self,
        frame_indices: torch.Tensor,
        targets: torch.Tensor,
        rays: torch.Tensor | None = None,
    ) -> dict[str, float]:
        aux = self.height_sv_aux_batch(frame_indices, targets, rays)
        if aux is None:
            return {}
        contrib = aux.contrib
        point_error = aux.point_error
        visible_tensor = aux.visible_mask.to(dtype=aux.normal_distance.dtype)
        normal_distance_tensor = aux.normal_distance
        normal_norm_tensor = aux.normal_norm
        median_depth_tensor = aux.median_depth
        ema_frames = frame_indices.to(device=aux.normal_distance.device, dtype=torch.long)
        for row, frame in enumerate(ema_frames):
            visible_alpha = 0.99 * visible_tensor[row] + (1.0 - visible_tensor[row])
            self.contrib_ema[frame] = visible_alpha * self.contrib_ema[frame] + (1.0 - visible_alpha) * contrib[row]
            self.point_error_ema[frame] = 0.99 * self.point_error_ema[frame] + 0.01 * point_error[row]
        valid_depth = median_depth_tensor >= 0.0
        metrics = {
            "aux_mean_contrib": float(contrib.mean().detach().cpu()),
            "aux_max_contrib": float(contrib.max().detach().cpu()),
            "aux_mean_point_error": float(point_error.mean().detach().cpu()),
            "aux_max_point_error": float(point_error.max().detach().cpu()),
            "aux_mean_contrib_ema": float(self.contrib_ema[ema_frames].mean().detach().cpu()),
            "aux_mean_point_error_ema": float(self.point_error_ema[ema_frames].mean().detach().cpu()),
            "aux_visible_fraction": float(visible_tensor.mean().detach().cpu()),
            "aux_visible_cell_frame_events": float(visible_tensor.sum().detach().cpu()),
            "aux_possible_cell_frame_events": float(visible_tensor.numel()),
            "aux_mean_visible_cells_per_frame": float(
                visible_tensor.sum(dim=1).mean().detach().cpu()
            ),
            "aux_mean_normal_distance": float(normal_distance_tensor.mean().detach().cpu()),
            "aux_mean_normal_norm": float(normal_norm_tensor.mean().detach().cpu()),
            "aux_median_depth_valid_fraction": float(valid_depth.to(dtype=aux.normal_distance.dtype).mean().detach().cpu()),
        }
        if bool(valid_depth.any().detach().cpu()):
            metrics["aux_mean_median_depth"] = float(median_depth_tensor[valid_depth].mean().detach().cpu())
        return metrics

    def _gather_cells(self, values: torch.Tensor, new_indices: torch.Tensor) -> torch.Tensor:
        frame_ids = torch.arange(new_indices.shape[0], device=values.device, dtype=torch.long)[:, None]
        indices = new_indices.to(device=values.device, dtype=torch.long)
        return values[frame_ids, indices]

    def _reindex_parameter_cells(
        self,
        param: nn.Parameter | None,
        new_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if param is None:
            return
        param.data.copy_(self._gather_cells(param.data, new_indices))
        state = optimizer.state.get(param)
        if state is None:
            return
        for value in state.values():
            if torch.is_tensor(value) and value.shape[:2] == param.shape[:2]:
                value.copy_(self._gather_cells(value, new_indices))

    def _replace_optimizer_parameter(
        self,
        old_param: nn.Parameter,
        new_param: nn.Parameter,
        new_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> nn.Parameter:
        for group in optimizer.param_groups:
            group["params"] = [new_param if param is old_param else param for param in group["params"]]
        state = optimizer.state.pop(old_param, None)
        if state is not None:
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.shape[:2] == old_param.shape[:2]:
                    state[key] = self._gather_cells(value, new_indices).clone()
            optimizer.state[new_param] = state
        return new_param

    def _resize_parameter_cells(
        self,
        param: nn.Parameter | None,
        new_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> nn.Parameter | None:
        if param is None:
            return None
        resized = self._gather_cells(param.data, new_indices).contiguous()
        if resized.shape == param.shape:
            param.data.copy_(resized)
            state = optimizer.state.get(param)
            if state is not None:
                for value in state.values():
                    if torch.is_tensor(value) and value.shape[:2] == param.shape[:2]:
                        value.copy_(self._gather_cells(value, new_indices))
            return param
        return self._replace_optimizer_parameter(param, nn.Parameter(resized), new_indices, optimizer)

    def _resize_buffer_cells(self, name: str, new_indices: torch.Tensor) -> None:
        value = getattr(self, name)
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == new_indices.shape[0]:
            setattr(self, name, self._gather_cells(value, new_indices).contiguous())

    @torch.no_grad()
    def resample_from_ema(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        target_cells: int | None = None,
        perturb_scale: float = 0.05,
    ) -> dict[str, float]:
        frame_count, cell_count = self.contrib_ema.shape
        target_count = cell_count if target_cells is None else int(target_cells)
        if target_count < 1:
            raise ValueError("target_cells must be positive")
        if cell_count < 2:
            return {
                "resample_replaced": 0.0,
                "resample_valid_mean": float(cell_count),
                "resample_cell_count": float(cell_count),
            }

        new_rows: list[torch.Tensor] = []
        duplicate_count_rows: list[torch.Tensor] = []
        duplicate_mask_rows: list[torch.Tensor] = []
        total_replaced = 0
        total_pruned = 0
        total_invalid_pruned = 0
        points, radii, densities, features, normals = self.decoded_parameters()
        for frame in range(frame_count):
            contrib = self.contrib_ema[frame]
            point_error = self.point_error_ema[frame]
            finite_mask = (
                torch.isfinite(points[frame]).all(dim=-1)
                & torch.isfinite(radii[frame])
                & torch.isfinite(densities[frame])
                & torch.isfinite(features[frame].reshape(cell_count, -1)).all(dim=-1)
                & torch.isfinite(contrib)
                & torch.isfinite(point_error)
            )
            if normals is not None:
                finite_mask = finite_mask & torch.isfinite(normals[frame]).all(dim=-1)
            total_invalid_pruned += int(cell_count - int(finite_mask.sum().detach().cpu()))
            finite_indices = torch.nonzero(finite_mask, as_tuple=False).flatten()
            if finite_indices.numel() == 0:
                raise RuntimeError("PowerFoam resampling found no finite cells to keep.")
            finite_contrib = contrib[finite_indices]
            contrib_q = torch.quantile(finite_contrib, torch.tensor([0.1, 0.99], device=contrib.device), dim=0)
            threshold = torch.minimum(
                torch.tensor(1.0 / (float(cell_count) * 25.0), device=contrib.device, dtype=contrib.dtype),
                contrib_q[0],
            )
            valid_indices = torch.nonzero(finite_mask & (contrib > threshold), as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                finite_scores = finite_contrib.nan_to_num(nan=-float("inf"), neginf=-float("inf"))
                valid_indices = finite_indices[torch.topk(finite_scores, k=1, largest=True).indices]
            if valid_indices.numel() >= target_count:
                order = torch.argsort(contrib[valid_indices], descending=True, stable=True)
                new_indices = valid_indices[order[:target_count]]
                total_pruned += cell_count - target_count
                duplicate_count = torch.ones(target_count, device=contrib.device, dtype=contrib.dtype)
                duplicate_mask = torch.zeros(target_count, device=contrib.device, dtype=torch.bool)
            else:
                num_samples = target_count - int(valid_indices.numel())
                total_replaced += num_samples
                point_error_q = torch.quantile(point_error, 0.99, dim=0)
                prob = point_error[valid_indices].clamp(min=0.0, max=float(point_error_q.detach().cpu()))
                if float(prob.sum().detach().cpu()) <= 0.0:
                    prob = torch.ones_like(prob)
                sampled_pos = torch.multinomial(
                    prob,
                    num_samples,
                    replacement=num_samples > int(valid_indices.numel()),
                )
                duplicate_count_valid = torch.ones(valid_indices.numel(), device=contrib.device, dtype=contrib.dtype)
                duplicate_count_valid.index_add_(0, sampled_pos, torch.ones_like(sampled_pos, dtype=contrib.dtype))
                new_indices = torch.cat([valid_indices, valid_indices[sampled_pos]], dim=0)
                duplicate_count = torch.cat([duplicate_count_valid, duplicate_count_valid[sampled_pos]], dim=0)
                duplicate_mask = duplicate_count > 1.0
            new_rows.append(new_indices)
            duplicate_count_rows.append(duplicate_count)
            duplicate_mask_rows.append(duplicate_mask)

        new_indices_tensor = torch.stack(new_rows, dim=0).to(device=self.raw_xy.device, dtype=torch.long)
        duplicate_count_tensor = torch.stack(duplicate_count_rows, dim=0).to(device=self.raw_xy.device, dtype=self.raw_xy.dtype)
        duplicate_mask_tensor = torch.stack(duplicate_mask_rows, dim=0).to(device=self.raw_xy.device)

        self.raw_xy = self._resize_parameter_cells(self.raw_xy, new_indices_tensor, optimizer)
        self.raw_z = self._resize_parameter_cells(self.raw_z, new_indices_tensor, optimizer)
        self.raw_radii = self._resize_parameter_cells(self.raw_radii, new_indices_tensor, optimizer)
        self.raw_densities = self._resize_parameter_cells(self.raw_densities, new_indices_tensor, optimizer)
        self.raw_features = self._resize_parameter_cells(self.raw_features, new_indices_tensor, optimizer)
        self.raw_normals = self._resize_parameter_cells(self.raw_normals, new_indices_tensor, optimizer)
        self.raw_tangents = self._resize_parameter_cells(self.raw_tangents, new_indices_tensor, optimizer)
        self.raw_quaternions = self._resize_parameter_cells(self.raw_quaternions, new_indices_tensor, optimizer)
        self.raw_texel_sites = self._resize_parameter_cells(self.raw_texel_sites, new_indices_tensor, optimizer)
        self.raw_texel_heights = self._resize_parameter_cells(self.raw_texel_heights, new_indices_tensor, optimizer)
        self.raw_texel_sv_axis = self._resize_parameter_cells(self.raw_texel_sv_axis, new_indices_tensor, optimizer)
        self.raw_texel_sv_rgb = self._resize_parameter_cells(self.raw_texel_sv_rgb, new_indices_tensor, optimizer)

        self.contrib_ema = (self._gather_cells(self.contrib_ema, new_indices_tensor) / duplicate_count_tensor).contiguous()
        self.point_error_ema = (self._gather_cells(self.point_error_ema, new_indices_tensor) / duplicate_count_tensor).contiguous()
        for name in (
            "initial_points",
            "initial_radii",
            "initial_densities",
            "initial_features",
            "initial_normals",
            "initial_tangents",
            "initial_texel_sites",
            "initial_texel_heights",
            "initial_texel_sv_axis",
            "initial_texel_sv_rgb",
            "initial_quaternions",
        ):
            self._resize_buffer_cells(name, new_indices_tensor)

        if float(perturb_scale) > 0.0 and bool(duplicate_mask_tensor.any().detach().cpu()):
            points, radii, _densities, _features, normals = self.decoded_parameters()
            if normals is None:
                normals = torch.zeros_like(points)
                normals[..., 2] = -1.0
            direction = torch.randn_like(points)
            direction = direction - (direction * normals).sum(dim=-1, keepdim=True) * normals
            direction = F.normalize(direction, dim=-1, eps=1.0e-6)
            perturbed = points + float(perturb_scale) * radii[..., None] * direction
            xy = perturbed[..., :2].clamp(-self.xy_extent * 0.9999, self.xy_extent * 0.9999)
            z = perturbed[..., 2:].clamp(self.z_min + 1.0e-4, self.z_max - 1.0e-4)
            encoded_xy = torch.atanh((xy / self.xy_extent).clamp(-0.9999, 0.9999))
            encoded_z = logit_clamped((z - self.z_min) / (self.z_max - self.z_min))
            mask = duplicate_mask_tensor[..., None]
            self.raw_xy.data.copy_(torch.where(mask, encoded_xy, self.raw_xy.data))
            self.raw_z.data.copy_(torch.where(mask, encoded_z, self.raw_z.data))

        return {
            "resample_replaced": float(total_replaced),
            "resample_pruned": float(total_pruned),
            "resample_invalid_pruned": float(total_invalid_pruned),
            "resample_valid_mean": float(target_count - (total_replaced / max(frame_count, 1))),
            "resample_cell_count": float(target_count),
        }


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_torch_device(str(cfg["train"]["device"]), auto_cuda=False)
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("powerfoam_metal requires torch MPS")

    output_dir: Path = cfg["logging"]["output_dir"]
    write_resolved_config(output_dir, cfg)

    training_data = load_powerfoam_training_data(cfg, device)
    targets = training_data["targets"]
    sample_frame_indices = training_data["sample_frame_indices"]
    sample_rays = training_data["sample_rays"]
    heldout_targets = training_data["heldout_targets"]
    heldout_frame_indices = training_data["heldout_frame_indices"]
    heldout_rays = training_data["heldout_rays"]
    cfg["video_fps"] = float(training_data["video_fps"])
    loaded_image_size = normalize_image_size(targets.shape[-2:])
    paper_enabled = bool(cfg["paper_protocol"]["enabled"])
    paper_stages = normalize_paper_stages(
        cfg["paper_protocol"]["stages"] if paper_enabled else None,
        total_steps=int(cfg["train"]["steps"]),
        default_image_size=loaded_image_size,
        default_primitive_count=int(cfg["model"]["cells"]),
        default_frames_per_step=int(cfg["train"]["frames_per_step"]),
    )
    initial_cell_count = paper_stages[0].primitive_count if paper_enabled else int(cfg["model"]["cells"])
    with wandb_run_lifecycle(cfg) as wandb_run:
        point_cloud_init = None
        if cfg["model"]["init_point_cloud_path"] is not None:
            point_cloud_coordinate_frame = str(cfg["model"]["init_point_cloud_coordinate_frame"])
            point_transform = None
            if point_cloud_coordinate_frame == "multicam_world":
                point_transform = training_data.get("world_to_model")
                if point_transform is None:
                    raise ValueError(
                        "model.init_point_cloud_coordinate_frame='multicam_world' requires multicam camera metadata."
                    )
            point_cloud_init = load_powerfoam_point_cloud_initialization(
                path=cfg["model"]["init_point_cloud_path"],
                frame_count=int(training_data["frame_count"]),
                cell_count=initial_cell_count,
                xy_extent=float(cfg["model"]["xy_extent"]),
                z_min=float(cfg["model"]["z_min"]),
                z_max=float(cfg["model"]["z_max"]),
                normalize_mode=str(cfg["model"]["init_point_cloud_normalize"]),
                coordinate_frame=point_cloud_coordinate_frame,
                point_transform=point_transform,
                visibility_filter=str(cfg["model"]["init_point_cloud_visibility_filter"]),
                min_visible_train_views=int(cfg["model"]["init_point_cloud_min_visible_train_views"]),
                visibility_train_K=training_data.get("point_cloud_visibility_train_K"),
                visibility_train_w2c=training_data.get("point_cloud_visibility_train_w2c"),
                visibility_train_lens_models=training_data.get("point_cloud_visibility_train_lens_models"),
                visibility_train_distortions=training_data.get("point_cloud_visibility_train_distortions"),
                visibility_render_size=loaded_image_size.width,
                sample_mode=str(cfg["model"]["init_point_cloud_sample_mode"]),
                duplicate_jitter=float(cfg["model"]["init_point_cloud_duplicate_jitter"]),
                seed=int(cfg["train"]["seed"]),
            )

        model = MetalPowerFoamVideo(
            frame_count=int(training_data["frame_count"]),
            cell_count=initial_cell_count,
            render_size=loaded_image_size.width,
            fov_degrees=float(cfg["render"]["fov_degrees"]),
            neighbor_count=int(cfg["model"]["neighbor_count"]),
            adjacency_mode=str(cfg["model"]["adjacency_mode"]),
            xy_extent=float(cfg["model"]["xy_extent"]),
            z_min=float(cfg["model"]["z_min"]),
            z_max=float(cfg["model"]["z_max"]),
            radius_init=float(cfg["model"]["radius_init"]),
            radius_min=float(cfg["model"]["radius_min"]),
            radius_scale=float(cfg["model"]["radius_scale"]),
            density_init=float(cfg["model"]["density_init"]),
            feature_mode=str(cfg["model"]["feature_mode"]),
            linear_coeff_init=float(cfg["model"]["linear_coeff_init"]),
            linear_coeff_scale=float(cfg["model"]["linear_coeff_scale"]),
            normal_init_jitter=float(cfg["model"]["normal_init_jitter"]),
            num_texel_sites=int(cfg["model"]["num_texel_sites"]),
            texel_site_scale=float(cfg["model"]["texel_site_scale"]),
            texel_height_scale=float(cfg["model"]["texel_height_scale"]),
            sv_dof=int(cfg["model"]["sv_dof"]),
            sv_axis_init=float(cfg["model"]["sv_axis_init"]),
            sv_axis_init_jitter=float(cfg["model"]["sv_axis_init_jitter"]),
            sv_rgb_init_jitter=float(cfg["model"]["sv_rgb_init_jitter"]),
            color_init_mode=str(cfg["model"]["color_init_mode"]),
            seed=int(cfg["train"]["seed"]),
            init_frames=training_data["init_frames"] if bool(cfg["model"]["init_from_video"]) else None,
            init_points=None if point_cloud_init is None else point_cloud_init.points,
            init_colors=None if point_cloud_init is None else point_cloud_init.colors,
            image_init_depth=None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"]),
            image_init_jitter=float(cfg["model"]["image_init_jitter"]),
            raster_config=make_raster_config(cfg["render"]),
            use_raytrace=bool(cfg["render"]["use_raytrace"]),
        ).to(device)
        optimizer = torch.optim.Adam(model.optimizer_param_groups(cfg["train"]), lr=float(cfg["train"]["lr"]))
        legacy_initial_cell_count = int(model.contrib_ema.shape[1])
        adjacency_stats = model.adjacency_diagnostics()

        print(
            {
                "arch": "powerfoam_metal",
                "device": str(device),
                "source": str(training_data["source_label"]),
                "frame_source": str(cfg["data"]["frame_source"]),
                "frames": int(training_data["frame_count"]),
                "samples": int(targets.size(0)),
                "train_views": training_data["train_views"],
                "heldout_views": training_data["heldout_views"],
                "pose_source": training_data["pose_source"],
                "image_size": loaded_image_size.as_list(),
                "cells": initial_cell_count,
                "final_cells": paper_stages[-1].primitive_count,
                "neighbors": int(cfg["model"]["neighbor_count"]),
                "adjacency_mode": str(cfg["model"]["adjacency_mode"]),
                "adjacency_avg_degree": adjacency_stats["adjacency_avg_degree"],
                "adjacency_max_degree": adjacency_stats["adjacency_max_degree"],
                "adjacency_required_overlap_edges": adjacency_stats["adjacency_required_overlap_edges"],
                "adjacency_missing_overlap_edges": adjacency_stats["adjacency_missing_overlap_edges"],
                "feature_mode": str(cfg["model"]["feature_mode"]),
                "render_backend": (
                    "raytrace"
                    if bool(cfg["render"]["use_raytrace"])
                    else ("tiled" if bool(cfg["render"]["use_tiled"]) else "streaming")
                ),
                "color_init_mode": str(cfg["model"]["color_init_mode"]),
                "background_mode": str(cfg["render"]["background_mode"]),
                "lr_schedule": str(cfg["train"]["lr_schedule"]),
                "init_point_cloud": None if point_cloud_init is None else str(point_cloud_init.source_path),
                "init_point_cloud_source_count": None if point_cloud_init is None else point_cloud_init.source_count,
                "init_point_cloud_normalize": None if point_cloud_init is None else point_cloud_init.normalize_mode,
                "init_point_cloud_coordinate_frame": None
                if point_cloud_init is None
                else point_cloud_init.coordinate_frame,
                "init_point_cloud_visibility_filter": None
                if point_cloud_init is None
                else point_cloud_init.visibility_filter,
                "init_point_cloud_sample_mode": None if point_cloud_init is None else point_cloud_init.sample_mode,
                "init_point_cloud_filtered_count": None if point_cloud_init is None else point_cloud_init.filtered_count,
                "steps": int(cfg["train"]["steps"]),
            }
        )
        log_wandb_run_payload(
            wandb_run,
            {f"adjacency/{key}": value for key, value in adjacency_stats.items()},
            step=0,
        )
        initial_metrics = log_artifacts(
            model,
            targets,
            cfg,
            0,
            output_dir,
            wandb_run,
            frame_indices=sample_frame_indices,
            rays=sample_rays,
            heldout_targets=heldout_targets,
            heldout_frame_indices=heldout_frame_indices,
            heldout_rays=heldout_rays,
        )
        best_metric_value: float | None = maybe_save_best_powerfoam_checkpoint(
            model,
            cfg,
            output_dir,
            step=0,
            metrics=initial_metrics,
            best_metric_value=None,
        )
        append_jsonl(output_dir / "eval_metrics_history.jsonl", {"step": 0, "metrics": initial_metrics})
        last_artifact_step = 0
        last_artifact_metrics = dict(initial_metrics)
        print({"step": 0, **initial_metrics})

        start_time = time.perf_counter()
        paper_sampler = (
            SpacetimeEpochSampler(
                view_count=int(training_data["train_view_count"]),
                frame_indices=range(int(training_data["frame_count"])),
                batch_size=max(stage.frames_per_step for stage in paper_stages),
                same_time_count=int(cfg["paper_protocol"]["same_time_count"]),
                local_time_count=int(cfg["paper_protocol"]["local_time_count"]),
                local_time_radius=int(cfg["paper_protocol"]["local_time_radius"]),
                seed=int(cfg["train"]["seed"]) + int(cfg["paper_protocol"]["sampler_seed_offset"]),
            )
            if paper_enabled
            else None
        )
        paper_costs = PaperCostTracker()
        paper_phase_timer = PaperPhaseTimer(device)
        paper_memory_sampler = DeviceMemorySampler(device)
        paper_memory_sampler.start()
        paper_optimizer_elapsed_s = 0.0
        active_paper_stage = None
        progress = trange(1, int(cfg["train"]["steps"]) + 1, desc="powerfoam_metal")
        for step in progress:
            paper_update_started_at = time.perf_counter()
            paper_stage = paper_stage_for_step(paper_stages, step - 1)
            if paper_enabled and active_paper_stage != paper_stage.label:
                if int(model.contrib_ema.shape[1]) != paper_stage.primitive_count:
                    transition_metrics = model.resample_from_ema(
                        optimizer,
                        target_cells=paper_stage.primitive_count,
                        perturb_scale=float(cfg["model"]["resample_perturb_scale"]),
                    )
                    print({"step": step, "paper_stage": paper_stage.label, **transition_metrics})
                active_paper_stage = paper_stage.label
            lr_by_group = update_powerfoam_learning_rates(
                optimizer,
                cfg["train"],
                step=step - 1,
                total_steps=int(cfg["train"]["steps"]),
            )
            if paper_enabled:
                paper_batch = paper_sampler.next_batch(paper_stage.frames_per_step)
                sample_indices = paper_batch.flat_indices(int(training_data["frame_count"]), device=device)
            else:
                paper_batch = None
                sample_indices = powerfoam_train_batch_indices(targets.size(0), cfg, device=device)
            if paper_stage.lr_multiplier != 1.0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= paper_stage.lr_multiplier
                lr_by_group = {name: value * paper_stage.lr_multiplier for name, value in lr_by_group.items()}
            frame_indices = sample_frame_indices[sample_indices]
            target = targets[sample_indices]
            batch_rays = None if sample_rays is None else sample_rays[sample_indices]
            if paper_stage.image_size != loaded_image_size:
                target = resize_video_frames(target, paper_stage.image_size)
                if batch_rays is None:
                    raise RuntimeError("progressive PowerFoam stages require calibrated per-sample rays")
                batch_rays = resize_ray_grids(batch_rays, paper_stage.image_size)
            loss_weights = scheduled_loss_weights(cfg["losses"], step - 1, int(cfg["train"]["steps"]))
            need_normal_distance = loss_weights["normal_weight"] > 0.0
            need_normal_map = loss_weights["normal_map_weight"] > 0.0
            paper_forward_started_at = paper_phase_timer.start("forward")
            if need_normal_distance or need_normal_map:
                render_result = model(
                    frame_indices,
                    rays=batch_rays,
                    return_normal_distance=need_normal_distance,
                    return_rendered_normal=need_normal_map,
                )
                rendered, alpha = render_result[:2]
                cursor = 2
                normal_distance = render_result[cursor] if need_normal_distance else None
                cursor += 1 if need_normal_distance else 0
                rendered_normal = render_result[cursor] if need_normal_map else None
            else:
                rendered, alpha = model(frame_indices, rays=batch_rays)
                normal_distance = None
                rendered_normal = None
            rendered = composite_powerfoam_background(
                rendered,
                alpha,
                training_background_tensor(rendered, cfg["render"]),
            )
            l1 = F.l1_loss(rendered, target)
            mse = F.mse_loss(rendered, target)
            ssim_loss = (
                powerfoam_ssim_loss(rendered, target, cfg["losses"])
                if loss_weights["ssim_weight"] > 0.0
                else rendered.new_zeros(())
            )
            _, radii, densities, _, _ = model.decoded_parameters()
            radius_l2 = radii.square().mean()
            density_l2 = densities.square().mean()
            normal_loss = (
                powerfoam_normal_distance_loss(normal_distance)
                if normal_distance is not None and loss_weights["normal_weight"] > 0.0
                else rendered.new_zeros(())
            )
            normal_map_valid_fraction = rendered.new_zeros(())
            if rendered_normal is not None and loss_weights["normal_map_weight"] > 0.0:
                aux = model.height_sv_aux_batch(frame_indices, target, batch_rays)
                if aux is None:
                    raise RuntimeError("normal-map supervision requires a height+SV PowerFoam aux path")
                normal_rays = (
                    model.rays.to(device=rendered.device, dtype=rendered.dtype)
                    if batch_rays is None
                    else batch_rays.to(device=rendered.device, dtype=rendered.dtype)
                )
                target_normal, normal_map_mask = normals_from_ray_depth(aux.median_depth, normal_rays)
                normal_map_mask = normal_map_mask & (alpha.detach() >= float(cfg["losses"]["normal_map_min_alpha"]))
                normal_map_loss = powerfoam_normal_map_loss(rendered_normal, target_normal.detach(), normal_map_mask)
                normal_map_valid_fraction = normal_map_mask.to(dtype=rendered.dtype).mean()
            else:
                normal_map_loss = rendered.new_zeros(())
            contribution_loss = (
                powerfoam_contribution_loss(alpha)
                if loss_weights["contribution_weight"] > 0.0
                else rendered.new_zeros(())
            )
            interpenetration_loss = (
                model.interpenetration_loss(frame_indices)
                if loss_weights["interpenetration_weight"] > 0.0
                else rendered.new_zeros(())
            )
            loss = (
                loss_weights["l1_weight"] * l1
                + loss_weights["mse_weight"] * mse
                + loss_weights["ssim_weight"] * ssim_loss
                + loss_weights["radius_l2_weight"] * radius_l2
                + loss_weights["density_l2_weight"] * density_l2
                + loss_weights["normal_weight"] * normal_loss
                + loss_weights["normal_map_weight"] * normal_map_loss
                + loss_weights["contribution_weight"] * contribution_loss
                + loss_weights["interpenetration_weight"] * interpenetration_loss
            )
            paper_phase_timer.stop("forward", paper_forward_started_at)
            optimizer.zero_grad(set_to_none=True)
            paper_backward_started_at = paper_phase_timer.start("backward")
            loss.backward()
            paper_phase_timer.stop("backward", paper_backward_started_at)
            paper_optimizer_started_at = paper_phase_timer.start("optimizer")
            optimizer.step()
            paper_phase_timer.stop("optimizer", paper_optimizer_started_at)
            paper_costs.record(
                stage=paper_stage,
                target_frames=int(frame_indices.numel()),
                rasterized_frames=int(frame_indices.numel()),
            )
            paper_optimizer_elapsed_s += time.perf_counter() - paper_update_started_at

            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", l1=f"{float(l1.detach().cpu()):.4f}")
            if should_log_scalar(cfg, step):
                elapsed = time.perf_counter() - start_time
                train_metrics = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "l1": float(l1.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "ssim_loss": float(ssim_loss.detach().cpu()),
                    "normal_loss": float(normal_loss.detach().cpu()),
                    "normal_weight": loss_weights["normal_weight"],
                    "normal_map_loss": float(normal_map_loss.detach().cpu()),
                    "normal_map_weight": loss_weights["normal_map_weight"],
                    "normal_map_valid_fraction": float(normal_map_valid_fraction.detach().cpu()),
                    "contribution_loss": float(contribution_loss.detach().cpu()),
                    "contribution_weight": loss_weights["contribution_weight"],
                    "interpenetration_loss": float(interpenetration_loss.detach().cpu()),
                    "interpenetration_weight": loss_weights["interpenetration_weight"],
                    "elapsed_s": elapsed,
                    "paper_stage": paper_stage.label,
                    "paper_epoch": None if paper_batch is None else paper_batch.epoch,
                    "paper_batch_index": None if paper_batch is None else paper_batch.batch_index,
                    "paper_epoch_complete": None if paper_batch is None else paper_batch.completes_epoch,
                    "paper_height": paper_stage.image_size.height,
                    "paper_width": paper_stage.image_size.width,
                    "paper_active_cells": int(model.contrib_ema.shape[1]),
                    "paper_target_pixels": int(frame_indices.numel()) * paper_stage.image_size.pixels,
                }
                for name, value in lr_by_group.items():
                    train_metrics[f"lr_{name}"] = value
                append_jsonl(output_dir / "train_metrics_history.jsonl", train_metrics)
                print(train_metrics)
                log_wandb_run_payload(
                    wandb_run,
                    {
                        **mapped_metric_payload(train_metrics, POWERFOAM_METAL_TRAIN_WANDB_KEYS),
                        **{f"LR/{name}": value for name, value in lr_by_group.items()},
                    },
                    step=step,
                )
            logged_artifacts = False
            if should_log_image(cfg, step):
                if step == int(cfg["train"]["steps"]):
                    save_powerfoam_checkpoint(
                        output_dir / "checkpoint_pre_final_eval.pt",
                        model,
                        cfg,
                        step=step,
                    )
                metrics = log_artifacts(
                    model,
                    targets,
                    cfg,
                    step,
                    output_dir,
                    wandb_run,
                    frame_indices=sample_frame_indices,
                    rays=sample_rays,
                    heldout_targets=heldout_targets,
                    heldout_frame_indices=heldout_frame_indices,
                    heldout_rays=heldout_rays,
                )
                best_metric_value = maybe_save_best_powerfoam_checkpoint(
                    model,
                    cfg,
                    output_dir,
                    step=step,
                    metrics=metrics,
                    best_metric_value=best_metric_value,
                )
                append_jsonl(output_dir / "eval_metrics_history.jsonl", {"step": int(step), "metrics": metrics})
                last_artifact_step = int(step)
                last_artifact_metrics = dict(metrics)
                logged_artifacts = True
                print({"step": step, **metrics})
            if should_resample_powerfoam_step(cfg, step):
                if not logged_artifacts:
                    model.aux_metrics(frame_indices, target, rays=batch_rays)
                target_cells = scheduled_resample_target_cells(
                    cfg["model"],
                    initial_cells=legacy_initial_cell_count,
                    current_cells=int(model.contrib_ema.shape[1]),
                    step=step,
                    total_steps=int(cfg["train"]["steps"]),
                )
                resample_metrics = model.resample_from_ema(
                    optimizer,
                    target_cells=target_cells,
                    perturb_scale=float(cfg["model"]["resample_perturb_scale"]),
                )
                print({"step": step, **resample_metrics})
                log_wandb_run_payload(
                    wandb_run,
                    {f"Resample/{key.removeprefix('resample_')}": value for key, value in resample_metrics.items()},
                    step=step,
                )

        final_step = int(cfg["train"]["steps"])
        paper_memory_sampler.stop()
        final_checkpoint_path = output_dir / "checkpoint_final.pt"
        save_powerfoam_checkpoint(
            final_checkpoint_path,
            model,
            cfg,
            step=final_step,
            metrics=last_artifact_metrics if last_artifact_step == final_step else None,
        )
        paper_summary = {
            "enabled": paper_enabled,
            "representation": "worldfoam",
            "kernel": MetalKernelSpec(
                representation="worldfoam",
                family="powerfoam_metal",
                forward="raytrace" if bool(cfg["render"]["use_raytrace"]) else "raster",
                backward="powerfoam_metal_autograd",
                deterministic=False,
                implementation="third_party/powerfoam-metal",
            ).as_dict(),
            "sampling": {
                "mode": "spacetime_epoch" if paper_enabled else "iid_with_replacement",
                "same_time_count": int(cfg["paper_protocol"]["same_time_count"]),
                "local_time_count": int(cfg["paper_protocol"]["local_time_count"]),
                "local_time_radius": int(cfg["paper_protocol"]["local_time_radius"]),
            },
            "stages": [stage.as_dict() for stage in paper_stages],
            "cost": paper_costs.snapshot(
                model=model,
                optimizer=optimizer,
                elapsed_s=paper_optimizer_elapsed_s,
                memory=paper_memory_sampler.stats(),
                serialized_checkpoint_bytes=final_checkpoint_path.stat().st_size,
            ).as_dict(),
            "timing": paper_phase_timer.snapshot(train_wall_s=time.perf_counter() - start_time),
            "wall_loop_elapsed_s": time.perf_counter() - start_time,
        }
        write_json(output_dir / "paper_protocol_summary.json", paper_summary)


__all__ = [
    "MetalPowerFoamVideo",
    "PowerFoamAuxBatch",
    "rays_for_sample_batch",
    "run_training",
]
