from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from checkpoint_utils import atomic_torch_save
from colorize import FeatureToColor
from config_utils import serialize_config_value
from dynamic_powerfoam_camera import (
    build_camera_decoder,
    camera_param_group,
    camera_regularization,
    compact_camera_metrics,
    decoded_powerfoam_rays,
    load_teacher_camera_to_world,
    prefit_camera_decoder_from_teacher,
)
from dynamic_powerfoam_initialization import (
    initialize_full_powerfoam_from_orbit_video,
    initialize_powerfoam_normals,
    make_texel_feature_init,
    transform_powerfoam_frame_to_camera,
    transform_points_camera_to_world,
)
from dynamic_powerfoam_metal_config import (
    CAMERA_DEFAULTS,
    COLORIZE_DEFAULTS,
    DATA_DEFAULTS,
    LOGGING_DEFAULTS,
    LOSS_DEFAULTS,
    MODEL_DEFAULTS,
    RENDER_DEFAULTS,
    TOKEN_RBF_FEATURE_MODE,
    TRAIN_DEFAULTS,
    resolve_config,
)
from dynamic_powerfoam_rendering import (
    per_frame_reconstruction_metrics,
    render_all,
    render_features_to_rgb,
    sample_background,
    temporal_alpha_metrics,
)
from dynamic_powerfoam_staging import apply_training_stage, camera_curriculum_active_frames
from external_paths import ensure_third_party_path
from powerfoam_colorizers import build_dynamic_powerfoam_colorizer
from powerfoam_diagnostics import powerfoam_parameter_delta_metrics
from dynamic_powerfoam_temporal import (
    atanh_clamped,
    fit_temporal_basis,
    make_gaussian_time_basis,
    temporal_accel,
    temporal_motion_metrics,
)
from powerfoam_raster_config import make_dynamic_powerfoam_metal_raster_config as make_raster_config
from powerfoam_training import powerfoam_train_batch_indices
from powerfoam_direct import (
    POWERFOAM_SOFTPLUS_BETA,
    initialize_full_powerfoam_from_video,
    initialize_random_full_powerfoam,
    inverse_softplus,
    logit_clamped,
)
from powerfoam_implicit_camera import PowerFoamImplicitCameraDecoder
from pipeline.diagnostics import reconstruction_l1_mse_metrics
from powerfoam_adjacency import build_csr_adjacency
from powerfoam_eval_render import powerfoam_eval_batch_size
from sequence_data import load_video_sequence
from train_artifacts import append_jsonl, write_json, write_resolved_config
from train_devices import resolve_torch_device
from train_logging import (
    log_wandb_run_payload,
    log_wandb_run_payload_lazy,
    mapped_metric_payload,
    should_log_image,
    should_log_scalar,
    should_log_video,
    wandb_run_lifecycle,
)
from train_optim import optimizer_backward_step
from wandb_media import (
    build_rgb_alpha_eval_media_payload,
)
from powerfoam_geometry import (
    make_pinhole_rays,
    orthonormal_surface_frame,
    stable_tangent_from_normals,
)
from video_io import save_rgb_alpha_eval_media, video_fps_from_config

DYNAMIC_POWERFOAM_METAL_ROOT = ensure_third_party_path("dynamic-powerfoam-metal")

from torch_dynamic_powerfoam_metal import (  # noqa: E402
    FoamRasterConfig,
    rasterize_power_foam_oriented_texel_surface,
)

DYNAMIC_POWERFOAM_TRAIN_WANDB_KEYS = (
    ("loss", "Train/Loss"),
    ("l1", "Train/L1"),
    ("mse", "Train/MSE"),
    ("temporal", "Train/TemporalLoss"),
    ("elapsed_s", "Timing/ElapsedSeconds"),
    ("stage_temporal_geometry_scale", "Stage/TemporalGeometryScale"),
    ("stage_temporal_feature_scale", "Stage/TemporalFeatureScale"),
    ("stage_camera_active_frames", "Stage/CameraActiveFrames"),
)

DYNAMIC_POWERFOAM_TRAIN_CAMERA_WANDB_KEYS = (
    ("camera_rotation_l2", "Train/CameraRotationL2"),
    ("camera_translation_l2", "Train/CameraTranslationL2"),
    ("camera_temporal_l2", "Train/CameraTemporalL2"),
    ("camera_global_l2", "Train/CameraGlobalL2"),
    ("camera_velocity_l2", "Train/CameraVelocityL2"),
    ("camera_acceleration_l2", "Train/CameraAccelerationL2"),
    ("camera_gimbal_l2", "Train/CameraGimbalL2"),
)


class DynamicMetalPowerFoamVideo(nn.Module):
    def __init__(
        self,
        *,
        frame_count: int,
        cell_count: int,
        render_size: int,
        fov_degrees: float,
        neighbor_count: int,
        adjacency_mode: str,
        dynamic_mode: str,
        time_basis_count: int,
        time_basis_sigma_scale: float,
        temporal_init_mode: str,
        dynamic_centers: bool,
        dynamic_radii: bool,
        dynamic_densities: bool,
        dynamic_features: bool,
        dynamic_normals: bool,
        dynamic_texel_sites: bool,
        xy_extent: float,
        z_min: float,
        z_max: float,
        radius_init: float,
        radius_min: float,
        radius_scale: float,
        density_init: float,
        normal_init_jitter: float,
        num_texel_sites: int,
        texel_site_scale: float,
        color_init_mode: str,
        video_init_mode: str,
        seed: int,
        init_frames: torch.Tensor | None,
        image_init_depth: float | None,
        image_init_jitter: float,
        raster_config: FoamRasterConfig,
        camera_decoder: PowerFoamImplicitCameraDecoder | None = None,
    ) -> None:
        super().__init__()
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if init_frames is None:
            init = initialize_random_full_powerfoam(
                frame_count=frame_count,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                radius_init=radius_init,
                radius_min=radius_min,
                num_texel_sites=int(num_texel_sites),
                sv_dof=1,
                sv_axis_init=1.0,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        elif str(video_init_mode) == "orbit_camera":
            if camera_decoder is None:
                raise ValueError("model.video_init_mode='orbit_camera' requires camera.enabled=true")
            init = initialize_full_powerfoam_from_orbit_video(
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
                image_init_jitter=image_init_jitter,
                texel_site_scale=texel_site_scale,
                camera_decoder=camera_decoder,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        else:
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
                sv_dof=1,
                sv_axis_init=1.0,
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        if str(color_init_mode) == "random":
            texel_colors_init = torch.rand(
                frame_count,
                cell_count,
                int(num_texel_sites),
                3,
                generator=generator,
                dtype=init_points.dtype,
            )

        init_density = torch.full((frame_count, cell_count), max(float(density_init), 1.0e-4))
        init_normals = initialize_powerfoam_normals(
            frame_count=frame_count,
            cell_count=cell_count,
            dtype=init_points.dtype,
            normal_init_jitter=normal_init_jitter,
            video_init_mode=video_init_mode,
            camera_decoder=camera_decoder,
            generator=generator,
        )
        init_tangents = stable_tangent_from_normals(init_normals)

        raw_xy = atanh_clamped(init_points[..., :2] / float(xy_extent))
        raw_z = logit_clamped((init_points[..., 2:] - float(z_min)) / (float(z_max) - float(z_min)))
        raw_radii = inverse_softplus((init_radii - float(radius_min)).clamp_min(1.0e-4), beta=POWERFOAM_SOFTPLUS_BETA)
        raw_densities = inverse_softplus(init_density, beta=POWERFOAM_SOFTPLUS_BETA)
        raw_features = logit_clamped(texel_colors_init.clamp(0.0, 1.0))
        raw_texel_sites = atanh_clamped(texel_sites_init / float(texel_site_scale))

        self.frame_count = int(frame_count)
        self.dynamic_mode = str(dynamic_mode)
        self.dynamic_centers = bool(dynamic_centers)
        self.dynamic_radii = bool(dynamic_radii)
        self.dynamic_densities = bool(dynamic_densities)
        self.dynamic_features = bool(dynamic_features)
        self.dynamic_normals = bool(dynamic_normals)
        self.dynamic_texel_sites = bool(dynamic_texel_sites)
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.radius_min = float(radius_min)
        self.neighbor_count = int(neighbor_count)
        self.adjacency_mode = str(adjacency_mode)
        self.texel_site_scale = float(texel_site_scale)
        self.render_size = int(render_size)
        self.fov_degrees = float(fov_degrees)
        self.raster_config = raster_config
        self.camera_decoder = camera_decoder
        self.register_buffer("rays", make_pinhole_rays(render_size, render_size, fov_degrees, torch.device("cpu")), persistent=False)
        basis = make_gaussian_time_basis(frame_count, int(time_basis_count), float(time_basis_sigma_scale))
        self.register_buffer("frame_basis", basis, persistent=False)

        if self.dynamic_mode == "per_frame_smooth":
            self.raw_xy = nn.Parameter(raw_xy)
            self.raw_z = nn.Parameter(raw_z)
            self.raw_radii = nn.Parameter(raw_radii)
            self.raw_densities = nn.Parameter(raw_densities)
            self.raw_features = nn.Parameter(raw_features)
            self.raw_normals = nn.Parameter(init_normals)
            self.raw_tangents = nn.Parameter(init_tangents)
            self.raw_texel_sites = nn.Parameter(raw_texel_sites)
        else:
            self.raw_xy0, self.raw_xy_coeff = self._make_temporal_param(
                raw_xy, dynamic=self.dynamic_centers, temporal_init_mode=temporal_init_mode
            )
            self.raw_z0, self.raw_z_coeff = self._make_temporal_param(
                raw_z, dynamic=self.dynamic_centers, temporal_init_mode=temporal_init_mode
            )
            self.raw_radii0, self.raw_radii_coeff = self._make_temporal_param(
                raw_radii, dynamic=self.dynamic_radii, temporal_init_mode=temporal_init_mode
            )
            self.raw_densities0, self.raw_densities_coeff = self._make_temporal_param(
                raw_densities, dynamic=self.dynamic_densities, temporal_init_mode=temporal_init_mode
            )
            self.raw_features0, self.raw_features_coeff = self._make_temporal_param(
                raw_features, dynamic=self.dynamic_features, temporal_init_mode=temporal_init_mode
            )
            self.raw_normals0, self.raw_normals_coeff = self._make_temporal_param(
                init_normals, dynamic=self.dynamic_normals, temporal_init_mode=temporal_init_mode
            )
            self.raw_tangents0, self.raw_tangents_coeff = self._make_temporal_param(
                init_tangents, dynamic=self.dynamic_normals, temporal_init_mode=temporal_init_mode
            )
            self.raw_texel_sites0, self.raw_texel_sites_coeff = self._make_temporal_param(
                raw_texel_sites, dynamic=self.dynamic_texel_sites, temporal_init_mode=temporal_init_mode
            )

        self.register_buffer("initial_points", init_points.clone(), persistent=False)
        self.register_buffer("initial_radii", init_radii.clone(), persistent=False)
        self.register_buffer("initial_densities", init_density.clone(), persistent=False)
        self.register_buffer("initial_features", texel_colors_init.clone(), persistent=False)
        self.register_buffer("initial_normals", init_normals.clone(), persistent=False)
        self.register_buffer("initial_tangents", init_tangents.clone(), persistent=False)
        self.register_buffer("initial_texel_sites", texel_sites_init.clone(), persistent=False)

    def _make_temporal_param(
        self,
        values: torch.Tensor,
        *,
        dynamic: bool,
        temporal_init_mode: str,
    ) -> tuple[nn.Parameter, nn.Parameter | None]:
        base, coeff = fit_temporal_basis(values, self.frame_basis.cpu(), mode=str(temporal_init_mode))
        base_param = nn.Parameter(base)
        if not dynamic:
            return base_param, None
        return base_param, nn.Parameter(coeff)

    def _decode_temporal(self, base: torch.Tensor, coeff: torch.Tensor | None, frame_indices: torch.Tensor | None) -> torch.Tensor:
        if self.dynamic_mode == "per_frame_smooth":
            raise RuntimeError("_decode_temporal is only used by rbf mode")
        if frame_indices is None:
            basis = self.frame_basis.to(device=base.device, dtype=base.dtype)
        else:
            basis = self.frame_basis[frame_indices.to(device=self.frame_basis.device, dtype=torch.long)].to(
                device=base.device,
                dtype=base.dtype,
            )
        values = base[None].expand(basis.shape[0], *base.shape)
        if coeff is not None:
            values = values + torch.einsum("tk,k...->t...", basis, coeff)
        return values

    def raw_parameter_tensors(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dynamic_mode == "per_frame_smooth":
            if frame_indices is None:
                return (
                    self.raw_xy,
                    self.raw_z,
                    self.raw_radii,
                    self.raw_densities,
                    self.raw_features,
                    self.raw_normals,
                    self.raw_tangents,
                    self.raw_texel_sites,
                )
            idx = frame_indices.to(device=self.raw_xy.device, dtype=torch.long)
            return (
                self.raw_xy[idx],
                self.raw_z[idx],
                self.raw_radii[idx],
                self.raw_densities[idx],
                self.raw_features[idx],
                self.raw_normals[idx],
                self.raw_tangents[idx],
                self.raw_texel_sites[idx],
            )
        return (
            self._decode_temporal(self.raw_xy0, self.raw_xy_coeff, frame_indices),
            self._decode_temporal(self.raw_z0, self.raw_z_coeff, frame_indices),
            self._decode_temporal(self.raw_radii0, self.raw_radii_coeff, frame_indices),
            self._decode_temporal(self.raw_densities0, self.raw_densities_coeff, frame_indices),
            self._decode_temporal(self.raw_features0, self.raw_features_coeff, frame_indices),
            self._decode_temporal(self.raw_normals0, self.raw_normals_coeff, frame_indices),
            self._decode_temporal(self.raw_tangents0, self.raw_tangents_coeff, frame_indices),
            self._decode_temporal(self.raw_texel_sites0, self.raw_texel_sites_coeff, frame_indices),
        )

    def decoded_parameters(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_xy, raw_z, raw_radii, raw_densities, raw_features, raw_normals, _raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        xy = torch.tanh(raw_xy) * self.xy_extent
        z = self.z_min + torch.sigmoid(raw_z) * (self.z_max - self.z_min)
        points = torch.cat([xy, z], dim=-1)
        radii = F.softplus(raw_radii, beta=POWERFOAM_SOFTPLUS_BETA) + self.radius_min
        densities = F.softplus(raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)
        features = torch.sigmoid(raw_features)
        normals = F.normalize(raw_normals, dim=-1, eps=1.0e-6)
        return points, radii, densities, features, normals

    def decoded_texel_sites(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        *_prefix, raw_texel_sites = self.raw_parameter_tensors(frame_indices)
        return self.texel_site_scale * torch.tanh(raw_texel_sites)

    def decoded_surface_frame(
        self,
        normals: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _raw_xy, _raw_z, _raw_radii, _raw_densities, _raw_features, _raw_normals, raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        return orthonormal_surface_frame(normals, raw_tangents)

    def optimizer_param_groups(self, train_cfg: dict[str, Any]) -> list[dict[str, object]]:
        lr = float(train_cfg["lr"])
        groups: list[dict[str, object]] = []

        def add(params: list[nn.Parameter | None], multiplier: str, name: str) -> None:
            clean = [param for param in params if param is not None]
            if clean:
                groups.append({"params": clean, "lr": float(train_cfg[multiplier]) * lr, "name": name})

        if self.dynamic_mode == "per_frame_smooth":
            add([self.raw_xy, self.raw_z], "point_lr_multiplier", "points")
            add([self.raw_radii], "radius_lr_multiplier", "radii")
            add([self.raw_densities], "density_lr_multiplier", "density")
            add([self.raw_features], "feature_lr_multiplier", "features")
            add([self.raw_normals, self.raw_tangents], "normal_lr_multiplier", "surface_frame")
            add([self.raw_texel_sites], "texel_site_lr_multiplier", "texel_sites")
            camera_group = camera_param_group(self.camera_decoder, train_cfg)
            if camera_group is not None:
                groups.append(camera_group)
            return groups

        add([self.raw_xy0, self.raw_z0], "point_lr_multiplier", "base_points")
        add([self.raw_radii0], "radius_lr_multiplier", "base_radii")
        add([self.raw_densities0], "density_lr_multiplier", "base_density")
        add([self.raw_features0], "feature_lr_multiplier", "base_features")
        add([self.raw_normals0, self.raw_tangents0], "normal_lr_multiplier", "base_surface_frame")
        add([self.raw_texel_sites0], "texel_site_lr_multiplier", "base_texel_sites")
        add([self.raw_xy_coeff, self.raw_z_coeff], "temporal_lr_multiplier", "temporal_points")
        add([self.raw_radii_coeff], "temporal_lr_multiplier", "temporal_radii")
        add([self.raw_densities_coeff], "temporal_lr_multiplier", "temporal_density")
        add([self.raw_features_coeff], "temporal_lr_multiplier", "temporal_features")
        add([self.raw_normals_coeff, self.raw_tangents_coeff], "temporal_lr_multiplier", "temporal_surface_frame")
        add([self.raw_texel_sites_coeff], "temporal_lr_multiplier", "temporal_texel_sites")
        camera_group = camera_param_group(self.camera_decoder, train_cfg)
        if camera_group is not None:
            groups.append(camera_group)
        return groups

    def temporal_regularization(self, loss_cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        points, radii, densities, features, _normals = self.decoded_parameters()
        terms = {
            "temporal_center_accel": temporal_accel(points),
            "temporal_radius_accel": temporal_accel(radii),
            "temporal_density_accel": temporal_accel(densities),
            "temporal_feature_accel": temporal_accel(features),
            "temporal_coeff_l2": points.new_zeros(()),
        }
        if self.dynamic_mode == "rbf":
            coeffs = [
                self.raw_xy_coeff,
                self.raw_z_coeff,
                self.raw_radii_coeff,
                self.raw_densities_coeff,
                self.raw_features_coeff,
                self.raw_normals_coeff,
                self.raw_tangents_coeff,
                self.raw_texel_sites_coeff,
            ]
            coeff_terms = [c.square().mean() for c in coeffs if c is not None]
            if coeff_terms:
                terms["temporal_coeff_l2"] = torch.stack(coeff_terms).mean()
        loss = points.new_zeros(())
        for key, value in terms.items():
            loss = loss + float(loss_cfg[f"{key}_weight"]) * value
        camera_loss, camera_terms = camera_regularization(self.camera_decoder, loss_cfg)
        if camera_loss is not None:
            loss = loss + camera_loss
            terms.update(camera_terms)
        return loss, terms

    @torch.no_grad()
    def parameter_drift_metrics(self) -> dict[str, float]:
        points, radii, densities, features, normals = self.decoded_parameters()
        texel_sites = self.decoded_texel_sites()
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
            texel_sites=texel_sites,
            initial_texel_sites=self.initial_texel_sites,
        )
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices()
        metrics.update(
            temporal_motion_metrics(
                points,
                features,
                render_size=self.render_size,
                fov_degrees=self.fov_degrees,
                camera_to_world=camera_to_world,
            )
        )
        if self.dynamic_mode == "rbf":
            coeff_abs = [
                c.abs().mean()
                for c in [
                    self.raw_xy_coeff,
                    self.raw_z_coeff,
                    self.raw_radii_coeff,
                    self.raw_densities_coeff,
                    self.raw_features_coeff,
                    self.raw_normals_coeff,
                    self.raw_tangents_coeff,
                    self.raw_texel_sites_coeff,
                ]
                if c is not None
            ]
            if coeff_abs:
                metrics["state_mean_temporal_coeff_abs"] = float(torch.stack(coeff_abs).mean().cpu())
        metrics.update(compact_camera_metrics(self.camera_decoder))
        return metrics

    def decoded_camera_rays(self, frame_indices: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return decoded_powerfoam_rays(
            self.camera_decoder,
            self.rays,
            frame_indices.to(device=next(self.parameters()).device, dtype=torch.long),
            height=self.render_size,
            width=self.render_size,
            device=next(self.parameters()).device,
            dtype=dtype,
        )

    def forward(self, frame_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if next(self.parameters()).device.type != "mps":
            raise RuntimeError("DynamicMetalPowerFoamVideo requires an MPS device")
        frame_indices = frame_indices.to(device=next(self.parameters()).device, dtype=torch.long)
        points, radii, densities, features, normals = self.decoded_parameters(frame_indices)
        texel_sites = self.decoded_texel_sites(frame_indices)
        tangents, bitangents = self.decoded_surface_frame(normals, frame_indices)
        rays = self.rays.to(device=points.device, dtype=points.dtype)
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices(frame_indices)
        renders = []
        alphas = []
        for local_index in range(int(frame_indices.numel())):
            point = points[local_index]
            normal = normals[local_index]
            tangent = tangents[local_index]
            bitangent = bitangents[local_index]
            if camera_to_world is not None:
                point, normal, tangent, bitangent = transform_powerfoam_frame_to_camera(
                    point,
                    normal,
                    tangent,
                    bitangent,
                    camera_to_world[local_index],
                )
            radius = radii[local_index]
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            out, alpha = rasterize_power_foam_oriented_texel_surface(
                point,
                radius,
                densities[local_index],
                texel_sites[local_index],
                features[local_index],
                normal,
                adjacency,
                offsets,
                rays,
                self.raster_config,
                tangents=tangent,
                bitangents=bitangent,
            )
            renders.append(out.permute(0, 3, 1, 2))
            alphas.append(alpha)
        return torch.cat(renders, dim=0), torch.cat(alphas, dim=0)


class TokenDynamicPowerFoamFeatures(nn.Module):
    """Token decoder for dynamic PowerFoam feature splatting.

    This is intentionally one token per cell for the first baseline: the token
    bank owns the canonical cell state plus RBF temporal coefficients, and the
    Metal rasterizer splats F-channel texel features before a colorizer maps
    features to RGB.
    """

    def __init__(
        self,
        *,
        frame_count: int,
        cell_count: int,
        render_size: int,
        fov_degrees: float,
        neighbor_count: int,
        adjacency_mode: str,
        time_basis_count: int,
        time_basis_sigma_scale: float,
        temporal_init_mode: str,
        dynamic_centers: bool,
        dynamic_radii: bool,
        dynamic_densities: bool,
        dynamic_features: bool,
        dynamic_normals: bool,
        dynamic_texel_sites: bool,
        feature_dim: int,
        feature_init_noise: float,
        feature_rgb_init: str,
        token_dim: int,
        token_hidden_dim: int,
        token_hidden_layers: int,
        token_init_std: float,
        token_output_init_std: float,
        token_point_residual_scale: float,
        token_z_residual_scale: float,
        token_radius_residual_scale: float,
        token_density_residual_scale: float,
        token_feature_residual_scale: float,
        token_normal_residual_scale: float,
        token_texel_site_residual_scale: float,
        token_temporal_residual_scale: float,
        static_dynamic_split: bool,
        dynamic_cells: int | None,
        dynamic_cell_fraction: float,
        xy_extent: float,
        z_min: float,
        z_max: float,
        radius_init: float,
        radius_min: float,
        radius_scale: float,
        density_init: float,
        normal_init_jitter: float,
        num_texel_sites: int,
        texel_site_scale: float,
        color_init_mode: str,
        video_init_mode: str,
        seed: int,
        init_frames: torch.Tensor | None,
        image_init_depth: float | None,
        image_init_jitter: float,
        raster_config: FoamRasterConfig,
        camera_decoder: PowerFoamImplicitCameraDecoder | None = None,
    ) -> None:
        super().__init__()
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        if int(feature_dim) < 3:
            raise ValueError("feature_dim must be at least 3")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if init_frames is None:
            init = initialize_random_full_powerfoam(
                frame_count=frame_count,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                radius_init=radius_init,
                radius_min=radius_min,
                num_texel_sites=int(num_texel_sites),
                sv_dof=1,
                sv_axis_init=1.0,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        elif str(video_init_mode) == "orbit_camera":
            if camera_decoder is None:
                raise ValueError("model.video_init_mode='orbit_camera' requires camera.enabled=true")
            init = initialize_full_powerfoam_from_orbit_video(
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
                image_init_jitter=image_init_jitter,
                texel_site_scale=texel_site_scale,
                camera_decoder=camera_decoder,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        else:
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
                sv_dof=1,
                sv_axis_init=1.0,
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        if str(color_init_mode) == "random":
            texel_colors_init = torch.rand(
                frame_count,
                cell_count,
                int(num_texel_sites),
                3,
                generator=generator,
                dtype=init_points.dtype,
            )

        init_density = torch.full((frame_count, cell_count), max(float(density_init), 1.0e-4))
        init_normals = initialize_powerfoam_normals(
            frame_count=frame_count,
            cell_count=cell_count,
            dtype=init_points.dtype,
            normal_init_jitter=normal_init_jitter,
            video_init_mode=video_init_mode,
            camera_decoder=camera_decoder,
            generator=generator,
        )
        init_tangents = stable_tangent_from_normals(init_normals)

        raw_xy = atanh_clamped(init_points[..., :2] / float(xy_extent))
        raw_z = logit_clamped((init_points[..., 2:] - float(z_min)) / (float(z_max) - float(z_min)))
        raw_radii = inverse_softplus((init_radii - float(radius_min)).clamp_min(1.0e-4), beta=POWERFOAM_SOFTPLUS_BETA)
        raw_densities = inverse_softplus(init_density, beta=POWERFOAM_SOFTPLUS_BETA)
        raw_features = make_texel_feature_init(
            texel_colors_init,
            feature_dim=int(feature_dim),
            rgb_init=str(feature_rgb_init),
            noise_std=float(feature_init_noise),
            generator=generator,
        )
        raw_texel_sites = atanh_clamped(texel_sites_init / float(texel_site_scale))

        self.frame_count = int(frame_count)
        self.dynamic_mode = TOKEN_RBF_FEATURE_MODE
        self.dynamic_centers = bool(dynamic_centers)
        self.dynamic_radii = bool(dynamic_radii)
        self.dynamic_densities = bool(dynamic_densities)
        self.dynamic_features = bool(dynamic_features)
        self.dynamic_normals = bool(dynamic_normals)
        self.dynamic_texel_sites = bool(dynamic_texel_sites)
        self.cell_count = int(cell_count)
        self.feature_dim = int(feature_dim)
        self.num_texel_sites = int(num_texel_sites)
        self.static_dynamic_split = bool(static_dynamic_split)
        if dynamic_cells is None:
            dynamic_cell_count = max(1, int(round(self.cell_count * float(dynamic_cell_fraction))))
        else:
            dynamic_cell_count = int(dynamic_cells)
        self.dynamic_cell_count = min(self.cell_count, max(1, dynamic_cell_count))
        self.static_cell_count = self.cell_count - self.dynamic_cell_count if self.static_dynamic_split else 0
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.radius_min = float(radius_min)
        self.neighbor_count = int(neighbor_count)
        self.adjacency_mode = str(adjacency_mode)
        self.texel_site_scale = float(texel_site_scale)
        self.render_size = int(render_size)
        self.fov_degrees = float(fov_degrees)
        self.raster_config = raster_config
        self.camera_decoder = camera_decoder
        self.register_buffer("rays", make_pinhole_rays(render_size, render_size, fov_degrees, torch.device("cpu")), persistent=False)
        basis = make_gaussian_time_basis(frame_count, int(time_basis_count), float(time_basis_sigma_scale))
        self.register_buffer("frame_basis", basis, persistent=False)
        dynamic_mask = torch.zeros(self.cell_count, dtype=torch.float32)
        if self.static_dynamic_split:
            dynamic_mask[self.static_cell_count :] = 1.0
        else:
            dynamic_mask[:] = 1.0
        self.register_buffer("dynamic_cell_mask", dynamic_mask, persistent=False)
        self.temporal_geometry_runtime_scale = 1.0
        self.temporal_feature_runtime_scale = 1.0

        def fit_init(values: torch.Tensor, *, dynamic: bool) -> tuple[torch.Tensor, torch.Tensor]:
            base, coeff = fit_temporal_basis(values, basis, mode=str(temporal_init_mode))
            if not dynamic:
                coeff = torch.zeros_like(coeff)
            return base, coeff

        init_raw_xy0, init_raw_xy_coeff = fit_init(raw_xy, dynamic=self.dynamic_centers)
        init_raw_z0, init_raw_z_coeff = fit_init(raw_z, dynamic=self.dynamic_centers)
        init_raw_radii0, init_raw_radii_coeff = fit_init(raw_radii, dynamic=self.dynamic_radii)
        init_raw_densities0, init_raw_densities_coeff = fit_init(raw_densities, dynamic=self.dynamic_densities)
        init_raw_features0, init_raw_features_coeff = fit_init(raw_features, dynamic=self.dynamic_features)
        init_raw_normals0, init_raw_normals_coeff = fit_init(init_normals, dynamic=self.dynamic_normals)
        init_raw_tangents0, init_raw_tangents_coeff = fit_init(init_tangents, dynamic=self.dynamic_normals)
        init_raw_texel_sites0, init_raw_texel_sites_coeff = fit_init(raw_texel_sites, dynamic=self.dynamic_texel_sites)

        for name, value in {
            "init_raw_xy0": init_raw_xy0,
            "init_raw_xy_coeff": init_raw_xy_coeff,
            "init_raw_z0": init_raw_z0,
            "init_raw_z_coeff": init_raw_z_coeff,
            "init_raw_radii0": init_raw_radii0,
            "init_raw_radii_coeff": init_raw_radii_coeff,
            "init_raw_densities0": init_raw_densities0,
            "init_raw_densities_coeff": init_raw_densities_coeff,
            "init_raw_features0": init_raw_features0,
            "init_raw_features_coeff": init_raw_features_coeff,
            "init_raw_normals0": init_raw_normals0,
            "init_raw_normals_coeff": init_raw_normals_coeff,
            "init_raw_tangents0": init_raw_tangents0,
            "init_raw_tangents_coeff": init_raw_tangents_coeff,
            "init_raw_texel_sites0": init_raw_texel_sites0,
            "init_raw_texel_sites_coeff": init_raw_texel_sites_coeff,
        }.items():
            self.register_buffer(name, value.contiguous(), persistent=False)

        self.register_buffer("initial_points", init_points.clone(), persistent=False)
        self.register_buffer("initial_radii", init_radii.clone(), persistent=False)
        self.register_buffer("initial_densities", init_density.clone(), persistent=False)
        self.register_buffer("initial_features", raw_features.clone(), persistent=False)
        self.register_buffer("initial_normals", init_normals.clone(), persistent=False)
        self.register_buffer("initial_tangents", init_tangents.clone(), persistent=False)
        self.register_buffer("initial_texel_sites", texel_sites_init.clone(), persistent=False)

        basis_count = int(time_basis_count)
        temporal_scale = float(token_temporal_residual_scale)
        self._cell_chunks: list[tuple[str, tuple[int, ...], float]] = []

        def add_chunk(name: str, shape: tuple[int, ...], scale: float) -> None:
            self._cell_chunks.append((name, shape, float(scale)))

        add_chunk("raw_xy0", (2,), float(token_point_residual_scale))
        add_chunk("raw_z0", (1,), float(token_z_residual_scale))
        add_chunk("raw_radii0", (), float(token_radius_residual_scale))
        add_chunk("raw_densities0", (), float(token_density_residual_scale))
        add_chunk("raw_features0", (self.num_texel_sites, self.feature_dim), float(token_feature_residual_scale))
        add_chunk("raw_normals0", (3,), float(token_normal_residual_scale))
        add_chunk("raw_tangents0", (3,), float(token_normal_residual_scale))
        add_chunk("raw_texel_sites0", (self.num_texel_sites, 2), float(token_texel_site_residual_scale))
        if self.dynamic_centers:
            add_chunk("raw_xy_coeff", (basis_count, 2), float(token_point_residual_scale) * temporal_scale)
            add_chunk("raw_z_coeff", (basis_count, 1), float(token_z_residual_scale) * temporal_scale)
        if self.dynamic_radii:
            add_chunk("raw_radii_coeff", (basis_count,), float(token_radius_residual_scale) * temporal_scale)
        if self.dynamic_densities:
            add_chunk("raw_densities_coeff", (basis_count,), float(token_density_residual_scale) * temporal_scale)
        if self.dynamic_features:
            add_chunk(
                "raw_features_coeff",
                (basis_count, self.num_texel_sites, self.feature_dim),
                float(token_feature_residual_scale) * temporal_scale,
            )
        if self.dynamic_normals:
            add_chunk("raw_normals_coeff", (basis_count, 3), float(token_normal_residual_scale) * temporal_scale)
            add_chunk("raw_tangents_coeff", (basis_count, 3), float(token_normal_residual_scale) * temporal_scale)
        if self.dynamic_texel_sites:
            add_chunk(
                "raw_texel_sites_coeff",
                (basis_count, self.num_texel_sites, 2),
                float(token_texel_site_residual_scale) * temporal_scale,
            )

        self.total_decoder_dim = sum(math.prod(shape) if shape else 1 for _name, shape, _scale in self._cell_chunks)
        self.tokens = nn.Parameter(
            float(token_init_std) * torch.randn(self.cell_count, int(token_dim), generator=generator, dtype=init_points.dtype)
        )
        layers: list[nn.Module] = []
        in_dim = int(token_dim)
        for _ in range(int(token_hidden_layers)):
            layers.append(nn.Linear(in_dim, int(token_hidden_dim)))
            layers.append(nn.GELU())
            in_dim = int(token_hidden_dim)
        layers.append(nn.Linear(in_dim, self.total_decoder_dim))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=float(token_output_init_std))
        nn.init.zeros_(layers[-1].bias)
        self.decoder = nn.Sequential(*layers)

    def _token_residuals(self) -> dict[str, torch.Tensor]:
        flat = self.decoder(self.tokens)
        residuals: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, scale in self._cell_chunks:
            width = math.prod(shape) if shape else 1
            value = flat[:, offset : offset + width]
            offset += width
            if shape:
                value = value.reshape(self.cell_count, *shape)
            else:
                value = value.reshape(self.cell_count)
            value = value * scale
            if name.endswith("_coeff"):
                value = value.movedim(1, 0).contiguous()
            residuals[name] = value
        return residuals

    def _mask_temporal_coeff(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not name.endswith("_coeff"):
            return value
        mask = self.dynamic_cell_mask.to(device=value.device, dtype=value.dtype).view(1, self.cell_count, *([1] * (value.dim() - 2)))
        scale = self.temporal_feature_runtime_scale if name == "raw_features_coeff" else self.temporal_geometry_runtime_scale
        return value * mask * float(scale)

    def set_training_controls(self, *, temporal_geometry_scale: float, temporal_feature_scale: float) -> None:
        self.temporal_geometry_runtime_scale = float(temporal_geometry_scale)
        self.temporal_feature_runtime_scale = float(temporal_feature_scale)

    def _raw_state(self) -> dict[str, torch.Tensor]:
        residuals = self._token_residuals()
        names = (
            "raw_xy0",
            "raw_xy_coeff",
            "raw_z0",
            "raw_z_coeff",
            "raw_radii0",
            "raw_radii_coeff",
            "raw_densities0",
            "raw_densities_coeff",
            "raw_features0",
            "raw_features_coeff",
            "raw_normals0",
            "raw_normals_coeff",
            "raw_tangents0",
            "raw_tangents_coeff",
            "raw_texel_sites0",
            "raw_texel_sites_coeff",
        )
        state = {}
        for name in names:
            init = getattr(self, f"init_{name}")
            state[name] = self._mask_temporal_coeff(name, init + residuals.get(name, init.new_zeros(())))
        return state

    def _decode_temporal(self, base: torch.Tensor, coeff: torch.Tensor, frame_indices: torch.Tensor | None) -> torch.Tensor:
        if frame_indices is None:
            basis = self.frame_basis.to(device=base.device, dtype=base.dtype)
        else:
            basis = self.frame_basis[frame_indices.to(device=self.frame_basis.device, dtype=torch.long)].to(
                device=base.device,
                dtype=base.dtype,
            )
        return base[None].expand(basis.shape[0], *base.shape) + torch.einsum("tk,k...->t...", basis, coeff)

    def raw_parameter_tensors(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._raw_state()
        return (
            self._decode_temporal(state["raw_xy0"], state["raw_xy_coeff"], frame_indices),
            self._decode_temporal(state["raw_z0"], state["raw_z_coeff"], frame_indices),
            self._decode_temporal(state["raw_radii0"], state["raw_radii_coeff"], frame_indices),
            self._decode_temporal(state["raw_densities0"], state["raw_densities_coeff"], frame_indices),
            self._decode_temporal(state["raw_features0"], state["raw_features_coeff"], frame_indices),
            self._decode_temporal(state["raw_normals0"], state["raw_normals_coeff"], frame_indices),
            self._decode_temporal(state["raw_tangents0"], state["raw_tangents_coeff"], frame_indices),
            self._decode_temporal(state["raw_texel_sites0"], state["raw_texel_sites_coeff"], frame_indices),
        )

    def decoded_parameters(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_xy, raw_z, raw_radii, raw_densities, raw_features, raw_normals, _raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        xy = torch.tanh(raw_xy) * self.xy_extent
        z = self.z_min + torch.sigmoid(raw_z) * (self.z_max - self.z_min)
        points = torch.cat([xy, z], dim=-1)
        radii = F.softplus(raw_radii, beta=POWERFOAM_SOFTPLUS_BETA) + self.radius_min
        densities = F.softplus(raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)
        normals = F.normalize(raw_normals, dim=-1, eps=1.0e-6)
        return points, radii, densities, raw_features, normals

    def decoded_texel_sites(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        *_prefix, raw_texel_sites = self.raw_parameter_tensors(frame_indices)
        return self.texel_site_scale * torch.tanh(raw_texel_sites)

    def decoded_surface_frame(
        self,
        normals: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _raw_xy, _raw_z, _raw_radii, _raw_densities, _raw_features, _raw_normals, raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        return orthonormal_surface_frame(normals, raw_tangents)

    def optimizer_param_groups(self, train_cfg: dict[str, Any]) -> list[dict[str, object]]:
        lr = float(train_cfg["lr"])
        groups: list[dict[str, object]] = [
            {"params": [self.tokens], "lr": float(train_cfg["token_lr_multiplier"]) * lr, "name": "tokens"},
            {
                "params": list(self.decoder.parameters()),
                "lr": float(train_cfg["decoder_lr_multiplier"]) * lr,
                "name": "decoder",
            },
        ]
        camera_group = camera_param_group(self.camera_decoder, train_cfg)
        if camera_group is not None:
            groups.append(camera_group)
        return groups

    def temporal_regularization(self, loss_cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        points, radii, densities, features, _normals = self.decoded_parameters()
        state = self._raw_state()
        coeffs = [
            state["raw_xy_coeff"],
            state["raw_z_coeff"],
            state["raw_radii_coeff"],
            state["raw_densities_coeff"],
            state["raw_features_coeff"],
            state["raw_normals_coeff"],
            state["raw_tangents_coeff"],
            state["raw_texel_sites_coeff"],
        ]
        terms = {
            "temporal_center_accel": temporal_accel(points),
            "temporal_radius_accel": temporal_accel(radii),
            "temporal_density_accel": temporal_accel(densities),
            "temporal_feature_accel": temporal_accel(features),
            "temporal_coeff_l2": torch.stack([coeff.square().mean() for coeff in coeffs]).mean(),
        }
        loss = points.new_zeros(())
        for key, value in terms.items():
            loss = loss + float(loss_cfg[f"{key}_weight"]) * value
        camera_loss, camera_terms = camera_regularization(self.camera_decoder, loss_cfg)
        if camera_loss is not None:
            loss = loss + camera_loss
            terms.update(camera_terms)
        return loss, terms

    @torch.no_grad()
    def parameter_drift_metrics(self) -> dict[str, float]:
        points, radii, densities, features, normals = self.decoded_parameters()
        texel_sites = self.decoded_texel_sites()
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
            texel_sites=texel_sites,
            initial_texel_sites=self.initial_texel_sites,
        )
        metrics.update(
            {
                "state_token_rms": float(self.tokens.detach().square().mean().sqrt().cpu()),
                "state_static_cell_count": float(self.static_cell_count),
                "state_dynamic_cell_count": float(
                    self.dynamic_cell_count if self.static_dynamic_split else self.cell_count
                ),
                "state_dynamic_cell_fraction": float(
                    (self.dynamic_cell_count if self.static_dynamic_split else self.cell_count) / self.cell_count
                ),
            }
        )
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices()
        metrics.update(
            temporal_motion_metrics(
                points,
                features,
                render_size=self.render_size,
                fov_degrees=self.fov_degrees,
                camera_to_world=camera_to_world,
            )
        )
        state = self._raw_state()
        coeff_abs = [
            state["raw_xy_coeff"].abs().mean(),
            state["raw_z_coeff"].abs().mean(),
            state["raw_radii_coeff"].abs().mean(),
            state["raw_densities_coeff"].abs().mean(),
            state["raw_features_coeff"].abs().mean(),
            state["raw_normals_coeff"].abs().mean(),
            state["raw_tangents_coeff"].abs().mean(),
            state["raw_texel_sites_coeff"].abs().mean(),
        ]
        metrics["state_mean_temporal_coeff_abs"] = float(torch.stack(coeff_abs).mean().cpu())
        metrics.update(compact_camera_metrics(self.camera_decoder))
        return metrics

    def decoded_camera_rays(self, frame_indices: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return decoded_powerfoam_rays(
            self.camera_decoder,
            self.rays,
            frame_indices.to(device=next(self.parameters()).device, dtype=torch.long),
            height=self.render_size,
            width=self.render_size,
            device=next(self.parameters()).device,
            dtype=dtype,
        )

    def forward(self, frame_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if next(self.parameters()).device.type != "mps":
            raise RuntimeError("TokenDynamicPowerFoamFeatures requires an MPS device")
        frame_indices = frame_indices.to(device=next(self.parameters()).device, dtype=torch.long)
        points, radii, densities, features, normals = self.decoded_parameters(frame_indices)
        texel_sites = self.decoded_texel_sites(frame_indices)
        tangents, bitangents = self.decoded_surface_frame(normals, frame_indices)
        rays = self.rays.to(device=points.device, dtype=points.dtype)
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices(frame_indices)
        renders = []
        alphas = []
        for local_index in range(int(frame_indices.numel())):
            point = points[local_index]
            normal = normals[local_index]
            tangent = tangents[local_index]
            bitangent = bitangents[local_index]
            if camera_to_world is not None:
                point, normal, tangent, bitangent = transform_powerfoam_frame_to_camera(
                    point,
                    normal,
                    tangent,
                    bitangent,
                    camera_to_world[local_index],
                )
            radius = radii[local_index]
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            out, alpha = rasterize_power_foam_oriented_texel_surface(
                point,
                radius,
                densities[local_index],
                texel_sites[local_index],
                features[local_index],
                normal,
                adjacency,
                offsets,
                rays,
                self.raster_config,
                tangents=tangent,
                bitangents=bitangent,
            )
            renders.append(out.permute(0, 3, 1, 2))
            alphas.append(alpha)
        return torch.cat(renders, dim=0), torch.cat(alphas, dim=0)


FoamModel = DynamicMetalPowerFoamVideo | TokenDynamicPowerFoamFeatures


def log_artifacts(
    model: FoamModel,
    colorizer: FeatureToColor | None,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
) -> dict[str, float]:
    model.eval()
    if colorizer is not None:
        colorizer.eval()
    renders, features, alphas = render_all(
        model,
        targets.size(0),
        batch_size=powerfoam_eval_batch_size(cfg),
        cfg=cfg,
        colorizer=colorizer,
    )
    frame_metrics = per_frame_reconstruction_metrics(renders, targets.cpu())
    metrics = {
        **reconstruction_l1_mse_metrics(renders, targets.cpu(), prefix="eval"),
        "eval_frame_psnr_mean": frame_metrics["summary"]["frame_psnr_mean"],
        "eval_frame_psnr_min": frame_metrics["summary"]["frame_psnr_min"],
        "eval_frame_snr_mean": frame_metrics["summary"]["frame_snr_mean"],
        "eval_frame_snr_min": frame_metrics["summary"]["frame_snr_min"],
        "eval_alpha_mean": float(alphas.mean().cpu()),
        "eval_feature_mean": float(features.mean().cpu()),
        "eval_feature_std": float(features.std().cpu()),
    }
    metrics.update(temporal_alpha_metrics(alphas))
    metrics.update(model.parameter_drift_metrics())
    write_json(output_dir / f"per_frame_metrics_step_{step:04d}.json", frame_metrics)
    save_rgb_alpha_eval_media(
        output_dir,
        step,
        renders,
        targets,
        alphas,
        fps=video_fps_from_config(cfg),
        save_videos=should_log_video(cfg, step),
    )

    def _wandb_payload() -> dict[str, Any]:
        fps = video_fps_from_config(cfg)
        payload: dict[str, Any] = mapped_metric_payload(
            metrics,
            (
                ("eval_l1", "Eval/L1"),
                ("eval_mse", "Eval/MSE"),
                ("eval_alpha_mean", "Eval/AlphaMean"),
                ("eval_feature_mean", "Eval/FeatureMean"),
                ("eval_feature_std", "Eval/FeatureStd"),
                ("state_mean_center_delta", "State/MeanCenterDelta"),
                ("state_p95_center_delta", "State/P95CenterDelta"),
                ("state_max_center_delta", "State/MaxCenterDelta"),
                ("state_mean_xy_delta", "State/MeanXYDelta"),
                ("state_mean_z_delta", "State/MeanZDelta"),
                ("state_mean_radius_delta", "State/MeanRadiusDelta"),
                ("state_mean_density_delta", "State/MeanDensityDelta"),
                ("state_mean_feature_delta", "State/MeanFeatureDelta"),
                ("state_mean_normal_delta", "State/MeanNormalDelta"),
                ("state_mean_texel_site_delta", "State/MeanTexelSiteDelta"),
                ("state_mean_temporal_xy_delta", "State/MeanTemporalXYDelta"),
                ("state_p95_temporal_xy_delta", "State/P95TemporalXYDelta"),
                ("state_mean_temporal_z_delta", "State/MeanTemporalZDelta"),
                ("state_mean_temporal_screen_delta_px", "State/MeanTemporalScreenDeltaPx"),
                ("state_p95_temporal_screen_delta_px", "State/P95TemporalScreenDeltaPx"),
                ("state_temporal_screen_valid_fraction", "State/TemporalScreenValidFraction"),
                ("state_mean_temporal_feature_abs_delta", "State/MeanTemporalFeatureAbsDelta"),
            ),
        )
        payload.update(
            mapped_metric_payload(
                metrics,
                (
                    ("state_mean_temporal_coeff_abs", "State/MeanTemporalCoeffAbs"),
                    ("state_camera_fov_degrees", "Camera/FovDegrees"),
                    ("state_camera_radius", "Camera/Radius"),
                    ("state_camera_rotation_delta_mean_degrees", "Camera/RotationDeltaMeanDegrees"),
                    ("state_camera_translation_delta_mean", "Camera/TranslationDeltaMean"),
                    ("state_camera_origin_delta_mean", "Camera/OriginDeltaMean"),
                    ("state_camera_forward_delta_mean", "Camera/ForwardDeltaMean"),
                    ("state_camera_global_residual_l2", "Camera/GlobalResidualL2"),
                    ("state_camera_active_frames", "Camera/ActiveFrames"),
                    ("state_camera_velocity_l2", "Camera/VelocityL2"),
                    ("state_camera_acceleration_l2", "Camera/AccelerationL2"),
                    ("state_camera_gimbal_l2", "Camera/GimbalL2"),
                ),
                require=False,
            )
        )
        payload.update(
            build_rgb_alpha_eval_media_payload(
                renders,
                targets,
                alphas,
                step=step,
                fps=fps,
                include_videos=should_log_video(cfg, step),
            )
        )
        return payload

    log_wandb_run_payload_lazy(wandb_run, _wandb_payload, step=step)
    model.train()
    if colorizer is not None:
        colorizer.train()
    return metrics


def dynamic_geometry_summary(
    cfg: dict[str, Any],
    history: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    eval_rows = [row for row in history if row.get("kind") == "eval"]
    train_rows = [row for row in history if row.get("kind") == "train"]
    initial = eval_rows[0] if eval_rows else {}
    final = eval_rows[-1] if eval_rows else {}
    best = min(eval_rows, key=lambda row: float(row.get("eval_l1", float("inf"))), default={})
    return {
        "schema_version": "dynamic_powerfoam_geometry_summary_v1",
        "status": "ok" if final else "missing_eval",
        "output_dir": str(output_dir),
        "config": {
            "video_path": str(cfg["data"]["video_path"]),
            "frames": int(cfg["data"]["max_frames"]),
            "render_size": int(cfg["render"]["render_size"]),
            "steps": int(cfg["train"]["steps"]),
            "cells": int(cfg["model"]["cells"]),
            "dynamic_mode": str(cfg["model"]["dynamic_mode"]),
            "dynamic_centers": bool(cfg["model"]["dynamic_centers"]),
            "dynamic_radii": bool(cfg["model"]["dynamic_radii"]),
            "dynamic_densities": bool(cfg["model"]["dynamic_densities"]),
            "dynamic_features": bool(cfg["model"]["dynamic_features"]),
            "dynamic_normals": bool(cfg["model"]["dynamic_normals"]),
            "dynamic_texel_sites": bool(cfg["model"]["dynamic_texel_sites"]),
            "static_dynamic_split": bool(cfg["model"]["static_dynamic_split"]),
            "dynamic_cells": cfg["model"]["dynamic_cells"],
            "dynamic_cell_fraction": float(cfg["model"]["dynamic_cell_fraction"]),
            "video_init_mode": str(cfg["model"]["video_init_mode"]),
            "camera_enabled": bool(cfg["camera"]["enabled"]),
            "camera_mode": str(cfg["camera"]["mode"]),
            "camera_base_path_mode": str(cfg["camera"]["base_path_mode"]),
            "camera_path_parameterization": str(cfg["camera"]["path_parameterization"]),
            "camera_init_teacher_steps": int(cfg["camera"]["init_teacher_steps"]),
            "camera_initial_zoom_steps": int(cfg["camera"]["initial_zoom_steps"]),
            "camera_initial_zoom_translation": float(cfg["camera"]["initial_zoom_translation"]),
            "static_only_steps": int(cfg["train"]["static_only_steps"]),
            "no_repaint_steps": int(cfg["train"]["no_repaint_steps"]),
            "camera_curriculum_enabled": bool(cfg["train"]["camera_curriculum_enabled"]),
            "camera_curriculum_schedule": cfg["train"]["camera_curriculum_schedule"],
        },
        "initial_eval": initial,
        "final_eval": final,
        "best_eval": best,
        "last_train": train_rows[-1] if train_rows else {},
        "motion_vs_repaint": {
            "state_mean_temporal_screen_delta_px": float(final.get("state_mean_temporal_screen_delta_px", 0.0)),
            "state_p95_temporal_screen_delta_px": float(final.get("state_p95_temporal_screen_delta_px", 0.0)),
            "state_temporal_screen_valid_fraction": float(final.get("state_temporal_screen_valid_fraction", 0.0)),
            "state_mean_temporal_feature_abs_delta": float(final.get("state_mean_temporal_feature_abs_delta", 0.0)),
            "eval_mean_temporal_alpha_delta": float(final.get("eval_mean_temporal_alpha_delta", 0.0)),
            "eval_mean_temporal_support_delta": float(final.get("eval_mean_temporal_support_delta", 0.0)),
            "state_camera_rotation_delta_mean_degrees": float(
                final.get("state_camera_rotation_delta_mean_degrees", 0.0)
            ),
            "state_camera_translation_delta_mean": float(final.get("state_camera_translation_delta_mean", 0.0)),
        },
        "artifacts": {
            "history": str(output_dir / "train_metrics_history.jsonl"),
            "checkpoint": str(output_dir / "checkpoint_final.pt"),
            "final_per_frame_metrics": str(output_dir / f"per_frame_metrics_step_{int(final.get('step', 0)):04d}.json")
            if final
            else None,
            "final_render": str(output_dir / f"render_step_{int(final.get('step', 0)):04d}.mp4") if final else None,
            "final_side_by_side": str(output_dir / f"side_by_side_step_{int(final.get('step', 0)):04d}.mp4") if final else None,
        },
    }


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_torch_device(str(cfg["train"]["device"]), auto_cuda=False)
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("dynamic_powerfoam_metal requires torch MPS")

    output_dir: Path = cfg["logging"]["output_dir"]
    write_resolved_config(output_dir, cfg)
    history_path = output_dir / "train_metrics_history.jsonl"
    summary_path = output_dir / "dynamic_geometry_summary.json"
    for path in (history_path, summary_path):
        if path.exists():
            path.unlink()
    metrics_history: list[dict[str, Any]] = []

    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=int(cfg["render"]["render_size"]),
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=str(cfg["data"]["frame_source"]),
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    cfg["video_fps"] = float(sequence.video_fps)
    with wandb_run_lifecycle(cfg) as wandb_run:
        camera_decoder = build_camera_decoder(cfg, frame_count=int(targets.size(0)))
        teacher_init_metrics = prefit_camera_decoder_from_teacher(
            camera_decoder,
            cfg,
            frame_count=int(targets.size(0)),
            device=device,
            output_dir=output_dir,
        )
        if teacher_init_metrics:
            print({"camera_teacher_init": teacher_init_metrics})

        model_kwargs = {
            "frame_count": targets.size(0),
            "cell_count": int(cfg["model"]["cells"]),
            "render_size": int(cfg["render"]["render_size"]),
            "fov_degrees": float(cfg["render"]["fov_degrees"]),
            "neighbor_count": int(cfg["model"]["neighbor_count"]),
            "adjacency_mode": str(cfg["model"]["adjacency_mode"]),
            "time_basis_count": int(cfg["model"]["time_basis_count"]),
            "time_basis_sigma_scale": float(cfg["model"]["time_basis_sigma_scale"]),
            "temporal_init_mode": str(cfg["model"]["temporal_init_mode"]),
            "dynamic_centers": bool(cfg["model"]["dynamic_centers"]),
            "dynamic_radii": bool(cfg["model"]["dynamic_radii"]),
            "dynamic_densities": bool(cfg["model"]["dynamic_densities"]),
            "dynamic_features": bool(cfg["model"]["dynamic_features"]),
            "dynamic_normals": bool(cfg["model"]["dynamic_normals"]),
            "dynamic_texel_sites": bool(cfg["model"]["dynamic_texel_sites"]),
            "xy_extent": float(cfg["model"]["xy_extent"]),
            "z_min": float(cfg["model"]["z_min"]),
            "z_max": float(cfg["model"]["z_max"]),
            "radius_init": float(cfg["model"]["radius_init"]),
            "radius_min": float(cfg["model"]["radius_min"]),
            "radius_scale": float(cfg["model"]["radius_scale"]),
            "density_init": float(cfg["model"]["density_init"]),
            "normal_init_jitter": float(cfg["model"]["normal_init_jitter"]),
            "num_texel_sites": int(cfg["model"]["num_texel_sites"]),
            "texel_site_scale": float(cfg["model"]["texel_site_scale"]),
            "color_init_mode": str(cfg["model"]["color_init_mode"]),
            "video_init_mode": str(cfg["model"]["video_init_mode"]),
            "seed": int(cfg["train"]["seed"]),
            "init_frames": targets.detach().cpu() if bool(cfg["model"]["init_from_video"]) else None,
            "image_init_depth": None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"]),
            "image_init_jitter": float(cfg["model"]["image_init_jitter"]),
            "raster_config": make_raster_config(cfg["render"]),
            "camera_decoder": camera_decoder,
        }
        if str(cfg["model"]["dynamic_mode"]) == TOKEN_RBF_FEATURE_MODE:
            model = TokenDynamicPowerFoamFeatures(
                **model_kwargs,
                feature_dim=int(cfg["model"]["feature_dim"]),
                feature_init_noise=float(cfg["model"]["feature_init_noise"]),
                feature_rgb_init=str(cfg["model"]["feature_rgb_init"]),
                token_dim=int(cfg["model"]["token_dim"]),
                token_hidden_dim=int(cfg["model"]["token_hidden_dim"]),
                token_hidden_layers=int(cfg["model"]["token_hidden_layers"]),
                token_init_std=float(cfg["model"]["token_init_std"]),
                token_output_init_std=float(cfg["model"]["token_output_init_std"]),
                token_point_residual_scale=float(cfg["model"]["token_point_residual_scale"]),
                token_z_residual_scale=float(cfg["model"]["token_z_residual_scale"]),
                token_radius_residual_scale=float(cfg["model"]["token_radius_residual_scale"]),
                token_density_residual_scale=float(cfg["model"]["token_density_residual_scale"]),
                token_feature_residual_scale=float(cfg["model"]["token_feature_residual_scale"]),
                token_normal_residual_scale=float(cfg["model"]["token_normal_residual_scale"]),
                token_texel_site_residual_scale=float(cfg["model"]["token_texel_site_residual_scale"]),
                token_temporal_residual_scale=float(cfg["model"]["token_temporal_residual_scale"]),
                static_dynamic_split=bool(cfg["model"]["static_dynamic_split"]),
                dynamic_cells=None if cfg["model"]["dynamic_cells"] is None else int(cfg["model"]["dynamic_cells"]),
                dynamic_cell_fraction=float(cfg["model"]["dynamic_cell_fraction"]),
            ).to(device)
        else:
            model = DynamicMetalPowerFoamVideo(
                **model_kwargs,
                dynamic_mode=str(cfg["model"]["dynamic_mode"]),
            ).to(device)
        colorizer = build_dynamic_powerfoam_colorizer(
            cfg,
            device=device,
            feature_dynamic_mode=TOKEN_RBF_FEATURE_MODE,
        )
        param_groups = model.optimizer_param_groups(cfg["train"])
        if colorizer is not None:
            param_groups.append(
                {
                    "params": list(colorizer.parameters()),
                    "lr": float(cfg["train"]["colorize_lr_multiplier"]) * float(cfg["train"]["lr"]),
                    "name": "colorize",
                }
            )
        optimizer = torch.optim.Adam(param_groups, lr=float(cfg["train"]["lr"]))

        print(
            {
                "arch": "dynamic_powerfoam_metal",
                "dynamic_mode": str(cfg["model"]["dynamic_mode"]),
                "device": str(device),
                "video_path": str(cfg["data"]["video_path"]),
                "frames": int(targets.size(0)),
                "render_size": int(cfg["render"]["render_size"]),
                "cells": int(cfg["model"]["cells"]),
                "feature_dim": int(cfg["model"]["feature_dim"]),
                "camera_mode": str(cfg["camera"]["mode"]),
                "camera_enabled": bool(cfg["camera"]["enabled"]),
                "camera_path_parameterization": str(cfg["camera"]["path_parameterization"]),
                "train_background_mode": str(cfg["render"]["train_background_mode"]),
                "eval_background_mode": str(cfg["render"]["eval_background_mode"]),
                "neighbors": int(cfg["model"]["neighbor_count"]),
                "steps": int(cfg["train"]["steps"]),
            }
        )
        apply_training_stage(model, cfg, 0)
        if teacher_init_metrics:
            init_record = {"kind": "camera_teacher_init", "step": 0, **teacher_init_metrics}
            metrics_history.append(init_record)
            append_jsonl(history_path, init_record)
        initial_metrics = log_artifacts(model, colorizer, targets, cfg, 0, output_dir, wandb_run)
        initial_record = {"kind": "eval", "step": 0, **initial_metrics}
        metrics_history.append(initial_record)
        append_jsonl(history_path, initial_record)
        print({"step": 0, **initial_metrics})

        start_time = time.perf_counter()
        progress = trange(1, int(cfg["train"]["steps"]) + 1, desc=f"dynamic_powerfoam_{cfg['model']['dynamic_mode']}")
        for step in progress:
            stage_controls = apply_training_stage(model, cfg, step)
            frame_indices = powerfoam_train_batch_indices(
                int(stage_controls["stage_camera_active_frames"]),
                cfg,
                device=device,
            )
            target = targets[frame_indices]
            features, alpha = model(frame_indices)
            background = sample_background(
                cfg["render"],
                phase="train",
                batch_size=int(features.shape[0]),
                device=features.device,
                dtype=features.dtype,
            )
            rendered = render_features_to_rgb(
                features,
                alpha,
                colorizer,
                background,
                normalize_features_by_alpha=bool(cfg["render"]["normalize_features_by_alpha"]),
                eps=float(cfg["render"]["eps"]),
            )
            l1 = F.l1_loss(rendered, target)
            mse = F.mse_loss(rendered, target)
            _points, radii, densities, _features, _normals = model.decoded_parameters()
            temporal_loss, temporal_terms = model.temporal_regularization(cfg["losses"])
            loss = (
                float(cfg["losses"]["l1_weight"]) * l1
                + float(cfg["losses"]["mse_weight"]) * mse
                + float(cfg["losses"]["radius_l2_weight"]) * radii.square().mean()
                + float(cfg["losses"]["density_l2_weight"]) * densities.square().mean()
                + temporal_loss
            )
            optimizer_backward_step(optimizer, loss)

            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", l1=f"{float(l1.detach().cpu()):.4f}")
            if should_log_scalar(cfg, step):
                elapsed = time.perf_counter() - start_time
                train_metrics = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "l1": float(l1.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "temporal": float(temporal_loss.detach().cpu()),
                    "elapsed_s": elapsed,
                    **stage_controls,
                }
                for key in (
                    "camera_rotation_l2",
                    "camera_translation_l2",
                    "camera_temporal_l2",
                    "camera_global_l2",
                    "camera_velocity_l2",
                    "camera_acceleration_l2",
                    "camera_gimbal_l2",
                ):
                    if key in temporal_terms:
                        train_metrics[key] = float(temporal_terms[key].detach().cpu())
                metrics_history.append({"kind": "train", **train_metrics})
                append_jsonl(history_path, {"kind": "train", **train_metrics})
                print(train_metrics)
                def _wandb_train_payload() -> dict[str, Any]:
                    payload = mapped_metric_payload(train_metrics, DYNAMIC_POWERFOAM_TRAIN_WANDB_KEYS)
                    payload.update(
                        {
                            "Train/AlphaMean": float(alpha.detach().mean().cpu()),
                            "Train/FeatureMean": float(features.detach().mean().cpu()),
                            "Train/FeatureStd": float(features.detach().std().cpu()),
                            "Train/TemporalCenterAccel": float(
                                temporal_terms["temporal_center_accel"].detach().cpu()
                            ),
                            "Train/TemporalCoeffL2": float(temporal_terms["temporal_coeff_l2"].detach().cpu()),
                        }
                    )
                    payload.update(
                        mapped_metric_payload(
                            train_metrics,
                            DYNAMIC_POWERFOAM_TRAIN_CAMERA_WANDB_KEYS,
                            require=False,
                        )
                    )
                    return payload

                log_wandb_run_payload_lazy(wandb_run, _wandb_train_payload, step=step)
            if should_log_image(cfg, step):
                metrics = log_artifacts(model, colorizer, targets, cfg, step, output_dir, wandb_run)
                eval_record = {"kind": "eval", "step": step, **metrics}
                metrics_history.append(eval_record)
                append_jsonl(history_path, eval_record)
                print({"step": step, **metrics})

        checkpoint: dict[str, Any] = {"model": model.state_dict(), "config": serialize_config_value(cfg)}
        if colorizer is not None:
            checkpoint["colorizer"] = colorizer.state_dict()
        atomic_torch_save(checkpoint, output_dir / "checkpoint_final.pt")
        write_json(summary_path, dynamic_geometry_summary(cfg, metrics_history, output_dir))
