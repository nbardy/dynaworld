from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional, Dict

import torch
import torch.nn as nn
import wandb


DYNAWORLD_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from config_utils import apply_defaults, load_config_file, path_or_none, resolved_config, serialize_config_value
from multicam_val_data import load_multicam_val_manifest, load_multicam_val_sample
from sequence_data import load_uncalibrated_sequence, select_window_indices
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video


DEFAULT_CONFIG_PATH = "src/train_configs/local_mac_gauge_fields_material_surfel_128_16f_512el.jsonc"
SUPPORT_MODES = {"screen_disk", "oriented_slab", "rank_adaptive_metric"}
OPACITY_TRANSFERS = {"linear", "optical_thickness"}
SUPPORT_LOG_SCALE_MIN = -12.0
SUPPORT_LOG_SCALE_MAX = 4.0

DATA_DEFAULTS = {
    "sequence_dir": "test_data",
    "frames_dir": None,
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 0,
    "frame_indices": None,
    "multicam_manifest": "data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl",
    "multicam_split": "val",
    "multicam_sample_id": None,
    "multicam_sample_index": 0,
}

MODEL_DEFAULTS = {
    "num_elements": 512,
    "num_basis": 8,
    "support_mode": "screen_disk",
    "support_knn_k": 8,
    "support_jacobian_lambda": 1e-4,
    "init_basis_std": 0.001,
    "init_coeff_std": 0.0,
    "init_depth": 3.0,
    "init_radius": 0.05,
    "init_thickness": 0.005,
    "init_alpha_logit": -1.2,
    "slab_rotation_init_std": 0.0,
    "metric_offdiag_scale": 0.01,
}

CAMERA_DEFAULTS = {
    "lens_model": "pinhole",
    "base_fov_degrees": 60.0,
    "multicam_pose_source": "auto",
}

RENDER_DEFAULTS = {
    "render_size": 128,
    "background": [1.0, 1.0, 1.0],
    "near_plane": 1e-3,
    "far_plane": 1e4,
    "min_radius_px": 0.75,
    "max_radius_px": 24.0,
    "max_alpha_per_element": 0.95,
    "opacity_transfer": "linear",
    "pixel_chunk": 2048,
}

TRAIN_DEFAULTS = {
    "steps": 250,
    "lr": 2e-3,
    "device": "auto",
    "seed": 0,
    "frames_per_step": 1,
    "train_frame_count": 16,
}

LOSS_DEFAULTS = {
    "rgb_weight": 1.0,
    "query_weight": 0.25,
    "flow_weight": 0.0,
    "depth_weight": 0.0,
    "arap_weight": 0.05,
    "smooth_weight": 0.02,
    "mass_weight": 1e-3,
    "radius_weight": 1e-3,
}

LOGGING_DEFAULTS = {
    "log_every": 25,
    "log_to_wandb": True,
    "wandb_project": "dynaworld",
    "wandb_run_name": "gauge-fields-material-surfel-128-16f-512el",
    "wandb_tags": ["gauge-fields", "material-surfel", "128px"],
    "wandb_mode": "online",
    "output_dir": "outputs/gauge_fields/material_surfel_128_16f_512el",
}

DIAGNOSTIC_DEFAULTS = {
    "projection_stats": True,
    "xmap_metrics": True,
    "motion_stats": True,
    "flow_stats": False,
    "xmap_bins": 16,
    "xmap_alpha_min": 0.05,
}


# ----------------------------
# Utilities
# ----------------------------

@dataclass
class RenderConfig:
    H: int
    W: int
    near: float = 1e-3
    far: float = 1e4
    bg: float = 1.0
    min_radius_px: float = 0.75
    max_radius_px: float = 24.0
    max_alpha_per_element: float = 0.95
    opacity_transfer: str = "linear"
    pixel_chunk: int = 4096


def make_pixel_grid(H: int, W: int, device):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    return torch.stack([xx.reshape(-1).float(), yy.reshape(-1).float()], dim=-1)


def project_points(
    x_world: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    near: float = 1e-3,
    far: float = 1e4,
):
    """
    x_world: [N,3]
    K: [3,3]
    w2c: [4,4], camera convention: camera looks along +Z.
    Returns:
        uv: [N,2] pixel coordinates
        z: [N]
        valid: [N]
    """
    N = x_world.shape[0]
    ones = torch.ones(N, 1, device=x_world.device, dtype=x_world.dtype)
    xh = torch.cat([x_world, ones], dim=-1)

    x_cam = (xh @ w2c.T)[..., :3]
    z = x_cam[:, 2].clamp_min(1e-8)

    x_norm = x_cam / z[:, None]
    uvh = x_norm @ K.T
    uv = uvh[:, :2]

    valid = (x_cam[:, 2] > near) & (x_cam[:, 2] < far)
    return uv, x_cam[:, 2], valid


def project_points_with_jacobian(
    x_world: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    near: float = 1e-3,
    far: float = 1e4,
):
    """
    Returns projection plus d(pixel)/d(world_xyz) for world-space support kernels.
    The camera convention matches project_points: camera looks along +Z.
    """
    N = x_world.shape[0]
    ones = torch.ones(N, 1, device=x_world.device, dtype=x_world.dtype)
    xh = torch.cat([x_world, ones], dim=-1)
    x_cam = (xh @ w2c.T)[..., :3]
    z = x_cam[:, 2]
    z_safe = z.clamp_min(float(near))

    x_norm = x_cam / z_safe[:, None]
    uvh = x_norm @ K.T
    uv = uvh[:, :2]

    fx = K[0, 0]
    fy = K[1, 1]
    j_cam = torch.zeros(N, 2, 3, device=x_world.device, dtype=x_world.dtype)
    j_cam[:, 0, 0] = fx / z_safe
    j_cam[:, 0, 2] = -fx * x_cam[:, 0] / (z_safe * z_safe)
    j_cam[:, 1, 1] = fy / z_safe
    j_cam[:, 1, 2] = -fy * x_cam[:, 1] / (z_safe * z_safe)
    j_world = j_cam @ w2c[:3, :3]

    valid = (z > near) & (z < far)
    return uv, z, valid, x_cam, j_world


def build_knn_index(x0: torch.Tensor, k: int = 8) -> torch.Tensor:
    if x0.shape[0] <= 1 or k <= 0:
        return torch.empty(x0.shape[0], 0, device=x0.device, dtype=torch.long)
    kk = min(int(k), x0.shape[0] - 1)
    with torch.no_grad():
        d = torch.cdist(x0.detach(), x0.detach())
        return d.topk(k=kk + 1, largest=False).indices[:, 1:]


def build_knn_edges(x0: torch.Tensor, k: int = 8):
    """
    Builds canonical neighborhood edges once.
    x0: [N,3]
    Returns:
        edges: [E,2]
        rest_lengths: [E]
    """
    with torch.no_grad():
        d = torch.cdist(x0, x0)
        idx = d.topk(k=k + 1, largest=False).indices[:, 1:]  # skip self
        src = torch.arange(x0.shape[0], device=x0.device)[:, None].expand_as(idx)
        edges = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=-1)
        rest = (x0[edges[:, 0]] - x0[edges[:, 1]]).norm(dim=-1)
    return edges, rest


def validate_support_mode(value: str) -> str:
    mode = str(value)
    if mode not in SUPPORT_MODES:
        raise ValueError(f"model.support_mode must be one of {sorted(SUPPORT_MODES)}, got {value!r}.")
    return mode


def validate_opacity_transfer(value: str) -> str:
    transfer = str(value)
    if transfer not in OPACITY_TRANSFERS:
        raise ValueError(
            f"render.opacity_transfer must be one of {sorted(OPACITY_TRANSFERS)}, got {value!r}."
        )
    return transfer


def estimate_local_deformation_jacobian(
    x0: torch.Tensor,
    xt: torch.Tensor,
    knn_idx: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Estimate D Phi_t at each material element from fixed canonical neighbors.

    This uses an identity-plus-displacement fit instead of the raw Q P^T
    expression. That preserves the unconstrained normal direction when the
    initialized points are nearly planar.
    """
    N = x0.shape[0]
    eye = torch.eye(3, device=x0.device, dtype=x0.dtype).expand(N, 3, 3)
    if knn_idx.numel() == 0:
        return eye

    p = x0[knn_idx] - x0[:, None, :]      # [N,K,3]
    q = xt[knn_idx] - xt[:, None, :]      # [N,K,3]
    d = q - p

    p_m = p.transpose(-1, -2)             # [N,3,K]
    d_m = d.transpose(-1, -2)             # [N,3,K]
    gram = p_m @ p_m.transpose(-1, -2)
    gram = gram + float(lam) * torch.eye(3, device=x0.device, dtype=x0.dtype)
    update = d_m @ p_m.transpose(-1, -2) @ torch.linalg.inv(gram)
    return eye + update


def bound_projected_covariance(cov2: torch.Tensor, cfg: RenderConfig) -> torch.Tensor:
    max_var = float(cfg.max_radius_px) ** 2
    min_var = float(cfg.min_radius_px) ** 2
    cov2 = torch.nan_to_num(cov2, nan=0.0, posinf=max_var, neginf=0.0)
    cov2 = 0.5 * (cov2 + cov2.transpose(-1, -2))
    a = cov2[:, 0, 0]
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1]
    trace = a + c
    disc = torch.sqrt(((a - c) * (a - c) + 4.0 * b * b).clamp_min(0.0))
    lambda_min = (0.5 * (trace - disc)).clamp(min=min_var, max=max_var)
    lambda_max = (0.5 * (trace + disc)).clamp(min=min_var, max=max_var)

    v_raw = torch.stack([b, lambda_max - a], dim=-1)
    fallback_x = torch.tensor([1.0, 0.0], device=cov2.device, dtype=cov2.dtype).expand_as(v_raw)
    fallback_y = torch.tensor([0.0, 1.0], device=cov2.device, dtype=cov2.dtype).expand_as(v_raw)
    near_diagonal = b.abs() < 1e-8
    fallback = torch.where((a >= c)[:, None], fallback_x, fallback_y)
    v_raw = torch.where(near_diagonal[:, None], fallback, v_raw)
    v = v_raw / v_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    vv = v[:, :, None] @ v[:, None, :]
    eye2 = torch.eye(2, device=cov2.device, dtype=cov2.dtype).expand(cov2.shape[0], 2, 2)
    out = lambda_max[:, None, None] * vv + lambda_min[:, None, None] * (eye2 - vv)
    return torch.nan_to_num(out, nan=min_var, posinf=max_var, neginf=min_var)


def covariance_equivalent_radius_and_anisotropy(cov2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cov2 = torch.nan_to_num(cov2, nan=0.0, posinf=1.0e6, neginf=0.0)
    a = cov2[:, 0, 0]
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1]
    trace = a + c
    disc = torch.sqrt(((a - c) * (a - c) + 4.0 * b * b).clamp_min(0.0))
    lambda_min = (0.5 * (trace - disc)).clamp_min(1e-8)
    lambda_max = (0.5 * (trace + disc)).clamp_min(1e-8)
    radius_eq = torch.sqrt(torch.sqrt((lambda_min * lambda_max).clamp_min(1e-16)))
    anisotropy = torch.sqrt(lambda_max / lambda_min)
    return radius_eq, anisotropy


def exp_bounded_log_scale(log_scale: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_scale.clamp(SUPPORT_LOG_SCALE_MIN, SUPPORT_LOG_SCALE_MAX)).clamp_min(1e-6)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta_sq = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp_min(1e-12))
    theta_safe = theta.clamp_min(1e-6)
    theta_sq_safe = theta_sq.clamp_min(1e-12)
    small = theta_sq < 1e-8
    sin_over_theta = torch.where(
        small,
        1.0 - theta_sq / 6.0,
        torch.sin(theta) / theta_safe,
    )
    one_minus_cos_over_theta_sq = torch.where(
        small,
        0.5 - theta_sq / 24.0,
        (1.0 - torch.cos(theta)) / theta_sq_safe,
    )

    wx, wy, wz = axis_angle.unbind(dim=-1)
    K = torch.zeros(*axis_angle.shape[:-1], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[..., 0, 1] = -wz
    K[..., 0, 2] = wy
    K[..., 1, 0] = wz
    K[..., 1, 2] = -wx
    K[..., 2, 0] = -wy
    K[..., 2, 1] = wx

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(K)
    return eye + sin_over_theta[..., None] * K + one_minus_cos_over_theta_sq[..., None] * (K @ K)


def robust_l1(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()


def tensor_scalar(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def finite_values(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().reshape(-1).float().cpu()
    return values[torch.isfinite(values)]


def stats_for_tensor(prefix: str, values: torch.Tensor) -> dict[str, float]:
    finite = finite_values(values)
    if finite.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_p05": float(torch.quantile(finite, 0.05)),
        f"{prefix}_p50": float(torch.quantile(finite, 0.50)),
        f"{prefix}_p95": float(torch.quantile(finite, 0.95)),
        f"{prefix}_max": float(finite.max()),
    }


def mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row and math.isfinite(row[key])]
        out[key] = float(sum(vals) / max(len(vals), 1))
    return out


# ----------------------------
# Model: transported material samples
# ----------------------------

class MaterialSurfelField(nn.Module):
    """
    Universal material elements.

    This is not free 4D splat soup:
      - x_i(t) is derived from canonical material x_i^0 and shared low-rank motion.
      - color and opacity are persistent.
      - radius is persistent.
      - there is no free covariance per frame.
    """

    def __init__(
        self,
        init_x0: torch.Tensor,   # [N,3]
        num_frames: int,
        num_basis: int = 8,
        support_mode: str = "screen_disk",
        support_knn_k: int = 8,
        support_jacobian_lambda: float = 1e-4,
        init_radius: float = 0.015,
        init_thickness: float = 0.005,
        init_color: Optional[torch.Tensor] = None,
        init_alpha_logit: float = -2.0,
        init_basis_std: float = 0.001,
        init_coeff_std: float = 0.0,
        slab_rotation_init_std: float = 0.0,
        metric_offdiag_scale: float = 0.01,
    ):
        super().__init__()
        N = init_x0.shape[0]
        self.N = N
        self.T = num_frames
        self.L = num_basis
        self.support_mode = validate_support_mode(support_mode)
        self.support_knn_k = max(0, min(int(support_knn_k), max(0, N - 1)))
        self.support_jacobian_lambda = float(support_jacobian_lambda)
        self.metric_offdiag_scale = float(metric_offdiag_scale)

        self.x0 = nn.Parameter(init_x0.clone())                     # [N,3]
        if init_color is None:
            color_logits = torch.zeros(N, 3, device=init_x0.device, dtype=init_x0.dtype)
        else:
            color = init_color.to(device=init_x0.device, dtype=init_x0.dtype).clamp(1e-4, 1.0 - 1e-4)
            color_logits = torch.logit(color)
        self.color_logits = nn.Parameter(color_logits)              # [N,3]
        self.raw_alpha = nn.Parameter(torch.full((N, 1), init_alpha_logit, device=init_x0.device))  # [N,1]
        self.log_radius = nn.Parameter(
            torch.full((N, 1), math.log(init_radius), device=init_x0.device, dtype=init_x0.dtype)
        )
        slab_scales = torch.tensor(
            [init_radius, init_radius, init_thickness],
            device=init_x0.device,
            dtype=init_x0.dtype,
        ).clamp_min(1e-6)
        self.slab_log_scales = nn.Parameter(torch.log(slab_scales).expand(N, 3).clone())
        self.slab_raw_rot = nn.Parameter(
            float(slab_rotation_init_std) * torch.randn(N, 3, device=init_x0.device, dtype=init_x0.dtype)
        )
        self.metric_log_diag = nn.Parameter(
            torch.full((N, 3), math.log(init_radius), device=init_x0.device, dtype=init_x0.dtype)
        )
        self.metric_offdiag = nn.Parameter(torch.zeros(N, 3, device=init_x0.device, dtype=init_x0.dtype))

        # Low-rank material deformation:
        # x_i(t) = x_i^0 + sum_l coeff[t,l] * basis[i,l,:]
        self.nr_basis = nn.Parameter(
            float(init_basis_std) * torch.randn(N, num_basis, 3, device=init_x0.device, dtype=init_x0.dtype)
        )
        self.nr_coeff = nn.Parameter(
            float(init_coeff_std) * torch.randn(num_frames, num_basis, device=init_x0.device, dtype=init_x0.dtype)
        )
        self.register_buffer(
            "support_knn_idx",
            build_knn_index(init_x0, self.support_knn_k),
            persistent=True,
        )

    def positions(self, t: int):
        if self.L == 0:
            return self.x0
        delta = torch.einsum("nlc,l->nc", self.nr_basis, self.nr_coeff[t])
        return self.x0 + delta

    def colors(self):
        return torch.sigmoid(self.color_logits)

    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def radius(self):
        return exp_bounded_log_scale(self.log_radius)

    def slab_scales(self):
        return exp_bounded_log_scale(self.slab_log_scales)

    def slab_rotation(self):
        return axis_angle_to_matrix(self.slab_raw_rot)

    def metric_covariance(self):
        diag = exp_bounded_log_scale(self.metric_log_diag)
        offdiag = self.metric_offdiag * self.metric_offdiag_scale
        L = torch.zeros(self.N, 3, 3, device=self.x0.device, dtype=self.x0.dtype)
        L[:, 0, 0] = diag[:, 0]
        L[:, 1, 0] = offdiag[:, 0]
        L[:, 1, 1] = diag[:, 1]
        L[:, 2, 0] = offdiag[:, 1]
        L[:, 2, 1] = offdiag[:, 2]
        L[:, 2, 2] = diag[:, 2]
        return L @ L.transpose(-1, -2)

    def world_support_covariance(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.support_mode == "oriented_slab":
            R = self.slab_rotation()
            local_cov = torch.diag_embed(self.slab_scales().square())
            base_cov = R @ local_cov @ R.transpose(-1, -2)
        elif self.support_mode == "rank_adaptive_metric":
            base_cov = self.metric_covariance()
        else:
            raise ValueError("screen_disk does not use a world-space support covariance.")

        J = estimate_local_deformation_jacobian(
            self.x0,
            x_t,
            self.support_knn_idx,
            self.support_jacobian_lambda,
        )
        eye3 = torch.eye(3, device=self.x0.device, dtype=self.x0.dtype).expand(self.N, 3, 3)
        return J @ base_cov @ J.transpose(-1, -2) + 1e-10 * eye3

    def motion_smoothness_loss(self):
        if self.L == 0:
            return self.x0.new_zeros(())
        # velocity + acceleration smoothness on shared temporal coefficients
        vel = self.nr_coeff[1:] - self.nr_coeff[:-1]
        loss = (vel ** 2).mean()

        if self.T >= 3:
            acc = self.nr_coeff[2:] - 2 * self.nr_coeff[1:-1] + self.nr_coeff[:-2]
            loss = loss + 5.0 * (acc ** 2).mean()

        # control basis magnitude to avoid free per-element teleportation
        loss = loss + 1e-2 * (self.nr_basis ** 2).mean()
        return loss

    def mass_loss(self):
        return self.alpha().mean()

    def radius_loss(self):
        # prevents huge image-space blobs
        if self.support_mode == "screen_disk":
            return (self.radius() ** 2).mean()
        if self.support_mode == "oriented_slab":
            return (self.slab_scales() ** 2).mean()
        trace = self.metric_covariance().diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return trace.mean() / 3.0


# ----------------------------
# Differentiable soft renderer
# ----------------------------

def projected_support(
    model: MaterialSurfelField,
    x_t: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    cfg: RenderConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    uv, z, valid, _x_cam, j_world = project_points_with_jacobian(x_t, K, w2c, cfg.near, cfg.far)

    if model.support_mode == "screen_disk":
        fx = K[0, 0].abs().clamp_min(1e-6)
        radius_px = model.radius()[:, 0] * fx / z.abs().clamp_min(cfg.near)
        radius_px = radius_px.clamp(cfg.min_radius_px, cfg.max_radius_px)
        cov2 = torch.zeros(model.N, 2, 2, device=x_t.device, dtype=x_t.dtype)
        cov2[:, 0, 0] = radius_px.square()
        cov2[:, 1, 1] = radius_px.square()
        return uv, z, valid, cov2

    sigma3 = model.world_support_covariance(x_t)
    cov2 = j_world @ sigma3 @ j_world.transpose(-1, -2)
    cov2 = bound_projected_covariance(cov2, cfg)
    return uv, z, valid, cov2


def render_material_field(
    model: MaterialSurfelField,
    t: int,
    K: torch.Tensor,
    w2c: torch.Tensor,
    cfg: RenderConfig,
    K_next: Optional[torch.Tensor] = None,
    w2c_next: Optional[torch.Tensor] = None,
    t_next: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Pure Torch soft projected-kernel renderer.

    For production, replace with a faster renderer.
    For research, this is enough to test whether the representation trains.

    Returns:
        rgb:   [H,W,3]
        alpha: [H,W]
        depth: [H,W]
        flow:  [H,W,2] if next frame/camera supplied
        xmap:  [H,W,3] alpha-normalized canonical coordinate
    """
    device = model.x0.device
    H, W = cfg.H, cfg.W
    P = H * W

    x_t = model.positions(t)                         # [N,3]
    uv, z, valid, cov2 = projected_support(model, x_t, K, w2c, cfg)

    grid = make_pixel_grid(H, W, device=device)      # [P,2]

    alpha_i = model.alpha()[:, 0]                    # [N]

    # Approximate global depth sort. This is not exact per-pixel ordering,
    # but good enough for a toy loop.
    z_sort_key = torch.where(valid, z, torch.full_like(z, cfg.far + 1.0))
    order = torch.argsort(z_sort_key, dim=0)         # near to far

    uv_s = uv[order]                                 # [N,2]
    z_s = z[order].clamp_min(cfg.near)               # [N]
    cov2_s = cov2[order]                             # [N,2,2]
    alpha_s = alpha_i[order]                         # [N]
    valid_s = valid[order]                           # [N]
    color_s = model.colors()[order]                  # [N,3]
    x0_s = model.x0[order]                           # [N,3]

    cov_a = cov2_s[:, 0, 0]
    cov_b = cov2_s[:, 0, 1]
    cov_c = cov2_s[:, 1, 1]
    det = (cov_a * cov_c - cov_b * cov_b).clamp_min(1e-8)
    inv00 = cov_c / det
    inv01 = -cov_b / det
    inv11 = cov_a / det

    flow_s = None
    if K_next is not None and w2c_next is not None and t_next is not None:
        x_next = model.positions(t_next)
        uv_next, _, valid_next = project_points(x_next, K_next, w2c_next, cfg.near, cfg.far)
        flow_i = uv_next - uv                         # [N,2]
        flow_i = flow_i * (valid & valid_next)[:, None].float()
        flow_s = flow_i[order]

    rgb_chunks = []
    alpha_chunks = []
    depth_chunks = []
    xmap_chunks = []
    flow_chunks = []

    pixel_chunk = P if cfg.pixel_chunk <= 0 else min(cfg.pixel_chunk, P)
    for start in range(0, P, pixel_chunk):
        end = min(start + pixel_chunk, P)
        pix = grid[start:end]                         # [C,2]

        diff = pix[None, :, :] - uv_s[:, None, :]     # [N,C,2]
        dx = diff[..., 0]
        dy = diff[..., 1]
        maha = inv00[:, None] * dx * dx + 2.0 * inv01[:, None] * dx * dy + inv11[:, None] * dy * dy
        kernel = torch.exp(-0.5 * maha.clamp_min(0.0))

        if cfg.opacity_transfer == "optical_thickness":
            a_s = 1.0 - torch.exp(-alpha_s[:, None].clamp_min(0.0) * kernel)
        else:
            a_s = alpha_s[:, None] * kernel
        a_s = a_s.clamp(0.0, cfg.max_alpha_per_element)
        a_s = a_s * valid_s[:, None].float()

        one_minus = (1.0 - a_s).clamp(1e-5, 1.0)
        trans = torch.cumprod(
            torch.cat([torch.ones(1, end - start, device=device), one_minus[:-1]], dim=0),
            dim=0,
        )
        weights = trans * a_s                         # [N,C]

        alpha_map = weights.sum(dim=0).clamp(0.0, 1.0)
        rgb = weights.T @ color_s
        rgb = rgb + (1.0 - alpha_map)[:, None] * cfg.bg

        depth_num = (weights * z_s[:, None]).sum(dim=0)
        depth = depth_num / alpha_map.clamp_min(1e-6)

        xmap_num = weights.T @ x0_s
        xmap = xmap_num / alpha_map[:, None].clamp_min(1e-6)

        rgb_chunks.append(rgb)
        alpha_chunks.append(alpha_map)
        depth_chunks.append(depth)
        xmap_chunks.append(xmap)

        if flow_s is not None:
            flow_num = weights.T @ flow_s
            flow = flow_num / alpha_map[:, None].clamp_min(1e-6)
            flow_chunks.append(flow)

    out = {
        "rgb": torch.cat(rgb_chunks, dim=0).reshape(H, W, 3),
        "alpha": torch.cat(alpha_chunks, dim=0).reshape(H, W),
        "depth": torch.cat(depth_chunks, dim=0).reshape(H, W),
        "xmap": torch.cat(xmap_chunks, dim=0).reshape(H, W, 3),
    }

    # Render induced optical flow from material transport if requested.
    if flow_s is not None:
        out["flow"] = torch.cat(flow_chunks, dim=0).reshape(H, W, 2)

    return out


# ----------------------------
# Losses
# ----------------------------

def arap_loss(
    model: MaterialSurfelField,
    t: int,
    edges: torch.Tensor,
    rest_lengths: torch.Tensor,
):
    """
    Local distance preservation in material neighborhoods.
    This is the cloth/isometry-ish contract.
    """
    x = model.positions(t)
    d = (x[edges[:, 0]] - x[edges[:, 1]]).norm(dim=-1)
    return ((d - rest_lengths) ** 2).mean()


def flow_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    alpha: torch.Tensor,
    alpha_min: float = 0.05,
):
    """
    pred_flow: [H,W,2]
    gt_flow: [H,W,2]
    alpha: [H,W]
    """
    mask = (alpha.detach() > alpha_min) & torch.isfinite(gt_flow).all(dim=-1)
    if mask.sum() == 0:
        return pred_flow.new_zeros(())
    return robust_l1(pred_flow[mask] - gt_flow[mask])


def scale_shift_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    alpha: torch.Tensor,
    alpha_min: float = 0.05,
):
    """
    Monocular depth is usually only affine/scale meaningful.
    Align pred to target by detached least squares, then apply L1.
    """
    mask = (
        (alpha.detach() > alpha_min)
        & torch.isfinite(pred_depth)
        & torch.isfinite(target_depth)
        & (target_depth > 0)
    )
    if mask.sum() < 32:
        return pred_depth.new_zeros(())

    x = pred_depth[mask].reshape(-1)
    y = target_depth[mask].reshape(-1)

    A = torch.stack([x.detach(), torch.ones_like(x)], dim=-1)
    sol = torch.linalg.lstsq(A, y.detach()).solution
    a, b = sol[0], sol[1]

    aligned = a * pred_depth + b
    return robust_l1(aligned[mask] - target_depth[mask])


# ----------------------------
# Training loop
# ----------------------------

@dataclass
class TrainWeights:
    rgb: float = 1.0
    query: float = 0.25
    flow: float = 0.05
    depth: float = 0.02
    arap: float = 0.05
    smooth: float = 0.02
    mass: float = 1e-3
    radius: float = 1e-3


def train_material_surfel_field(
    video: torch.Tensor,              # [T,H,W,3], float in [0,1]
    K: torch.Tensor,                  # [3,3] or [T,3,3]
    w2c: torch.Tensor,                # [T,4,4]
    init_x0: torch.Tensor,            # [N,3]
    init_color: Optional[torch.Tensor] = None,  # [N,3]
    flow: Optional[torch.Tensor] = None,   # [T-1,H,W,2], optional
    depth_prior: Optional[torch.Tensor] = None,  # [T,H,W], optional
    num_steps: int = 2000,
    batch_size: int = 1,
    train_frame_count: int = 0,
    num_basis: int = 8,
    support_mode: str = "screen_disk",
    support_knn_k: int = 8,
    support_jacobian_lambda: float = 1e-4,
    init_radius: float = 0.04,
    init_thickness: float = 0.005,
    init_alpha_logit: float = -1.2,
    init_basis_std: float = 0.001,
    init_coeff_std: float = 0.0,
    slab_rotation_init_std: float = 0.0,
    metric_offdiag_scale: float = 0.01,
    lr: float = 2e-3,
    weights: TrainWeights = TrainWeights(),
    query_every: int = 4,
    render_cfg: Optional[RenderConfig] = None,
    log_every: int = 50,
):
    """
    Minimal single-video 4D material-field train loop.

    Important:
      - This is a toy loop.
      - It is O(N*H*W), so keep H/W/N small at first.
      - Good first config: H=W=64, N=256-1024, T=20-100.
    """
    device = video.device
    T, H, W, _ = video.shape
    cfg = render_cfg if render_cfg is not None else RenderConfig(H=H, W=W)
    if cfg.H != H or cfg.W != W:
        raise ValueError(f"RenderConfig size {(cfg.H, cfg.W)} does not match video size {(H, W)}.")

    model = MaterialSurfelField(
        init_x0=init_x0,
        num_frames=T,
        num_basis=num_basis,
        support_mode=support_mode,
        support_knn_k=support_knn_k,
        support_jacobian_lambda=support_jacobian_lambda,
        init_color=init_color,
        init_radius=init_radius,
        init_thickness=init_thickness,
        init_alpha_logit=init_alpha_logit,
        init_basis_std=init_basis_std,
        init_coeff_std=init_coeff_std,
        slab_rotation_init_std=slab_rotation_init_std,
        metric_offdiag_scale=metric_offdiag_scale,
    ).to(device)

    edges, rest_lengths = build_knn_edges(init_x0.detach(), k=8)
    edges = edges.to(device)
    rest_lengths = rest_lengths.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def get_K(t: int):
        return K[t] if K.ndim == 3 else K

    logs = []

    for step in range(num_steps):
        opt.zero_grad(set_to_none=True)

        # Context frames: sample from the same contiguous-window idea used by
        # the video-token baselines, but keep frames_per_step small for this
        # pure Torch toy renderer.
        if train_frame_count > 0 and train_frame_count < T:
            window = select_window_indices(T, train_frame_count, device=device)
            local = torch.randint(0, window.numel(), (batch_size,), device=device)
            frames = window[local]
        else:
            frames = torch.randint(0, T, (batch_size,), device=device)

        total_loss = video.new_zeros(())
        rgb_meter = 0.0

        for tb in frames.tolist():
            use_flow = flow is not None and tb < T - 1

            out = render_material_field(
                model=model,
                t=tb,
                K=get_K(tb),
                w2c=w2c[tb],
                cfg=cfg,
                K_next=get_K(tb + 1) if use_flow else None,
                w2c_next=w2c[tb + 1] if use_flow else None,
                t_next=tb + 1 if use_flow else None,
            )

            rgb_l = robust_l1(out["rgb"] - video[tb])
            rgb_meter += float(rgb_l.detach())

            loss = weights.rgb * rgb_l
            loss = loss + weights.arap * arap_loss(model, tb, edges, rest_lengths)

            if use_flow:
                loss = loss + weights.flow * flow_loss(out["flow"], flow[tb], out["alpha"])

            if depth_prior is not None:
                loss = loss + weights.depth * scale_shift_depth_loss(
                    out["depth"], depth_prior[tb], out["alpha"]
                )

            total_loss = total_loss + loss

        total_loss = total_loss / float(batch_size)

        # Omitted/query frame loss.
        # Very simple version: every few steps, render a frame not in the sampled context batch.
        # No private paths exist in this toy model, so query loss directly pressures shared material geometry.
        if weights.query > 0 and (step % query_every == 0):
            tq = torch.randint(0, T, (1,), device=device).item()
            out_q = render_material_field(
                model=model,
                t=tq,
                K=get_K(tq),
                w2c=w2c[tq],
                cfg=cfg,
            )
            query_l = robust_l1(out_q["rgb"] - video[tq])
            total_loss = total_loss + weights.query * query_l

        total_loss = total_loss + weights.smooth * model.motion_smoothness_loss()
        total_loss = total_loss + weights.mass * model.mass_loss()
        total_loss = total_loss + weights.radius * model.radius_loss()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == num_steps - 1:
            log = {
                "step": step,
                "loss": float(total_loss.detach()),
                "rgb_l1": rgb_meter / float(batch_size),
                "mass": float(model.mass_loss().detach()),
                "motion_smooth": float(model.motion_smoothness_loss().detach()),
                "radius": float(model.radius().mean().detach()),
                "support_radius_loss": float(model.radius_loss().detach()),
            }
            logs.append(log)
            print(log)

    return model, logs


# ----------------------------
# Baseline video harness
# ----------------------------


def resolve_dynaworld_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return DYNAWORLD_ROOT / value


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class GaugeVideoBundle:
    video: torch.Tensor
    K: torch.Tensor
    w2c: torch.Tensor
    fps: float = 4.0
    source_path: str | None = None
    metadata: dict[str, Any] | None = None
    heldout_video: torch.Tensor | None = None
    heldout_K: torch.Tensor | None = None
    heldout_w2c: torch.Tensor | None = None
    heldout_pose_source: str | None = None


def make_fixed_pinhole_camera(
    num_frames: int,
    H: int,
    W: int,
    fov_degrees: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fov = math.radians(fov_degrees)
    focal = 0.5 * float(W) / math.tan(0.5 * fov)
    K = torch.tensor(
        [
            [focal, 0.0, float(W) * 0.5],
            [0.0, focal, float(H) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    w2c = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
    return K, w2c


def make_scaled_intrinsics(
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    source_width: float,
    source_height: float,
    target_width: int,
    target_height: int,
    device: torch.device,
) -> torch.Tensor:
    sx = float(target_width) / float(source_width)
    sy = float(target_height) / float(source_height)
    return torch.tensor(
        [
            [float(fx) * sx, 0.0, float(cx) * sx],
            [0.0, float(fy) * sy, float(cy) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def rodrigues_matrix(axis_angle: list[float] | tuple[float, ...], device: torch.device) -> torch.Tensor:
    r = torch.tensor(axis_angle, dtype=torch.float32, device=device)
    theta = torch.linalg.norm(r).clamp_min(1e-8)
    rx, ry, rz = r
    skew = torch.stack(
        [
            torch.stack([r.new_zeros(()), -rz, ry]),
            torch.stack([rz, r.new_zeros(()), -rx]),
            torch.stack([-ry, rx, r.new_zeros(())]),
        ]
    )
    eye = torch.eye(3, dtype=torch.float32, device=device)
    return eye + (torch.sin(theta) / theta) * skew + ((1.0 - torch.cos(theta)) / (theta * theta)) * (skew @ skew)


def deepview_camera_from_models(
    record: dict[str, Any],
    camera_name: str,
    *,
    H: int,
    W: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    models_path = Path(record["models_path"])
    models = json.loads(models_path.read_text(encoding="utf-8"))
    by_name = {str(model["name"]): model for model in models}
    if camera_name not in by_name:
        raise KeyError(f"DeepView camera {camera_name!r} not found in {models_path}.")
    model = by_name[camera_name]

    focal = float(model["focal_length"])
    pixel_aspect = float(model.get("pixel_aspect_ratio", 1.0))
    principal = model["principal_point"]
    K = make_scaled_intrinsics(
        fx=focal,
        fy=focal * pixel_aspect,
        cx=float(principal[0]),
        cy=float(principal[1]),
        source_width=float(model["width"]),
        source_height=float(model["height"]),
        target_width=W,
        target_height=H,
        device=device,
    )

    # DeepView stores a Rodrigues world-to-camera rotation for an OpenGL-style
    # camera. Convert it to the gauge renderer's camera frame: +x right,
    # +y down, +z forward.
    w2c_gl_rot = rodrigues_matrix(model["orientation"], device=device)
    c2w_gl_rot = w2c_gl_rot.T
    gl_to_plus_z = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32, device=device))
    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, :3] = c2w_gl_rot @ gl_to_plus_z
    c2w[:3, 3] = torch.tensor(model["position"], dtype=torch.float32, device=device)
    return K, c2w


def make_multicam_pair_cameras(
    record: dict[str, Any],
    *,
    T: int,
    H: int,
    W: int,
    fov_degrees: float,
    pose_source: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if pose_source not in {"auto", "deepview", "source_proxy"}:
        raise ValueError("camera.multicam_pose_source must be one of: auto, deepview, source_proxy")

    if pose_source in {"auto", "deepview"} and record.get("dataset") == "deepview_video" and record.get("models_path"):
        source_K, source_c2w = deepview_camera_from_models(
            record,
            str(record["source_camera"]),
            H=H,
            W=W,
            device=device,
        )
        target_K, target_c2w = deepview_camera_from_models(
            record,
            str(record["target_camera"]),
            H=H,
            W=W,
            device=device,
        )
        source_w2c = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(T, 1, 1)
        target_w2c = torch.linalg.inv(target_c2w) @ source_c2w
        target_w2c = target_w2c.unsqueeze(0).repeat(T, 1, 1)
        return source_K, source_w2c, target_K, target_w2c, "deepview_models_relative_pinhole"

    if pose_source == "deepview":
        raise ValueError(
            f"Requested DeepView camera calibration for non-DeepView record {record.get('sample_id')!r}."
        )

    source_K, source_w2c = make_fixed_pinhole_camera(
        num_frames=T,
        H=H,
        W=W,
        fov_degrees=fov_degrees,
        device=device,
    )
    target_K = source_K.clone()
    target_w2c = source_w2c.clone()
    return source_K, source_w2c, target_K, target_w2c, "source_camera_proxy_uncalibrated"


def load_baseline_video(
    sequence_dir: Path,
    frames_dir: Optional[Path],
    video_path: Optional[Path],
    frame_source: str,
    render_size: int,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    sequence = load_uncalibrated_sequence(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        target_size=render_size,
        max_frames=max_frames,
        frame_source=frame_source,
        device=device,
    )
    return sequence.frames.permute(0, 2, 3, 1).contiguous()


def select_multicam_record(data_cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_path = resolve_dynaworld_path(data_cfg["multicam_manifest"])
    records = load_multicam_val_manifest(manifest_path, split=data_cfg["multicam_split"])
    sample_id = data_cfg.get("multicam_sample_id")
    if sample_id:
        for record in records:
            if str(record.get("sample_id")) == str(sample_id):
                return record
        raise ValueError(f"multicam_sample_id={sample_id!r} was not found in {manifest_path}.")

    index = int(data_cfg.get("multicam_sample_index", 0))
    if index < 0 or index >= len(records):
        raise IndexError(f"multicam_sample_index={index} out of range for {len(records)} records.")
    return records[index]


def select_configured_frames(video: torch.Tensor, frame_indices: Any) -> torch.Tensor:
    if frame_indices is None:
        return video
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError("data.frame_indices must be a non-empty list of integer frame indices when provided.")
    indices = torch.as_tensor(frame_indices, dtype=torch.long, device=video.device)
    if bool((indices < 0).any()) or bool((indices >= video.shape[0]).any()):
        raise IndexError(f"data.frame_indices {frame_indices!r} out of range for {video.shape[0]} loaded frames.")
    return video[indices].contiguous()


def load_gauge_video_bundle(
    *,
    data_cfg: dict[str, Any],
    camera_cfg: dict[str, Any],
    render_size: int,
    device: torch.device,
) -> GaugeVideoBundle:
    frame_source = str(data_cfg["frame_source"])
    if frame_source == "multicam_val":
        record = select_multicam_record(data_cfg)
        sample = load_multicam_val_sample(record, target_size=render_size, device=device)
        video = sample.source_frames.permute(0, 2, 3, 1).contiguous()
        heldout_video = sample.target_frames.permute(0, 2, 3, 1).contiguous()
        max_frames = int(data_cfg["max_frames"])
        if max_frames > 0:
            video = video[:max_frames].contiguous()
            heldout_video = heldout_video[:max_frames].contiguous()
        video = select_configured_frames(video, data_cfg["frame_indices"])
        heldout_video = select_configured_frames(heldout_video, data_cfg["frame_indices"])
        if video.shape != heldout_video.shape:
            raise ValueError(
                f"Source/heldout frame shape mismatch for {record.get('sample_id')}: "
                f"{tuple(video.shape)} vs {tuple(heldout_video.shape)}"
            )
        T, H, W, _ = video.shape
        K, w2c, heldout_K, heldout_w2c, pose_note = make_multicam_pair_cameras(
            record,
            T=T,
            H=H,
            W=W,
            fov_degrees=float(camera_cfg["base_fov_degrees"]),
            pose_source=str(camera_cfg["multicam_pose_source"]),
            device=device,
        )
        return GaugeVideoBundle(
            video=video,
            K=K,
            w2c=w2c,
            fps=float(record.get("fps", 4.0)),
            source_path=str(record.get("source_video_path")),
            metadata=record,
            heldout_video=heldout_video,
            heldout_K=heldout_K,
            heldout_w2c=heldout_w2c,
            heldout_pose_source=pose_note,
        )

    sequence_dir = resolve_dynaworld_path(data_cfg["sequence_dir"])
    frames_dir = path_or_none(data_cfg["frames_dir"])
    if frames_dir is not None:
        frames_dir = resolve_dynaworld_path(frames_dir)
    video_path = path_or_none(data_cfg["video_path"])
    if video_path is not None:
        video_path = resolve_dynaworld_path(video_path)
    video = load_baseline_video(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        frame_source=frame_source,
        render_size=render_size,
        max_frames=int(data_cfg["max_frames"]),
        device=device,
    )
    video = select_configured_frames(video, data_cfg["frame_indices"])
    T, H, W, _ = video.shape
    K, w2c = make_fixed_pinhole_camera(
        num_frames=T,
        H=H,
        W=W,
        fov_degrees=float(camera_cfg["base_fov_degrees"]),
        device=device,
    )
    return GaugeVideoBundle(
        video=video,
        K=K,
        w2c=w2c,
        fps=4.0,
        source_path=str(video_path) if video_path is not None else None,
        metadata=None,
    )


def initialize_material_points_from_first_frame(
    video: torch.Tensor,
    K: torch.Tensor,
    num_elements: int,
    init_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, H, W, _ = video.shape
    device = video.device

    grid_x = max(1, math.ceil(math.sqrt(float(num_elements) * float(W) / float(H))))
    grid_y = max(1, math.ceil(float(num_elements) / float(grid_x)))

    xs = torch.linspace(0.5, float(W) - 0.5, grid_x, device=device)
    ys = torch.linspace(0.5, float(H) - 0.5, grid_y, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:num_elements]

    z = torch.full((pixels.shape[0],), init_depth, device=device)
    x = (pixels[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) * z / K[1, 1]
    x0 = torch.stack([x, y, z], dim=-1)

    px = pixels[:, 0].round().long().clamp(0, W - 1)
    py = pixels[:, 1].round().long().clamp(0, H - 1)
    color = video[0, py, px]
    return x0, color


@torch.no_grad()
def render_sequence(
    model: MaterialSurfelField,
    K: torch.Tensor,
    w2c: torch.Tensor,
    cfg: RenderConfig,
    include_flow: bool = False,
) -> dict[str, torch.Tensor]:
    rgbs = []
    alphas = []
    depths = []
    xmaps = []
    flows = []
    for t in range(model.T):
        use_flow = include_flow and t < model.T - 1
        out = render_material_field(
            model,
            t=t,
            K=K[t] if K.ndim == 3 else K,
            w2c=w2c[t],
            cfg=cfg,
            K_next=(K[t + 1] if K.ndim == 3 else K) if use_flow else None,
            w2c_next=w2c[t + 1] if use_flow else None,
            t_next=t + 1 if use_flow else None,
        )
        rgbs.append(out["rgb"])
        alphas.append(out["alpha"])
        depths.append(out["depth"])
        xmaps.append(out["xmap"])
        if use_flow:
            flows.append(out["flow"])
    rendered = {
        "rgb": torch.stack(rgbs, dim=0),
        "alpha": torch.stack(alphas, dim=0),
        "depth": torch.stack(depths, dim=0),
        "xmap": torch.stack(xmaps, dim=0),
    }
    if flows:
        rendered["flow"] = torch.stack(flows, dim=0)
    return rendered


def video_metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = rendered - target
    l1 = diff.abs().mean()
    mse = (diff ** 2).mean().clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    return {
        "eval_l1": float(l1.detach().cpu()),
        "eval_mse": float(mse.detach().cpu()),
        "eval_psnr": float(psnr.detach().cpu()),
    }


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def alpha_metrics(alpha: torch.Tensor) -> dict[str, float]:
    return {
        "alpha_mean": float(alpha.mean().detach().cpu()),
        "alpha_coverage_005": float((alpha > 0.05).float().mean().detach().cpu()),
        "alpha_coverage_050": float((alpha > 0.50).float().mean().detach().cpu()),
        "alpha_coverage_090": float((alpha > 0.90).float().mean().detach().cpu()),
        "alpha_hole_fraction": float((alpha < 0.05).float().mean().detach().cpu()),
        "alpha_max": float(alpha.max().detach().cpu()),
    }


def model_metrics(model: MaterialSurfelField) -> dict[str, float]:
    metrics = {
        "model_alpha_mean": float(model.alpha().mean().detach().cpu()),
        "model_radius_mean": float(model.radius().mean().detach().cpu()),
        "model_radius_max": float(model.radius().max().detach().cpu()),
        "model_support_radius_loss": float(model.radius_loss().detach().cpu()),
        "model_motion_smooth": float(model.motion_smoothness_loss().detach().cpu()),
    }
    if model.support_mode == "oriented_slab":
        scales = model.slab_scales().detach()
        metrics.update(
            {
                "model_slab_r1_mean": float(scales[:, 0].mean().cpu()),
                "model_slab_r2_mean": float(scales[:, 1].mean().cpu()),
                "model_slab_thickness_mean": float(scales[:, 2].mean().cpu()),
                "model_slab_thickness_ratio_mean": float(
                    (scales[:, 2] / scales[:, :2].mean(dim=-1).clamp_min(1e-8)).mean().cpu()
                ),
            }
        )
    if model.support_mode == "rank_adaptive_metric":
        cov = model.metric_covariance().detach()
        diag = cov.diagonal(dim1=-2, dim2=-1)
        metrics.update(
            {
                "model_metric_trace_mean": float(diag.sum(dim=-1).mean().cpu()),
                "model_metric_diag_x_mean": float(diag[:, 0].mean().cpu()),
                "model_metric_diag_y_mean": float(diag[:, 1].mean().cpu()),
                "model_metric_diag_z_mean": float(diag[:, 2].mean().cpu()),
            }
        )
    return metrics


@torch.no_grad()
def projection_health_metrics(
    model: MaterialSurfelField,
    K: torch.Tensor,
    w2c: torch.Tensor,
    cfg: RenderConfig,
    frames: Optional[list[int]] = None,
) -> dict[str, float]:
    frames = list(range(model.T)) if frames is None else frames
    rows: list[dict[str, float]] = []
    for t in frames:
        K_t = K[t] if K.ndim == 3 else K
        x = model.positions(int(t))
        _, z, valid, cov2 = projected_support(model, x, K_t, w2c[int(t)], cfg)
        radius_eq, anisotropy = covariance_equivalent_radius_and_anisotropy(cov2)
        valid_radius = radius_eq[valid]
        valid_anisotropy = anisotropy[valid]
        if valid_radius.numel() == 0:
            coverage_budget = 0.0
        else:
            coverage_budget = tensor_scalar(math.pi * valid_radius.square().sum() / float(cfg.H * cfg.W))
        row = {
            "projection_valid_fraction": tensor_scalar(valid.float().mean()),
            "projection_radius_min_clamp_fraction": tensor_scalar((radius_eq <= cfg.min_radius_px).float().mean()),
            "projection_radius_max_clamp_fraction": tensor_scalar((radius_eq >= cfg.max_radius_px).float().mean()),
            "projection_coverage_budget": coverage_budget,
        }
        row.update(stats_for_tensor("projection_radius_px", valid_radius))
        row.update(stats_for_tensor("projection_anisotropy", valid_anisotropy))
        row.update(stats_for_tensor("projection_depth", z[valid]))
        rows.append(row)
    return mean_metric_rows(rows)


@torch.no_grad()
def motion_health_metrics(model: MaterialSurfelField) -> dict[str, float]:
    if model.L == 0:
        return {
            "motion_delta_mean": 0.0,
            "motion_delta_p50": 0.0,
            "motion_delta_p95": 0.0,
            "motion_delta_max": 0.0,
            "motion_basis_norm_mean": 0.0,
            "motion_basis_norm_p95": 0.0,
            "motion_coeff_norm_mean": 0.0,
            "motion_coeff_velocity_mean": 0.0,
            "motion_coeff_acceleration_mean": 0.0,
        }

    deltas = []
    for t in range(model.T):
        deltas.append((model.positions(t) - model.x0).norm(dim=-1))
    delta = torch.stack(deltas, dim=0)
    basis_norm = model.nr_basis.norm(dim=-1).reshape(-1)
    coeff_abs = model.nr_coeff.abs().reshape(-1)
    velocity = model.nr_coeff[1:] - model.nr_coeff[:-1]

    metrics = {
        **stats_for_tensor("motion_delta", delta),
        "motion_basis_norm_mean": float(finite_values(basis_norm).mean()) if basis_norm.numel() else 0.0,
        "motion_basis_norm_p95": float(torch.quantile(finite_values(basis_norm), 0.95)) if basis_norm.numel() else 0.0,
        "motion_coeff_norm_mean": float(finite_values(coeff_abs).mean()) if coeff_abs.numel() else 0.0,
        "motion_coeff_velocity_mean": float(finite_values(velocity.norm(dim=-1)).mean()) if velocity.numel() else 0.0,
        "motion_coeff_acceleration_mean": 0.0,
    }
    if model.T >= 3:
        acc = model.nr_coeff[2:] - 2 * model.nr_coeff[1:-1] + model.nr_coeff[:-2]
        metrics["motion_coeff_acceleration_mean"] = float(finite_values(acc.norm(dim=-1)).mean())
    return metrics


@torch.no_grad()
def xmap_health_metrics(
    xmap: torch.Tensor,
    alpha: torch.Tensor,
    canonical_x0: torch.Tensor,
    bins: int = 16,
    alpha_min: float = 0.05,
) -> dict[str, float]:
    mask = alpha > alpha_min
    valid_fraction = tensor_scalar(mask.float().mean())
    if int(mask.sum().detach().cpu()) < 16:
        return {
            "xmap_valid_fraction": valid_fraction,
            "xmap_occ": 0.0,
            "xmap_entropy": 0.0,
            "xmap_eff_bins": 0.0,
            "xmap_variance_x": 0.0,
            "xmap_variance_y": 0.0,
            "xmap_variance_z": 0.0,
            "xmap_local_smoothness": 0.0,
        }

    x = xmap[mask].detach()
    lo = canonical_x0.detach().amin(dim=0)
    hi = canonical_x0.detach().amax(dim=0)
    xn = (x - lo) / (hi - lo).clamp_min(1e-8)
    idx = (xn * int(bins)).long().clamp(0, int(bins) - 1)
    flat = idx[:, 0] * int(bins) * int(bins) + idx[:, 1] * int(bins) + idx[:, 2]
    counts = torch.bincount(flat.detach().cpu(), minlength=int(bins) ** 3).float()
    probs = counts / counts.sum().clamp_min(1e-8)
    nonzero = probs > 0
    entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
    var = x.var(dim=0, unbiased=False).detach().cpu()

    alpha_pair_x = (alpha[..., 1:] > alpha_min) & (alpha[..., :-1] > alpha_min)
    alpha_pair_y = (alpha[:, 1:, :] > alpha_min) & (alpha[:, :-1, :] > alpha_min)
    dx = (xmap[..., 1:, :] - xmap[..., :-1, :]).norm(dim=-1)
    dy = (xmap[:, 1:, :, :] - xmap[:, :-1, :, :]).norm(dim=-1)
    smooth_terms = []
    if bool(alpha_pair_x.any().detach().cpu()):
        smooth_terms.append(dx[alpha_pair_x].mean())
    if bool(alpha_pair_y.any().detach().cpu()):
        smooth_terms.append(dy[alpha_pair_y].mean())
    smoothness = torch.stack(smooth_terms).mean() if smooth_terms else xmap.new_zeros(())

    return {
        "xmap_valid_fraction": valid_fraction,
        "xmap_occ": float((counts > 0).float().mean()),
        "xmap_entropy": float(entropy),
        "xmap_eff_bins": float(entropy.exp()),
        "xmap_variance_x": float(var[0]),
        "xmap_variance_y": float(var[1]),
        "xmap_variance_z": float(var[2]),
        "xmap_local_smoothness": tensor_scalar(smoothness),
    }


@torch.no_grad()
def flow_health_metrics(flow: torch.Tensor, alpha: torch.Tensor, alpha_min: float = 0.05) -> dict[str, float]:
    alpha_for_flow = alpha[:-1] if alpha.shape[0] == flow.shape[0] + 1 else alpha
    valid = alpha_for_flow > alpha_min
    magnitude = flow.norm(dim=-1)
    if not bool(valid.any().detach().cpu()):
        return {
            "flow_valid_fraction": 0.0,
            "flow_magnitude_mean": 0.0,
            "flow_magnitude_p50": 0.0,
            "flow_magnitude_p95": 0.0,
            "flow_magnitude_max": 0.0,
        }
    metrics = {"flow_valid_fraction": tensor_scalar(valid.float().mean())}
    metrics.update(stats_for_tensor("flow_magnitude", magnitude[valid]))
    return metrics


def hwc_video_to_chw(video: torch.Tensor) -> torch.Tensor:
    return video.permute(0, 3, 1, 2).contiguous()


def wandb_log_training_logs(logs: list[dict[str, float]]) -> None:
    for log in logs:
        step = int(log["step"])
        wandb.log(
            {
                "Loss": log["loss"],
                "Loss/RGBL1": log["rgb_l1"],
                "Model/Mass": log["mass"],
                "Model/MotionSmooth": log["motion_smooth"],
                "Model/RadiusMean": log["radius"],
                "Model/SupportRadiusLoss": log["support_radius_loss"],
            },
            step=step,
        )


def wandb_final_payload(
    video: torch.Tensor,
    rendered: torch.Tensor,
    metrics: dict[str, float],
    fps: float,
) -> dict[str, Any]:
    target_chw = hwc_video_to_chw(video)
    rendered_chw = hwc_video_to_chw(rendered)
    payload: dict[str, Any] = {
        "Eval/L1": metrics["eval_l1"],
        "Eval/MSE": metrics["eval_mse"],
        "Eval/PSNR": metrics["eval_psnr"],
        "Eval/AlphaMean": metrics["alpha_mean"],
        "Eval/AlphaCoverage005": metrics["alpha_coverage_005"],
        "Eval/AlphaCoverage050": metrics["alpha_coverage_050"],
        "Eval/AlphaMax": metrics["alpha_max"],
        "Model/AlphaMeanFinal": metrics["model_alpha_mean"],
        "Model/RadiusMeanFinal": metrics["model_radius_mean"],
        "Model/RadiusMaxFinal": metrics["model_radius_max"],
        "Model/MotionSmoothFinal": metrics["model_motion_smooth"],
        "Render_GT_vs_Pred": make_preview_image(
            target_chw[0],
            rendered_chw[0],
            caption="Final frame 0",
        ),
        "GT_Video": make_wandb_video(target_chw, fps),
    }
    logged_keys = {
        "eval_l1",
        "eval_mse",
        "eval_psnr",
        "alpha_mean",
        "alpha_coverage_005",
        "alpha_coverage_050",
        "alpha_max",
        "model_alpha_mean",
        "model_radius_mean",
        "model_radius_max",
        "model_motion_smooth",
    }
    for key, value in metrics.items():
        if key not in logged_keys and isinstance(value, (int, float)):
            payload[f"Diag/{key}"] = value
    payload.update(build_validation_video_payload(rendered_chw, target_chw, fps))
    return payload


def tensor_to_uint8_image(image: torch.Tensor) -> Any:
    array = (image.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    from PIL import Image

    return Image.fromarray(array)


def save_preview_strip(
    path: Path,
    target: torch.Tensor,
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    max_frames: int = 4,
) -> None:
    T, H, W, _ = target.shape
    count = min(max_frames, T)
    indices = torch.linspace(0, T - 1, count).round().long().tolist()
    rows = []
    for index in indices:
        tgt = target[index]
        ren = rendered[index]
        diff = (ren - tgt).abs()
        a = alpha[index][..., None].expand(H, W, 3)
        row = torch.cat([tgt, ren, diff, a], dim=1)
        rows.append(row)

    canvas = torch.cat(rows, dim=0)
    image = tensor_to_uint8_image(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

    legend_path = path.with_name(path.stem + "_columns.txt")
    legend_path.write_text("columns: target | render | abs_error | alpha\n")


def save_side_by_side_mp4(
    path: Path,
    target: torch.Tensor,
    rendered: torch.Tensor,
    fps: float = 4.0,
) -> None:
    import cv2

    frames = torch.cat([target, rendered], dim=2)
    frames_u8 = (frames.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    T, H, W, _ = frames_u8.shape

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    for frame in frames_u8:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def init_wandb_if_enabled(logging_cfg: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if not bool(logging_cfg["log_to_wandb"]):
        return False

    mode = logging_cfg.get("wandb_mode")
    kwargs = {
        "project": logging_cfg["wandb_project"],
        "name": logging_cfg["wandb_run_name"],
        "tags": logging_cfg.get("wandb_tags"),
        "config": serialize_config_value(cfg),
    }
    if mode:
        kwargs["mode"] = mode
    wandb.init(**kwargs)
    return True


def scalar_background(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 3:
        channels = [float(channel) for channel in value]
        if max(channels) - min(channels) > 1e-6:
            raise ValueError("The toy gauge-field renderer only supports grayscale background values.")
        return channels[0]
    raise TypeError(f"Unsupported background value: {value!r}")


def gauge_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(
        config,
        sections=("data", "model", "camera", "render", "train", "losses", "logging"),
    )
    cfg.setdefault("diagnostics", {})
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["camera"], CAMERA_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    apply_defaults(cfg["diagnostics"], DIAGNOSTIC_DEFAULTS)
    cfg["model"]["support_mode"] = validate_support_mode(cfg["model"]["support_mode"])
    cfg["model"]["support_knn_k"] = int(cfg["model"]["support_knn_k"])
    cfg["model"]["support_jacobian_lambda"] = float(cfg["model"]["support_jacobian_lambda"])
    cfg["model"]["init_thickness"] = float(cfg["model"]["init_thickness"])
    cfg["model"]["slab_rotation_init_std"] = float(cfg["model"]["slab_rotation_init_std"])
    cfg["model"]["metric_offdiag_scale"] = float(cfg["model"]["metric_offdiag_scale"])
    cfg["render"]["opacity_transfer"] = validate_opacity_transfer(cfg["render"]["opacity_transfer"])
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit the material-coordinate gauge-field toy renderer from a Dynaworld JSONC config."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default=None, help="Override train.device from the config.")
    parser.add_argument("--steps", type=int, default=None, help="Override train.steps from the config.")
    parser.add_argument("--support-mode", default=None, choices=sorted(SUPPORT_MODES), help="Override model.support_mode.")
    parser.add_argument(
        "--opacity-transfer",
        default=None,
        choices=sorted(OPACITY_TRANSFERS),
        help="Override render.opacity_transfer.",
    )
    parser.add_argument("--output-dir", default=None, help="Override logging.output_dir from the config.")
    parser.add_argument("--wandb-mode", default=None, help="Override logging.wandb_mode from the config.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for local probes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_dynaworld_path(args.config)
    cfg = gauge_config(load_config_file(config_path))
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.support_mode is not None:
        cfg["model"]["support_mode"] = validate_support_mode(args.support_mode)
    if args.opacity_transfer is not None:
        cfg["render"]["opacity_transfer"] = validate_opacity_transfer(args.opacity_transfer)
    if args.output_dir is not None:
        cfg["logging"]["output_dir"] = args.output_dir
    if args.wandb_mode is not None:
        cfg["logging"]["wandb_mode"] = args.wandb_mode
    if args.no_wandb:
        cfg["logging"]["log_to_wandb"] = False

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    camera_cfg = cfg["camera"]
    render_cfg_values = cfg["render"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["losses"]
    logging_cfg = cfg["logging"]
    diagnostics_cfg = cfg["diagnostics"]

    torch.manual_seed(int(train_cfg["seed"]))

    device = resolve_device(str(train_cfg["device"]))
    output_dir = resolve_dynaworld_path(logging_cfg["output_dir"])

    bundle = load_gauge_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=camera_cfg,
        render_size=int(render_cfg_values["render_size"]),
        device=device,
    )
    video = bundle.video
    K = bundle.K
    w2c = bundle.w2c
    heldout_video = bundle.heldout_video
    heldout_K = bundle.heldout_K
    heldout_w2c = bundle.heldout_w2c
    T, H, W, _ = video.shape
    init_x0, init_color = initialize_material_points_from_first_frame(
        video=video,
        K=K,
        num_elements=int(model_cfg["num_elements"]),
        init_depth=float(model_cfg["init_depth"]),
    )

    render_cfg = RenderConfig(
        H=H,
        W=W,
        near=float(render_cfg_values["near_plane"]),
        far=float(render_cfg_values["far_plane"]),
        bg=scalar_background(render_cfg_values["background"]),
        min_radius_px=float(render_cfg_values["min_radius_px"]),
        max_radius_px=float(render_cfg_values["max_radius_px"]),
        max_alpha_per_element=float(render_cfg_values["max_alpha_per_element"]),
        opacity_transfer=str(render_cfg_values["opacity_transfer"]),
        pixel_chunk=int(render_cfg_values["pixel_chunk"]),
    )
    weights = TrainWeights(
        rgb=float(loss_cfg["rgb_weight"]),
        query=float(loss_cfg["query_weight"]),
        flow=float(loss_cfg["flow_weight"]),
        depth=float(loss_cfg["depth_weight"]),
        arap=float(loss_cfg["arap_weight"]),
        smooth=float(loss_cfg["smooth_weight"]),
        mass=float(loss_cfg["mass_weight"]),
        radius=float(loss_cfg["radius_weight"]),
    )

    print(
        "Gauge-field overfit "
        f"config={config_path} video={bundle.source_path} frames={T}/{data_cfg['max_frames'] or 'all'} size={H}x{W} "
        f"elements={model_cfg['num_elements']} basis={model_cfg['num_basis']} "
        f"support_mode={model_cfg['support_mode']} opacity_transfer={render_cfg.opacity_transfer} "
        f"train_frame_count={train_cfg['train_frame_count']} frames_per_step={train_cfg['frames_per_step']} "
        f"steps={train_cfg['steps']} device={device}"
    )
    if heldout_video is not None:
        print(
            "Held-out camera eval "
            f"sample={bundle.metadata.get('sample_id') if bundle.metadata else None} "
            f"dataset={bundle.metadata.get('dataset') if bundle.metadata else None} "
            f"source={bundle.metadata.get('source_camera') if bundle.metadata else None} "
            f"target={bundle.metadata.get('target_camera') if bundle.metadata else None} "
            f"pose_source={bundle.heldout_pose_source}"
        )

    wandb_enabled = init_wandb_if_enabled(logging_cfg, cfg)
    try:
        model, logs = train_material_surfel_field(
            video=video,
            K=K,
            w2c=w2c,
            init_x0=init_x0,
            init_color=init_color,
            num_steps=int(train_cfg["steps"]),
            batch_size=int(train_cfg["frames_per_step"]),
            train_frame_count=int(train_cfg["train_frame_count"]),
            num_basis=int(model_cfg["num_basis"]),
            support_mode=str(model_cfg["support_mode"]),
            support_knn_k=int(model_cfg["support_knn_k"]),
            support_jacobian_lambda=float(model_cfg["support_jacobian_lambda"]),
            init_radius=float(model_cfg["init_radius"]),
            init_thickness=float(model_cfg["init_thickness"]),
            init_alpha_logit=float(model_cfg["init_alpha_logit"]),
            init_basis_std=float(model_cfg["init_basis_std"]),
            init_coeff_std=float(model_cfg["init_coeff_std"]),
            slab_rotation_init_std=float(model_cfg["slab_rotation_init_std"]),
            metric_offdiag_scale=float(model_cfg["metric_offdiag_scale"]),
            lr=float(train_cfg["lr"]),
            weights=weights,
            render_cfg=render_cfg,
            log_every=int(logging_cfg["log_every"]),
        )

        rendered = render_sequence(
            model,
            K=K,
            w2c=w2c,
            cfg=render_cfg,
            include_flow=bool(diagnostics_cfg["flow_stats"]),
        )
        metrics = {
            **video_metrics(rendered["rgb"], video),
            **alpha_metrics(rendered["alpha"]),
            **model_metrics(model),
        }
        if bool(diagnostics_cfg["projection_stats"]):
            metrics.update(projection_health_metrics(model, K=K, w2c=w2c, cfg=render_cfg))
        if bool(diagnostics_cfg["motion_stats"]):
            metrics.update(motion_health_metrics(model))
        if bool(diagnostics_cfg["xmap_metrics"]):
            metrics.update(
                xmap_health_metrics(
                    rendered["xmap"],
                    rendered["alpha"],
                    canonical_x0=model.x0,
                    bins=int(diagnostics_cfg["xmap_bins"]),
                    alpha_min=float(diagnostics_cfg["xmap_alpha_min"]),
                )
            )
        heldout_rendered = None
        if heldout_video is not None:
            if heldout_K is None or heldout_w2c is None:
                raise RuntimeError("heldout_video is present but heldout camera tensors are missing.")
            heldout_rendered = render_sequence(
                model,
                K=heldout_K,
                w2c=heldout_w2c,
                cfg=render_cfg,
                include_flow=False,
            )
            heldout_metrics = prefix_metrics(
                "heldout",
                {
                    **video_metrics(heldout_rendered["rgb"], heldout_video),
                    **alpha_metrics(heldout_rendered["alpha"]),
                },
            )
            heldout_metrics["heldout_pose_is_calibrated"] = float(
                bundle.heldout_pose_source == "deepview_models_relative_pinhole"
            )
            if bool(diagnostics_cfg["projection_stats"]):
                heldout_metrics.update(
                    prefix_metrics(
                        "heldout",
                        projection_health_metrics(model, K=heldout_K, w2c=heldout_w2c, cfg=render_cfg),
                    )
                )
            if bool(diagnostics_cfg["xmap_metrics"]):
                heldout_metrics.update(
                    prefix_metrics(
                        "heldout",
                        xmap_health_metrics(
                            heldout_rendered["xmap"],
                            heldout_rendered["alpha"],
                            canonical_x0=model.x0,
                            bins=int(diagnostics_cfg["xmap_bins"]),
                            alpha_min=float(diagnostics_cfg["xmap_alpha_min"]),
                        ),
                    )
                )
            metrics.update(heldout_metrics)
        if bool(diagnostics_cfg["flow_stats"]) and "flow" in rendered:
            metrics.update(
                flow_health_metrics(
                    rendered["flow"],
                    rendered["alpha"],
                    alpha_min=float(diagnostics_cfg["xmap_alpha_min"]),
                )
            )
        print({"final": metrics})

        if wandb_enabled:
            wandb_log_training_logs(logs)
            wandb.log(
                wandb_final_payload(
                    video=video,
                    rendered=rendered["rgb"],
                    metrics=metrics,
                    fps=4.0,
                ),
                step=int(train_cfg["steps"]),
            )
    finally:
        if wandb_enabled:
            wandb.finish()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", serialize_config_value(cfg))
    write_json(output_dir / "logs.json", logs)
    write_json(output_dir / "metrics.json", metrics)
    torch.save(
        {
            "model": model.state_dict(),
            "K": K.detach().cpu(),
            "w2c": w2c.detach().cpu(),
            "heldout_K": heldout_K.detach().cpu() if heldout_K is not None else None,
            "heldout_w2c": heldout_w2c.detach().cpu() if heldout_w2c is not None else None,
            "heldout_pose_source": bundle.heldout_pose_source,
            "render_config": render_cfg.__dict__,
            "config": serialize_config_value(cfg),
            "metrics": metrics,
        },
        output_dir / "checkpoint.pt",
    )
    save_preview_strip(
        output_dir / "preview.png",
        target=video,
        rendered=rendered["rgb"],
        alpha=rendered["alpha"],
    )
    save_side_by_side_mp4(
        output_dir / "side_by_side.mp4",
        target=video,
        rendered=rendered["rgb"],
        fps=4.0,
    )
    if heldout_video is not None and heldout_rendered is not None:
        save_preview_strip(
            output_dir / "heldout_preview.png",
            target=heldout_video,
            rendered=heldout_rendered["rgb"],
            alpha=heldout_rendered["alpha"],
        )
        save_side_by_side_mp4(
            output_dir / "heldout_side_by_side.mp4",
            target=heldout_video,
            rendered=heldout_rendered["rgb"],
            fps=bundle.fps,
        )
    print(f"Wrote gauge-field outputs to {output_dir}")


if __name__ == "__main__":
    main()
