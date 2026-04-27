from __future__ import annotations

import math

import torch


INCIDENCE_MODES = {"projected_conic", "ray_gaussian_line_peak", "ray_gaussian_line_mass"}
RAY_GAUSSIAN_LINE_MODES = {"ray_gaussian_line_peak", "ray_gaussian_line_mass"}

WORLD_COV_MIN_VAR = 1.0e-8
WORLD_COV_MAX_VAR = 1.0e2


def validate_incidence_mode(value: str) -> str:
    mode = str(value)
    if mode not in INCIDENCE_MODES:
        raise ValueError(f"render.incidence_mode must be one of {sorted(INCIDENCE_MODES)}, got {value!r}.")
    return mode


def uses_ray_gaussian_line(mode: str) -> bool:
    return validate_incidence_mode(mode) in RAY_GAUSSIAN_LINE_MODES


def bound_world_covariance(cov3: torch.Tensor) -> torch.Tensor:
    cov3 = torch.nan_to_num(cov3, nan=0.0, posinf=WORLD_COV_MAX_VAR, neginf=0.0)
    cov3 = 0.5 * (cov3 + cov3.transpose(-1, -2))
    trace = cov3.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(WORLD_COV_MIN_VAR)
    max_trace = 3.0 * WORLD_COV_MAX_VAR
    scale = (max_trace / trace).clamp_max(1.0)
    cov3 = cov3 * scale[:, None, None]
    eye3 = torch.eye(3, device=cov3.device, dtype=cov3.dtype).expand(cov3.shape[0], 3, 3)
    return cov3 + WORLD_COV_MIN_VAR * eye3


def make_world_rays_for_pixels(
    pixels: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fx = K[0, 0].clamp_min(1e-6)
    fy = K[1, 1].clamp_min(1e-6)
    cx = K[0, 2]
    cy = K[1, 2]
    dirs_cam = torch.stack(
        [
            (pixels[:, 0] - cx) / fx,
            (pixels[:, 1] - cy) / fy,
            torch.ones_like(pixels[:, 0]),
        ],
        dim=-1,
    )
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c2w = torch.linalg.inv(w2c)
    origin_world = c2w[:3, 3]
    dirs_world = dirs_cam @ w2c[:3, :3]
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return origin_world, dirs_world


def centerline_strength_to_mass(strength: torch.Tensor, cov3: torch.Tensor) -> torch.Tensor:
    """
    Convert an initialized center-line optical strength into comparable total mass.

    For isotropic covariance with variance sigma^2, the center whole-line
    optical depth for a mass-normalized Gaussian is mass / (2*pi*sigma^2).
    This inverse keeps ray_gaussian_line_mass from starting fully saturated
    when reusing projected-conic alpha initialization.
    """
    bounded_cov3 = bound_world_covariance(cov3)
    eq_var = torch.linalg.det(bounded_cov3).clamp_min(WORLD_COV_MIN_VAR ** 3).pow(1.0 / 3.0)
    return strength * (2.0 * math.pi) * eq_var


def ray_gaussian_line_optical_depth(
    ray_origin: torch.Tensor,
    ray_dirs: torch.Tensor,
    s0: float,
    s1: float,
    mu: torch.Tensor,
    cov3: torch.Tensor,
    strength: torch.Tensor,
    *,
    mass_normalized: bool,
) -> torch.Tensor:
    """
    Closed-form optical depth for Gaussian event support along finite camera rays.

    ray_origin: [3]
    ray_dirs: [C,3], unit world directions
    mu: [N,3]
    cov3: [N,3,3], SPD world support
    strength: [N], peak density or total optical mass depending on mode
    returns: [N,C] optical depths
    """
    cov3 = bound_world_covariance(cov3)
    inv_cov = torch.linalg.inv(cov3)
    v = ray_origin[None, :] - mu
    av = torch.einsum("nij,nj->ni", inv_cov, v)

    a = torch.einsum("cd,ndk,ck->nc", ray_dirs, inv_cov, ray_dirs).clamp_min(1e-8)
    b = torch.einsum("nd,cd->nc", av, ray_dirs)
    c = (v * av).sum(dim=-1)[:, None]
    d2 = (c - b.square() / a).clamp_min(0.0)

    sqrt_half_a = torch.sqrt(0.5 * a)
    center = b / a
    erf_hi = torch.erf(sqrt_half_a * (float(s1) + center))
    erf_lo = torch.erf(sqrt_half_a * (float(s0) + center))
    line_factor = torch.exp(-0.5 * d2) * torch.sqrt(math.pi / (2.0 * a)) * (erf_hi - erf_lo)

    if mass_normalized:
        det = torch.linalg.det(cov3).clamp_min(WORLD_COV_MIN_VAR ** 3)
        norm = strength.clamp_min(0.0) / (((2.0 * math.pi) ** 1.5) * torch.sqrt(det))
    else:
        norm = strength.clamp_min(0.0)

    tau = norm[:, None] * line_factor
    return torch.nan_to_num(tau, nan=0.0, posinf=1.0e6, neginf=0.0).clamp_min(0.0)
