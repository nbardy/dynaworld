from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None


@dataclass(frozen=True)
class FoamRasterConfig:
    near_plane: float = 1.0e-4
    alpha_threshold: float = 0.0
    transmittance_threshold: float = 1.0e-4
    max_alpha: float = 0.999
    eps: float = 1.0e-8
    texel_temperature: float = 10.0


def _normalize_rays(rays: Tensor) -> tuple[Tensor, bool]:
    if rays.ndim == 3:
        if rays.shape[-1] != 6:
            raise ValueError("rays must have shape [H,W,6] or [B,H,W,6]")
        return rays.unsqueeze(0), False
    if rays.ndim == 4 and rays.shape[-1] == 6:
        return rays, True
    raise ValueError("rays must have shape [H,W,6] or [B,H,W,6]")


def _check_float_mps(name: str, tensor: Tensor, ndim: int | None = None) -> None:
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on MPS")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_int_mps(name: str, tensor: Tensor, ndim: int | None = None) -> None:
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on MPS")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _make_meta(
    rays_b: Tensor,
    points: Tensor,
    features: Tensor,
    config: FoamRasterConfig,
    *,
    output_dim: int | None = None,
    feature_mode: int = 0,
) -> tuple[Tensor, Tensor]:
    batch_size, height, width = rays_b.shape[:3]
    cell_count = points.shape[0]
    feature_dim = features.shape[1]
    if output_dim is None:
        output_dim = feature_dim
    meta_i32 = torch.tensor(
        [
            batch_size,
            height,
            width,
            cell_count,
            feature_dim,
            int(output_dim),
            int(feature_mode),
        ],
        device=rays_b.device,
        dtype=torch.int32,
    )
    meta_f32 = torch.tensor(
        [
            float(config.near_plane),
            float(config.alpha_threshold),
            float(config.transmittance_threshold),
            float(config.max_alpha),
            float(config.eps),
            float(config.texel_temperature),
        ],
        device=rays_b.device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def _full_screen_bounds(rays_b: Tensor, points: Tensor) -> Tensor:
    batch_size, height, width = rays_b.shape[:3]
    cell_count = points.shape[0]
    bounds = torch.tensor([0, 0, width - 1, height - 1], device=rays_b.device, dtype=torch.int32)
    return bounds.view(1, 1, 4).expand(batch_size, cell_count, 4).contiguous()


def _default_sorted_ids(points: Tensor, radii: Tensor, rays_b: Tensor) -> Tensor:
    origins = rays_b[:, 0, 0, :3]
    power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    return torch.argsort(power.detach(), dim=1, stable=True).to(torch.int32).contiguous()


def _check_inputs(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays_b: Tensor,
    sorted_ids: Tensor,
    output_dim: int,
    feature_mode: int,
) -> None:
    _check_float_mps("points", points, 2)
    _check_float_mps("radii", radii, 1)
    _check_float_mps("densities", densities, 1)
    _check_float_mps("features", features, 2)
    _check_float_mps("rays", rays_b, 4)
    _check_int_mps("adjacency", adjacency, 1)
    _check_int_mps("offsets", offsets, 1)
    _check_int_mps("sorted_ids", sorted_ids, 2)

    cell_count = points.shape[0]
    if points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if radii.shape[0] != cell_count or densities.shape[0] != cell_count or features.shape[0] != cell_count:
        raise ValueError("points/radii/densities/features must agree on N")
    if features.shape[1] <= 0:
        raise ValueError("features must have a positive feature dimension")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if feature_mode == 0:
        if features.shape[1] != output_dim:
            raise ValueError("constant feature mode requires features.shape[1] == output_dim")
    elif feature_mode in {1, 2}:
        if features.shape[1] != output_dim * 4:
            raise ValueError("linear feature mode requires flattened features.shape[1] == output_dim * 4")
    elif feature_mode == 3:
        if features.shape[1] != output_dim * 4 + 3:
            raise ValueError("oriented surface-linear feature mode requires flattened features.shape[1] == output_dim * 4 + 3")
    elif feature_mode == 4:
        stride = output_dim + 2
        if features.shape[1] <= 9 or (features.shape[1] - 9) % stride != 0:
            raise ValueError("oriented texel-surface feature mode requires features.shape[1] == S * (output_dim + 2) + 9")
    else:
        raise ValueError("feature_mode must be 0, 1, 2, 3, or 4")
    if offsets.shape[0] != cell_count + 1:
        raise ValueError("offsets must have shape [N+1]")
    if rays_b.shape[-1] != 6:
        raise ValueError("rays must have trailing dimension 6")
    if sorted_ids.shape != (rays_b.shape[0], cell_count):
        raise ValueError("sorted_ids must have shape [B,N]")


class _RasterizeDynamicPowerFoamFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points: Tensor,
        radii: Tensor,
        densities: Tensor,
        features: Tensor,
        adjacency: Tensor,
        offsets: Tensor,
        sorted_ids: Tensor,
        screen_bounds: Tensor,
        rays_b: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ctx.output_dim = int(meta_i32.detach().cpu()[5])
        out, alpha, log_t, pixel_stop = torch.ops.dynamic_powerfoam_metal.rasterize_train_forward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            meta_i32,
            meta_f32,
        )
        ctx.save_for_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            log_t,
            pixel_stop,
            meta_i32,
            meta_f32,
        )
        return out, alpha

    @staticmethod
    def backward(ctx, grad_out: Tensor | None, grad_alpha: Tensor | None):
        (
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            log_t,
            pixel_stop,
            meta_i32,
            meta_f32,
        ) = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros(
                (*rays_b.shape[:3], ctx.output_dim),
                device=points.device,
                dtype=torch.float32,
            )
        if grad_alpha is None:
            grad_alpha = torch.zeros(rays_b.shape[:3], device=points.device, dtype=torch.float32)
        grad_points, grad_radii, grad_densities, grad_features = torch.ops.dynamic_powerfoam_metal.rasterize_train_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            log_t,
            pixel_stop,
            grad_out.contiguous(),
            grad_alpha.contiguous(),
            meta_i32,
            meta_f32,
        )
        return (
            grad_points,
            grad_radii,
            grad_densities,
            grad_features,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_power_foam(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells along rays.

    Each cell is clipped by its bounding sphere and by radical planes against
    the supplied neighbor graph, then alpha-composited with constant per-cell
    features. The Metal path has a custom replay backward for points, radii,
    densities, and features.
    """
    if not hasattr(torch.ops, "dynamic_powerfoam_metal"):
        raise RuntimeError("dynamic_powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    features = features.contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=features.shape[1],
        feature_mode=0,
    )
    meta_i32, meta_f32 = _make_meta(rays_b, points, features, config)
    screen_bounds = _full_screen_bounds(rays_b, points)
    out, alpha = _RasterizeDynamicPowerFoamFunction.apply(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
        return out[0], alpha[0]


def _frame_from_normals(normals: Tensor) -> tuple[Tensor, Tensor]:
    normals = F.normalize(normals, dim=-1, eps=1.0e-6)
    z_axis = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    y_axis = normals.new_tensor([0.0, 1.0, 0.0]).expand_as(normals)
    helper = torch.where(normals[..., 2:3].abs() < 0.9, z_axis, y_axis)
    tangents = F.normalize(torch.cross(helper, normals, dim=-1), dim=-1, eps=1.0e-6)
    bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-6)
    return tangents, bitangents


def rasterize_power_foam_linear(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells with local linear per-cell features.

    `features` is `[N, C, 4]`: base feature plus x/y/z coefficients evaluated
    at the ray midpoint in radius-normalized local cell coordinates.
    """
    if not hasattr(torch.ops, "dynamic_powerfoam_metal"):
        raise RuntimeError("dynamic_powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if features.ndim != 3 or features.shape[2] != 4:
        raise ValueError("linear features must have shape [N,C,4]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(features.shape[1])
    features_flat = features.permute(0, 2, 1).contiguous().reshape(features.shape[0], output_dim * 4)
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=1,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=1,
    )
    screen_bounds = _full_screen_bounds(rays_b, points)
    out, alpha = _RasterizeDynamicPowerFoamFunction.apply(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_surface_linear(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells with a fixed camera-facing surface plane.

    This uses the same `[N,C,4]` feature layout as `rasterize_power_foam_linear`,
    but clips each cell by a camera-facing `-z` plane through the cell center and
    evaluates the local-linear feature at that surface intersection.
    """
    if not hasattr(torch.ops, "dynamic_powerfoam_metal"):
        raise RuntimeError("dynamic_powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if features.ndim != 3 or features.shape[2] != 4:
        raise ValueError("surface-linear features must have shape [N,C,4]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(features.shape[1])
    features_flat = features.permute(0, 2, 1).contiguous().reshape(features.shape[0], output_dim * 4)
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=2,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=2,
    )
    screen_bounds = _full_screen_bounds(rays_b, points)
    out, alpha = _RasterizeDynamicPowerFoamFunction.apply(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_oriented_surface_linear(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells with learned per-cell surface normals.

    `features` is `[N,C,4]`. `normals` is `[N,3]` and is expected to already be
    normalized by the caller so autograd can own the normalization gradient.
    """
    if not hasattr(torch.ops, "dynamic_powerfoam_metal"):
        raise RuntimeError("dynamic_powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if features.ndim != 3 or features.shape[2] != 4:
        raise ValueError("oriented surface-linear features must have shape [N,C,4]")
    if normals.ndim != 2 or normals.shape != (features.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(features.shape[1])
    features_flat = features.permute(0, 2, 1).contiguous().reshape(features.shape[0], output_dim * 4)
    features_flat = torch.cat([features_flat, normals.contiguous()], dim=1).contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=3,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=3,
    )
    screen_bounds = _full_screen_bounds(rays_b, points)
    out, alpha = _RasterizeDynamicPowerFoamFunction.apply(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_oriented_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_features: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    tangents: Tensor | None = None,
    bitangents: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize oriented surface cells with learned local detail sites.

    `texel_sites` is `[N,S,2]` in radius-normalized local surface coordinates,
    `texel_features` is `[N,S,C]`, and `normals` is `[N,3]`. Optional
    `tangents`/`bitangents` define the material-frame axes used to turn the 3D
    surface hit into texel coordinates; when omitted they are derived from the
    normals with a stable but roll-free frame.
    """
    if not hasattr(torch.ops, "dynamic_powerfoam_metal"):
        raise RuntimeError("dynamic_powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if texel_sites.ndim != 3 or texel_sites.shape[2] != 2:
        raise ValueError("texel_sites must have shape [N,S,2]")
    if texel_features.ndim != 3:
        raise ValueError("texel_features must have shape [N,S,C]")
    if texel_sites.shape[:2] != texel_features.shape[:2]:
        raise ValueError("texel_sites and texel_features must agree on [N,S]")
    if normals.ndim != 2 or normals.shape != (texel_sites.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")
    if (tangents is None) != (bitangents is None):
        raise ValueError("tangents and bitangents must be provided together")
    if tangents is None or bitangents is None:
        tangents, bitangents = _frame_from_normals(normals)
    if tangents.ndim != 2 or tangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("tangents must have shape [N,3]")
    if bitangents.ndim != 2 or bitangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("bitangents must have shape [N,3]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(texel_features.shape[2])
    texel_flat = torch.cat([texel_sites.contiguous(), texel_features.contiguous()], dim=-1)
    features_flat = torch.cat(
        [
            texel_flat.reshape(texel_sites.shape[0], -1),
            normals.contiguous(),
            tangents.contiguous(),
            bitangents.contiguous(),
        ],
        dim=1,
    ).contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=4,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=4,
    )
    screen_bounds = _full_screen_bounds(rays_b, points)
    out, alpha = _RasterizeDynamicPowerFoamFunction.apply(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]
