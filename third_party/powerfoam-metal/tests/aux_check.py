from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_powerfoam_metal import (
    FoamRasterConfig,
    rasterize_power_foam,
    rasterize_power_foam_aux,
    rasterize_power_foam_oriented_height_sv_texel_surface,
    rasterize_power_foam_oriented_height_sv_texel_surface_aux,
)
from torch_powerfoam_metal.random_scene import make_pinhole_rays


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, threshold: float) -> None:
    err = float((got.detach().cpu() - ref.detach().cpu()).abs().max())
    print(f"{name} max error:", err)
    if err > threshold:
        raise AssertionError(f"{name} max error {err} exceeded {threshold}")


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Metal aux check")
    device = torch.device("mps")
    points = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float32)
    radii = torch.tensor([0.85], device=device, dtype=torch.float32)
    densities = torch.tensor([1.7], device=device, dtype=torch.float32)
    features = torch.tensor([[0.2, 0.5, 0.7]], device=device, dtype=torch.float32)
    adjacency = torch.empty(0, device=device, dtype=torch.int32)
    offsets = torch.zeros(2, device=device, dtype=torch.int32)
    rays = make_pinhole_rays(batch_size=1, height=6, width=5, device=device, fov_degrees=42.0)
    target = torch.zeros((1, 6, 5, 3), device=device, dtype=torch.float32)
    config = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=1.0e-5, use_tiled=True)

    out, alpha = rasterize_power_foam(points, radii, densities, features, adjacency, offsets, rays, config)
    aux = rasterize_power_foam_aux(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays,
        target,
        config,
        depth_quantiles=[0.25, 0.5, 0.75],
    )

    normal_ref = torch.zeros_like(aux.normal)
    normal_ref[:, 2] = -alpha
    dirs = rays[..., 3:]
    oc = rays[..., :3] - points.view(1, 1, 1, 3)
    b = 2.0 * (dirs * oc).sum(dim=-1)
    c = oc.square().sum(dim=-1) - radii.view(1, 1, 1).square()
    disc = b.square() - 4.0 * c
    t_near = ((-b - disc.clamp_min(0.0).sqrt()) * 0.5).clamp_min(config.near_plane)
    quantile_values = torch.tensor([0.25, 0.5, 0.75], device=device, dtype=torch.float32)
    thresholds = 1.0 - quantile_values
    depth_refs = []
    for threshold in thresholds:
        depth_refs.append(
            torch.where(
                alpha < 1.0 - threshold,
                torch.full_like(alpha, -1.0),
                t_near + torch.log(1.0 / threshold) / densities[0],
            )
        )
    quantile_ref = torch.stack(depth_refs, dim=-1)
    median_ref = quantile_ref[..., 1]
    contrib_ref = alpha.reshape(1, -1).sum(dim=1, keepdim=True) / float(alpha.numel())
    point_error_ref = (alpha * features.abs().sum()).reshape(1, -1).sum(dim=1, keepdim=True) / float(alpha.numel())

    assert_close("normal_distance", aux.normal_distance, torch.zeros_like(aux.normal_distance), 1.0e-6)
    assert_close("normal", aux.normal, normal_ref, 1.0e-6)
    assert_close("median_depth", aux.median_depth, median_ref, 1.0e-6)
    if aux.depth_quantile_depths is None:
        raise AssertionError("expected depth_quantile_depths to be populated")
    assert_close("depth_quantile_depths", aux.depth_quantile_depths, quantile_ref, 1.0e-6)
    assert_close("contrib", aux.contrib, contrib_ref, 1.0e-6)
    assert_close("point_error", aux.point_error, point_error_ref, 1.0e-6)
    if not bool(aux.visible_mask[0, 0].item()):
        raise AssertionError("expected the single cell to be visible")
    if not torch.isfinite(out).all():
        raise AssertionError("render output contains non-finite values")

    texel_sites = torch.zeros((1, 1, 2), device=device, dtype=torch.float32)
    texel_heights = torch.zeros((1, 1), device=device, dtype=torch.float32)
    texel_sv_axis = torch.tensor([[[[0.0, 0.0, 1.0]]]], device=device, dtype=torch.float32)
    texel_sv_rgb = torch.tensor([[[[0.4, 0.6, 0.8]]]], device=device, dtype=torch.float32)
    normals = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32)
    tangents = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    bitangents = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=torch.float32)
    sv_out, sv_alpha = rasterize_power_foam_oriented_height_sv_texel_surface(
        points,
        radii,
        densities,
        texel_sites,
        texel_heights,
        texel_sv_axis,
        texel_sv_rgb,
        normals,
        adjacency,
        offsets,
        rays,
        config,
        tangents=tangents,
        bitangents=bitangents,
    )
    sv_aux = rasterize_power_foam_oriented_height_sv_texel_surface_aux(
        points,
        radii,
        densities,
        texel_sites,
        texel_heights,
        texel_sv_axis,
        texel_sv_rgb,
        normals,
        adjacency,
        offsets,
        rays,
        target,
        config,
        tangents=tangents,
        bitangents=bitangents,
    )
    assert_close("height_sv contrib", sv_aux.contrib, sv_alpha.reshape(1, -1).sum(dim=1, keepdim=True) / float(sv_alpha.numel()), 1.0e-6)
    assert_close(
        "height_sv point_error",
        sv_aux.point_error,
        sv_out.abs().sum(dim=-1).reshape(1, -1).sum(dim=1, keepdim=True) / float(sv_alpha.numel()),
        1.0e-6,
    )
    print("powerfoam Metal aux check passed")


if __name__ == "__main__":
    main()
