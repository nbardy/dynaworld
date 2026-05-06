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
    rasterize_power_foam_oriented_height_sv_texel_surface,
)
from torch_powerfoam_metal.random_scene import make_adjacency, make_pinhole_rays, make_power_sorted_ids, make_random_foam


def clone_with_grad(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def check_grads(name: str, got: list[torch.Tensor], ref: list[torch.Tensor], labels: list[str], tol: float) -> None:
    for label, got_tensor, ref_tensor in zip(labels, got, ref):
        err = float((got_tensor.grad - ref_tensor.grad).abs().max().detach().cpu())
        print(f"{name} {label} grad max error:", err)
        if err > tol:
            raise AssertionError(f"{name} {label} grad max error {err} exceeded tolerance {tol}")


def check_constant(device: torch.device) -> None:
    points, radii, densities, features = make_random_foam(cell_count=12, feature_dim=3, device=device, seed=123)
    adjacency, offsets = make_adjacency(points, radii, mode="knn", neighbors=4)
    rays = make_pinhole_rays(batch_size=1, height=24, width=20, device=device)
    sorted_ids = make_power_sorted_ids(points, radii, rays)
    stream_cfg = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=0.0, use_tiled=False)
    tiled_cfg = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=0.0, use_tiled=True)

    stream_params = clone_with_grad([points, radii, densities, features])
    tiled_params = clone_with_grad([points, radii, densities, features])
    stream_out, stream_alpha = rasterize_power_foam(
        *stream_params,
        adjacency,
        offsets,
        rays,
        stream_cfg,
        sorted_ids=sorted_ids,
    )
    tiled_out, tiled_alpha = rasterize_power_foam(
        *tiled_params,
        adjacency,
        offsets,
        rays,
        tiled_cfg,
        sorted_ids=sorted_ids,
    )
    stream_loss = stream_out.square().mean() + 0.3 * stream_alpha.square().mean()
    tiled_loss = tiled_out.square().mean() + 0.3 * tiled_alpha.square().mean()
    stream_loss.backward()
    tiled_loss.backward()

    out_err = float((stream_out - tiled_out).abs().max().detach().cpu())
    alpha_err = float((stream_alpha - tiled_alpha).abs().max().detach().cpu())
    print("constant tiled features max error:", out_err)
    print("constant tiled alpha max error:", alpha_err)
    if out_err > 3.0e-6 or alpha_err > 1.0e-6:
        raise AssertionError("constant tiled forward mismatch")
    check_grads("constant tiled", tiled_params, stream_params, ["points", "radii", "densities", "features"], 1.0e-5)


def check_height_sv(device: torch.device) -> None:
    points, radii, densities, _features = make_random_foam(cell_count=10, feature_dim=3, device=device, seed=321)
    gen = torch.Generator(device=device).manual_seed(999)
    texel_sites = torch.rand((10, 4, 2), device=device, dtype=torch.float32, generator=gen) - 0.5
    texel_heights = 0.03 * torch.randn((10, 4), device=device, dtype=torch.float32, generator=gen)
    texel_sv_axis = torch.randn((10, 4, 3, 3), device=device, dtype=torch.float32, generator=gen)
    texel_sv_rgb = torch.rand((10, 4, 3, 3), device=device, dtype=torch.float32, generator=gen) - 0.5
    normals = torch.nn.functional.normalize(
        torch.randn((10, 3), device=device, dtype=torch.float32, generator=gen),
        dim=-1,
        eps=1.0e-6,
    )
    adjacency, offsets = make_adjacency(points, radii, mode="knn", neighbors=4)
    rays = make_pinhole_rays(batch_size=1, height=18, width=22, device=device)
    sorted_ids = make_power_sorted_ids(points, radii, rays)
    stream_cfg = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=0.0, use_tiled=False)
    tiled_cfg = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=0.0, use_tiled=True)

    base = [points, radii, densities, texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, normals]
    stream_params = clone_with_grad(base)
    tiled_params = clone_with_grad(base)
    stream_out, stream_alpha = rasterize_power_foam_oriented_height_sv_texel_surface(
        *stream_params,
        adjacency,
        offsets,
        rays,
        stream_cfg,
        sorted_ids=sorted_ids,
    )
    tiled_out, tiled_alpha = rasterize_power_foam_oriented_height_sv_texel_surface(
        *tiled_params,
        adjacency,
        offsets,
        rays,
        tiled_cfg,
        sorted_ids=sorted_ids,
    )
    stream_loss = stream_out.square().mean() + 0.3 * stream_alpha.square().mean()
    tiled_loss = tiled_out.square().mean() + 0.3 * tiled_alpha.square().mean()
    stream_loss.backward()
    tiled_loss.backward()

    out_err = float((stream_out - tiled_out).abs().max().detach().cpu())
    alpha_err = float((stream_alpha - tiled_alpha).abs().max().detach().cpu())
    print("height_sv tiled features max error:", out_err)
    print("height_sv tiled alpha max error:", alpha_err)
    if out_err > 3.0e-6 or alpha_err > 3.0e-6:
        raise AssertionError("height_sv tiled forward mismatch")
    check_grads(
        "height_sv tiled",
        tiled_params,
        stream_params,
        ["points", "radii", "densities", "texel_sites", "texel_heights", "texel_sv_axis", "texel_sv_rgb", "normals"],
        1.0e-5,
    )


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the PowerFoam Metal tiled check")
    device = torch.device("mps")
    check_constant(device)
    check_height_sv(device)
    torch.mps.synchronize()


if __name__ == "__main__":
    main()
