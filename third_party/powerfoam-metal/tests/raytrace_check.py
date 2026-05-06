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
    raytrace_power_foam,
    raytrace_power_foam_oriented_height_sv_texel_surface,
)
from torch_powerfoam_metal.random_scene import make_adjacency, make_pinhole_rays, make_power_sorted_ids, make_random_foam


def all_pairs_adjacency(cell_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    offsets = [0]
    for i in range(cell_count):
        rows.extend(j for j in range(cell_count) if j != i)
        offsets.append(len(rows))
    return (
        torch.tensor(rows, device=device, dtype=torch.int32),
        torch.tensor(offsets, device=device, dtype=torch.int32),
    )


def two_cell_scene(device: torch.device) -> tuple[torch.Tensor, ...]:
    points = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], device=device, dtype=torch.float32)
    radii = torch.tensor([1.15, 1.15], device=device, dtype=torch.float32)
    densities = torch.tensor([0.8, 1.1], device=device, dtype=torch.float32)
    features = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]], device=device, dtype=torch.float32)
    adjacency = torch.tensor([1, 0], device=device, dtype=torch.int32)
    offsets = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    return points, radii, densities, features, adjacency, offsets


def unsupported_origin_start_scene(device: torch.device) -> tuple[torch.Tensor, ...]:
    rays = make_pinhole_rays(batch_size=1, height=7, width=7, device=device, fov_degrees=26.0)
    origin = torch.tensor([-0.35, 0.0, 0.0], device=device, dtype=torch.float32)
    rays[..., :3] = origin
    rays[..., 3:] = torch.nn.functional.normalize(
        rays[..., 3:] + torch.tensor([0.18, 0.0, 0.0], device=device, dtype=torch.float32),
        dim=-1,
        eps=1.0e-6,
    )
    center_ray = rays[0, rays.shape[1] // 2, rays.shape[2] // 2, 3:]
    points = torch.stack(
        [
            origin + torch.tensor([0.85, 0.0, 0.15], device=device, dtype=torch.float32),
            origin + 2.35 * center_ray,
        ],
        dim=0,
    ).contiguous()
    radii = torch.tensor([0.055, 0.62], device=device, dtype=torch.float32)
    densities = torch.tensor([2.0, 1.4], device=device, dtype=torch.float32)
    adjacency = torch.empty((0,), device=device, dtype=torch.int32)
    offsets = torch.zeros((3,), device=device, dtype=torch.int32)
    return points, radii, densities, adjacency, offsets, rays


def maybe_regular_triangulation_adjacency(
    points: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        return make_adjacency(points, radii, mode="regular_triangulation", neighbors=0)
    except ImportError as exc:
        print("raytrace regular_triangulation skipped:", exc)
        return None


def origin_power_start_ids(points: torch.Tensor, radii: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    if rays.ndim == 3:
        rays = rays.unsqueeze(0)
    origins = rays[:, 0, 0, :3]
    power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    return torch.argmin(power.detach(), dim=1).to(device=points.device, dtype=torch.int32).contiguous()


def check_height_sv_material_uses_height_endpoint(device: torch.device) -> None:
    points = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    radii = torch.tensor([5.0], device=device, dtype=torch.float32)
    densities = torch.tensor([4.0], device=device, dtype=torch.float32)
    adjacency = torch.empty((0,), device=device, dtype=torch.int32)
    offsets = torch.tensor([0, 0], device=device, dtype=torch.int32)
    rays = torch.tensor([[[[0.0, -4.0, -2.0, 0.0, 1.0, 1.0]]]], device=device, dtype=torch.float32)
    texel_sites = torch.tensor([[[-0.5, 0.0], [-0.2, 0.0], [0.2, 0.0], [0.5, 0.0]]], device=device)
    texel_heights = torch.ones((1, 4), device=device, dtype=torch.float32)
    texel_sv_axis = torch.zeros((1, 4, 1, 3), device=device, dtype=torch.float32)
    texel_sv_axis[..., 2] = 1.0
    texel_sv_rgb = torch.tensor(
        [[[[0.4, -0.4, -0.4]], [[-0.4, 0.4, -0.4]], [[-0.4, -0.4, 0.4]], [[0.4, 0.4, -0.4]]]],
        device=device,
        dtype=torch.float32,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    tangents = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=torch.float32)
    bitangents = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    cfg = FoamRasterConfig(
        alpha_threshold=0.0,
        transmittance_threshold=0.0,
        max_alpha=0.99,
        texel_temperature=80.0,
        use_tiled=True,
    )
    stream_out, stream_alpha = rasterize_power_foam_oriented_height_sv_texel_surface(
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
        cfg,
        tangents=tangents,
        bitangents=bitangents,
    )
    ray_out, ray_alpha = raytrace_power_foam_oriented_height_sv_texel_surface(
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
        cfg,
        tangents=tangents,
        bitangents=bitangents,
    )
    torch.mps.synchronize()
    stream_color = stream_out[0, 0, 0] / stream_alpha[0, 0, 0].clamp_min(1.0e-6)
    ray_color = ray_out[0, 0, 0] / ray_alpha[0, 0, 0].clamp_min(1.0e-6)
    color_err = float((stream_color - ray_color).abs().max().detach().cpu())
    print("raytrace height_sv material endpoint color:", stream_color.detach().cpu().tolist())
    print("raytrace height_sv material endpoint color max error:", color_err)
    if color_err > 2.0e-5:
        raise AssertionError("raytrace height_sv material endpoint color mismatch")
    if not (float(stream_color[1].detach().cpu()) > 0.75 and float(stream_color[0].detach().cpu()) < 0.25):
        raise AssertionError("height+SV material did not sample the height-clipped endpoint")


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the PowerFoam Metal raytrace check")
    device = torch.device("mps")
    check_height_sv_material_uses_height_endpoint(device)
    points, radii, densities, features, adjacency, offsets = two_cell_scene(device)
    rays = make_pinhole_rays(batch_size=1, height=9, width=9, device=device, fov_degrees=30.0)
    sorted_ids = make_power_sorted_ids(points, radii, rays)
    cfg = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=0.0, near_plane=0.0)
    cfg_tiled = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=0.0, near_plane=0.0, use_tiled=True)

    stream_out, stream_alpha = rasterize_power_foam(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays,
        cfg,
        sorted_ids=sorted_ids,
    )
    ray_out, ray_alpha, steps = raytrace_power_foam(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays,
        cfg,
        return_steps=True,
    )
    torch.mps.synchronize()

    out_err = float((stream_out - ray_out).abs().max().detach().cpu())
    alpha_err = float((stream_alpha - ray_alpha).abs().max().detach().cpu())
    max_steps = int(steps.max().detach().cpu())
    min_steps = int(steps.min().detach().cpu())
    print("raytrace two-cell features max error:", out_err)
    print("raytrace two-cell alpha max error:", alpha_err)
    print("raytrace two-cell steps range:", min_steps, max_steps)
    if out_err > 2.0e-5 or alpha_err > 2.0e-5:
        raise AssertionError("raytrace two-cell forward mismatch")
    if max_steps < 2:
        raise AssertionError("raytrace did not walk across the two-cell adjacency")

    points, radii, densities, features = make_random_foam(
        cell_count=8,
        feature_dim=3,
        device=device,
        seed=707,
        depth_min=1.8,
        depth_max=3.2,
    )
    radii = radii.mul(2.5).clamp_min(0.35)
    adjacency, offsets = all_pairs_adjacency(int(points.shape[0]), device)
    rays = make_pinhole_rays(batch_size=1, height=14, width=16, device=device, fov_degrees=34.0)
    sorted_ids = make_power_sorted_ids(points, radii, rays)
    stream_out, stream_alpha = rasterize_power_foam(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays,
        cfg,
        sorted_ids=sorted_ids,
    )
    ray_out, ray_alpha, steps = raytrace_power_foam(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays,
        cfg,
        return_steps=True,
    )
    torch.mps.synchronize()
    out_err = float((stream_out - ray_out).abs().max().detach().cpu())
    alpha_err = float((stream_alpha - ray_alpha).abs().max().detach().cpu())
    print("raytrace all-pairs features max error:", out_err)
    print("raytrace all-pairs alpha max error:", alpha_err)
    print("raytrace all-pairs mean/max steps:", float(steps.float().mean().detach().cpu()), int(steps.max().detach().cpu()))
    if out_err > 2.0e-5 or alpha_err > 2.0e-5:
        raise AssertionError("raytrace all-pairs forward mismatch")

    cell_count = 6
    points, radii, densities, _features = make_random_foam(
        cell_count=cell_count,
        feature_dim=3,
        device=device,
        seed=909,
        depth_min=1.8,
        depth_max=3.0,
    )
    radii = radii.mul(2.8).clamp_min(0.4)
    gen = torch.Generator(device=device).manual_seed(910)
    texel_sites = torch.rand((cell_count, 4, 2), device=device, dtype=torch.float32, generator=gen) - 0.5
    texel_heights = 0.03 * torch.randn((cell_count, 4), device=device, dtype=torch.float32, generator=gen)
    texel_sv_axis = torch.nn.functional.normalize(
        torch.randn((cell_count, 4, 3, 3), device=device, dtype=torch.float32, generator=gen),
        dim=-1,
        eps=1.0e-6,
    )
    texel_sv_rgb = torch.rand((cell_count, 4, 3, 3), device=device, dtype=torch.float32, generator=gen) - 0.5
    normals = torch.zeros((cell_count, 3), device=device, dtype=torch.float32)
    normals[:, 2] = -1.0
    adjacency, offsets = all_pairs_adjacency(cell_count, device)
    rays = make_pinhole_rays(batch_size=1, height=11, width=13, device=device, fov_degrees=28.0)
    sorted_ids = make_power_sorted_ids(points, radii, rays)
    stream_out, stream_alpha, stream_normal_distance = rasterize_power_foam_oriented_height_sv_texel_surface(
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
        cfg_tiled,
        sorted_ids=sorted_ids,
        return_normal_distance=True,
    )
    ray_out, ray_alpha, ray_normal_distance, steps = raytrace_power_foam_oriented_height_sv_texel_surface(
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
        cfg,
        return_normal_distance=True,
        return_steps=True,
    )
    torch.mps.synchronize()
    out_err = float((stream_out - ray_out).abs().max().detach().cpu())
    alpha_err = float((stream_alpha - ray_alpha).abs().max().detach().cpu())
    normal_distance_err = float((stream_normal_distance - ray_normal_distance).abs().max().detach().cpu())
    print("raytrace height_sv all-pairs features max error:", out_err)
    print("raytrace height_sv all-pairs alpha max error:", alpha_err)
    print("raytrace height_sv all-pairs normal_distance max error:", normal_distance_err)
    print("raytrace height_sv all-pairs mean/max steps:", float(steps.float().mean().detach().cpu()), int(steps.max().detach().cpu()))
    if out_err > 2.0e-5 or alpha_err > 2.0e-5 or normal_distance_err > 2.0e-5:
        raise AssertionError("raytrace height_sv all-pairs forward mismatch")

    def clone_req(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().clone().requires_grad_(True)

    raster_tensors = [
        clone_req(tensor)
        for tensor in (points, radii, densities, texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, normals)
    ]
    raytrace_tensors = [
        clone_req(tensor)
        for tensor in (points, radii, densities, texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, normals)
    ]
    sorted_ids = make_power_sorted_ids(raster_tensors[0].detach(), raster_tensors[1].detach(), rays)
    stream_out, stream_alpha, stream_normal_distance = rasterize_power_foam_oriented_height_sv_texel_surface(
        *raster_tensors[:7],
        raster_tensors[7],
        adjacency,
        offsets,
        rays,
        cfg_tiled,
        sorted_ids=sorted_ids,
        return_normal_distance=True,
    )
    ray_out, ray_alpha, ray_normal_distance, _steps = raytrace_power_foam_oriented_height_sv_texel_surface(
        *raytrace_tensors[:7],
        raytrace_tensors[7],
        adjacency,
        offsets,
        rays,
        cfg,
        return_normal_distance=True,
        return_steps=True,
    )
    (stream_out.square().mean() + stream_alpha.square().mean() + stream_normal_distance.square().mean()).backward()
    (ray_out.square().mean() + ray_alpha.square().mean() + ray_normal_distance.square().mean()).backward()
    torch.mps.synchronize()
    names = [
        "points",
        "radii",
        "densities",
        "texel_sites",
        "texel_heights",
        "texel_sv_axis",
        "texel_sv_rgb",
        "normals",
    ]
    grad_errors = {}
    for name, raster_tensor, raytrace_tensor in zip(names, raster_tensors, raytrace_tensors):
        grad_errors[name] = float((raster_tensor.grad - raytrace_tensor.grad).abs().max().detach().cpu())
    print("raytrace height_sv grad max errors:", grad_errors)
    if max(grad_errors.values()) > 2.0e-5:
        raise AssertionError("raytrace height_sv backward mismatch")

    points, radii, densities, adjacency, offsets, rays = unsupported_origin_start_scene(device)
    gen = torch.Generator(device=device).manual_seed(1313)
    texel_sites = torch.zeros((2, 3, 2), device=device, dtype=torch.float32)
    texel_heights = torch.zeros((2, 3), device=device, dtype=torch.float32)
    texel_sv_axis = torch.nn.functional.normalize(
        torch.randn((2, 3, 2, 3), device=device, dtype=torch.float32, generator=gen),
        dim=-1,
        eps=1.0e-6,
    )
    texel_sv_rgb = torch.rand((2, 3, 2, 3), device=device, dtype=torch.float32, generator=gen) - 0.5
    normals = torch.zeros((2, 3), device=device, dtype=torch.float32)
    normals[:, 2] = -1.0
    sorted_ids = make_power_sorted_ids(points, radii, rays)
    stream_out, stream_alpha, stream_normal_distance = rasterize_power_foam_oriented_height_sv_texel_surface(
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
        cfg_tiled,
        sorted_ids=sorted_ids,
        return_normal_distance=True,
    )
    old_start = torch.zeros((1,), device=device, dtype=torch.int32)
    _old_out, old_alpha, _old_normal_distance, _old_steps = raytrace_power_foam_oriented_height_sv_texel_surface(
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
        cfg,
        old_start,
        return_normal_distance=True,
        return_steps=True,
    )
    ray_out, ray_alpha, ray_normal_distance, steps = raytrace_power_foam_oriented_height_sv_texel_surface(
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
        cfg,
        return_normal_distance=True,
        return_steps=True,
    )
    torch.mps.synchronize()
    out_err = float((stream_out - ray_out).abs().max().detach().cpu())
    alpha_err = float((stream_alpha - ray_alpha).abs().max().detach().cpu())
    normal_distance_err = float((stream_normal_distance - ray_normal_distance).abs().max().detach().cpu())
    old_alpha_max = float(old_alpha.max().detach().cpu())
    ray_alpha_max = float(ray_alpha.max().detach().cpu())
    print("raytrace unsupported-origin old-start alpha max:", old_alpha_max)
    print("raytrace unsupported-origin default alpha max:", ray_alpha_max)
    print("raytrace unsupported-origin features max error:", out_err)
    print("raytrace unsupported-origin alpha max error:", alpha_err)
    print("raytrace unsupported-origin normal_distance max error:", normal_distance_err)
    print("raytrace unsupported-origin steps max:", int(steps.max().detach().cpu()))
    if old_alpha_max > 1.0e-6:
        raise AssertionError("unsupported-origin fixture no longer exercises the old zero-alpha start")
    if ray_alpha_max <= 0.05 or int(steps.max().detach().cpu()) < 1:
        raise AssertionError("raytrace default start did not recover visible unsupported-origin support")
    if out_err > 2.0e-5 or alpha_err > 2.0e-5 or normal_distance_err > 2.0e-5:
        raise AssertionError("raytrace unsupported-origin forward mismatch")

    raster_tensors = [
        clone_req(tensor)
        for tensor in (points, radii, densities, texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, normals)
    ]
    raytrace_tensors = [
        clone_req(tensor)
        for tensor in (points, radii, densities, texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, normals)
    ]
    sorted_ids = make_power_sorted_ids(raster_tensors[0].detach(), raster_tensors[1].detach(), rays)
    stream_out, stream_alpha, stream_normal_distance = rasterize_power_foam_oriented_height_sv_texel_surface(
        *raster_tensors[:7],
        raster_tensors[7],
        adjacency,
        offsets,
        rays,
        cfg_tiled,
        sorted_ids=sorted_ids,
        return_normal_distance=True,
    )
    ray_out, ray_alpha, ray_normal_distance, _steps = raytrace_power_foam_oriented_height_sv_texel_surface(
        *raytrace_tensors[:7],
        raytrace_tensors[7],
        adjacency,
        offsets,
        rays,
        cfg,
        return_normal_distance=True,
        return_steps=True,
    )
    (stream_out.square().mean() + stream_alpha.square().mean() + stream_normal_distance.square().mean()).backward()
    (ray_out.square().mean() + ray_alpha.square().mean() + ray_normal_distance.square().mean()).backward()
    torch.mps.synchronize()
    grad_errors = {}
    for name, raster_tensor, raytrace_tensor in zip(names, raster_tensors, raytrace_tensors):
        grad_errors[name] = float((raster_tensor.grad - raytrace_tensor.grad).abs().max().detach().cpu())
    print("raytrace unsupported-origin grad max errors:", grad_errors)
    if max(grad_errors.values()) > 2.0e-5:
        raise AssertionError("raytrace unsupported-origin backward mismatch")

    points, radii, densities, features = make_random_foam(
        cell_count=10,
        feature_dim=3,
        device=device,
        seed=1117,
        depth_min=1.6,
        depth_max=3.4,
    )
    radii = torch.full_like(radii, 1.1)
    regular = maybe_regular_triangulation_adjacency(points, radii)
    if regular is not None:
        regular_adjacency, regular_offsets = regular
        dense_adjacency, dense_offsets = all_pairs_adjacency(int(points.shape[0]), device)
        rays = make_pinhole_rays(batch_size=1, height=13, width=15, device=device, fov_degrees=32.0)
        sorted_ids = make_power_sorted_ids(points, radii, rays)
        regular_start_ids = origin_power_start_ids(points, radii, rays)
        dense_out, dense_alpha = rasterize_power_foam(
            points,
            radii,
            densities,
            features,
            dense_adjacency,
            dense_offsets,
            rays,
            cfg,
            sorted_ids=sorted_ids,
        )
        regular_ray_out, regular_ray_alpha, regular_steps = raytrace_power_foam(
            points,
            radii,
            densities,
            features,
            regular_adjacency,
            regular_offsets,
            rays,
            cfg,
            regular_start_ids,
            return_steps=True,
        )
        torch.mps.synchronize()
        out_err = float((dense_out - regular_ray_out).abs().max().detach().cpu())
        alpha_err = float((dense_alpha - regular_ray_alpha).abs().max().detach().cpu())
        print("raytrace regular_triangulation dense features max error:", out_err)
        print("raytrace regular_triangulation dense alpha max error:", alpha_err)
        print(
            "raytrace regular_triangulation avg degree/mean/max steps:",
            float(regular_adjacency.numel()) / float(points.shape[0]),
            float(regular_steps.float().mean().detach().cpu()),
            int(regular_steps.max().detach().cpu()),
        )
        if out_err > 2.0e-5 or alpha_err > 2.0e-5:
            raise AssertionError("raytrace regular_triangulation dense forward mismatch")

        reg_cell_count = 8
        points, radii, densities, _features = make_random_foam(
            cell_count=reg_cell_count,
            feature_dim=3,
            device=device,
            seed=1219,
            depth_min=1.6,
            depth_max=3.3,
        )
        radii = torch.full_like(radii, 1.0)
        gen = torch.Generator(device=device).manual_seed(1220)
        texel_sites = torch.rand((reg_cell_count, 4, 2), device=device, dtype=torch.float32, generator=gen) - 0.5
        texel_heights = 0.02 * torch.randn((reg_cell_count, 4), device=device, dtype=torch.float32, generator=gen)
        texel_sv_axis = torch.nn.functional.normalize(
            torch.randn((reg_cell_count, 4, 3, 3), device=device, dtype=torch.float32, generator=gen),
            dim=-1,
            eps=1.0e-6,
        )
        texel_sv_rgb = torch.rand((reg_cell_count, 4, 3, 3), device=device, dtype=torch.float32, generator=gen) - 0.5
        normals = torch.zeros((reg_cell_count, 3), device=device, dtype=torch.float32)
        normals[:, 2] = -1.0
        regular_adjacency, regular_offsets = make_adjacency(points, radii, mode="regular_triangulation", neighbors=0)
        dense_adjacency, dense_offsets = all_pairs_adjacency(reg_cell_count, device)
        rays = make_pinhole_rays(batch_size=1, height=11, width=12, device=device, fov_degrees=30.0)
        regular_start_ids = origin_power_start_ids(points, radii, rays)

        raster_tensors = [
            clone_req(tensor)
            for tensor in (points, radii, densities, texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, normals)
        ]
        raytrace_tensors = [
            clone_req(tensor)
            for tensor in (points, radii, densities, texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, normals)
        ]
        sorted_ids = make_power_sorted_ids(raster_tensors[0].detach(), raster_tensors[1].detach(), rays)
        dense_out, dense_alpha, dense_normal_distance = rasterize_power_foam_oriented_height_sv_texel_surface(
            *raster_tensors[:7],
            raster_tensors[7],
            dense_adjacency,
            dense_offsets,
            rays,
            cfg_tiled,
            sorted_ids=sorted_ids,
            return_normal_distance=True,
        )
        regular_ray_out, regular_ray_alpha, regular_ray_normal_distance, regular_steps = (
            raytrace_power_foam_oriented_height_sv_texel_surface(
                *raytrace_tensors[:7],
                raytrace_tensors[7],
                regular_adjacency,
                regular_offsets,
                rays,
                cfg,
                regular_start_ids,
                return_normal_distance=True,
                return_steps=True,
            )
        )
        dense_loss = dense_out.square().mean() + dense_alpha.square().mean() + dense_normal_distance.square().mean()
        regular_loss = (
            regular_ray_out.square().mean()
            + regular_ray_alpha.square().mean()
            + regular_ray_normal_distance.square().mean()
        )
        dense_loss.backward()
        regular_loss.backward()
        torch.mps.synchronize()
        out_err = float((dense_out - regular_ray_out).abs().max().detach().cpu())
        alpha_err = float((dense_alpha - regular_ray_alpha).abs().max().detach().cpu())
        normal_distance_err = float((dense_normal_distance - regular_ray_normal_distance).abs().max().detach().cpu())
        grad_errors = {}
        for name, raster_tensor, raytrace_tensor in zip(names, raster_tensors, raytrace_tensors):
            grad_errors[name] = float((raster_tensor.grad - raytrace_tensor.grad).abs().max().detach().cpu())
        print("raytrace regular_triangulation height_sv dense features max error:", out_err)
        print("raytrace regular_triangulation height_sv dense alpha max error:", alpha_err)
        print("raytrace regular_triangulation height_sv dense normal_distance max error:", normal_distance_err)
        print("raytrace regular_triangulation height_sv dense grad max errors:", grad_errors)
        print(
            "raytrace regular_triangulation height_sv avg degree/mean/max steps:",
            float(regular_adjacency.numel()) / float(points.shape[0]),
            float(regular_steps.float().mean().detach().cpu()),
            int(regular_steps.max().detach().cpu()),
        )
        if out_err > 2.0e-5 or alpha_err > 2.0e-5 or normal_distance_err > 2.0e-5:
            raise AssertionError("raytrace regular_triangulation height_sv dense forward mismatch")
        if max(grad_errors.values()) > 2.0e-5:
            raise AssertionError("raytrace regular_triangulation height_sv dense backward mismatch")


if __name__ == "__main__":
    main()
