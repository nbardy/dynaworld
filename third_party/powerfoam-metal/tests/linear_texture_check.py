from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_powerfoam_metal import (
    FoamRasterConfig,
    quaternion_frames,
    rasterize_power_foam_linear,
    rasterize_power_foam_oriented_height_sv_texel_surface,
    rasterize_power_foam_oriented_height_texel_surface,
    rasterize_power_foam_oriented_surface_linear,
    rasterize_power_foam_oriented_texel_surface,
    rasterize_power_foam_quaternion_height_sv_texel_surface,
    rasterize_power_foam_quaternion_height_texel_surface,
    rasterize_power_foam_quaternion_texel_surface,
    rasterize_power_foam_surface_linear,
)
from torch_powerfoam_metal.random_scene import make_pinhole_rays, make_power_sorted_ids


def fully_connected_adjacency(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    offsets = [0]
    for i in range(n):
        rows.extend(j for j in range(n) if j != i)
        offsets.append(len(rows))
    return (
        torch.tensor(rows, device=device, dtype=torch.int32),
        torch.tensor(offsets, device=device, dtype=torch.int32),
    )


def frame_from_normals(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_axis = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    y_axis = normals.new_tensor([0.0, 1.0, 0.0]).expand_as(normals)
    helper = torch.where(normals[..., 2:3].abs() < 0.9, z_axis, y_axis)
    tangents = F.normalize(torch.cross(helper, normals, dim=-1), dim=-1, eps=1.0e-6)
    bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-6)
    return tangents, bitangents


def torch_linear_reference(
    points: torch.Tensor,
    radii: torch.Tensor,
    densities: torch.Tensor,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    sorted_ids: torch.Tensor,
    rays: torch.Tensor,
    config: FoamRasterConfig,
    *,
    surface: bool = False,
    normals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, height, width = rays.shape[:3]
    output_dim = features.shape[1]
    out = torch.zeros(batch_size, height, width, output_dim, device=points.device, dtype=points.dtype)
    alpha_out = torch.zeros(batch_size, height, width, device=points.device, dtype=points.dtype)

    for batch in range(batch_size):
        for y in range(height):
            for x in range(width):
                origin = rays[batch, y, x, :3]
                direction = rays[batch, y, x, 3:]
                direction = direction / direction.norm().clamp_min(config.eps)
                transmittance = points.new_tensor(1.0)
                rgb = torch.zeros(output_dim, device=points.device, dtype=points.dtype)
                for cell_i in [int(v) for v in sorted_ids[batch].detach().cpu().tolist()]:
                    sigma = densities[cell_i].clamp_min(0.0)
                    radius = radii[cell_i]
                    center = points[cell_i]
                    oc = origin - center
                    qa = torch.dot(direction, direction)
                    qb = 2.0 * torch.dot(oc, direction)
                    qc = torch.dot(oc, oc) - radius * radius
                    disc = qb * qb - 4.0 * qa * qc
                    if bool((disc < 0.0).detach().cpu()) or bool((qa <= config.eps).detach().cpu()):
                        continue
                    root = disc.clamp_min(0.0).sqrt()
                    t_near = (-qb - root) * 0.5 / qa
                    t_far = (-qb + root) * 0.5 / qa
                    if bool((t_far <= config.near_plane).detach().cpu()):
                        continue
                    t_near = torch.maximum(t_near, points.new_tensor(config.near_plane))
                    if bool((t_far <= t_near).detach().cpu()):
                        continue

                    inside = True
                    for edge in range(int(offsets[cell_i]), int(offsets[cell_i + 1])):
                        neighbor_i = int(adjacency[edge])
                        if neighbor_i == cell_i:
                            continue
                        neighbor_center = points[neighbor_i]
                        n = neighbor_center - center
                        h = 0.5 * (
                            torch.dot(neighbor_center, neighbor_center)
                            - torch.dot(center, center)
                            + radius * radius
                            - radii[neighbor_i] * radii[neighbor_i]
                        )
                        dp = torch.dot(direction, n)
                        num = h - torch.dot(origin, n)
                        if bool((dp.abs() <= config.eps).detach().cpu()):
                            if bool((num < -config.eps).detach().cpu()):
                                inside = False
                                break
                            continue
                        t_face = num / dp
                        if bool((dp > 0.0).detach().cpu()):
                            t_far = torch.minimum(t_far, t_face)
                        else:
                            t_near = torch.maximum(t_near, t_face)
                        if bool((t_far <= t_near).detach().cpu()):
                            inside = False
                            break
                    if not inside or bool((t_far <= t_near).detach().cpu()):
                        continue

                    if surface:
                        if normals is None:
                            normal = torch.tensor([0.0, 0.0, -1.0], device=points.device, dtype=points.dtype)
                        else:
                            normal = normals[cell_i]
                        dp_surface = torch.dot(direction, normal)
                        if bool((dp_surface.abs() <= config.eps).detach().cpu()):
                            continue
                        t_surface = (torch.dot(center, normal) - torch.dot(origin, normal)) / dp_surface
                        if bool((dp_surface >= 0.0).detach().cpu()):
                            t_far = torch.minimum(t_far, t_surface)
                        else:
                            t_near = torch.maximum(t_near, t_surface)
                        if bool((t_far <= t_near).detach().cpu()):
                            continue

                    alpha = (1.0 - torch.exp(-sigma * (t_far - t_near))).clamp(0.0, config.max_alpha)
                    if bool((alpha < config.alpha_threshold).detach().cpu()):
                        continue
                    if surface:
                        t_sample = t_surface
                    else:
                        t_sample = 0.5 * (t_near + t_far)
                    local = (origin + direction * t_sample - center) / radius.clamp_min(config.eps)
                    color = (
                        features[cell_i, :, 0]
                        + local[0] * features[cell_i, :, 1]
                        + local[1] * features[cell_i, :, 2]
                        + local[2] * features[cell_i, :, 3]
                    )
                    weight = transmittance * alpha
                    rgb = rgb + weight * color
                    transmittance = transmittance * (1.0 - alpha)
                out[batch, y, x] = rgb
                alpha_out[batch, y, x] = 1.0 - transmittance
    return out, alpha_out


def torch_texel_reference(
    points: torch.Tensor,
    radii: torch.Tensor,
    densities: torch.Tensor,
    texel_sites: torch.Tensor,
    texel_features: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor | None,
    bitangents: torch.Tensor | None,
    texel_heights: torch.Tensor | None,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    sorted_ids: torch.Tensor,
    rays: torch.Tensor,
    config: FoamRasterConfig,
    texel_sv_axis: torch.Tensor | None = None,
    texel_sv_rgb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, height, width = rays.shape[:3]
    output_dim = 3 if texel_sv_axis is not None else texel_features.shape[2]
    out = torch.zeros(batch_size, height, width, output_dim, device=points.device, dtype=points.dtype)
    alpha_out = torch.zeros(batch_size, height, width, device=points.device, dtype=points.dtype)
    if tangents is None or bitangents is None:
        frame_tangents, frame_bitangents = frame_from_normals(normals)
    else:
        frame_tangents, frame_bitangents = tangents, bitangents

    for batch in range(batch_size):
        for y in range(height):
            for x in range(width):
                origin = rays[batch, y, x, :3]
                direction = rays[batch, y, x, 3:]
                direction = direction / direction.norm().clamp_min(config.eps)
                transmittance = points.new_tensor(1.0)
                rgb = torch.zeros(output_dim, device=points.device, dtype=points.dtype)
                for cell_i in [int(v) for v in sorted_ids[batch].detach().cpu().tolist()]:
                    sigma = densities[cell_i].clamp_min(0.0)
                    radius = radii[cell_i]
                    center = points[cell_i]
                    oc = origin - center
                    qa = torch.dot(direction, direction)
                    qb = 2.0 * torch.dot(oc, direction)
                    qc = torch.dot(oc, oc) - radius * radius
                    disc = qb * qb - 4.0 * qa * qc
                    if bool((disc < 0.0).detach().cpu()) or bool((qa <= config.eps).detach().cpu()):
                        continue
                    root = disc.clamp_min(0.0).sqrt()
                    t_near = (-qb - root) * 0.5 / qa
                    t_far = (-qb + root) * 0.5 / qa
                    if bool((t_far <= config.near_plane).detach().cpu()):
                        continue
                    t_near = torch.maximum(t_near, points.new_tensor(config.near_plane))

                    inside = True
                    for edge in range(int(offsets[cell_i]), int(offsets[cell_i + 1])):
                        neighbor_i = int(adjacency[edge])
                        if neighbor_i == cell_i:
                            continue
                        neighbor_center = points[neighbor_i]
                        n = neighbor_center - center
                        h = 0.5 * (
                            torch.dot(neighbor_center, neighbor_center)
                            - torch.dot(center, center)
                            + radius * radius
                            - radii[neighbor_i] * radii[neighbor_i]
                        )
                        dp = torch.dot(direction, n)
                        num = h - torch.dot(origin, n)
                        if bool((dp.abs() <= config.eps).detach().cpu()):
                            if bool((num < -config.eps).detach().cpu()):
                                inside = False
                                break
                            continue
                        t_face = num / dp
                        if bool((dp > 0.0).detach().cpu()):
                            t_far = torch.minimum(t_far, t_face)
                        else:
                            t_near = torch.maximum(t_near, t_face)
                        if bool((t_far <= t_near).detach().cpu()):
                            inside = False
                            break
                    if not inside or bool((t_far <= t_near).detach().cpu()):
                        continue

                    normal = normals[cell_i]
                    dp_surface = torch.dot(direction, normal)
                    if bool((dp_surface.abs() <= config.eps).detach().cpu()):
                        continue
                    if texel_heights is None:
                        t_surface = (torch.dot(center, normal) - torch.dot(origin, normal)) / dp_surface
                        t_color = t_surface
                    else:
                        t_flat = (torch.dot(center, normal) - torch.dot(origin, normal)) / dp_surface
                        t_query0 = torch.where(dp_surface >= 0.0, t_near, torch.maximum(t_near, t_flat))
                        local0 = (origin + direction * t_query0 - center) / radius.clamp_min(config.eps)
                        texel_coord0 = torch.stack(
                            [
                                torch.dot(local0, frame_tangents[cell_i]),
                                torch.dot(local0, frame_bitangents[cell_i]),
                            ],
                            dim=0,
                        )
                        diff0 = texel_coord0[None, :] - texel_sites[cell_i]
                        weights0 = torch.exp(-config.texel_temperature * diff0.square().sum(dim=-1))
                        height_value = (weights0 * texel_heights[cell_i]).sum() / weights0.sum().clamp_min(config.eps)
                        t_surface = (torch.dot(center, normal) - torch.dot(origin, normal) + height_value) / dp_surface
                        height_clipped_far = bool(
                            (dp_surface >= 0.0).detach().cpu()
                        ) and bool((t_surface < t_far).detach().cpu())
                    if bool((dp_surface >= 0.0).detach().cpu()):
                        t_far = torch.minimum(t_far, t_surface)
                    else:
                        t_near = torch.maximum(t_near, t_surface)
                    if bool((t_far <= t_near).detach().cpu()):
                        continue
                    if texel_heights is not None:
                        t_color = t_far if height_clipped_far else t_near

                    alpha = (1.0 - torch.exp(-sigma * (t_far - t_near))).clamp(0.0, config.max_alpha)
                    if bool((alpha < config.alpha_threshold).detach().cpu()):
                        continue
                    local = (origin + direction * t_color - center) / radius.clamp_min(config.eps)
                    texel_coord = torch.stack(
                        [torch.dot(local, frame_tangents[cell_i]), torch.dot(local, frame_bitangents[cell_i])],
                        dim=0,
                    )
                    diff = texel_coord[None, :] - texel_sites[cell_i]
                    weights = torch.exp(-config.texel_temperature * diff.square().sum(dim=-1))
                    if texel_sv_axis is None:
                        texel_color = texel_features[cell_i]
                    else:
                        sites = texel_sites[cell_i]
                        texel_world = (
                            center[None, :]
                            + radius
                            * (
                                sites[:, 0:1] * frame_tangents[cell_i][None, :]
                                + sites[:, 1:2] * frame_bitangents[cell_i][None, :]
                            )
                        ).detach()
                        view_dirs = F.normalize(texel_world - origin[None, :], dim=-1, eps=config.eps).unsqueeze(1)
                        temps = texel_sv_axis[cell_i].norm(dim=-1).clamp_min(config.eps)
                        axes = texel_sv_axis[cell_i] / temps[..., None]
                        dist = (view_dirs - axes).norm(dim=-1)
                        sv_weights = torch.exp(-temps * dist)
                        sv_weight_sum = sv_weights.sum(dim=-1, keepdim=True).clamp_min(config.eps)
                        texel_color = (
                            (sv_weights[..., None] * texel_sv_rgb[cell_i]).sum(dim=-2) / sv_weight_sum + 0.5
                        ).clamp_min(0.0)
                    color = (weights[:, None] * texel_color).sum(dim=0) / weights.sum().clamp_min(config.eps)
                    weight = transmittance * alpha
                    rgb = rgb + weight * color
                    transmittance = transmittance * (1.0 - alpha)
                out[batch, y, x] = rgb
                alpha_out[batch, y, x] = 1.0 - transmittance
    return out, alpha_out


def check_height_sv_material_uses_height_endpoint_against_reference(device: torch.device) -> None:
    points = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    radii = torch.tensor([5.0], device=device, dtype=torch.float32)
    densities = torch.tensor([4.0], device=device, dtype=torch.float32)
    adjacency = torch.empty((0,), device=device, dtype=torch.int32)
    offsets = torch.tensor([0, 0], device=device, dtype=torch.int32)
    sorted_ids = torch.tensor([[0]], device=device, dtype=torch.int32)
    rays = torch.tensor([[[[0.0, -4.0, -2.0, 0.0, 1.0, 1.0]]]], device=device, dtype=torch.float32)
    texel_sites = torch.tensor([[[-0.5, 0.0], [-0.2, 0.0], [0.2, 0.0], [0.5, 0.0]]], device=device)
    texel_heights = torch.ones((1, 4), device=device, dtype=torch.float32)
    texel_features = torch.zeros((1, 4, 1), device=device, dtype=torch.float32)
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
    metal_out, metal_alpha = rasterize_power_foam_oriented_height_sv_texel_surface(
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
        sorted_ids=sorted_ids,
        tangents=tangents,
        bitangents=bitangents,
    )
    ref_out, ref_alpha = torch_texel_reference(
        points,
        radii,
        densities,
        texel_sites,
        texel_features,
        normals,
        tangents,
        bitangents,
        texel_heights,
        adjacency,
        offsets,
        sorted_ids,
        rays,
        cfg,
        texel_sv_axis=texel_sv_axis,
        texel_sv_rgb=texel_sv_rgb,
    )
    torch.mps.synchronize()
    metal_color = metal_out[0, 0, 0] / metal_alpha[0, 0, 0].clamp_min(1.0e-6)
    ref_color = ref_out[0, 0, 0] / ref_alpha[0, 0, 0].clamp_min(1.0e-6)
    color_err = float((metal_color - ref_color).abs().max().detach().cpu())
    print("linear height_sv material endpoint color:", metal_color.detach().cpu().tolist())
    print("linear height_sv material endpoint reference max error:", color_err)
    if color_err > 1.0e-2:
        raise AssertionError("height+SV material endpoint disagreed with torch reference")
    if not (float(metal_color[1].detach().cpu()) > 0.75 and float(metal_color[0].detach().cpu()) < 0.25):
        raise AssertionError("height+SV material did not sample the height-clipped endpoint")


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Metal linear texture check")
    device = torch.device("mps")
    check_height_sv_material_uses_height_endpoint_against_reference(device)
    config = FoamRasterConfig(alpha_threshold=0.0, max_alpha=0.95, transmittance_threshold=1.0e-5)
    base = (
        torch.tensor([[-0.12, -0.08, 2.0], [0.08, -0.04, 2.25], [-0.03, 0.12, 2.5]], device=device),
        torch.tensor([0.65, 0.60, 0.58], device=device),
        torch.tensor([0.7, 0.9, 0.6], device=device),
        torch.tensor(
            [
                [[0.20, 0.05, -0.02, 0.01], [0.70, -0.04, 0.03, 0.02]],
                [[0.80, 0.01, 0.04, -0.03], [0.10, 0.02, -0.05, 0.01]],
                [[0.30, -0.03, 0.02, 0.04], [0.40, 0.04, 0.01, -0.02]],
            ],
            device=device,
        ),
    )
    normals = torch.nn.functional.normalize(
        torch.tensor(
            [[0.06, 0.02, -1.0], [-0.03, 0.04, -1.0], [0.02, -0.05, -1.0]],
            device=device,
        ),
        dim=-1,
    )
    adjacency, offsets = fully_connected_adjacency(3, device)
    rays = make_pinhole_rays(batch_size=1, height=5, width=4, device=device, fov_degrees=30.0)
    sorted_ids = make_power_sorted_ids(base[0], base[1], rays)

    def grads(fn, *, include_normals: bool = False):
        tensors = (*base, normals) if include_normals else base
        params = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    def check_case(name: str, metal_fn, *, surface: bool, oriented: bool = False) -> None:
        metal_loss, metal_grads, metal_out, metal_alpha = grads(
            lambda *params: metal_fn(*params, adjacency, offsets, rays, config, sorted_ids=sorted_ids),
            include_normals=oriented,
        )
        ref_loss, ref_grads, ref_out, ref_alpha = grads(
            lambda *params: torch_linear_reference(
                params[0],
                params[1],
                params[2],
                params[3],
                adjacency,
                offsets,
                sorted_ids,
                rays,
                config,
                surface=surface,
                normals=params[4] if oriented else None,
            ),
            include_normals=oriented,
        )

        print(name, "loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
        print(name, "features max error:", float((metal_out - ref_out).abs().max()))
        print(name, "alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
        names = ("points", "radii", "densities", "features", "normals") if oriented else (
            "points",
            "radii",
            "densities",
            "features",
        )
        for param_name, got, ref in zip(names, metal_grads, ref_grads):
            err = float((got - ref).abs().max())
            print(f"{name} {param_name} grad max error:", err)
            if err > 1.0e-4:
                raise AssertionError(f"{name} {param_name} grad max error {err} exceeded tolerance")

    check_case("linear", rasterize_power_foam_linear, surface=False)
    check_case("surface_linear", rasterize_power_foam_surface_linear, surface=True)
    check_case("oriented_surface_linear", rasterize_power_foam_oriented_surface_linear, surface=True, oriented=True)

    texel_sites = torch.tensor(
        [
            [[-0.25, -0.20], [0.20, -0.15], [0.05, 0.22]],
            [[-0.18, 0.18], [0.22, 0.16], [0.00, -0.24]],
            [[-0.20, 0.00], [0.18, 0.22], [0.12, -0.20]],
        ],
        device=device,
    )
    texel_features = torch.tensor(
        [
            [[0.20, 0.70], [0.35, 0.55], [0.12, 0.82]],
            [[0.80, 0.10], [0.65, 0.24], [0.92, 0.18]],
            [[0.30, 0.40], [0.44, 0.28], [0.22, 0.52]],
        ],
        device=device,
    )
    texel_heights = torch.tensor(
        [
            [0.035, -0.015, 0.020],
            [-0.025, 0.030, 0.010],
            [0.018, -0.020, 0.026],
        ],
        device=device,
    )
    texel_sv_axis = torch.tensor(
        [
            [
                [[1.10, 0.10, 0.20], [0.15, 1.25, -0.10], [-0.20, 0.25, 1.05]],
                [[1.00, -0.25, 0.15], [0.25, 1.10, 0.05], [0.10, -0.15, 1.20]],
                [[0.95, 0.18, -0.05], [-0.10, 1.30, 0.20], [0.22, 0.10, 1.15]],
            ],
            [
                [[1.20, -0.05, 0.10], [0.20, 1.00, 0.15], [-0.15, 0.10, 1.25]],
                [[1.05, 0.22, -0.12], [-0.18, 1.18, 0.08], [0.30, -0.05, 1.10]],
                [[0.90, -0.20, 0.25], [0.12, 1.22, -0.18], [-0.05, 0.20, 1.18]],
            ],
            [
                [[1.15, 0.12, -0.16], [-0.12, 1.05, 0.24], [0.18, -0.22, 1.30]],
                [[1.00, -0.18, 0.22], [0.22, 1.16, -0.04], [-0.20, 0.14, 1.08]],
                [[1.08, 0.05, 0.18], [-0.15, 1.25, -0.15], [0.10, 0.18, 1.12]],
            ],
        ],
        device=device,
    )
    texel_sv_rgb = torch.tensor(
        [
            [
                [[-0.20, 0.10, 0.05], [0.05, -0.12, 0.14], [0.12, 0.03, -0.08]],
                [[0.18, -0.08, 0.02], [-0.04, 0.16, -0.10], [0.08, 0.06, 0.12]],
                [[-0.10, 0.14, -0.06], [0.15, -0.02, 0.10], [0.02, 0.08, -0.12]],
            ],
            [
                [[0.10, -0.14, 0.08], [0.16, 0.04, -0.06], [-0.08, 0.12, 0.10]],
                [[-0.16, 0.06, 0.14], [0.12, -0.10, 0.04], [0.04, 0.18, -0.08]],
                [[0.14, 0.02, -0.12], [-0.06, 0.16, 0.06], [0.18, -0.04, 0.08]],
            ],
            [
                [[0.06, 0.12, -0.10], [-0.14, 0.10, 0.04], [0.16, -0.06, 0.12]],
                [[-0.08, 0.18, 0.06], [0.10, -0.14, 0.10], [0.04, 0.12, -0.04]],
                [[0.12, -0.04, 0.16], [-0.10, 0.08, -0.06], [0.06, 0.14, 0.02]],
            ],
        ],
        device=device,
    )

    def texel_grads(fn):
        params = [tensor.detach().clone().requires_grad_(True) for tensor in (*base[:3], texel_sites, texel_features, normals)]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = texel_grads(
        lambda p, r, d, s, tf, n: rasterize_power_foam_oriented_texel_surface(
            p,
            r,
            d,
            s,
            tf,
            n,
            adjacency,
            offsets,
            rays,
            config,
            sorted_ids=sorted_ids,
        )
    )
    ref_loss, ref_grads, ref_out, ref_alpha = texel_grads(
        lambda p, r, d, s, tf, n: torch_texel_reference(
            p,
            r,
            d,
            s,
            tf,
            n,
            None,
            None,
            None,
            adjacency,
            offsets,
            sorted_ids,
            rays,
            config,
        )
    )
    print("oriented_texel_surface loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("oriented_texel_surface features max error:", float((metal_out - ref_out).abs().max()))
    print("oriented_texel_surface alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for param_name, got, ref in zip(
        ("points", "radii", "densities", "texel_sites", "texel_features", "normals"),
        metal_grads,
        ref_grads,
    ):
        err = float((got - ref).abs().max())
        print(f"oriented_texel_surface {param_name} grad max error:", err)
        if err > 1.0e-4:
            raise AssertionError(f"oriented_texel_surface {param_name} grad max error {err} exceeded tolerance")

    default_tangents, default_bitangents = frame_from_normals(normals)
    angle = torch.tensor(0.43, device=device)
    rolled_tangents = torch.cos(angle) * default_tangents + torch.sin(angle) * default_bitangents
    rolled_bitangents = -torch.sin(angle) * default_tangents + torch.cos(angle) * default_bitangents

    def explicit_frame_grads(fn):
        params = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (*base[:3], texel_sites, texel_features, normals, rolled_tangents, rolled_bitangents)
        ]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = explicit_frame_grads(
        lambda p, r, d, s, tf, n, t, b: rasterize_power_foam_oriented_texel_surface(
            p,
            r,
            d,
            s,
            tf,
            n,
            adjacency,
            offsets,
            rays,
            config,
            sorted_ids=sorted_ids,
            tangents=t,
            bitangents=b,
        )
    )
    ref_loss, ref_grads, ref_out, ref_alpha = explicit_frame_grads(
        lambda p, r, d, s, tf, n, t, b: torch_texel_reference(
            p,
            r,
            d,
            s,
            tf,
            n,
            t,
            b,
            None,
            adjacency,
            offsets,
            sorted_ids,
            rays,
            config,
        )
    )
    print("oriented_texel_surface explicit-frame loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("oriented_texel_surface explicit-frame features max error:", float((metal_out - ref_out).abs().max()))
    print("oriented_texel_surface explicit-frame alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for param_name, got, ref in zip(
        ("points", "radii", "densities", "texel_sites", "texel_features", "normals", "tangents", "bitangents"),
        metal_grads,
        ref_grads,
    ):
        err = float((got - ref).abs().max())
        print(f"oriented_texel_surface explicit-frame {param_name} grad max error:", err)
        if err > 1.0e-4:
            raise AssertionError(
                f"oriented_texel_surface explicit-frame {param_name} grad max error {err} exceeded tolerance"
            )

    def height_texel_grads(fn):
        params = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (
                *base[:3],
                texel_sites,
                texel_heights,
                texel_features,
                normals,
                rolled_tangents,
                rolled_bitangents,
            )
        ]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = height_texel_grads(
        lambda p, r, d, s, h, tf, n, t, b: rasterize_power_foam_oriented_height_texel_surface(
            p,
            r,
            d,
            s,
            h,
            tf,
            n,
            adjacency,
            offsets,
            rays,
            config,
            sorted_ids=sorted_ids,
            tangents=t,
            bitangents=b,
        )
    )
    ref_loss, ref_grads, ref_out, ref_alpha = height_texel_grads(
        lambda p, r, d, s, h, tf, n, t, b: torch_texel_reference(
            p,
            r,
            d,
            s,
            tf,
            n,
            t,
            b,
            h,
            adjacency,
            offsets,
            sorted_ids,
            rays,
            config,
        )
    )
    print("oriented_height_texel_surface explicit-frame loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("oriented_height_texel_surface explicit-frame features max error:", float((metal_out - ref_out).abs().max()))
    print("oriented_height_texel_surface explicit-frame alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for param_name, got, ref in zip(
        (
            "points",
            "radii",
            "densities",
            "texel_sites",
            "texel_heights",
            "texel_features",
            "normals",
            "tangents",
            "bitangents",
        ),
        metal_grads,
        ref_grads,
    ):
        err = float((got - ref).abs().max())
        print(f"oriented_height_texel_surface explicit-frame {param_name} grad max error:", err)
        if err > 2.0e-4:
            raise AssertionError(
                f"oriented_height_texel_surface explicit-frame {param_name} grad max error {err} exceeded tolerance"
            )

    def height_sv_texel_grads(fn):
        params = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (
                *base[:3],
                texel_sites,
                texel_heights,
                texel_sv_axis,
                texel_sv_rgb,
                normals,
                rolled_tangents,
                rolled_bitangents,
            )
        ]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = height_sv_texel_grads(
        lambda p, r, d, s, h, sva, svr, n, t, b: rasterize_power_foam_oriented_height_sv_texel_surface(
            p,
            r,
            d,
            s,
            h,
            sva,
            svr,
            n,
            adjacency,
            offsets,
            rays,
            config,
            sorted_ids=sorted_ids,
            tangents=t,
            bitangents=b,
        )
    )
    ref_loss, ref_grads, ref_out, ref_alpha = height_sv_texel_grads(
        lambda p, r, d, s, h, sva, svr, n, t, b: torch_texel_reference(
            p,
            r,
            d,
            s,
            texel_features,
            n,
            t,
            b,
            h,
            adjacency,
            offsets,
            sorted_ids,
            rays,
            config,
            texel_sv_axis=sva,
            texel_sv_rgb=svr,
        )
    )
    print("oriented_height_sv_texel_surface explicit-frame loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("oriented_height_sv_texel_surface explicit-frame features max error:", float((metal_out - ref_out).abs().max()))
    print("oriented_height_sv_texel_surface explicit-frame alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for param_name, got, ref in zip(
        (
            "points",
            "radii",
            "densities",
            "texel_sites",
            "texel_heights",
            "texel_sv_axis",
            "texel_sv_rgb",
            "normals",
            "tangents",
            "bitangents",
        ),
        metal_grads,
        ref_grads,
    ):
        err = float((got - ref).abs().max())
        print(f"oriented_height_sv_texel_surface explicit-frame {param_name} grad max error:", err)
        if err > 3.0e-4:
            raise AssertionError(
                f"oriented_height_sv_texel_surface explicit-frame {param_name} grad max error {err} exceeded tolerance"
            )

    quaternions = torch.nn.functional.normalize(
        torch.tensor(
            [
                [0.92, 0.08, -0.37, 0.03],
                [0.88, -0.02, -0.45, 0.10],
                [0.95, 0.04, -0.30, -0.05],
            ],
            device=device,
        ),
        dim=-1,
    )

    def quaternion_frame_grads(fn):
        params = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (*base[:3], texel_sites, texel_features, quaternions)
        ]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = quaternion_frame_grads(
        lambda p, r, d, s, tf, q: rasterize_power_foam_quaternion_texel_surface(
            p,
            r,
            d,
            s,
            tf,
            q,
            adjacency,
            offsets,
            rays,
            config,
            sorted_ids=sorted_ids,
        )
    )
    ref_loss, ref_grads, ref_out, ref_alpha = quaternion_frame_grads(
        lambda p, r, d, s, tf, q: torch_texel_reference(
            p,
            r,
            d,
            s,
            tf,
            *quaternion_frames(q, eps=config.eps),
            None,
            adjacency,
            offsets,
            sorted_ids,
            rays,
            config,
        )
    )
    print("quaternion_texel_surface loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("quaternion_texel_surface features max error:", float((metal_out - ref_out).abs().max()))
    print("quaternion_texel_surface alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for param_name, got, ref in zip(
        ("points", "radii", "densities", "texel_sites", "texel_features", "quaternions"),
        metal_grads,
        ref_grads,
    ):
        err = float((got - ref).abs().max())
        print(f"quaternion_texel_surface {param_name} grad max error:", err)
        if err > 1.0e-4:
            raise AssertionError(f"quaternion_texel_surface {param_name} grad max error {err} exceeded tolerance")

    def quaternion_height_grads(fn):
        params = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (*base[:3], texel_sites, texel_heights, texel_features, quaternions)
        ]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = quaternion_height_grads(
        lambda p, r, d, s, h, tf, q: rasterize_power_foam_quaternion_height_texel_surface(
            p,
            r,
            d,
            s,
            h,
            tf,
            q,
            adjacency,
            offsets,
            rays,
            config,
            sorted_ids=sorted_ids,
        )
    )
    ref_loss, ref_grads, ref_out, ref_alpha = quaternion_height_grads(
        lambda p, r, d, s, h, tf, q: torch_texel_reference(
            p,
            r,
            d,
            s,
            tf,
            *quaternion_frames(q, eps=config.eps),
            h,
            adjacency,
            offsets,
            sorted_ids,
            rays,
            config,
        )
    )
    print("quaternion_height_texel_surface loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("quaternion_height_texel_surface features max error:", float((metal_out - ref_out).abs().max()))
    print("quaternion_height_texel_surface alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for param_name, got, ref in zip(
        ("points", "radii", "densities", "texel_sites", "texel_heights", "texel_features", "quaternions"),
        metal_grads,
        ref_grads,
    ):
        err = float((got - ref).abs().max())
        print(f"quaternion_height_texel_surface {param_name} grad max error:", err)
        if err > 2.0e-4:
            raise AssertionError(f"quaternion_height_texel_surface {param_name} grad max error {err} exceeded tolerance")

    def quaternion_height_sv_grads(fn):
        params = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in (*base[:3], texel_sites, texel_heights, texel_sv_axis, texel_sv_rgb, quaternions)
        ]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = quaternion_height_sv_grads(
        lambda p, r, d, s, h, sva, svr, q: rasterize_power_foam_quaternion_height_sv_texel_surface(
            p,
            r,
            d,
            s,
            h,
            sva,
            svr,
            q,
            adjacency,
            offsets,
            rays,
            config,
            sorted_ids=sorted_ids,
        )
    )
    ref_loss, ref_grads, ref_out, ref_alpha = quaternion_height_sv_grads(
        lambda p, r, d, s, h, sva, svr, q: torch_texel_reference(
            p,
            r,
            d,
            s,
            texel_features,
            *quaternion_frames(q, eps=config.eps),
            h,
            adjacency,
            offsets,
            sorted_ids,
            rays,
            config,
            texel_sv_axis=sva,
            texel_sv_rgb=svr,
        )
    )
    print("quaternion_height_sv_texel_surface loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("quaternion_height_sv_texel_surface features max error:", float((metal_out - ref_out).abs().max()))
    print("quaternion_height_sv_texel_surface alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for param_name, got, ref in zip(
        (
            "points",
            "radii",
            "densities",
            "texel_sites",
            "texel_heights",
            "texel_sv_axis",
            "texel_sv_rgb",
            "quaternions",
        ),
        metal_grads,
        ref_grads,
    ):
        err = float((got - ref).abs().max())
        print(f"quaternion_height_sv_texel_surface {param_name} grad max error:", err)
        if err > 3.0e-4:
            raise AssertionError(f"quaternion_height_sv_texel_surface {param_name} grad max error {err} exceeded tolerance")
    print("powerfoam Metal linear texture checks passed")


if __name__ == "__main__":
    main()
