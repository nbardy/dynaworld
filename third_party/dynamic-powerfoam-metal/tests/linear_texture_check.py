from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_dynamic_powerfoam_metal import (
    FoamRasterConfig,
    rasterize_power_foam_linear,
    rasterize_power_foam_oriented_surface_linear,
    rasterize_power_foam_oriented_texel_surface,
    rasterize_power_foam_surface_linear,
)
from torch_dynamic_powerfoam_metal.random_scene import make_pinhole_rays, make_power_sorted_ids


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
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    sorted_ids: torch.Tensor,
    rays: torch.Tensor,
    config: FoamRasterConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, height, width = rays.shape[:3]
    output_dim = texel_features.shape[2]
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
                    local = (origin + direction * t_surface - center) / radius.clamp_min(config.eps)
                    texel_coord = torch.stack(
                        [torch.dot(local, frame_tangents[cell_i]), torch.dot(local, frame_bitangents[cell_i])],
                        dim=0,
                    )
                    diff = texel_coord[None, :] - texel_sites[cell_i]
                    weights = torch.exp(-config.texel_temperature * diff.square().sum(dim=-1))
                    color = (weights[:, None] * texel_features[cell_i]).sum(dim=0) / weights.sum().clamp_min(config.eps)
                    weight = transmittance * alpha
                    rgb = rgb + weight * color
                    transmittance = transmittance * (1.0 - alpha)
                out[batch, y, x] = rgb
                alpha_out[batch, y, x] = 1.0 - transmittance
    return out, alpha_out


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Metal linear texture check")
    device = torch.device("mps")
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
    print("powerfoam Metal linear texture checks passed")


if __name__ == "__main__":
    main()
