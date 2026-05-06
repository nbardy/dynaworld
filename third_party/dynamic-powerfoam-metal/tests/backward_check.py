from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_dynamic_powerfoam_metal import FoamRasterConfig, rasterize_power_foam
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


def torch_reference(
    points: torch.Tensor,
    radii: torch.Tensor,
    densities: torch.Tensor,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    sorted_ids: torch.Tensor,
    rays: torch.Tensor,
    config: FoamRasterConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, height, width = rays.shape[:3]
    _, feature_dim = features.shape
    out = torch.zeros(batch_size, height, width, feature_dim, device=points.device, dtype=points.dtype)
    alpha_out = torch.zeros(batch_size, height, width, device=points.device, dtype=points.dtype)

    for batch in range(batch_size):
        for y in range(height):
            for x in range(width):
                origin = rays[batch, y, x, :3]
                direction = rays[batch, y, x, 3:]
                direction = direction / direction.norm().clamp_min(config.eps)
                transmittance = points.new_tensor(1.0)
                rgb = torch.zeros(feature_dim, device=points.device, dtype=points.dtype)
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

                    alpha = (1.0 - torch.exp(-sigma * (t_far - t_near))).clamp(0.0, config.max_alpha)
                    if bool((alpha < config.alpha_threshold).detach().cpu()):
                        continue
                    weight = transmittance * alpha
                    rgb = rgb + weight * features[cell_i]
                    transmittance = transmittance * (1.0 - alpha)
                out[batch, y, x] = rgb
                alpha_out[batch, y, x] = 1.0 - transmittance
    return out, alpha_out


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Metal backward check")
    device = torch.device("mps")
    config = FoamRasterConfig(alpha_threshold=0.0, max_alpha=0.95, transmittance_threshold=1.0e-5)
    base = (
        torch.tensor([[-0.12, -0.08, 2.0], [0.08, -0.04, 2.25], [-0.03, 0.12, 2.5]], device=device),
        torch.tensor([0.65, 0.60, 0.58], device=device),
        torch.tensor([0.7, 0.9, 0.6], device=device),
        torch.tensor([[0.2, 0.7], [0.8, 0.1], [0.3, 0.4]], device=device),
    )
    adjacency, offsets = fully_connected_adjacency(3, device)
    rays = make_pinhole_rays(batch_size=1, height=5, width=4, device=device, fov_degrees=30.0)
    sorted_ids = make_power_sorted_ids(base[0], base[1], rays)

    def grads(fn):
        params = [tensor.detach().clone().requires_grad_(True) for tensor in base]
        out, alpha = fn(*params)
        loss = out.square().mean() + 0.37 * alpha.square().mean()
        loss.backward()
        return loss.detach(), [param.grad.detach().cpu() for param in params], out.detach().cpu(), alpha.detach().cpu()

    metal_loss, metal_grads, metal_out, metal_alpha = grads(
        lambda p, r, d, f: rasterize_power_foam(p, r, d, f, adjacency, offsets, rays, config, sorted_ids=sorted_ids)
    )
    ref_loss, ref_grads, ref_out, ref_alpha = grads(
        lambda p, r, d, f: torch_reference(p, r, d, f, adjacency, offsets, sorted_ids, rays, config)
    )

    print("loss:", float(metal_loss.cpu()), float(ref_loss.cpu()))
    print("features max error:", float((metal_out - ref_out).abs().max()))
    print("alpha max error:", float((metal_alpha - ref_alpha).abs().max()))
    for name, got, ref in zip(("points", "radii", "densities", "features"), metal_grads, ref_grads):
        err = float((got - ref).abs().max())
        print(f"{name} grad max error:", err)
        if err > 5.0e-5:
            raise AssertionError(f"{name} grad max error {err} exceeded tolerance")
    print("powerfoam Metal backward check passed")


if __name__ == "__main__":
    main()
