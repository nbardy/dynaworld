from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_powerfoam_metal import FoamRasterConfig, rasterize_power_foam


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


def make_rays(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-0.28, 0.28, height, device=device, dtype=torch.float32)
    xs = torch.linspace(-0.28, 0.28, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    origins = torch.zeros_like(dirs)
    rays = torch.cat([origins, dirs], dim=-1)
    return rays.unsqueeze(0).repeat(batch, 1, 1, 1).contiguous()


def sorted_ids_for(points: torch.Tensor, radii: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    origins = rays[:, 0, 0, :3]
    power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    return torch.argsort(power.detach(), dim=1, stable=True).to(torch.int32).contiguous()


def reference(
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
    points = points.cpu()
    radii = radii.cpu()
    densities = densities.cpu()
    features = features.cpu()
    adjacency = adjacency.cpu()
    offsets = offsets.cpu()
    sorted_ids = sorted_ids.cpu()
    rays = rays.cpu()

    bsz, height, width = rays.shape[:3]
    n, fdim = features.shape
    out = torch.zeros((bsz, height, width, fdim), dtype=torch.float32)
    alpha_out = torch.zeros((bsz, height, width), dtype=torch.float32)

    for b in range(bsz):
        for y in range(height):
            for x in range(width):
                origin = rays[b, y, x, :3]
                direction = rays[b, y, x, 3:]
                transmittance = torch.tensor(1.0)
                for cell_i in sorted_ids[b].tolist():
                    if transmittance <= config.transmittance_threshold:
                        break
                    if cell_i < 0 or cell_i >= n:
                        continue
                    density = torch.clamp(densities[cell_i], min=0.0)
                    radius = radii[cell_i]
                    if density <= 0.0 or radius <= 0.0:
                        continue

                    center = points[cell_i]
                    oc = origin - center
                    a = torch.dot(direction, direction)
                    bb = 2.0 * torch.dot(oc, direction)
                    c = torch.dot(oc, oc) - radius * radius
                    disc = bb * bb - 4.0 * a * c
                    if disc < 0.0 or a <= config.eps:
                        continue
                    root = torch.sqrt(torch.clamp(disc, min=0.0))
                    t0 = (-bb - root) * 0.5 / a
                    t1 = (-bb + root) * 0.5 / a
                    if t1 <= config.near_plane:
                        continue
                    t0 = torch.maximum(t0, torch.tensor(config.near_plane))
                    if t1 <= t0:
                        continue

                    inside = True
                    for edge in range(int(offsets[cell_i]), int(offsets[cell_i + 1])):
                        neighbor_i = int(adjacency[edge])
                        if neighbor_i < 0 or neighbor_i >= n or neighbor_i == cell_i:
                            continue
                        pj = points[neighbor_i]
                        nvec = pj - center
                        rhs = torch.dot(pj, pj) - torch.dot(center, center) + radius ** 2 - radii[neighbor_i] ** 2
                        limit = rhs - 2.0 * torch.dot(origin, nvec)
                        denom = 2.0 * torch.dot(direction, nvec)
                        if torch.abs(denom) <= config.eps:
                            if limit < -config.eps:
                                inside = False
                                break
                            continue
                        split = limit / denom
                        if denom > 0.0:
                            t1 = torch.minimum(t1, split)
                        else:
                            t0 = torch.maximum(t0, split)
                        if t1 <= t0:
                            inside = False
                            break
                    if not inside or t1 <= t0:
                        continue

                    cell_alpha = 1.0 - torch.exp(-density * (t1 - t0))
                    cell_alpha = torch.clamp(cell_alpha, 0.0, config.max_alpha)
                    if cell_alpha < config.alpha_threshold:
                        continue
                    weight = transmittance * cell_alpha
                    out[b, y, x] += weight * features[cell_i]
                    transmittance = transmittance * (1.0 - cell_alpha)
                alpha_out[b, y, x] = 1.0 - transmittance
    return out, alpha_out


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, threshold: float) -> None:
    err = float((got.detach().cpu() - ref.detach().cpu()).abs().max())
    print(f"{name} max error:", err)
    if err > threshold:
        raise AssertionError(f"{name} max error {err} exceeded {threshold}")


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Metal reference check")
    device = torch.device("mps")
    torch.manual_seed(12)

    points = torch.tensor(
        [
            [-0.10, 0.00, 2.05],
            [0.18, -0.02, 2.23],
            [-0.20, 0.14, 2.42],
            [0.08, 0.17, 2.78],
        ],
        device=device,
        dtype=torch.float32,
    )
    radii = torch.tensor([0.72, 0.66, 0.61, 0.76], device=device, dtype=torch.float32)
    densities = torch.tensor([1.20, 0.85, 1.45, 0.55], device=device, dtype=torch.float32)
    features = torch.rand((points.shape[0], 5), device=device, dtype=torch.float32)
    adjacency, offsets = fully_connected_adjacency(points.shape[0], device)
    rays = make_rays(batch=2, height=10, width=9, device=device)
    sorted_ids = sorted_ids_for(points, radii, rays)
    config = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=1.0e-5)

    got, got_alpha = rasterize_power_foam(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays,
        config,
        sorted_ids=sorted_ids,
    )
    ref, ref_alpha = reference(points, radii, densities, features, adjacency, offsets, sorted_ids, rays, config)

    assert_close("features", got, ref, 2.0e-5)
    assert_close("alpha", got_alpha, ref_alpha, 2.0e-5)
    print("powerfoam Metal reference check passed")


if __name__ == "__main__":
    main()
