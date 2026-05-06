from __future__ import annotations

import math

import torch
from torch import Tensor


def make_pinhole_rays(
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    fov_degrees: float = 48.0,
) -> Tensor:
    half_y = math.tan(math.radians(fov_degrees) * 0.5)
    half_x = half_y * (float(width) / float(height))
    ys = torch.linspace(half_y, -half_y, height, device=device, dtype=torch.float32)
    xs = torch.linspace(-half_x, half_x, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    origins = torch.zeros_like(dirs)
    rays = torch.cat([origins, dirs], dim=-1)
    return rays.unsqueeze(0).repeat(batch_size, 1, 1, 1).contiguous()


def make_random_foam(
    *,
    cell_count: int,
    feature_dim: int,
    device: torch.device,
    seed: int,
    depth_min: float = 1.6,
    depth_max: float = 4.4,
    fov_degrees: float = 48.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.rand(cell_count, device=device, dtype=torch.float32, generator=gen)
    z = depth_min + (depth_max - depth_min) * z
    half_y = math.tan(math.radians(fov_degrees) * 0.5) * z
    half_x = 1.25 * half_y
    x = (torch.rand(cell_count, device=device, dtype=torch.float32, generator=gen) * 2.0 - 1.0) * half_x
    y = (torch.rand(cell_count, device=device, dtype=torch.float32, generator=gen) * 2.0 - 1.0) * half_y
    points = torch.stack([x, y, z], dim=-1).contiguous()

    # Keep sphere sizes large enough to produce visible cells but small enough
    # that random scenes do not collapse into one opaque blob.
    base = 0.18 * (256.0 / max(float(cell_count), 1.0)) ** (1.0 / 3.0)
    radii = torch.rand(cell_count, device=device, dtype=torch.float32, generator=gen)
    radii = (base * (0.75 + 1.65 * radii)).clamp_min(0.035).contiguous()
    densities = torch.rand(cell_count, device=device, dtype=torch.float32, generator=gen)
    densities = (1.0 + 3.0 * densities).contiguous()
    features = torch.rand(cell_count, feature_dim, device=device, dtype=torch.float32, generator=gen)
    return points, radii, densities, features.contiguous()


def make_power_sorted_ids(points: Tensor, radii: Tensor, rays: Tensor) -> Tensor:
    if rays.ndim == 3:
        rays = rays.unsqueeze(0)
    origins = rays[:, 0, 0, :3]
    power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    return torch.argsort(power.detach(), dim=1, stable=True).to(torch.int32).contiguous()


def make_overlap_adjacency(points: Tensor, radii: Tensor, *, max_neighbors: int = 64) -> tuple[Tensor, Tensor]:
    points_cpu = points.detach().cpu()
    radii_cpu = radii.detach().cpu()
    rows: list[int] = []
    offsets = [0]
    for i in range(points_cpu.shape[0]):
        dist = torch.linalg.vector_norm(points_cpu - points_cpu[i], dim=-1)
        overlap = dist < (radii_cpu + radii_cpu[i])
        overlap[i] = False
        ids = torch.nonzero(overlap, as_tuple=False).flatten()
        if ids.numel() > max_neighbors:
            _, order = torch.topk(dist[ids], k=max_neighbors, largest=False)
            ids = ids[order]
        rows.extend(int(v) for v in ids.tolist())
        offsets.append(len(rows))
    device = points.device
    return (
        torch.tensor(rows, device=device, dtype=torch.int32),
        torch.tensor(offsets, device=device, dtype=torch.int32),
    )


def make_knn_adjacency(points: Tensor, *, neighbors: int) -> tuple[Tensor, Tensor]:
    points_cpu = points.detach().cpu()
    n = points_cpu.shape[0]
    k = min(max(int(neighbors), 0), max(n - 1, 0))
    if k == 0:
        device = points.device
        return torch.empty((0,), device=device, dtype=torch.int32), torch.zeros((n + 1,), device=device, dtype=torch.int32)
    rows: list[int] = []
    offsets = [0]
    for i in range(n):
        dist = torch.linalg.vector_norm(points_cpu - points_cpu[i], dim=-1)
        dist[i] = float("inf")
        ids = torch.topk(dist, k=k, largest=False).indices
        rows.extend(int(v) for v in ids.tolist())
        offsets.append(len(rows))
    device = points.device
    return (
        torch.tensor(rows, device=device, dtype=torch.int32),
        torch.tensor(offsets, device=device, dtype=torch.int32),
    )


def make_adjacency(points: Tensor, radii: Tensor, *, mode: str, neighbors: int) -> tuple[Tensor, Tensor]:
    if mode == "overlap":
        return make_overlap_adjacency(points, radii, max_neighbors=neighbors)
    if mode == "knn":
        return make_knn_adjacency(points, neighbors=neighbors)
    raise ValueError("adjacency mode must be 'overlap' or 'knn'")
