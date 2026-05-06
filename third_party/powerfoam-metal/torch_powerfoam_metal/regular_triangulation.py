from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from torch import Tensor


def _require_scipy_spatial():
    try:
        from scipy.spatial import ConvexHull, QhullError  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without optional scipy.
        raise ImportError(
            "regular_triangulation adjacency requires scipy.spatial/Qhull. "
            "Install scipy or choose adjacency_mode='cech_aabb'."
        ) from exc
    return ConvexHull, QhullError


def regular_triangulation_edges_numpy(
    points: np.ndarray,
    radii: np.ndarray,
    *,
    lower_tol: float = 1.0e-10,
) -> set[tuple[int, int]]:
    """Return undirected weighted-Delaunay edges from the lower lifted hull."""
    points = np.asarray(points, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    if radii.shape != (points.shape[0],):
        raise ValueError("radii must be [N]")
    cell_count = int(points.shape[0])
    if cell_count < 2:
        return set()
    if cell_count <= 4:
        return {(i, j) for i in range(cell_count) for j in range(i + 1, cell_count)}

    ConvexHull, QhullError = _require_scipy_spatial()
    lifted_height = np.sum(points * points, axis=1) - radii * radii
    lifted = np.concatenate([points, lifted_height[:, None]], axis=1)
    try:
        hull = ConvexHull(lifted)
    except QhullError:
        hull = ConvexHull(lifted, qhull_options="QJ")

    edges: set[tuple[int, int]] = set()
    for simplex, equation in zip(hull.simplices, hull.equations):
        # equation[:-1] is the outward facet normal. Lower facets have an
        # outward component toward negative lifted height.
        if float(equation[-2]) >= -float(lower_tol):
            continue
        for a, b in combinations((int(v) for v in simplex), 2):
            if a != b:
                edges.add((min(a, b), max(a, b)))
    if not edges:
        raise RuntimeError("regular triangulation produced no lower-hull edges")
    return edges


def make_regular_triangulation_adjacency(points: Tensor, radii: Tensor) -> tuple[Tensor, Tensor]:
    points_cpu = points.detach().cpu().to(dtype=torch.float64).numpy()
    radii_cpu = radii.detach().cpu().to(dtype=torch.float64).numpy()
    edges = regular_triangulation_edges_numpy(points_cpu, radii_cpu)
    neighbor_sets = [set() for _ in range(int(points_cpu.shape[0]))]
    for a, b in edges:
        neighbor_sets[a].add(b)
        neighbor_sets[b].add(a)

    rows: list[int] = []
    offsets = [0]
    for cell, neighbors in enumerate(neighbor_sets):
        ids = sorted(
            neighbors,
            key=lambda idx: (float(np.linalg.norm(points_cpu[idx] - points_cpu[cell])), int(idx)),
        )
        rows.extend(ids)
        offsets.append(len(rows))
    return (
        torch.tensor(rows, device=points.device, dtype=torch.int32),
        torch.tensor(offsets, device=points.device, dtype=torch.int32),
    )
