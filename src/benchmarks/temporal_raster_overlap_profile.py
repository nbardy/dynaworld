from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ProjectedState:
    means2d: torch.Tensor
    radius_px: torch.Tensor
    opacities: torch.Tensor
    feature_dim: int


def _quantiles(values: torch.Tensor, qs: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[str, float]:
    flat = values.detach().flatten().to(torch.float32)
    if flat.numel() == 0:
        return {f"p{int(q * 100):02d}": 0.0 for q in qs}
    q_tensor = torch.tensor(qs, device=flat.device, dtype=flat.dtype)
    return {f"p{int(q * 100):02d}": float(v.item()) for q, v in zip(qs, torch.quantile(flat, q_tensor))}


def make_synthetic_projected_state(
    *,
    frames: int,
    gaussians: int,
    height: int,
    width: int,
    feature_dim: int,
    radius_px: float,
    radius_jitter_px: float,
    motion_px: float,
    noise_px: float,
    opacity: float,
    seed: int,
) -> ProjectedState:
    if min(frames, gaussians, height, width, feature_dim) < 1:
        raise ValueError("frames, gaussians, height, width, and feature_dim must be positive.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    margin = max(float(radius_px) * 2.5, 4.0)
    centers = torch.empty((gaussians, 2), dtype=torch.float32)
    centers[:, 0] = torch.rand(gaussians, generator=generator) * max(float(width) - 2.0 * margin, 1.0) + margin
    centers[:, 1] = torch.rand(gaussians, generator=generator) * max(float(height) - 2.0 * margin, 1.0) + margin

    directions = torch.randn((gaussians, 2), generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    frame_times = torch.linspace(-0.5, 0.5, frames, dtype=torch.float32).view(frames, 1, 1)
    drift = frame_times * directions.view(1, gaussians, 2) * float(motion_px)
    noise = torch.randn((frames, gaussians, 2), generator=generator) * float(noise_px)
    means2d = centers.view(1, gaussians, 2) + drift + noise

    base_radius = torch.full((frames, gaussians), float(radius_px), dtype=torch.float32)
    if radius_jitter_px > 0.0:
        base_radius = base_radius + torch.randn((frames, gaussians), generator=generator) * float(radius_jitter_px)
    radii = base_radius.clamp_min(0.1)
    opacities = torch.full((frames, gaussians), float(opacity), dtype=torch.float32).clamp(0.0, 1.0)
    return ProjectedState(means2d=means2d, radius_px=radii, opacities=opacities, feature_dim=int(feature_dim))


def bounds_from_projected_state(
    state: ProjectedState,
    *,
    height: int,
    width: int,
    alpha_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means = state.means2d
    radii = state.radius_px
    if alpha_threshold > 0.0:
        opacity_factor = torch.sqrt(
            torch.clamp(
                2.0 * torch.log(state.opacities.clamp_min(alpha_threshold) / float(alpha_threshold)),
                min=0.0,
            )
        )
        bound_radius = torch.minimum(radii, radii * opacity_factor / 3.0)
    else:
        bound_radius = radii
    min_x = torch.floor(means[..., 0] - bound_radius).clamp(0, width - 1).to(torch.int64)
    max_x = torch.ceil(means[..., 0] + bound_radius).clamp(0, width - 1).to(torch.int64)
    min_y = torch.floor(means[..., 1] - bound_radius).clamp(0, height - 1).to(torch.int64)
    max_y = torch.ceil(means[..., 1] + bound_radius).clamp(0, height - 1).to(torch.int64)
    valid = (
        (state.opacities > alpha_threshold)
        & (max_x >= min_x)
        & (max_y >= min_y)
        & (means[..., 0] + bound_radius >= 0)
        & (means[..., 0] - bound_radius < width)
        & (means[..., 1] + bound_radius >= 0)
        & (means[..., 1] - bound_radius < height)
    )
    return min_x, max_x, min_y, max_y, valid


def active_sets_from_bounds(
    min_x: torch.Tensor,
    max_x: torch.Tensor,
    min_y: torch.Tensor,
    max_y: torch.Tensor,
    valid: torch.Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[set[int]], torch.Tensor, torch.Tensor]:
    frames, gaussians = valid.shape
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    tile_masks = torch.zeros((frames, tiles_y * tiles_x), dtype=torch.bool)
    pair_sets: list[set[int]] = []
    per_tile_counts = torch.zeros((frames, tiles_y * tiles_x), dtype=torch.int64)
    per_gaussian_tile_counts = torch.zeros((frames, gaussians), dtype=torch.int64)
    visible_masks = valid.to(torch.bool)
    for frame in range(frames):
        frame_pairs: set[int] = set()
        ids = torch.nonzero(valid[frame], as_tuple=False).flatten()
        for gaussian in ids.tolist():
            tx0 = int(min_x[frame, gaussian].item() // tile_size)
            tx1 = int(max_x[frame, gaussian].item() // tile_size)
            ty0 = int(min_y[frame, gaussian].item() // tile_size)
            ty1 = int(max_y[frame, gaussian].item() // tile_size)
            tile_ids = [
                ty * tiles_x + tx
                for ty in range(max(ty0, 0), min(ty1, tiles_y - 1) + 1)
                for tx in range(max(tx0, 0), min(tx1, tiles_x - 1) + 1)
            ]
            if tile_ids:
                tile_tensor = torch.tensor(tile_ids, dtype=torch.int64)
                per_tile_counts[frame, tile_tensor] += 1
                per_gaussian_tile_counts[frame, gaussian] = len(tile_ids)
                tile_masks[frame, tile_tensor] = True
                frame_pairs.update(gaussian * tiles_x * tiles_y + tile_id for tile_id in tile_ids)
        pair_sets.append(frame_pairs)
    return visible_masks, tile_masks, pair_sets, per_tile_counts, per_gaussian_tile_counts


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.bool).flatten()
    b = b.to(torch.bool).flatten()
    union = a | b
    if not bool(union.any()):
        return 1.0
    return float(((a & b).sum().to(torch.float32) / union.sum().to(torch.float32)).item())


def _retention(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.bool).flatten()
    if not bool(a.any()):
        return 1.0
    return float(((a & b.to(torch.bool).flatten()).sum().to(torch.float32) / a.sum().to(torch.float32)).item())


def _set_jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b))


def _set_retention(a: set[int], b: set[int]) -> float:
    if not a:
        return 1.0
    return float(len(a & b) / len(a))


def _pairwise_temporal_stats(mask: torch.Tensor, *, prefix: str) -> dict[str, Any]:
    frames = int(mask.shape[0])
    adjacent_jaccards = []
    adjacent_retentions = []
    all_jaccards = []
    for frame in range(frames - 1):
        adjacent_jaccards.append(_jaccard(mask[frame], mask[frame + 1]))
        adjacent_retentions.append(_retention(mask[frame], mask[frame + 1]))
    for left in range(frames):
        for right in range(left + 1, frames):
            all_jaccards.append(_jaccard(mask[left], mask[right]))
    adjacent = torch.tensor(adjacent_jaccards, dtype=torch.float32)
    all_pairs = torch.tensor(all_jaccards, dtype=torch.float32)
    retentions = torch.tensor(adjacent_retentions, dtype=torch.float32)
    return {
        f"{prefix}_adjacent_jaccard_mean": float(adjacent.mean().item()) if adjacent.numel() else 1.0,
        f"{prefix}_adjacent_jaccard_min": float(adjacent.min().item()) if adjacent.numel() else 1.0,
        f"{prefix}_adjacent_retention_mean": float(retentions.mean().item()) if retentions.numel() else 1.0,
        f"{prefix}_all_pairs_jaccard_mean": float(all_pairs.mean().item()) if all_pairs.numel() else 1.0,
        f"{prefix}_all_pairs_jaccard_min": float(all_pairs.min().item()) if all_pairs.numel() else 1.0,
    }


def _pairwise_set_temporal_stats(sets: list[set[int]], *, prefix: str) -> dict[str, Any]:
    frames = len(sets)
    adjacent_jaccards = []
    adjacent_retentions = []
    all_jaccards = []
    for frame in range(frames - 1):
        adjacent_jaccards.append(_set_jaccard(sets[frame], sets[frame + 1]))
        adjacent_retentions.append(_set_retention(sets[frame], sets[frame + 1]))
    for left in range(frames):
        for right in range(left + 1, frames):
            all_jaccards.append(_set_jaccard(sets[left], sets[right]))
    adjacent = torch.tensor(adjacent_jaccards, dtype=torch.float32)
    all_pairs = torch.tensor(all_jaccards, dtype=torch.float32)
    retentions = torch.tensor(adjacent_retentions, dtype=torch.float32)
    return {
        f"{prefix}_adjacent_jaccard_mean": float(adjacent.mean().item()) if adjacent.numel() else 1.0,
        f"{prefix}_adjacent_jaccard_min": float(adjacent.min().item()) if adjacent.numel() else 1.0,
        f"{prefix}_adjacent_retention_mean": float(retentions.mean().item()) if retentions.numel() else 1.0,
        f"{prefix}_all_pairs_jaccard_mean": float(all_pairs.mean().item()) if all_pairs.numel() else 1.0,
        f"{prefix}_all_pairs_jaccard_min": float(all_pairs.min().item()) if all_pairs.numel() else 1.0,
    }


def profile_projected_state(
    state: ProjectedState,
    *,
    height: int,
    width: int,
    tile_size: int,
    alpha_threshold: float,
) -> dict[str, Any]:
    min_x, max_x, min_y, max_y, valid = bounds_from_projected_state(
        state,
        height=height,
        width=width,
        alpha_threshold=alpha_threshold,
    )
    visible_masks, tile_masks, pair_sets, per_tile_counts, per_gaussian_tile_counts = active_sets_from_bounds(
        min_x,
        max_x,
        min_y,
        max_y,
        valid,
        height=height,
        width=width,
        tile_size=tile_size,
    )
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    tiles_total = tiles_x * tiles_y
    visible_counts = visible_masks.sum(dim=1).to(torch.float32)
    active_tile_counts = tile_masks.sum(dim=1).to(torch.float32)
    active_pair_counts = torch.tensor([len(pair_set) for pair_set in pair_sets], dtype=torch.float32)
    tile_work_dense = float(tiles_total * int(state.means2d.shape[1]))
    pair_density = active_pair_counts / max(tile_work_dense, 1.0)
    tile_density = active_tile_counts / max(float(tiles_total), 1.0)
    active_per_tile_counts = per_tile_counts[per_tile_counts > 0].to(torch.float32)
    active_per_gaussian_tile_counts = per_gaussian_tile_counts[per_gaussian_tile_counts > 0].to(torch.float32)
    metrics = {
        "input_mode": "synthetic_projected_approximation",
        "frames": int(state.means2d.shape[0]),
        "gaussians": int(state.means2d.shape[1]),
        "height": int(height),
        "width": int(width),
        "tile_size": int(tile_size),
        "tiles_x": int(tiles_x),
        "tiles_y": int(tiles_y),
        "tiles_total": int(tiles_total),
        "feature_dim": int(state.feature_dim),
        "alpha_threshold": float(alpha_threshold),
        "visible_gaussians_mean": float(visible_counts.mean().item()),
        "visible_gaussians_min": float(visible_counts.min().item()),
        "active_tiles_mean": float(active_tile_counts.mean().item()),
        "active_tiles_min": float(active_tile_counts.min().item()),
        "active_tiles_fraction_mean": float(tile_density.mean().item()),
        "gaussian_tile_pairs_mean": float(active_pair_counts.mean().item()),
        "gaussian_tile_pair_density_mean": float(pair_density.mean().item()),
        "radius_px": _quantiles(state.radius_px),
        "active_gaussians_per_tile": _quantiles(active_per_tile_counts),
        "active_tiles_per_gaussian": _quantiles(active_per_gaussian_tile_counts),
    }
    metrics.update(_pairwise_temporal_stats(visible_masks, prefix="visible_gaussian"))
    metrics.update(_pairwise_temporal_stats(tile_masks, prefix="active_tile"))
    metrics.update(_pairwise_set_temporal_stats(pair_sets, prefix="gaussian_tile_pair"))
    metrics["frames_detail"] = [
        {
            "frame": int(frame),
            "visible_gaussians": int(visible_counts[frame].item()),
            "active_tiles": int(active_tile_counts[frame].item()),
            "active_tiles_fraction": float(tile_density[frame].item()),
            "gaussian_tile_pairs": int(active_pair_counts[frame].item()),
            "gaussian_tile_pair_density": float(pair_density[frame].item()),
        }
        for frame in range(int(state.means2d.shape[0]))
    ]
    return metrics


def print_summary(metrics: dict[str, Any]) -> None:
    summary_keys = (
        "input_mode",
        "frames",
        "gaussians",
        "height",
        "width",
        "tile_size",
        "feature_dim",
        "visible_gaussians_mean",
        "active_tiles_fraction_mean",
        "gaussian_tile_pair_density_mean",
        "visible_gaussian_adjacent_jaccard_mean",
        "active_tile_adjacent_jaccard_mean",
        "gaussian_tile_pair_adjacent_jaccard_mean",
        "gaussian_tile_pair_adjacent_retention_mean",
    )
    for key in summary_keys:
        print(f"{key}: {metrics[key]}")
    print("frames:")
    for row in metrics["frames_detail"]:
        print(
            "  frame={frame} visible={visible_gaussians} active_tiles={active_tiles} "
            "tile_frac={active_tiles_fraction:.4f} pairs={gaussian_tile_pairs} "
            "pair_density={gaussian_tile_pair_density:.6f}".format(**row)
        )


def _csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile temporal active-set overlap for projected Gaussian raster work. "
            "Currently synthetic only; metrics are diagnostic and do not enable pruning."
        )
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--gaussians", type=_csv_ints, default=_csv_ints("8192"))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--radius-px", type=_csv_floats, default=_csv_floats("3.5"))
    parser.add_argument("--radius-jitter-px", type=float, default=0.2)
    parser.add_argument("--motion-px", type=_csv_floats, default=_csv_floats("2.0"))
    parser.add_argument("--noise-px", type=float, default=0.25)
    parser.add_argument("--opacity", type=float, default=0.8)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 128.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print only JSON metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for gaussians, radius_px, motion_px in itertools.product(args.gaussians, args.radius_px, args.motion_px):
        state = make_synthetic_projected_state(
            frames=args.frames,
            gaussians=gaussians,
            height=args.height,
            width=args.width,
            feature_dim=args.feature_dim,
            radius_px=radius_px,
            radius_jitter_px=args.radius_jitter_px,
            motion_px=motion_px,
            noise_px=args.noise_px,
            opacity=args.opacity,
            seed=args.seed,
        )
        metrics = profile_projected_state(
            state,
            height=args.height,
            width=args.width,
            tile_size=args.tile_size,
            alpha_threshold=args.alpha_threshold,
        )
        metrics["synthetic_radius_px_requested"] = float(radius_px)
        metrics["synthetic_motion_px_requested"] = float(motion_px)
        rows.append(metrics)
    if args.json:
        payload: dict[str, Any] | list[dict[str, Any]]
        payload = rows[0] if len(rows) == 1 else {"cases": rows}
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif len(rows) == 1:
        metrics = rows[0]
        print_summary(metrics)
        print(json.dumps(metrics, sort_keys=True))
    else:
        for row in rows:
            print(
                "case gaussians={gaussians} radius={synthetic_radius_px_requested:g} "
                "motion={synthetic_motion_px_requested:g} tile_frac={active_tiles_fraction_mean:.4f} "
                "pair_density={gaussian_tile_pair_density_mean:.6f} "
                "pair_adj_jaccard={gaussian_tile_pair_adjacent_jaccard_mean:.4f} "
                "pair_adj_retention={gaussian_tile_pair_adjacent_retention_mean:.4f}".format(**row)
            )
        print(json.dumps({"cases": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
