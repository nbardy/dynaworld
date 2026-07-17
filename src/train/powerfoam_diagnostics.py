from __future__ import annotations

import torch


def powerfoam_parameter_delta_metrics(
    *,
    points: torch.Tensor,
    initial_points: torch.Tensor,
    radii: torch.Tensor | None = None,
    initial_radii: torch.Tensor | None = None,
    densities: torch.Tensor | None = None,
    initial_densities: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    initial_features: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    initial_normals: torch.Tensor | None = None,
    texel_sites: torch.Tensor | None = None,
    initial_texel_sites: torch.Tensor | None = None,
    include_cell_count: bool = False,
) -> dict[str, float]:
    center_offset = points - initial_points.to(points.device)
    center_delta = torch.linalg.vector_norm(center_offset, dim=-1)
    metrics = {}
    if include_cell_count:
        metrics["state_cell_count"] = float(points.shape[1])
    metrics.update(
        {
            "state_mean_center_delta": float(center_delta.mean().cpu()),
            "state_p95_center_delta": float(center_delta.flatten().quantile(0.95).cpu()),
            "state_max_center_delta": float(center_delta.max().cpu()),
            "state_mean_xy_delta": float(torch.linalg.vector_norm(center_offset[..., :2], dim=-1).mean().cpu()),
            "state_mean_z_delta": float(center_offset[..., 2].abs().mean().cpu()),
        }
    )
    if radii is not None and initial_radii is not None:
        metrics["state_mean_radius_delta"] = float((radii - initial_radii.to(radii.device)).abs().mean().cpu())
    if densities is not None and initial_densities is not None:
        metrics["state_mean_density_delta"] = float(
            (densities - initial_densities.to(densities.device)).abs().mean().cpu()
        )
    if features is not None and initial_features is not None:
        metrics["state_mean_feature_delta"] = float(
            (features - initial_features.to(features.device)).abs().mean().cpu()
        )
    if normals is not None and initial_normals is not None:
        metrics["state_mean_normal_delta"] = float(
            (normals - initial_normals.to(normals.device)).norm(dim=-1).mean().cpu()
        )
    if texel_sites is not None and initial_texel_sites is not None:
        metrics["state_mean_texel_site_delta"] = float(
            (texel_sites - initial_texel_sites.to(texel_sites.device)).norm(dim=-1).mean().cpu()
        )
    return metrics


__all__ = ["powerfoam_parameter_delta_metrics"]
