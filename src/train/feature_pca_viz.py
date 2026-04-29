"""PCA-based visualization for F-channel feature splat maps.

Projects an F-channel rasterized feature map onto its top-3 principal
components, min-max normalized per-channel to [0, 1], for W&B image/video
logging. Detached from autograd — visualization only.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def feature_pca_to_rgb(features: torch.Tensor, num_components: int = 3) -> torch.Tensor:
    """Project [..., F, H, W] feature maps to [..., 3, H, W] via PCA.

    Args:
        features: tensor with at least 3 dims; trailing dims are (F, H, W).
                  Common shapes: [T, F, H, W] (one clip) or [B, T, F, H, W].
        num_components: number of principal components (must be 3 for RGB output).

    Returns:
        Tensor of the same leading shape, with C=3 instead of F. Values in [0, 1].
        If features has fewer than 3 unique components after centering, the
        output channels are zero-padded so downstream image logging still works.
    """
    if num_components != 3:
        raise ValueError(f"feature_pca_to_rgb only supports num_components=3 (RGB), got {num_components}")
    if features.dim() < 3:
        raise ValueError(f"features must have at least 3 dims (..., F, H, W); got shape {tuple(features.shape)}")

    F = features.shape[-3]
    H = features.shape[-2]
    W = features.shape[-1]
    leading_shape = features.shape[:-3]

    # Flatten leading dims and spatial dims into a single sample axis: [N, F]
    x = features.reshape(-1, F, H * W).permute(0, 2, 1).reshape(-1, F)  # [N, F]
    x = x.detach().to(dtype=torch.float32)

    # Center; SVD on the centered matrix; take top-3 singular vectors as components
    mean = x.mean(dim=0, keepdim=True)  # [1, F]
    centered = x - mean
    # Use torch.linalg.svd; full_matrices=False for efficiency on F-tall matrices
    # On MPS, SVD may not be supported; fall back to CPU for the SVD step.
    svd_device = centered.device
    if svd_device.type == "mps":
        centered_for_svd = centered.cpu()
    else:
        centered_for_svd = centered
    _, _, vh = torch.linalg.svd(centered_for_svd, full_matrices=False)  # vh: [min(N,F), F]
    components = vh[:3].to(svd_device)  # [<=3, F]

    # Project: [N, F] @ [F, k] -> [N, k] where k <= 3
    projected = centered @ components.t()  # [N, k]
    k = projected.shape[-1]
    if k < 3:
        pad = torch.zeros(projected.shape[0], 3 - k, device=projected.device, dtype=projected.dtype)
        projected = torch.cat([projected, pad], dim=-1)

    # Per-channel min-max normalize to [0, 1] over all pixels in this batch
    pmin = projected.min(dim=0, keepdim=True).values  # [1, 3]
    pmax = projected.max(dim=0, keepdim=True).values
    span = (pmax - pmin).clamp_min(1.0e-8)
    normalized = (projected - pmin) / span  # [N, 3] in [0, 1]

    # Reshape back: [N, 3] -> [..., H, W, 3] -> [..., 3, H, W]
    out = normalized.reshape(*leading_shape, H, W, 3).movedim(-1, -3)
    return out.contiguous()
