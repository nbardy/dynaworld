from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_broadcast_param(
    value: float | torch.Tensor,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.view(1, 1, 1, 1)
    if tensor.numel() == batch_size:
        return tensor.reshape(batch_size, 1, 1, 1)
    if tensor.shape == (batch_size, 1, 1, 1):
        return tensor
    raise ValueError(
        f"{name} must be scalar, [B], or [B,1,1,1]; got shape {tuple(tensor.shape)}"
    )


def depth_aware_defocus_blur(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    alpha: torch.Tensor | None = None,
    inv_focus_depth: float | torch.Tensor,
    blur_strength: float | torch.Tensor | None = None,
    log_q: float | torch.Tensor | None = None,
    max_radius: int = 8,
    depth_edge_sigma: float | torch.Tensor | None = None,
    detach_depth: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Apply a differentiable image-space depth-of-field blur.

    This is a Torch reference prototype, not a physically exact lens model. It
    uses a fixed unfold window of radius ``max_radius`` and continuously varies
    Gaussian weights from a per-pixel circle-of-confusion radius:

    ``radius_px = aperture_scale * abs(1 / depth - inv_focus_depth)``.

    Set ``detach_depth=True`` when the renderer depth path should only guide the
    postprocess and should not receive the dense windowed backward pass.
    """

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [B,3,H,W]; got {tuple(rgb.shape)}")
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must have shape [B,1,H,W]; got {tuple(depth.shape)}")
    if depth.shape[0] != rgb.shape[0] or depth.shape[-2:] != rgb.shape[-2:]:
        raise ValueError("depth must share batch and spatial dimensions with rgb")
    if alpha is not None and (
        alpha.ndim != 4
        or alpha.shape[1] != 1
        or alpha.shape[0] != rgb.shape[0]
        or alpha.shape[-2:] != rgb.shape[-2:]
    ):
        raise ValueError(f"alpha must have shape [B,1,H,W]; got {tuple(alpha.shape)}")
    if (blur_strength is None) == (log_q is None):
        raise ValueError("pass exactly one of blur_strength or log_q")
    if max_radius < 0:
        raise ValueError("max_radius must be non-negative")
    if max_radius == 0:
        return rgb

    batch_size, _, height, width = rgb.shape
    device = rgb.device
    work_dtype = rgb.dtype if rgb.is_floating_point() else torch.float32
    rgb_work = rgb.to(dtype=work_dtype)
    depth_work = depth.to(device=device, dtype=work_dtype)
    if detach_depth:
        depth_work = depth_work.detach()

    inv_depth = depth_work.clamp_min(eps).reciprocal()
    inv_focus = _as_broadcast_param(
        inv_focus_depth,
        name="inv_focus_depth",
        batch_size=batch_size,
        device=device,
        dtype=work_dtype,
    )
    if blur_strength is None:
        aperture_scale = _as_broadcast_param(
            log_q,
            name="log_q",
            batch_size=batch_size,
            device=device,
            dtype=work_dtype,
        ).exp()
    else:
        aperture_scale = _as_broadcast_param(
            blur_strength,
            name="blur_strength",
            batch_size=batch_size,
            device=device,
            dtype=work_dtype,
        ).clamp_min(0.0)

    radius = (aperture_scale * (inv_depth - inv_focus).abs()).clamp(max=float(max_radius))
    kernel_size = max_radius * 2 + 1
    sample_count = kernel_size * kernel_size
    pad_mode = "reflect" if max_radius < min(height, width) else "replicate"

    padded_rgb = F.pad(rgb_work, (max_radius, max_radius, max_radius, max_radius), mode=pad_mode)
    rgb_samples = F.unfold(padded_rgb, kernel_size=kernel_size).view(
        batch_size, 3, sample_count, height, width
    )

    alpha_samples: torch.Tensor | None = None
    if alpha is not None:
        alpha_work = alpha.to(device=device, dtype=work_dtype).clamp(0.0, 1.0)
        padded_alpha = F.pad(
            alpha_work,
            (max_radius, max_radius, max_radius, max_radius),
            mode=pad_mode,
        )
        alpha_samples = F.unfold(padded_alpha, kernel_size=kernel_size).view(
            batch_size, 1, sample_count, height, width
        )

    offsets = torch.arange(-max_radius, max_radius + 1, device=device, dtype=work_dtype)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_distance_sq = (xx.square() + yy.square()).reshape(1, 1, sample_count, 1, 1)

    sigma = (radius * 0.5).clamp_min(0.5 * eps).unsqueeze(2)
    weights = torch.exp(-0.5 * offset_distance_sq / sigma.square())

    if depth_edge_sigma is not None:
        edge_sigma = _as_broadcast_param(
            depth_edge_sigma,
            name="depth_edge_sigma",
            batch_size=batch_size,
            device=device,
            dtype=work_dtype,
        )
        if torch.any(edge_sigma <= 0):
            raise ValueError("depth_edge_sigma must be positive when provided")
        padded_inv_depth = F.pad(
            inv_depth,
            (max_radius, max_radius, max_radius, max_radius),
            mode=pad_mode,
        )
        inv_depth_samples = F.unfold(padded_inv_depth, kernel_size=kernel_size).view(
            batch_size, 1, sample_count, height, width
        )
        edge_delta = inv_depth_samples - inv_depth.unsqueeze(2)
        weights = weights * torch.exp(-0.5 * edge_delta.square() / edge_sigma.unsqueeze(2).square())

    if alpha_samples is not None:
        weights = weights * alpha_samples

    weights_sum = weights.sum(dim=2, keepdim=True).clamp_min(eps)
    weights = weights / weights_sum
    return (rgb_samples * weights).sum(dim=2)


def _self_test() -> None:
    torch.manual_seed(0)
    rgb = torch.rand(2, 3, 8, 9, requires_grad=True)
    depth = (torch.rand(2, 1, 8, 9) + 0.2).requires_grad_()
    inv_focus_depth = torch.tensor(1.2, requires_grad=True)
    log_q = torch.tensor(0.25, requires_grad=True)
    out = depth_aware_defocus_blur(
        rgb,
        depth,
        inv_focus_depth=inv_focus_depth,
        log_q=log_q,
        max_radius=2,
        depth_edge_sigma=0.5,
        detach_depth=True,
    )
    assert out.shape == rgb.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert inv_focus_depth.grad is not None and torch.isfinite(inv_focus_depth.grad).all()
    assert log_q.grad is not None and torch.isfinite(log_q.grad).all()
    assert depth.grad is None


if __name__ == "__main__":
    _self_test()
