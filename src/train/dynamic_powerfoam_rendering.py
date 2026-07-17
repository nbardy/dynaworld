from __future__ import annotations

from typing import Any

import torch

from colorize import FeatureToColor


def sample_background(
    render_cfg: dict[str, Any],
    *,
    phase: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    mode = str(render_cfg["train_background_mode"] if phase == "train" else render_cfg["eval_background_mode"])
    if mode == "none":
        return None
    if mode == "black":
        rgb = torch.zeros(batch_size, 3, 1, 1, device=device, dtype=dtype)
    elif mode == "white":
        rgb = torch.ones(batch_size, 3, 1, 1, device=device, dtype=dtype)
    elif mode == "fixed_rgb":
        rgb = torch.tensor(render_cfg["background"], device=device, dtype=dtype).view(1, 3, 1, 1)
        rgb = rgb.expand(batch_size, -1, -1, -1).contiguous()
    elif mode == "random_rgb":
        low = float(render_cfg["random_background_min"])
        high = float(render_cfg["random_background_max"])
        rgb = low + (high - low) * torch.rand(batch_size, 3, 1, 1, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown background mode {mode!r}")
    return rgb


def render_features_to_rgb(
    features: torch.Tensor,
    alpha: torch.Tensor,
    colorizer: FeatureToColor | None,
    background: torch.Tensor | None,
    *,
    normalize_features_by_alpha: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    color_features = features
    if normalize_features_by_alpha and (colorizer is not None or background is not None):
        color_features = features / alpha.unsqueeze(1).clamp_min(float(eps))
    rgb = color_features if colorizer is None else colorizer(color_features)
    if background is None:
        return rgb
    return alpha.unsqueeze(1).to(device=rgb.device, dtype=rgb.dtype) * rgb + (1.0 - alpha.unsqueeze(1)) * background


@torch.no_grad()
def render_all(
    model: Any,
    frame_count: int,
    batch_size: int,
    cfg: dict[str, Any],
    colorizer: FeatureToColor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    renders = []
    features_out = []
    alphas = []
    device = next(model.parameters()).device
    for start in range(0, frame_count, batch_size):
        indices = torch.arange(start, min(start + batch_size, frame_count), device=device)
        features, alpha = model(indices)
        background = sample_background(
            cfg["render"],
            phase="eval",
            batch_size=int(features.shape[0]),
            device=features.device,
            dtype=features.dtype,
        )
        rendered = render_features_to_rgb(
            features,
            alpha,
            colorizer,
            background,
            normalize_features_by_alpha=bool(cfg["render"]["normalize_features_by_alpha"]),
            eps=float(cfg["render"]["eps"]),
        )
        renders.append(rendered.clamp(0.0, 1.0).detach().cpu())
        features_out.append(features.detach().cpu())
        alphas.append(alpha.detach().cpu())
    return torch.cat(renders, dim=0), torch.cat(features_out, dim=0), torch.cat(alphas, dim=0)


def per_frame_reconstruction_metrics(renders: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    diff = renders.float() - targets.float()
    mse = diff.square().flatten(1).mean(dim=1)
    l1 = diff.abs().flatten(1).mean(dim=1)
    signal = targets.float().square().flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
    snr = 10.0 * torch.log10((signal / mse.clamp_min(1.0e-12)).clamp_min(1.0e-12))
    rows = []
    for frame_index in range(int(renders.shape[0])):
        rows.append(
            {
                "frame_index": frame_index,
                "mse": float(mse[frame_index].cpu()),
                "l1": float(l1[frame_index].cpu()),
                "psnr": float(psnr[frame_index].cpu()),
                "snr": float(snr[frame_index].cpu()),
                "signal_power": float(signal[frame_index].cpu()),
            }
        )
    return {
        "summary": {
            "frame_psnr_mean": float(psnr.mean().cpu()),
            "frame_psnr_min": float(psnr.min().cpu()),
            "frame_snr_mean": float(snr.mean().cpu()),
            "frame_snr_min": float(snr.min().cpu()),
            "frame_l1_mean": float(l1.mean().cpu()),
            "frame_l1_max": float(l1.max().cpu()),
            "frame_mse_mean": float(mse.mean().cpu()),
            "frame_mse_max": float(mse.max().cpu()),
        },
        "per_frame": rows,
    }


def temporal_alpha_metrics(alphas: torch.Tensor) -> dict[str, float]:
    if alphas.shape[0] < 2:
        return {
            "eval_mean_temporal_alpha_delta": 0.0,
            "eval_max_temporal_alpha_delta": 0.0,
            "eval_mean_temporal_support_delta": 0.0,
        }
    delta = (alphas[1:] - alphas[:-1]).abs()
    support = alphas > 1.0e-4
    support_delta = (support[1:].float() - support[:-1].float()).abs()
    return {
        "eval_mean_temporal_alpha_delta": float(delta.mean().cpu()),
        "eval_max_temporal_alpha_delta": float(delta.max().cpu()),
        "eval_mean_temporal_support_delta": float(support_delta.mean().cpu()),
    }
