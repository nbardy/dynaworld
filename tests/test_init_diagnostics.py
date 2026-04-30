from __future__ import annotations

import torch

from gs_models.blocks import GaussianParameterHeads
from init_diagnostics import decoded_gaussian_init_diagnostics


GAUSSIAN_FIELDS = ("xyz", "scales", "quats", "opacities", "rgbs")


def _decoded_mapping(heads: GaussianParameterHeads, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return dict(zip(GAUSSIAN_FIELDS, heads(tokens), strict=True))


def _metrics_for(
    heads: GaussianParameterHeads,
    tokens: torch.Tensor,
    *,
    gaussians_per_token: int,
) -> dict[str, float]:
    return decoded_gaussian_init_diagnostics(
        _decoded_mapping(heads, tokens),
        token_count=tokens.shape[1],
        gaussians_per_token=gaussians_per_token,
        valid_ranges={"rgbs": (0.0, 1.0)},
    )


def test_rgb_uniform_bias_init_covers_full_color_range_without_token_spread() -> None:
    torch.manual_seed(17)
    gaussians_per_token = 256
    heads = GaussianParameterHeads(
        feat_dim=16,
        gaussians_per_token=gaussians_per_token,
        head_hidden_layers=0,
        head_output_init_std=0.0,
        rotation_init="identity",
        rgb_init="uniform",
        rgb_init_min=0.0,
        rgb_init_max=1.0,
    )
    tokens = torch.zeros(1, 4, 16)

    assert torch.isfinite(heads.rgb_head[-1].bias).all()
    metrics = _metrics_for(heads, tokens, gaussians_per_token=gaussians_per_token)

    assert metrics["InitDiag/RGB/Coverage"] > 0.98
    assert metrics["InitDiag/RGB/Entropy01"] > 0.92
    assert metrics["InitDiag/Spread/RGB/WithinTokenRangeMean"] > 0.95
    assert metrics["InitDiag/Spread/RGB/SameSplitCrossTokenStdMean"] < 1.0e-6


def test_token_and_head_scale_increase_same_split_cross_token_spread() -> None:
    torch.manual_seed(23)
    gaussians_per_token = 16
    heads = GaussianParameterHeads(
        feat_dim=32,
        gaussians_per_token=gaussians_per_token,
        head_hidden_layers=0,
        head_output_init_std=0.12,
        position_init_extent_coverage=0.0,
        rotation_init="identity",
    )
    base_tokens = torch.randn(1, 64, 32)

    low = _metrics_for(heads, base_tokens * 0.3, gaussians_per_token=gaussians_per_token)
    high = _metrics_for(heads, base_tokens * 0.8, gaussians_per_token=gaussians_per_token)

    assert high["InitDiag/Spread/RGB/SameSplitCrossTokenStdMean"] > (
        low["InitDiag/Spread/RGB/SameSplitCrossTokenStdMean"] * 1.8
    )
    assert high["InitDiag/Spread/XYZ/SameSplitCrossTokenStdMean"] > (
        low["InitDiag/Spread/XYZ/SameSplitCrossTokenStdMean"] * 1.8
    )
