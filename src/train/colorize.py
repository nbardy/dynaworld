"""Per-pixel feature-to-RGB colorization for feature splatting.

Applied after rasterization: takes an F-channel feature map produced by the
rasterizer and produces a 3-channel RGB image.

Architecture knobs (all configurable from the train config's `colorize` section):
    hidden_dim: None -> single Conv2d(F, 3); int -> Conv2d(F, h) -> GELU -> Conv2d(h, 3).
    activation: 'sigmoid' (default; clamps to (0, 1)) or 'identity'.
    pre_norm: when True, LayerNorm over the F channel-dim per pixel before the conv.
        Insulates the colorize MLP from upstream feature magnitude.
    weight_init: 'kaiming' (PyTorch default) or 'orthogonal' (variance-preserving).
    weight_init_gain: scalar; for orthogonal it's the row-norm of the matrix.
        For sigmoid output to engage its non-linear region the post-conv std should be
        ~1.5+, which (with LayerNorm pre_norm giving unit-variance input) needs gain ~2,
        and (without pre_norm with raw raster output ~0.2 std) needs gain ~7-8.
        For the parity case (F=3 + hidden_dim=None + kaiming + gain=1.0 + no pre_norm)
        the conv is identity-initialized so legacy RGB behaviour is preserved.
    view_condition: 'none' (default), 'camera_center_ray', or 'pixel_ray'. When
        enabled, append 3 world-ray direction channels after optional feature
        pre_norm and before the 1x1 color MLP.

Forward accepts [B, F, H, W] or [B, T, F, H, W]; T is folded into B internally.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


VIEW_CONDITION_ALIASES = {
    "none": "none",
    "off": "none",
    "false": "none",
    "no_ray": "none",
    "no_rays": "none",
    "camera_center_ray": "camera_center_ray",
    "center_ray": "camera_center_ray",
    "single_ray": "camera_center_ray",
    "camera_ray": "camera_center_ray",
    "pixel_ray": "pixel_ray",
    "pixel_rays": "pixel_ray",
    "per_pixel_ray": "pixel_ray",
    "per_pixel_rays": "pixel_ray",
}


def normalize_view_condition(value: str | None) -> str:
    mode = "none" if value is None else str(value).lower()
    if mode not in VIEW_CONDITION_ALIASES:
        expected = ", ".join(sorted({"none", "camera_center_ray", "pixel_ray"}))
        raise ValueError(f"colorize.view_condition must be one of {expected}; got {value!r}")
    return VIEW_CONDITION_ALIASES[mode]


class FeatureToColor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        activation: str = "sigmoid",
        pre_norm: bool = False,
        weight_init: str = "kaiming",
        weight_init_gain: float = 1.0,
        view_condition: str | None = "none",
    ):
        super().__init__()
        if activation not in {"sigmoid", "identity"}:
            raise ValueError(f"colorize activation must be 'sigmoid' or 'identity', got {activation!r}")
        if weight_init not in {"kaiming", "orthogonal"}:
            raise ValueError(f"colorize weight_init must be 'kaiming' or 'orthogonal', got {weight_init!r}")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)
        self.activation = activation
        self.weight_init = weight_init
        self.weight_init_gain = float(weight_init_gain)
        self.view_condition = normalize_view_condition(view_condition)
        self.input_dim = self.feature_dim + (3 if self.view_condition != "none" else 0)

        # Optional per-pixel LayerNorm over the F channel-dim. nn.LayerNorm normalizes
        # the LAST dim of its input, so we permute the channel dim to the last position
        # in forward (and back).
        self.pre_norm = nn.LayerNorm(self.feature_dim) if pre_norm else None

        if self.hidden_dim is None:
            conv = nn.Conv2d(self.input_dim, 3, kernel_size=1, bias=True)
            self._init_conv(conv, last_layer=True)
            if self._is_legacy_parity_case():
                self._identity_init(conv)
            self.net = conv
        else:
            conv1 = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1, bias=True)
            conv2 = nn.Conv2d(self.hidden_dim, 3, kernel_size=1, bias=True)
            # First layer is followed by GELU; use sqrt(2) gain to compensate for the
            # activation cutting half the variance (the standard ReLU/GELU rule).
            # Final layer uses the user-supplied gain to control sigmoid engagement.
            self._init_conv(conv1, last_layer=False)
            self._init_conv(conv2, last_layer=True)
            self.net = nn.Sequential(conv1, nn.GELU(), conv2)

    def _is_legacy_parity_case(self) -> bool:
        return (
            self.feature_dim == 3
            and self.hidden_dim is None
            and self.weight_init == "kaiming"
            and self.weight_init_gain == 1.0
            and self.pre_norm is None
            and self.view_condition == "none"
        )

    @staticmethod
    def _identity_init(conv: nn.Conv2d) -> None:
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[0, 0, 0, 0] = 1.0
            conv.weight[1, 1, 0, 0] = 1.0
            conv.weight[2, 2, 0, 0] = 1.0
            assert conv.bias is not None
            conv.bias.zero_()

    def _init_conv(self, conv: nn.Conv2d, *, last_layer: bool) -> None:
        if self.weight_init == "orthogonal":
            gain = self.weight_init_gain if last_layer else math.sqrt(2.0)
            with torch.no_grad():
                nn.init.orthogonal_(conv.weight, gain=gain)
                assert conv.bias is not None
                conv.bias.zero_()
        elif self.weight_init == "kaiming":
            # Default Kaiming was applied during nn.Conv2d construction.
            # Only scale if the user asked for non-trivial gain on the LAST layer.
            if last_layer and self.weight_init_gain != 1.0:
                with torch.no_grad():
                    conv.weight.data.mul_(self.weight_init_gain)
                    assert conv.bias is not None
                    conv.bias.zero_()

    def _prepare_channels(self, features: torch.Tensor, view_dirs: torch.Tensor | None) -> torch.Tensor:
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} feature channels, got {features.shape[1]}.")
        x = features
        if self.pre_norm is not None:
            x = self.pre_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.view_condition == "none":
            if view_dirs is not None:
                raise ValueError("view_dirs was provided but colorize.view_condition is 'none'.")
            return x
        if view_dirs is None:
            raise ValueError(f"colorize.view_condition={self.view_condition!r} requires view_dirs.")
        if view_dirs.dim() != 4:
            raise ValueError(f"view_dirs must have shape [N, 3, H, W], got {tuple(view_dirs.shape)}.")
        if view_dirs.shape[0] != x.shape[0] or view_dirs.shape[1] != 3 or view_dirs.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                "view_dirs must match feature batch/spatial shape with 3 channels; "
                f"got features={tuple(x.shape)}, view_dirs={tuple(view_dirs.shape)}."
            )
        view_dirs = view_dirs.to(device=x.device, dtype=x.dtype)
        return torch.cat([x, view_dirs], dim=1)

    def _run(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Helper named `_run` (not `_apply`) to avoid shadowing nn.Module._apply.
        if features.dim() == 5:
            B, T, F, H, W = features.shape
            x = features.reshape(B * T, F, H, W)
            view = None
            if view_dirs is not None:
                if view_dirs.dim() != 5:
                    raise ValueError(f"view_dirs must have shape [B, T, 3, H, W], got {tuple(view_dirs.shape)}.")
                if view_dirs.shape[:2] != (B, T) or view_dirs.shape[2] != 3 or view_dirs.shape[-2:] != (H, W):
                    raise ValueError(
                        "view_dirs must match feature batch/time/spatial shape with 3 channels; "
                        f"got features={tuple(features.shape)}, view_dirs={tuple(view_dirs.shape)}."
                    )
                view = view_dirs.reshape(B * T, 3, H, W)
            x = self._prepare_channels(x, view)
            logits = self.net(x)
            rgb = torch.sigmoid(logits) if self.activation == "sigmoid" else logits
            return rgb.reshape(B, T, 3, H, W), logits.reshape(B, T, 3, H, W)
        x = features
        x = self._prepare_channels(x, view_dirs)
        logits = self.net(x)
        rgb = torch.sigmoid(logits) if self.activation == "sigmoid" else logits
        return rgb, logits

    def forward(self, features: torch.Tensor, view_dirs: torch.Tensor | None = None) -> torch.Tensor:
        rgb, _ = self._run(features, view_dirs)
        return rgb

    def forward_with_logits(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (rgb_output, pre_sigmoid_logits) for diagnostic use."""
        return self._run(features, view_dirs)
