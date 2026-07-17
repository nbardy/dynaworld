from __future__ import annotations

import torch
import torch.nn as nn

from external_paths import ensure_module_path, third_party_path

FAST_MAC_V12A_DIR = (
    third_party_path("fast-mac-gsplat")
    / "variants"
    / "v12a_fused_colorize_l1_no_norm"
)


def _ensure_v12a_on_path() -> None:
    package_name = "torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm"
    ensure_module_path(
        package_name,
        FAST_MAC_V12A_DIR,
        missing_message=f"fast-mac v12a directory not found: {FAST_MAC_V12A_DIR}",
    )


def _v12a_fused_no_norm_l1_grad(
    features_nhwf: torch.Tensor,
    alpha_nhw: torch.Tensor,
    target_rgb: torch.Tensor,
    background_rgb: torch.Tensor,
    weight_3f: torch.Tensor,
    bias_3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_v12a_on_path()
    from torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm import fused_no_norm_l1_grad

    return fused_no_norm_l1_grad(
        features_nhwf,
        alpha_nhw,
        target_rgb,
        background_rgb,
        weight_3f,
        bias_3,
    )


def _require_supported_colorizer(colorizer: nn.Module) -> nn.Conv2d:
    pre_norm = getattr(colorizer, "pre_norm", None)
    hidden_dim = getattr(colorizer, "hidden_dim", None)
    activation = getattr(colorizer, "activation", None)
    view_condition = getattr(colorizer, "view_condition", None)
    net = getattr(colorizer, "net", None)
    if pre_norm is not None:
        raise ValueError("v12a fused L1 supports only colorize.pre_norm=false.")
    if hidden_dim is not None:
        raise ValueError("v12a fused L1 supports only colorize.hidden_dim=null.")
    if activation != "sigmoid":
        raise ValueError("v12a fused L1 supports only colorize.activation='sigmoid'.")
    if view_condition != "none":
        raise ValueError("v12a fused L1 supports only colorize.view_condition='none'.")
    if not isinstance(net, nn.Conv2d) or net.kernel_size != (1, 1) or net.out_channels != 3:
        raise ValueError("v12a fused L1 requires a single Conv2d(F, 3, kernel_size=1) colorizer.")
    if net.bias is None:
        raise ValueError("v12a fused L1 requires a colorizer bias.")
    return net


def _expand_rgb(value: torch.Tensor, *, name: str, batch: int, height: int, width: int) -> torch.Tensor:
    if value.dim() != 4 or int(value.shape[1]) != 3:
        raise ValueError(f"{name} must have shape [N,3,H,W] or broadcastable [1,3,1,1], got {tuple(value.shape)}")
    try:
        return torch.broadcast_to(value, (batch, 3, height, width)).contiguous()
    except RuntimeError as exc:
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} cannot broadcast to {(batch, 3, height, width)}"
        ) from exc


class _V12AFusedNoNormL1Mean(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features_nchw: torch.Tensor,
        alpha_nhw: torch.Tensor,
        target_rgb: torch.Tensor,
        background_rgb: torch.Tensor,
        weight_3f11: torch.Tensor,
        bias_3: torch.Tensor,
    ) -> torch.Tensor:
        batch, feature_dim, height, width = features_nchw.shape
        features_nhwf = features_nchw.permute(0, 2, 3, 1).contiguous()
        weight_3f = weight_3f11.reshape(3, feature_dim).contiguous()
        (
            loss_per_image,
            grad_features_nhwf,
            grad_alpha,
            grad_weight_3f,
            grad_bias,
        ) = _v12a_fused_no_norm_l1_grad(
            features_nhwf,
            alpha_nhw.contiguous(),
            target_rgb.contiguous(),
            background_rgb.contiguous(),
            weight_3f,
            bias_3.contiguous(),
        )
        ctx.save_for_backward(
            grad_features_nhwf.permute(0, 3, 1, 2).contiguous(),
            grad_alpha.contiguous(),
            grad_weight_3f.reshape_as(weight_3f11).contiguous(),
            grad_bias.contiguous(),
        )
        return loss_per_image.mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_features, grad_alpha, grad_weight, grad_bias = ctx.saved_tensors
        scale = grad_output.to(dtype=grad_features.dtype, device=grad_features.device)
        return (
            grad_features * scale,
            grad_alpha * scale,
            None,
            None,
            grad_weight * scale,
            grad_bias * scale,
        )


def fused_no_norm_l1_mean_loss(
    *,
    features_nchw: torch.Tensor,
    alpha_nhw: torch.Tensor | None,
    target_rgb: torch.Tensor,
    background_rgb: torch.Tensor | None,
    colorizer: nn.Module,
) -> torch.Tensor:
    if features_nchw.dim() != 4:
        raise ValueError(f"features_nchw must have shape [N,F,H,W], got {tuple(features_nchw.shape)}")
    if features_nchw.device.type != "mps":
        raise ValueError("v12a fused L1 currently requires MPS tensors.")
    if alpha_nhw is None:
        raise ValueError("v12a fused L1 requires raster alpha.")
    if background_rgb is None:
        raise ValueError("v12a fused L1 requires an explicit RGB background.")
    batch, _feature_dim, height, width = features_nchw.shape
    if tuple(alpha_nhw.shape) != (batch, height, width):
        raise ValueError(f"alpha_nhw must have shape {(batch, height, width)}, got {tuple(alpha_nhw.shape)}")
    target = _expand_rgb(target_rgb, name="target_rgb", batch=batch, height=height, width=width)
    background = _expand_rgb(background_rgb, name="background_rgb", batch=batch, height=height, width=width)
    conv = _require_supported_colorizer(colorizer)
    return _V12AFusedNoNormL1Mean.apply(
        features_nchw.contiguous(),
        alpha_nhw.contiguous(),
        target,
        background,
        conv.weight,
        conv.bias,
    )
