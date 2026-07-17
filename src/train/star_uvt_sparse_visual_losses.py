from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from colorize import FeatureToColor
from star_uvt_sparse_visual_sampling import (
    SPARSE_VISUAL_COMPOSITIONS,
    SPARSE_VISUAL_LOSS_BASES,
    SPARSE_VISUAL_LOSS_VJP_MODES,
)


def _gather_sparse_visual_rgb_values(target_rgb_chunk: torch.Tensor, pixel_ids: torch.Tensor) -> torch.Tensor:
    if target_rgb_chunk.dim() != 4 or int(target_rgb_chunk.shape[1]) != 3:
        raise ValueError(f"target_rgb_chunk must have shape [T,3,H,W], got {tuple(target_rgb_chunk.shape)}")
    flat = target_rgb_chunk.permute(0, 2, 3, 1).reshape(-1, 3)
    return flat.index_select(0, pixel_ids.to(device=target_rgb_chunk.device, dtype=torch.int64))


def _compose_sparse_visual_rgb(
    feature_values: torch.Tensor,
    alpha_values: torch.Tensor,
    colorizer: FeatureToColor,
    *,
    target_values: torch.Tensor | None = None,
    composition: str = "black",
) -> torch.Tensor:
    if feature_values.dim() != 2:
        raise ValueError(f"feature_values must have shape [M,F], got {tuple(feature_values.shape)}")
    if alpha_values.dim() != 1 or int(alpha_values.shape[0]) != int(feature_values.shape[0]):
        raise ValueError("alpha_values must have shape [M] matching feature_values")
    if composition not in SPARSE_VISUAL_COMPOSITIONS:
        expected = ", ".join(sorted(SPARSE_VISUAL_COMPOSITIONS))
        raise ValueError(f"sparse_visual.composition must be one of: {expected}")
    splat_rgb = colorizer(feature_values.transpose(0, 1).unsqueeze(0).unsqueeze(-1))
    splat_rgb = splat_rgb.squeeze(0).squeeze(-1).transpose(0, 1)
    return _compose_sparse_visual_rgb_values(
        splat_rgb,
        alpha_values,
        target_values=target_values,
        composition=composition,
    )


def _compose_sparse_visual_rgb_values(
    splat_rgb: torch.Tensor,
    alpha_values: torch.Tensor,
    *,
    target_values: torch.Tensor | None = None,
    composition: str = "black",
) -> torch.Tensor:
    if splat_rgb.dim() != 2 or int(splat_rgb.shape[1]) != 3:
        raise ValueError(f"splat_rgb must have shape [M,3], got {tuple(splat_rgb.shape)}")
    if alpha_values.dim() != 1 or int(alpha_values.shape[0]) != int(splat_rgb.shape[0]):
        raise ValueError("alpha_values must have shape [M] matching splat_rgb")
    if composition not in SPARSE_VISUAL_COMPOSITIONS:
        expected = ", ".join(sorted(SPARSE_VISUAL_COMPOSITIONS))
        raise ValueError(f"sparse_visual.composition must be one of: {expected}")
    alpha = alpha_values.to(dtype=splat_rgb.dtype).unsqueeze(1)
    if composition == "black":
        return alpha * splat_rgb
    if target_values is None:
        raise ValueError("sparse_visual.composition=target_background requires target_values")
    if target_values.shape != splat_rgb.shape:
        raise ValueError(
            "sparse_visual.composition=target_background requires target_values "
            f"shape {tuple(splat_rgb.shape)}, got {tuple(target_values.shape)}"
        )
    target = target_values.to(device=splat_rgb.device, dtype=splat_rgb.dtype)
    return target + alpha * (splat_rgb - target)


def _sparse_visual_rgb_alpha_grads_from_composition(
    *,
    grad_pred: torch.Tensor,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    target_values: torch.Tensor | None,
    composition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_col = alpha.unsqueeze(1).to(dtype=grad_pred.dtype)
    if composition == "black":
        return grad_pred * alpha_col, (grad_pred * rgb).sum(dim=1)
    if composition != "target_background":
        expected = ", ".join(sorted(SPARSE_VISUAL_COMPOSITIONS))
        raise ValueError(f"sparse_visual.composition must be one of: {expected}")
    if target_values is None:
        raise ValueError("sparse_visual.composition=target_background requires target_values")
    target = target_values.to(device=rgb.device, dtype=rgb.dtype)
    return grad_pred * alpha_col, (grad_pred * (rgb - target)).sum(dim=1)


def _gelu_exact_grad(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0))) + values * torch.exp(-0.5 * values.square()) / math.sqrt(
        2.0 * math.pi
    )


def _gelu_fast_sigmoid_grad(values: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(1.702 * values)
    return sigmoid + 1.702 * values * sigmoid * (1.0 - sigmoid)


def _hidden64_vjp_options(vjp_mode: str) -> tuple[bool, str]:
    if vjp_mode in {"manual_hidden", "manual_hidden64"}:
        return True, "exact"
    if vjp_mode in {"manual_hidden_fastgelu", "manual_hidden64_fastgelu"}:
        return True, "fast_sigmoid"
    if vjp_mode in {"manual_hidden_star_only", "manual_hidden64_star_only"}:
        return False, "exact"
    if vjp_mode in {"manual_hidden_star_only_fastgelu", "manual_hidden64_star_only_fastgelu"}:
        return False, "fast_sigmoid"
    raise ValueError(f"unsupported manual hidden sparse visual VJP mode: {vjp_mode}")


def _gelu_grad_for_mode(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "exact":
        return _gelu_exact_grad(values)
    if mode == "fast_sigmoid":
        return _gelu_fast_sigmoid_grad(values)
    raise ValueError(f"unsupported hidden64 GELU grad mode: {mode}")


def _native_target_area_backward_mode(vjp_mode: str) -> str:
    if vjp_mode in {
        "native_hidden_target_area_colorizer_simdreduce_vec4_wt",
        "native_hidden64_target_area_colorizer_simdreduce_vec4_wt",
    }:
        return "target_area_colorizer_simdreduce_vec4_wt"
    if vjp_mode in {
        "native_hidden_target_area_colorizer_vec4_wt",
        "native_hidden64_target_area_colorizer_vec4_wt",
    }:
        return "target_area_colorizer_vec4_wt"
    if vjp_mode in {
        "native_hidden_target_area_star_only_vec4_wt",
        "native_hidden64_target_area_star_only_vec4_wt",
    }:
        return "target_area_star_only_vec4_wt"
    if vjp_mode in {
        "native_hidden_target_area_star_only",
        "native_hidden64_target_area_star_only",
    }:
        return "target_area_star_only"
    raise ValueError(f"unsupported native target-area sparse visual VJP mode: {vjp_mode}")


def _add_param_grad(param: torch.Tensor, grad: torch.Tensor) -> None:
    detached = grad.detach()
    if param.grad is None:
        param.grad = detached.clone()
    else:
        param.grad.add_(detached)


def _hidden64_colorizer_layers(
    colorizer: FeatureToColor,
) -> tuple[nn.Conv2d, nn.GELU, nn.Conv2d]:
    if not isinstance(colorizer, FeatureToColor):
        raise ValueError("manual_hidden64 sparse visual VJP requires FeatureToColor")
    if colorizer.hidden_dim is None:
        raise ValueError("manual_hidden64 sparse visual VJP requires colorize.hidden_dim")
    if colorizer.pre_norm is not None:
        raise ValueError("manual_hidden64 sparse visual VJP requires colorize.pre_norm=false")
    if colorizer.view_condition != "none":
        raise ValueError("manual_hidden64 sparse visual VJP requires colorize.view_condition='none'")
    if not isinstance(colorizer.net, nn.Sequential) or len(colorizer.net) != 3:
        raise ValueError("manual_hidden64 sparse visual VJP requires Conv2d -> GELU -> Conv2d colorizer")
    conv1, gelu, conv2 = colorizer.net
    if not isinstance(conv1, nn.Conv2d) or not isinstance(gelu, nn.GELU) or not isinstance(conv2, nn.Conv2d):
        raise ValueError("manual_hidden64 sparse visual VJP requires Conv2d -> GELU -> Conv2d colorizer")
    if conv1.kernel_size != (1, 1) or conv2.kernel_size != (1, 1):
        raise ValueError("manual_hidden64 sparse visual VJP requires 1x1 colorizer convolutions")
    if conv1.bias is None or conv2.bias is None:
        raise ValueError("manual_hidden64 sparse visual VJP requires colorizer biases")
    if gelu.approximate != "none":
        raise ValueError("manual_hidden64 sparse visual VJP currently supports exact GELU only")
    if colorizer.activation not in {"sigmoid", "identity"}:
        raise ValueError("manual_hidden64 sparse visual VJP requires sigmoid or identity output activation")
    return conv1, gelu, conv2


def _linear_colorizer_layer(colorizer: FeatureToColor) -> nn.Conv2d:
    if not isinstance(colorizer, FeatureToColor):
        raise ValueError("manual_linear sparse visual VJP requires FeatureToColor")
    if colorizer.hidden_dim is not None:
        raise ValueError("manual_linear sparse visual VJP requires colorize.hidden_dim=null")
    if colorizer.pre_norm is not None:
        raise ValueError("manual_linear sparse visual VJP requires colorize.pre_norm=false")
    if colorizer.view_condition != "none":
        raise ValueError("manual_linear sparse visual VJP requires colorize.view_condition='none'")
    if not isinstance(colorizer.net, nn.Conv2d):
        raise ValueError("manual_linear sparse visual VJP requires one Conv2d colorizer")
    if colorizer.net.kernel_size != (1, 1):
        raise ValueError("manual_linear sparse visual VJP requires a 1x1 colorizer convolution")
    if colorizer.net.bias is None:
        raise ValueError("manual_linear sparse visual VJP requires colorizer bias")
    if colorizer.activation not in {"sigmoid", "identity"}:
        raise ValueError("manual_linear sparse visual VJP requires sigmoid or identity output activation")
    return colorizer.net


def _sparse_visual_loss_and_grad_pred_values(
    pred_values: torch.Tensor,
    target_values: torch.Tensor | None,
    *,
    total_loss_elems: int,
    loss_weight: float,
    loss_basis: str,
    sample_grid_shape: tuple[int, int, int] | None,
    patch_shape: tuple[int, int],
    target_rgb_chunk: torch.Tensor | None,
    local_frame_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if loss_basis in {"patch_mean", "target_area_mean"}:
        if sample_grid_shape is None:
            raise ValueError(f"sparse_visual.loss_basis={loss_basis} requires sample_grid_shape")
        patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("sparse_visual.patch_shape must be positive")
        grid_h, grid_w = int(sample_grid_shape[1]), int(sample_grid_shape[2])
        pixels_per_local_frame = grid_h * patch_h * grid_w * patch_w
        if pixels_per_local_frame <= 0 or int(pred_values.shape[0]) % pixels_per_local_frame != 0:
            raise ValueError("patch_mean sparse visual values do not match sample_grid_shape and patch_shape")
        local_frames = int(pred_values.shape[0]) // pixels_per_local_frame
        basis_shape = (local_frames, grid_h, patch_h, grid_w, patch_w, 3)
        pred_for_loss = pred_values.reshape(basis_shape).mean(dim=(2, 4))
        if loss_basis == "patch_mean":
            if target_values is None:
                raise ValueError("sparse_visual.loss_basis=patch_mean requires target_values")
            target_for_loss = target_values.reshape(basis_shape).mean(dim=(2, 4))
        else:
            if target_rgb_chunk is None or local_frame_ids is None:
                raise ValueError("sparse_visual.loss_basis=target_area_mean requires target_rgb_chunk and local_frame_ids")
            target_frames = target_rgb_chunk.index_select(
                0,
                local_frame_ids.to(device=target_rgb_chunk.device, dtype=torch.int64),
            )
            target_for_loss = F.interpolate(
                target_frames,
                size=(grid_h, grid_w),
                mode="area",
            ).permute(0, 2, 3, 1).contiguous()
        diff = pred_for_loss - target_for_loss
        loss = diff.square().sum() / float(total_loss_elems)
        patch_area = patch_h * patch_w
        grad_basis = (float(loss_weight) * 2.0 / float(total_loss_elems)) * diff
        grad_pred = (
            grad_basis[:, :, None, :, None, :]
            .expand(local_frames, grid_h, patch_h, grid_w, patch_w, 3)
            .reshape_as(pred_values)
            / float(patch_area)
        )
        return loss, grad_pred.contiguous()
    if loss_basis == "pixel":
        if target_values is None:
            raise ValueError("sparse_visual.loss_basis=pixel requires target_values")
        diff = pred_values - target_values
        loss = diff.square().sum() / float(total_loss_elems)
        return loss, ((float(loss_weight) * 2.0 / float(total_loss_elems)) * diff).contiguous()
    expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_BASES))
    raise ValueError(f"sparse_visual.loss_basis must be one of: {expected}")


def _sparse_visual_target_area_cells(
    target_rgb_chunk: torch.Tensor,
    local_frame_ids: torch.Tensor,
    *,
    sample_grid_shape: tuple[int, int, int],
) -> torch.Tensor:
    grid_h, grid_w = int(sample_grid_shape[1]), int(sample_grid_shape[2])
    target_frames = target_rgb_chunk.index_select(
        0,
        local_frame_ids.to(device=target_rgb_chunk.device, dtype=torch.int64),
    )
    return (
        F.interpolate(target_frames, size=(grid_h, grid_w), mode="area")
        .permute(0, 2, 3, 1)
        .reshape(-1, 3)
        .contiguous()
    )


def _sparse_visual_target_area_cell_ids(
    local_frame_count: int,
    *,
    sample_grid_shape: tuple[int, int, int],
    patch_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if local_frame_count <= 0:
        return torch.empty((0,), device=device, dtype=torch.int32)
    grid_h, grid_w = int(sample_grid_shape[1]), int(sample_grid_shape[2])
    patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
    frame_offsets = torch.arange(local_frame_count, device=device, dtype=torch.int64)[:, None, None, None, None]
    grid_y = torch.arange(grid_h, device=device, dtype=torch.int64)[None, :, None, None, None]
    grid_x = torch.arange(grid_w, device=device, dtype=torch.int64)[None, None, None, :, None]
    local_cells = frame_offsets * (grid_h * grid_w) + grid_y * grid_w + grid_x
    return (
        local_cells.expand(local_frame_count, grid_h, patch_h, grid_w, patch_w)
        .reshape(-1)
        .to(dtype=torch.int32)
        .contiguous()
    )


def _manual_hidden64_sparse_visual_rgb_loss_and_grads(
    feature_values: torch.Tensor,
    alpha_values: torch.Tensor,
    target_values: torch.Tensor | None,
    colorizer: FeatureToColor,
    *,
    total_loss_elems: int,
    loss_weight: float,
    loss_basis: str,
    sample_grid_shape: tuple[int, int, int] | None,
    patch_shape: tuple[int, int],
    target_rgb_chunk: torch.Tensor | None,
    local_frame_ids: torch.Tensor | None,
    accumulate_colorizer_grads: bool = True,
    gelu_grad_mode: str = "exact",
    composition: str = "black",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conv1, _gelu, conv2 = _hidden64_colorizer_layers(colorizer)
    if feature_values.dim() != 2:
        raise ValueError(f"feature_values must have shape [M,F], got {tuple(feature_values.shape)}")
    if alpha_values.dim() != 1 or int(alpha_values.shape[0]) != int(feature_values.shape[0]):
        raise ValueError("alpha_values must have shape [M] matching feature_values")
    x = feature_values.detach()
    alpha = alpha_values.detach()
    w1 = conv1.weight[:, :, 0, 0]
    b1 = conv1.bias
    w2 = conv2.weight[:, :, 0, 0]
    b2 = conv2.bias
    hidden_pre = x.matmul(w1.t()) + b1
    hidden = F.gelu(hidden_pre)
    logits = hidden.matmul(w2.t()) + b2
    rgb = torch.sigmoid(logits) if colorizer.activation == "sigmoid" else logits
    pred_values = _compose_sparse_visual_rgb_values(
        rgb,
        alpha,
        target_values=target_values,
        composition=composition,
    )
    loss, grad_pred = _sparse_visual_loss_and_grad_pred_values(
        pred_values,
        target_values,
        total_loss_elems=total_loss_elems,
        loss_weight=loss_weight,
        loss_basis=loss_basis,
        sample_grid_shape=sample_grid_shape,
        patch_shape=patch_shape,
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=local_frame_ids,
    )
    grad_rgb, grad_alpha = _sparse_visual_rgb_alpha_grads_from_composition(
        grad_pred=grad_pred,
        rgb=rgb,
        alpha=alpha,
        target_values=target_values,
        composition=composition,
    )
    grad_logits = grad_rgb * (rgb * (1.0 - rgb)) if colorizer.activation == "sigmoid" else grad_rgb
    grad_hidden = grad_logits.matmul(w2)
    grad_hidden_pre = grad_hidden * _gelu_grad_for_mode(hidden_pre, gelu_grad_mode)
    grad_feature = grad_hidden_pre.matmul(w1)
    if accumulate_colorizer_grads:
        grad_w2 = grad_logits.t().matmul(hidden)
        grad_b2 = grad_logits.sum(dim=0)
        grad_w1 = grad_hidden_pre.t().matmul(x)
        grad_b1 = grad_hidden_pre.sum(dim=0)
        _add_param_grad(conv1.weight, grad_w1.view_as(conv1.weight))
        _add_param_grad(conv1.bias, grad_b1)
        _add_param_grad(conv2.weight, grad_w2.view_as(conv2.weight))
        _add_param_grad(conv2.bias, grad_b2)
    return loss.detach(), grad_feature.contiguous(), grad_alpha.contiguous()


def _manual_linear_sparse_visual_rgb_loss_and_grads(
    feature_values: torch.Tensor,
    alpha_values: torch.Tensor,
    target_values: torch.Tensor | None,
    colorizer: FeatureToColor,
    *,
    total_loss_elems: int,
    loss_weight: float,
    loss_basis: str,
    sample_grid_shape: tuple[int, int, int] | None,
    patch_shape: tuple[int, int],
    target_rgb_chunk: torch.Tensor | None,
    local_frame_ids: torch.Tensor | None,
    composition: str = "black",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conv = _linear_colorizer_layer(colorizer)
    if feature_values.dim() != 2:
        raise ValueError(f"feature_values must have shape [M,F], got {tuple(feature_values.shape)}")
    if alpha_values.dim() != 1 or int(alpha_values.shape[0]) != int(feature_values.shape[0]):
        raise ValueError("alpha_values must have shape [M] matching feature_values")
    x = feature_values.detach()
    alpha = alpha_values.detach()
    weight = conv.weight[:, :, 0, 0]
    bias = conv.bias
    logits = x.matmul(weight.t()) + bias
    rgb = torch.sigmoid(logits) if colorizer.activation == "sigmoid" else logits
    pred_values = _compose_sparse_visual_rgb_values(
        rgb,
        alpha,
        target_values=target_values,
        composition=composition,
    )
    loss, grad_pred = _sparse_visual_loss_and_grad_pred_values(
        pred_values,
        target_values,
        total_loss_elems=total_loss_elems,
        loss_weight=loss_weight,
        loss_basis=loss_basis,
        sample_grid_shape=sample_grid_shape,
        patch_shape=patch_shape,
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=local_frame_ids,
    )
    grad_rgb, grad_alpha = _sparse_visual_rgb_alpha_grads_from_composition(
        grad_pred=grad_pred,
        rgb=rgb,
        alpha=alpha,
        target_values=target_values,
        composition=composition,
    )
    grad_logits = grad_rgb * (rgb * (1.0 - rgb)) if colorizer.activation == "sigmoid" else grad_rgb
    grad_weight = grad_logits.t().matmul(x)
    grad_bias = grad_logits.sum(dim=0)
    grad_feature = grad_logits.matmul(weight)
    _add_param_grad(conv.weight, grad_weight.view_as(conv.weight))
    _add_param_grad(conv.bias, grad_bias)
    return loss.detach(), grad_feature.contiguous(), grad_alpha.contiguous()


def _sparse_visual_rgb_loss_and_grads(
    feature_values: torch.Tensor,
    alpha_values: torch.Tensor,
    target_values: torch.Tensor | None,
    colorizer: FeatureToColor,
    *,
    total_loss_elems: int,
    loss_weight: float,
    loss_basis: str = "pixel",
    sample_grid_shape: tuple[int, int, int] | None = None,
    patch_shape: tuple[int, int] = (1, 1),
    target_rgb_chunk: torch.Tensor | None = None,
    local_frame_ids: torch.Tensor | None = None,
    vjp_mode: str = "autograd",
    composition: str = "black",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if total_loss_elems <= 0:
        raise ValueError("total_loss_elems must be positive")
    if loss_basis not in SPARSE_VISUAL_LOSS_BASES:
        expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_BASES))
        raise ValueError(f"sparse_visual.loss_basis must be one of: {expected}")
    if vjp_mode not in SPARSE_VISUAL_LOSS_VJP_MODES:
        expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_VJP_MODES))
        raise ValueError(f"sparse_visual.loss_vjp_mode must be one of: {expected}")
    if vjp_mode.startswith("manual_hidden"):
        accumulate_colorizer_grads, gelu_grad_mode = _hidden64_vjp_options(vjp_mode)
        return _manual_hidden64_sparse_visual_rgb_loss_and_grads(
            feature_values,
            alpha_values,
            target_values,
            colorizer,
            total_loss_elems=total_loss_elems,
            loss_weight=loss_weight,
            loss_basis=loss_basis,
            sample_grid_shape=sample_grid_shape,
            patch_shape=patch_shape,
            target_rgb_chunk=target_rgb_chunk,
            local_frame_ids=local_frame_ids,
            accumulate_colorizer_grads=accumulate_colorizer_grads,
            gelu_grad_mode=gelu_grad_mode,
            composition=composition,
        )
    if vjp_mode == "manual_linear":
        return _manual_linear_sparse_visual_rgb_loss_and_grads(
            feature_values,
            alpha_values,
            target_values,
            colorizer,
            total_loss_elems=total_loss_elems,
            loss_weight=loss_weight,
            loss_basis=loss_basis,
            sample_grid_shape=sample_grid_shape,
            patch_shape=patch_shape,
            target_rgb_chunk=target_rgb_chunk,
            local_frame_ids=local_frame_ids,
            composition=composition,
        )
    local_feature_values = feature_values.detach().requires_grad_(True)
    local_alpha_values = alpha_values.detach().requires_grad_(True)
    pred_values = _compose_sparse_visual_rgb(
        local_feature_values,
        local_alpha_values,
        colorizer,
        target_values=target_values,
        composition=composition,
    )
    if loss_basis in {"patch_mean", "target_area_mean"}:
        if sample_grid_shape is None:
            raise ValueError(f"sparse_visual.loss_basis={loss_basis} requires sample_grid_shape")
        patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("sparse_visual.patch_shape must be positive")
        grid_h, grid_w = int(sample_grid_shape[1]), int(sample_grid_shape[2])
        pixels_per_local_frame = grid_h * patch_h * grid_w * patch_w
        if pixels_per_local_frame <= 0 or int(pred_values.shape[0]) % pixels_per_local_frame != 0:
            raise ValueError("patch_mean sparse visual values do not match sample_grid_shape and patch_shape")
        local_frames = int(pred_values.shape[0]) // pixels_per_local_frame
        basis_shape = (local_frames, grid_h, patch_h, grid_w, patch_w, 3)
        pred_for_loss = pred_values.reshape(basis_shape).mean(dim=(2, 4))
        if loss_basis == "patch_mean":
            if target_values is None:
                raise ValueError("sparse_visual.loss_basis=patch_mean requires target_values")
            target_for_loss = target_values.reshape(basis_shape).mean(dim=(2, 4))
        else:
            if target_rgb_chunk is None or local_frame_ids is None:
                raise ValueError("sparse_visual.loss_basis=target_area_mean requires target_rgb_chunk and local_frame_ids")
            target_frames = target_rgb_chunk.index_select(
                0,
                local_frame_ids.to(device=target_rgb_chunk.device, dtype=torch.int64),
            )
            target_for_loss = F.interpolate(
                target_frames,
                size=(grid_h, grid_w),
                mode="area",
            ).permute(0, 2, 3, 1).contiguous()
    else:
        if target_values is None:
            raise ValueError("sparse_visual.loss_basis=pixel requires target_values")
        pred_for_loss = pred_values
        target_for_loss = target_values
    loss = (pred_for_loss - target_for_loss).square().sum() / float(total_loss_elems)
    weighted_loss = float(loss_weight) * loss
    weighted_loss.backward()
    if local_feature_values.grad is None or local_alpha_values.grad is None:
        raise RuntimeError("sparse visual RGB loss did not produce local feature/alpha gradients")
    return loss.detach(), local_feature_values.grad.contiguous(), local_alpha_values.grad.contiguous()


def _sparse_visual_alpha_loss_and_grad(
    alpha_values: torch.Tensor,
    *,
    target: float,
    total_loss_elems: int,
    loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_loss_elems <= 0:
        raise ValueError("total_loss_elems must be positive")
    if alpha_values.dim() != 1:
        raise ValueError(f"alpha_values must have shape [M], got {tuple(alpha_values.shape)}")
    diff = alpha_values.detach() - float(target)
    loss = diff.square().sum() / float(total_loss_elems)
    grad = (float(loss_weight) * 2.0 / float(total_loss_elems)) * diff
    return loss.detach(), grad.contiguous()


def _sparse_visual_black_hole_loss_and_grad(
    alpha_values: torch.Tensor,
    target_values: torch.Tensor | None,
    *,
    total_loss_elems: int,
    loss_weight: float,
    loss_basis: str,
    sample_grid_shape: tuple[int, int, int] | None = None,
    patch_shape: tuple[int, int] = (1, 1),
    target_rgb_chunk: torch.Tensor | None = None,
    local_frame_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if total_loss_elems <= 0:
        raise ValueError("total_loss_elems must be positive")
    if alpha_values.dim() != 1:
        raise ValueError(f"alpha_values must have shape [M], got {tuple(alpha_values.shape)}")
    if loss_basis not in SPARSE_VISUAL_LOSS_BASES:
        expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_BASES))
        raise ValueError(f"sparse_visual.loss_basis must be one of: {expected}")
    if loss_basis in {"patch_mean", "target_area_mean"}:
        if sample_grid_shape is None:
            raise ValueError(f"sparse_visual.loss_basis={loss_basis} requires sample_grid_shape")
        patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("sparse_visual.patch_shape must be positive")
        grid_h, grid_w = int(sample_grid_shape[1]), int(sample_grid_shape[2])
        pixels_per_local_frame = grid_h * patch_h * grid_w * patch_w
        if pixels_per_local_frame <= 0 or int(alpha_values.shape[0]) % pixels_per_local_frame != 0:
            raise ValueError("black-hole sparse visual values do not match sample_grid_shape and patch_shape")
        local_frames = int(alpha_values.shape[0]) // pixels_per_local_frame
        alpha_basis = alpha_values.detach().reshape(local_frames, grid_h, patch_h, grid_w, patch_w)
        alpha_for_loss = alpha_basis.mean(dim=(2, 4))
        if loss_basis == "patch_mean":
            if target_values is None:
                raise ValueError("sparse_visual.loss_basis=patch_mean requires target_values")
            target_basis = target_values.detach().reshape(local_frames, grid_h, patch_h, grid_w, patch_w, 3)
            target_for_loss = target_basis.mean(dim=(2, 4))
        else:
            if target_rgb_chunk is None or local_frame_ids is None:
                raise ValueError("sparse_visual.loss_basis=target_area_mean requires target_rgb_chunk and local_frame_ids")
            target_frames = target_rgb_chunk.index_select(
                0,
                local_frame_ids.to(device=target_rgb_chunk.device, dtype=torch.int64),
            )
            target_for_loss = F.interpolate(
                target_frames,
                size=(grid_h, grid_w),
                mode="area",
            ).permute(0, 2, 3, 1).contiguous()
        target_energy = target_for_loss.detach().square().mean(dim=-1)
        empty = 1.0 - alpha_for_loss
        loss = (empty.square() * target_energy).sum() / float(total_loss_elems)
        grad_basis = (-2.0 * float(loss_weight) / float(total_loss_elems)) * empty * target_energy
        patch_area = patch_h * patch_w
        grad_alpha = (
            grad_basis[:, :, None, :, None]
            .expand(local_frames, grid_h, patch_h, grid_w, patch_w)
            .reshape_as(alpha_values)
            / float(patch_area)
        )
        return loss.detach(), grad_alpha.contiguous()
    if loss_basis == "pixel":
        if target_values is None:
            raise ValueError("sparse_visual.loss_basis=pixel requires target_values")
        target_energy = target_values.detach().square().mean(dim=-1)
        empty = 1.0 - alpha_values.detach()
        loss = (empty.square() * target_energy).sum() / float(total_loss_elems)
        grad = (-2.0 * float(loss_weight) / float(total_loss_elems)) * empty * target_energy
        return loss.detach(), grad.contiguous()
    expected = ", ".join(sorted(SPARSE_VISUAL_LOSS_BASES))
    raise ValueError(f"sparse_visual.loss_basis must be one of: {expected}")


__all__ = [
    "_add_param_grad",
    "_compose_sparse_visual_rgb",
    "_compose_sparse_visual_rgb_values",
    "_gather_sparse_visual_rgb_values",
    "_gelu_fast_sigmoid_grad",
    "_gelu_grad_for_mode",
    "_hidden64_colorizer_layers",
    "_hidden64_vjp_options",
    "_linear_colorizer_layer",
    "_native_target_area_backward_mode",
    "_sparse_visual_alpha_loss_and_grad",
    "_sparse_visual_black_hole_loss_and_grad",
    "_sparse_visual_loss_and_grad_pred_values",
    "_sparse_visual_rgb_loss_and_grads",
    "_sparse_visual_target_area_cell_ids",
    "_sparse_visual_target_area_cells",
]
