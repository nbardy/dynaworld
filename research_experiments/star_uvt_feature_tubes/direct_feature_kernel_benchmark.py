from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    from .report_artifacts import split_csv_ints, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import split_csv_ints, write_report_json
from train_devices import sync_torch_device
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    brute_force_render_uvt_feature_tubes,
    direct_atomic_feature_backward,
    direct_atomic_feature_backward_cached_bins,
    direct_hidden_sigmoid_mse_backward,
    direct_linear_sigmoid_mse_backward,
    direct_logit_handoff_backward,
    linear_sigmoid_mse_logit_handoff_prep,
    render_uvt_feature_tubes,
)

CACHED_MODE_MAP = {
    "direct_atomic_cached_bins": "direct_atomic",
    "gradcache_cached_bins": "gradcache",
    "gradcache_reduce_feature_grad_cached_bins": "gradcache_reduce_feature_grad",
    "gradcache_reduce_feature_grad_vec4_cached_bins": "gradcache_reduce_feature_grad_vec4",
}
TWO_PASS_FEATURE_GRAD_MODE = "gradcache_two_pass_feature_grad"
TWO_PASS_FEATURE_GRAD_MODES = {
    TWO_PASS_FEATURE_GRAD_MODE: "gradcache_feature_grad_only",
    "gradcache_two_pass_feature_grad_reduce": "gradcache_feature_grad_only_reduce",
    "gradcache_two_pass_feature_grad_reduce_vec4": "gradcache_feature_grad_only_reduce_vec4",
}
LOGIT_HANDOFF_MODES = {
    "logit_handoff",
    "logit_handoff_reduce",
    "logit_handoff_reduce_vec4",
}
NATIVE_LOGIT_PREP_MODE_MAP = {
    "logit_handoff_native_prep": "logit_handoff",
    "logit_handoff_reduce_native_prep": "logit_handoff_reduce",
    "logit_handoff_reduce_vec4_native_prep": "logit_handoff_reduce_vec4",
}
HIDDEN_SIGMOID_MSE_MODES = {
    "hidden_sigmoid_mse_star_only",
    "hidden_sigmoid_mse_star_only_reduce_vec4",
}


def _sync() -> None:
    sync_torch_device(torch.device("mps"))


def _tiny_scene(feature_dim: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, UVTRenderConfig]:
    config = UVTRenderConfig(height=8, width=8, frames=2, tile_t=2, tile_capacity=128, alpha_threshold=1.0 / 255.0)
    ma = torch.tensor([[3.5, 3.5, -0.2], [4.5, 3.8, 0.1]], dtype=torch.float32)
    q = torch.tensor(
        [[0.30, 0.0, 0.0, 0.30, 0.0, 0.40], [0.24, 0.0, 0.03, 0.28, -0.02, 0.35]],
        dtype=torch.float32,
    )
    depth0 = torch.tensor([0.8, 1.2], dtype=torch.float32)
    depth_beta = torch.zeros((2, 3), dtype=torch.float32)
    opacity = torch.tensor([0.55, 0.45], dtype=torch.float32)
    feature = torch.linspace(-0.4, 0.5, 2 * feature_dim, dtype=torch.float32).view(2, feature_dim)
    return ma, q, depth0, depth_beta, opacity, feature, config


def _max_err(got: Tensor, expected: Tensor) -> float:
    return float((got.detach().cpu() - expected.detach().cpu()).abs().max().item())


def run_tiny_parity(feature_dim: int, *, backward_mode: str, hidden_dim: int = 16) -> dict[str, Any]:
    ma, q, depth0, depth_beta, opacity, feature, config = _tiny_scene(feature_dim)
    ref_feature, ref_alpha = brute_force_render_uvt_feature_tubes(ma, q, depth0, depth_beta, opacity, feature, config)
    mps_inputs = [tensor.to("mps").contiguous() for tensor in (ma, q, depth0, depth_beta, opacity, feature)]

    _sync()
    t0 = time.perf_counter()
    use_cached_bins = backward_mode in CACHED_MODE_MAP
    native_logit_prep = backward_mode in NATIVE_LOGIT_PREP_MODE_MAP
    kernel_backward_mode = CACHED_MODE_MAP.get(backward_mode, NATIVE_LOGIT_PREP_MODE_MAP.get(backward_mode, backward_mode))
    two_pass_feature_grad = backward_mode in TWO_PASS_FEATURE_GRAD_MODES
    kernel_call_mode = "gradcache" if two_pass_feature_grad else kernel_backward_mode
    metal = render_uvt_feature_tubes(*mps_inputs, config, return_bins=use_cached_bins)
    _sync()
    forward_ms = (time.perf_counter() - t0) * 1000.0

    fused_first3 = kernel_call_mode == "fused_first3_sigmoid_mse"
    linear_sigmoid_mse = kernel_call_mode in {"linear_sigmoid_mse", "linear_sigmoid_mse_skip_colorizer_grad"}
    logit_handoff = kernel_call_mode in LOGIT_HANDOFF_MODES
    hidden_sigmoid_mse = kernel_call_mode in HIDDEN_SIGMOID_MSE_MODES
    colorizer_grad_skipped = kernel_call_mode == "linear_sigmoid_mse_skip_colorizer_grad"
    grad_feature = torch.linspace(-0.25, 0.35, ref_feature.numel(), dtype=torch.float32).view_as(ref_feature)
    grad_alpha = torch.linspace(0.1, -0.2, ref_alpha.numel(), dtype=torch.float32).view_as(ref_alpha)
    target_rgb = torch.linspace(0.05, 0.85, ref_alpha.numel() * 3, dtype=torch.float32).view(
        ref_alpha.shape[0], 3, ref_alpha.shape[1], ref_alpha.shape[2]
    )
    target_padded = torch.zeros_like(ref_feature)
    target_padded[:, :3] = target_rgb
    ma_ref = ma.clone().requires_grad_(True)
    q_ref = q.clone().requires_grad_(True)
    opacity_ref = opacity.clone().requires_grad_(True)
    feature_ref = feature.clone().requires_grad_(True)
    color_weight = torch.linspace(-0.20, 0.25, 3 * feature_dim, dtype=torch.float32).view(3, feature_dim)
    color_bias = torch.tensor([-0.10, 0.05, 0.20], dtype=torch.float32)
    color_weight_ref = color_weight.clone().requires_grad_(True)
    color_bias_ref = color_bias.clone().requires_grad_(True)
    hidden_weight = torch.linspace(-0.18, 0.22, hidden_dim * feature_dim, dtype=torch.float32).view(
        hidden_dim, feature_dim
    )
    hidden_bias = torch.linspace(-0.08, 0.08, hidden_dim, dtype=torch.float32)
    output_weight = torch.linspace(-0.15, 0.20, 3 * hidden_dim, dtype=torch.float32).view(3, hidden_dim)
    output_bias = torch.tensor([-0.07, 0.03, 0.11], dtype=torch.float32)
    ref_feature_2, ref_alpha_2 = brute_force_render_uvt_feature_tubes(
        ma_ref, q_ref, depth0, depth_beta, opacity_ref, feature_ref, config
    )
    grad_logits_input = None
    grad_alpha_handoff_input = None
    if linear_sigmoid_mse or logit_handoff:
        logits = torch.einsum("cf,tfhw->tchw", color_weight_ref, ref_feature_2) + color_bias_ref.view(1, 3, 1, 1)
        sigmoid_logits = torch.sigmoid(logits)
        rgb = ref_alpha_2.unsqueeze(1) * sigmoid_logits
        loss = (rgb - target_rgb).square().mean()
        with torch.no_grad():
            grad_rgb = (2.0 / float(target_rgb.numel())) * (rgb.detach() - target_rgb)
            grad_logits_input = grad_rgb * ref_alpha_2.detach().unsqueeze(1) * sigmoid_logits.detach() * (1.0 - sigmoid_logits.detach())
            grad_alpha_handoff_input = (grad_rgb * sigmoid_logits.detach()).sum(dim=1)
        backward_feature_input = grad_feature
        backward_alpha_input = grad_alpha
    elif hidden_sigmoid_mse:
        hidden = torch.nn.functional.gelu(
            torch.einsum("hf,tfyw->thyw", hidden_weight, ref_feature_2) + hidden_bias.view(1, hidden_dim, 1, 1)
        )
        logits = torch.einsum("ch,thyw->tcyw", output_weight, hidden) + output_bias.view(1, 3, 1, 1)
        rgb = ref_alpha_2.unsqueeze(1) * torch.sigmoid(logits)
        loss = (rgb - target_rgb).square().mean()
        backward_feature_input = grad_feature
        backward_alpha_input = grad_alpha
    elif fused_first3:
        rgb = ref_alpha_2.unsqueeze(1) * torch.sigmoid(ref_feature_2[:, :3])
        loss = (rgb - target_rgb).square().mean()
        backward_feature_input = target_padded
        backward_alpha_input = torch.zeros_like(ref_alpha)
    else:
        loss = (ref_feature_2 * grad_feature).sum() + (ref_alpha_2 * grad_alpha).sum()
        backward_feature_input = grad_feature
        backward_alpha_input = grad_alpha
    loss.backward()

    _sync()
    t1 = time.perf_counter()
    if linear_sigmoid_mse:
        linear_result = direct_linear_sigmoid_mse_backward(
            *mps_inputs,
            target_rgb.to("mps"),
            color_weight.to("mps"),
            color_bias.to("mps"),
            config,
            compute_colorizer_grad=not colorizer_grad_skipped,
        )
        grad_ma = linear_result.grad_ma
        grad_q = linear_result.grad_q_uvt
        grad_opacity = linear_result.grad_opacity
        grad_feature_out = linear_result.grad_feature
        tile_unstable = linear_result.tile_unstable
    elif logit_handoff:
        assert grad_logits_input is not None and grad_alpha_handoff_input is not None
        if native_logit_prep:
            prep_result = linear_sigmoid_mse_logit_handoff_prep(
                metal.feature_image,
                metal.alpha,
                target_rgb.to("mps"),
                color_weight.to("mps"),
                color_bias.to("mps"),
                config,
            )
            grad_logits_for_handoff = prep_result.grad_logits_thw3
            grad_alpha_for_handoff = prep_result.grad_alpha
            grad_logits_layout = "thw3"
        else:
            prep_result = None
            grad_logits_for_handoff = grad_logits_input.to("mps")
            grad_alpha_for_handoff = grad_alpha_handoff_input.to("mps")
            grad_logits_layout = "tchw"
        logit_result = direct_logit_handoff_backward(
            *mps_inputs,
            grad_logits_for_handoff,
            grad_alpha_for_handoff,
            color_weight.to("mps"),
            config,
            backward_mode=kernel_call_mode,
            grad_logits_layout=grad_logits_layout,
        )
        grad_ma = logit_result.grad_ma
        grad_q = logit_result.grad_q_uvt
        grad_opacity = logit_result.grad_opacity
        grad_feature_out = logit_result.grad_feature
        tile_unstable = logit_result.tile_unstable
    elif hidden_sigmoid_mse:
        hidden_result = direct_hidden_sigmoid_mse_backward(
            *mps_inputs,
            target_rgb.to("mps"),
            hidden_weight.to("mps"),
            hidden_bias.to("mps"),
            output_weight.to("mps"),
            output_bias.to("mps"),
            config,
            backward_mode=kernel_call_mode,
        )
        grad_ma = hidden_result.grad_ma
        grad_q = hidden_result.grad_q_uvt
        grad_opacity = hidden_result.grad_opacity
        grad_feature_out = hidden_result.grad_feature
        tile_unstable = hidden_result.tile_unstable
    else:
        if two_pass_feature_grad:
            geom_grads = direct_atomic_feature_backward(
                *mps_inputs,
                backward_feature_input.to("mps"),
                backward_alpha_input.to("mps"),
                config,
                backward_mode="gradcache_skip_feature_grad",
            )
            feature_grads = direct_atomic_feature_backward(
                *mps_inputs,
                backward_feature_input.to("mps"),
                backward_alpha_input.to("mps"),
                config,
                backward_mode=TWO_PASS_FEATURE_GRAD_MODES[backward_mode],
            )
            grad_ma, grad_q, grad_opacity = geom_grads[:3]
            grad_feature_out = feature_grads[3]
            tile_unstable = geom_grads[-1]
        elif use_cached_bins:
            if metal.tile_tube_ids is None or metal.tile_depths is None:
                raise RuntimeError("cached-bin benchmark render did not return bins")
            grad_ma, grad_q, grad_opacity, grad_feature_out, tile_unstable = direct_atomic_feature_backward_cached_bins(
                *mps_inputs,
                backward_feature_input.to("mps"),
                backward_alpha_input.to("mps"),
                metal.tile_counts,
                metal.tile_tube_ids,
                metal.tile_depths,
                metal.tile_unstable,
                config,
                backward_mode=kernel_backward_mode,
            )
        else:
            grad_ma, grad_q, grad_opacity, grad_feature_out, tile_unstable = direct_atomic_feature_backward(
                *mps_inputs,
                backward_feature_input.to("mps"),
                backward_alpha_input.to("mps"),
                config,
                backward_mode=kernel_backward_mode,
            )
    _sync()
    backward_ms = (time.perf_counter() - t1) * 1000.0

    backward_errors = {
        "ma": _max_err(grad_ma, ma_ref.grad),
        "q": _max_err(grad_q, q_ref.grad),
        "opacity": _max_err(grad_opacity, opacity_ref.grad),
        "feature": _max_err(grad_feature_out, feature_ref.grad),
    }
    if linear_sigmoid_mse:
        if colorizer_grad_skipped:
            backward_errors.update(
                {
                    "color_weight": _max_err(linear_result.grad_color_weight, torch.zeros_like(color_weight_ref)),
                    "color_bias": _max_err(linear_result.grad_color_bias, torch.zeros_like(color_bias_ref)),
                }
            )
        else:
            backward_errors.update(
                {
                    "color_weight": _max_err(linear_result.grad_color_weight, color_weight_ref.grad),
                    "color_bias": _max_err(linear_result.grad_color_bias, color_bias_ref.grad),
                }
            )
    if logit_handoff and native_logit_prep:
        assert prep_result is not None and grad_logits_input is not None and grad_alpha_handoff_input is not None
        backward_errors.update(
            {
                "prep_grad_logits": _max_err(prep_result.grad_logits_thw3, grad_logits_input.permute(0, 2, 3, 1)),
                "prep_grad_alpha": _max_err(prep_result.grad_alpha, grad_alpha_handoff_input),
            }
        )
    feature_grad_skipped = "skip_feature_grad" in backward_mode
    feature_grad_only = "feature_grad_only" in backward_mode
    parity_errors = {
        key: value
        for key, value in backward_errors.items()
        if (
            (key != "feature" or not feature_grad_skipped)
            and (not feature_grad_only or key == "feature")
            and (key not in {"color_weight", "color_bias"} or not colorizer_grad_skipped)
        )
    }
    return {
        "feature_dim": feature_dim,
        "backward_mode": backward_mode,
        "kernel_backward_mode": kernel_backward_mode,
        "cached_bins": use_cached_bins,
        "two_pass_feature_grad": two_pass_feature_grad,
        "two_pass_feature_mode": TWO_PASS_FEATURE_GRAD_MODES.get(backward_mode),
        "feature_grad_skipped": feature_grad_skipped,
        "feature_grad_only": feature_grad_only,
        "colorizer_grad_skipped": colorizer_grad_skipped,
        "fused_first3_sigmoid_mse": fused_first3,
        "linear_sigmoid_mse": linear_sigmoid_mse,
        "logit_handoff": logit_handoff,
        "native_logit_prep": native_logit_prep,
        "hidden_sigmoid_mse": hidden_sigmoid_mse,
        "hidden_dim": hidden_dim if hidden_sigmoid_mse else None,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "forward_feature_max_abs_error": _max_err(metal.feature_image, ref_feature),
        "forward_alpha_max_abs_error": _max_err(metal.alpha, ref_alpha),
        "backward_max_abs_errors": backward_errors,
        "tile_overflow_sum": int(metal.tile_overflow.sum().cpu().item()),
        "tile_unstable_sum": int(tile_unstable.sum().cpu().item()),
        "pass": _max_err(metal.feature_image, ref_feature) <= 1.0e-5
        and _max_err(metal.alpha, ref_alpha) <= 1.0e-5
        and max(parity_errors.values()) <= 1.0e-4,
    }


def _random_timing_scene(
    *,
    frames: int,
    height: int,
    width: int,
    tubes: int,
    feature_dim: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, UVTRenderConfig]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    center_u = torch.rand((tubes,), generator=generator) * float(width)
    center_v = torch.rand((tubes,), generator=generator) * float(height)
    center_t = (torch.rand((tubes,), generator=generator) - 0.5) * float(frames)
    ma = torch.stack((center_u, center_v, center_t), dim=-1).to(dtype=torch.float32)
    precision_uv = torch.full((tubes,), 0.18, dtype=torch.float32)
    precision_t = torch.full((tubes,), 0.12, dtype=torch.float32)
    q = torch.stack(
        (
            precision_uv,
            torch.zeros_like(precision_uv),
            torch.zeros_like(precision_uv),
            precision_uv,
            torch.zeros_like(precision_uv),
            precision_t,
        ),
        dim=-1,
    )
    depth0 = torch.linspace(0.5, 1.5, tubes, dtype=torch.float32)
    depth_beta = torch.zeros((tubes, 3), dtype=torch.float32)
    opacity = torch.full((tubes,), 0.35, dtype=torch.float32)
    feature = torch.randn((tubes, feature_dim), generator=generator, dtype=torch.float32) * 0.1
    config = UVTRenderConfig(height=height, width=width, frames=frames, tile_t=2, tile_capacity=128)
    return ma, q, depth0, depth_beta, opacity, feature, config


def run_timing_case(
    *,
    frames: int,
    size: int,
    tubes: int,
    feature_dim: int,
    seed: int,
    backward_mode: str,
    warmup: int,
    repeat: int,
    hidden_dim: int,
) -> dict[str, Any]:
    cpu_inputs = _random_timing_scene(
        frames=frames,
        height=size,
        width=size,
        tubes=tubes,
        feature_dim=feature_dim,
        seed=seed,
    )
    ma, q, depth0, depth_beta, opacity, feature, config = cpu_inputs
    mps_inputs = [tensor.to("mps").contiguous() for tensor in (ma, q, depth0, depth_beta, opacity, feature)]
    generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
    target_rgb = torch.rand((frames, 3, size, size), generator=generator, dtype=torch.float32).to("mps").contiguous()
    color_weight = (torch.randn((3, feature_dim), generator=generator, dtype=torch.float32) * 0.1).to("mps").contiguous()
    color_bias = (torch.randn((3,), generator=generator, dtype=torch.float32) * 0.1).to("mps").contiguous()
    hidden_weight = (torch.randn((hidden_dim, feature_dim), generator=generator, dtype=torch.float32) * 0.1).to(
        "mps"
    ).contiguous()
    hidden_bias = (torch.randn((hidden_dim,), generator=generator, dtype=torch.float32) * 0.1).to("mps").contiguous()
    output_weight = (torch.randn((3, hidden_dim), generator=generator, dtype=torch.float32) * 0.1).to(
        "mps"
    ).contiguous()
    output_bias = (torch.randn((3,), generator=generator, dtype=torch.float32) * 0.1).to("mps").contiguous()
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    forward_samples: list[float] = []
    handoff_prep_samples: list[float] = []
    backward_samples: list[float] = []
    metal = None
    grads = None
    use_cached_bins = backward_mode in CACHED_MODE_MAP
    native_logit_prep = backward_mode in NATIVE_LOGIT_PREP_MODE_MAP
    kernel_backward_mode = CACHED_MODE_MAP.get(backward_mode, NATIVE_LOGIT_PREP_MODE_MAP.get(backward_mode, backward_mode))
    two_pass_feature_grad = backward_mode in TWO_PASS_FEATURE_GRAD_MODES
    kernel_call_mode = "gradcache" if two_pass_feature_grad else kernel_backward_mode
    for iteration in range(warmup + repeat):
        _sync()
        t0 = time.perf_counter()
        metal = render_uvt_feature_tubes(*mps_inputs, config, return_bins=use_cached_bins)
        _sync()
        forward_ms = (time.perf_counter() - t0) * 1000.0
        grad_feature = torch.randn_like(metal.feature_image)
        grad_alpha = torch.randn_like(metal.alpha)
        if kernel_call_mode == "fused_first3_sigmoid_mse":
            # The fused prototype interprets channels 0..2 as RGB targets and
            # computes the colorize/compose/MSE VJP inside the Metal backward.
            grad_feature[:, :3] = torch.sigmoid(metal.feature_image[:, :3]).detach()
            grad_alpha.zero_()
        handoff_prep_ms = 0.0
        if kernel_call_mode in LOGIT_HANDOFF_MODES:
            _sync()
            t_prep = time.perf_counter()
            if native_logit_prep:
                prep_result = linear_sigmoid_mse_logit_handoff_prep(
                    metal.feature_image,
                    metal.alpha,
                    target_rgb,
                    color_weight,
                    color_bias,
                    config,
                )
                grad_logits = prep_result.grad_logits_thw3
                grad_alpha_handoff = prep_result.grad_alpha
                grad_logits_layout = "thw3"
            else:
                logits = torch.einsum("cf,tfhw->tchw", color_weight, metal.feature_image) + color_bias.view(1, 3, 1, 1)
                sigmoid_logits = torch.sigmoid(logits)
                rgb = metal.alpha.unsqueeze(1) * sigmoid_logits
                grad_rgb = (2.0 / float(target_rgb.numel())) * (rgb - target_rgb)
                grad_logits = (grad_rgb * metal.alpha.unsqueeze(1) * sigmoid_logits * (1.0 - sigmoid_logits)).contiguous()
                grad_alpha_handoff = (grad_rgb * sigmoid_logits).sum(dim=1).contiguous()
                grad_logits_layout = "tchw"
            _sync()
            handoff_prep_ms = (time.perf_counter() - t_prep) * 1000.0
        _sync()
        t1 = time.perf_counter()
        if kernel_call_mode in LOGIT_HANDOFF_MODES:
            grads = direct_logit_handoff_backward(
                *mps_inputs,
                grad_logits,
                grad_alpha_handoff,
                color_weight,
                config,
                backward_mode=kernel_call_mode,
                grad_logits_layout=grad_logits_layout,
            )
            tile_unstable = grads.tile_unstable
            finite_grad_tensors = (grads.grad_ma, grads.grad_q_uvt, grads.grad_opacity, grads.grad_feature)
        elif kernel_call_mode in {"linear_sigmoid_mse", "linear_sigmoid_mse_skip_colorizer_grad"}:
            colorizer_grad_skipped = kernel_call_mode == "linear_sigmoid_mse_skip_colorizer_grad"
            grads = direct_linear_sigmoid_mse_backward(
                *mps_inputs,
                target_rgb,
                color_weight,
                color_bias,
                config,
                compute_colorizer_grad=not colorizer_grad_skipped,
            )
            tile_unstable = grads.tile_unstable
            finite_grad_tensors = (
                grads.grad_ma,
                grads.grad_q_uvt,
                grads.grad_opacity,
                grads.grad_feature,
            )
            if not colorizer_grad_skipped:
                finite_grad_tensors = finite_grad_tensors + (grads.grad_color_weight, grads.grad_color_bias)
        elif kernel_call_mode in HIDDEN_SIGMOID_MSE_MODES:
            grads = direct_hidden_sigmoid_mse_backward(
                *mps_inputs,
                target_rgb,
                hidden_weight,
                hidden_bias,
                output_weight,
                output_bias,
                config,
                backward_mode=kernel_call_mode,
            )
            tile_unstable = grads.tile_unstable
            finite_grad_tensors = (grads.grad_ma, grads.grad_q_uvt, grads.grad_opacity, grads.grad_feature)
        else:
            colorizer_grad_skipped = False
            if two_pass_feature_grad:
                geom_grads = direct_atomic_feature_backward(
                    *mps_inputs,
                    grad_feature,
                    grad_alpha,
                    config,
                    backward_mode="gradcache_skip_feature_grad",
                )
                feature_grads = direct_atomic_feature_backward(
                    *mps_inputs,
                    grad_feature,
                    grad_alpha,
                    config,
                    backward_mode=TWO_PASS_FEATURE_GRAD_MODES[backward_mode],
                )
                grads = (geom_grads[0], geom_grads[1], geom_grads[2], feature_grads[3], geom_grads[-1])
            elif use_cached_bins:
                if metal.tile_tube_ids is None or metal.tile_depths is None:
                    raise RuntimeError("cached-bin benchmark render did not return bins")
                grads = direct_atomic_feature_backward_cached_bins(
                    *mps_inputs,
                    grad_feature,
                    grad_alpha,
                    metal.tile_counts,
                    metal.tile_tube_ids,
                    metal.tile_depths,
                    metal.tile_unstable,
                    config,
                    backward_mode=kernel_backward_mode,
                )
            else:
                grads = direct_atomic_feature_backward(
                    *mps_inputs,
                    grad_feature,
                    grad_alpha,
                    config,
                    backward_mode=kernel_backward_mode,
                )
            tile_unstable = grads[-1]
            finite_grad_tensors = grads[:4]
        _sync()
        backward_ms = (time.perf_counter() - t1) * 1000.0
        if iteration >= warmup:
            forward_samples.append(forward_ms)
            handoff_prep_samples.append(handoff_prep_ms)
            backward_samples.append(backward_ms)
    assert metal is not None and grads is not None
    forward_ms = sum(forward_samples) / float(len(forward_samples))
    handoff_prep_ms = sum(handoff_prep_samples) / float(len(handoff_prep_samples))
    backward_ms = sum(backward_samples) / float(len(backward_samples))
    return {
        "frames": frames,
        "size": size,
        "tubes": tubes,
        "feature_dim": feature_dim,
        "backward_mode": backward_mode,
        "kernel_backward_mode": kernel_backward_mode,
        "cached_bins": use_cached_bins,
        "two_pass_feature_grad": two_pass_feature_grad,
        "two_pass_feature_mode": TWO_PASS_FEATURE_GRAD_MODES.get(backward_mode),
        "feature_grad_skipped": "skip_feature_grad" in backward_mode,
        "feature_grad_only": "feature_grad_only" in backward_mode,
        "colorizer_grad_skipped": backward_mode == "linear_sigmoid_mse_skip_colorizer_grad",
        "fused_first3_sigmoid_mse": kernel_backward_mode == "fused_first3_sigmoid_mse",
        "linear_sigmoid_mse": kernel_backward_mode in {"linear_sigmoid_mse", "linear_sigmoid_mse_skip_colorizer_grad"},
        "logit_handoff": kernel_backward_mode in LOGIT_HANDOFF_MODES,
        "native_logit_prep": native_logit_prep,
        "hidden_sigmoid_mse": kernel_backward_mode in HIDDEN_SIGMOID_MSE_MODES,
        "hidden_dim": hidden_dim if kernel_backward_mode in HIDDEN_SIGMOID_MSE_MODES else None,
        "warmup": warmup,
        "repeat": repeat,
        "forward_ms_samples": forward_samples,
        "handoff_prep_ms_samples": handoff_prep_samples,
        "backward_ms_samples": backward_samples,
        "forward_ms": forward_ms,
        "handoff_prep_ms": handoff_prep_ms,
        "backward_ms": backward_ms,
        "total_ms": forward_ms + handoff_prep_ms + backward_ms,
        "feature_image_shape": list(metal.feature_image.shape),
        "alpha_shape": list(metal.alpha.shape),
        "tile_overflow_sum": int(metal.tile_overflow.sum().cpu().item()),
        "tile_unstable_sum": int(tile_unstable.sum().cpu().item()),
        "finite": bool(torch.isfinite(metal.feature_image).all().cpu())
        and bool(torch.isfinite(metal.alpha).all().cpu())
        and all(bool(torch.isfinite(grad).all().cpu()) for grad in finite_grad_tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dims", default="4,32")
    parser.add_argument("--timing-frames", type=int, default=16)
    parser.add_argument("--timing-size", type=int, default=128)
    parser.add_argument("--timing-tubes", type=int, default=2048)
    parser.add_argument("--timing-feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--backward-mode",
        choices=(
            "direct_atomic",
            "direct_atomic_cached_bins",
            "gradcache",
            "gradcache_cached_bins",
            "direct_atomic_skip_feature_grad",
            "gradcache_skip_feature_grad",
            "direct_atomic_feature_grad_only",
            "gradcache_feature_grad_only",
            "gradcache_feature_grad_only_reduce",
            "gradcache_feature_grad_only_reduce_vec4",
            TWO_PASS_FEATURE_GRAD_MODE,
            "gradcache_two_pass_feature_grad_reduce",
            "gradcache_two_pass_feature_grad_reduce_vec4",
            "gradcache_reduce_feature_grad",
            "gradcache_reduce_feature_grad_cached_bins",
            "gradcache_reduce_feature_grad_vec4",
            "gradcache_reduce_feature_grad_vec4_cached_bins",
            "fused_first3_sigmoid_mse",
            "linear_sigmoid_mse",
            "linear_sigmoid_mse_skip_colorizer_grad",
            "logit_handoff",
            "logit_handoff_reduce",
            "logit_handoff_reduce_vec4",
            "logit_handoff_native_prep",
            "logit_handoff_reduce_native_prep",
            "logit_handoff_reduce_vec4_native_prep",
            "hidden_sigmoid_mse_star_only",
            "hidden_sigmoid_mse_star_only_reduce_vec4",
        ),
        default="direct_atomic",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--timing-warmup", type=int, default=0)
    parser.add_argument("--timing-repeat", type=int, default=1)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("direct feature kernel benchmark requires MPS")

    feature_dims = split_csv_ints(args.feature_dims)
    result: dict[str, Any] = {
        "gate": "star_uvt_feature_direct_metal_gate1",
        "backward_mode": args.backward_mode,
        "tiny_parity": [
            run_tiny_parity(feature_dim, backward_mode=args.backward_mode, hidden_dim=args.hidden_dim)
            for feature_dim in feature_dims
        ],
    }
    if not args.skip_timing:
        result["timing"] = run_timing_case(
            frames=args.timing_frames,
            size=args.timing_size,
            tubes=args.timing_tubes,
            feature_dim=args.timing_feature_dim,
            seed=args.seed,
            backward_mode=args.backward_mode,
            warmup=args.timing_warmup,
            repeat=args.timing_repeat,
            hidden_dim=args.hidden_dim,
        )
    result["pass"] = all(row["pass"] for row in result["tiny_parity"]) and result.get("timing", {}).get("finite", True)

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json is not None:
        write_report_json(args.out_json, result)
    print(payload)


if __name__ == "__main__":
    main()
