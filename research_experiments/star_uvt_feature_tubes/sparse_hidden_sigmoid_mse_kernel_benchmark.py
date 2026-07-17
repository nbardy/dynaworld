from __future__ import annotations

import argparse
import json
import math
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
    direct_atomic_feature_sparse_pixels_backward_cached_bins,
    direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins,
    render_uvt_feature_sparse_pixels_with_bins,
)


def _sync() -> None:
    sync_torch_device(torch.device("mps"))


def _max_err(got: Tensor, expected: Tensor) -> float:
    return float((got.detach().cpu() - expected.detach().cpu()).abs().max().item())


def _tiny_scene(feature_dim: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, UVTRenderConfig]:
    config = UVTRenderConfig(height=8, width=8, frames=2, tile_t=2, tile_capacity=128, alpha_threshold=1.0 / 255.0)
    ma = torch.tensor([[3.5, 3.5, -0.2], [4.5, 3.8, 0.1]], dtype=torch.float32)
    q_uvt = torch.tensor(
        [[0.30, 0.0, 0.0, 0.30, 0.0, 0.40], [0.24, 0.0, 0.03, 0.28, -0.02, 0.35]],
        dtype=torch.float32,
    )
    depth0 = torch.tensor([0.8, 1.2], dtype=torch.float32)
    depth_beta = torch.zeros((2, 3), dtype=torch.float32)
    opacity = torch.tensor([0.55, 0.45], dtype=torch.float32)
    feature = torch.linspace(-0.4, 0.5, 2 * feature_dim, dtype=torch.float32).view(2, feature_dim)
    return ma, q_uvt, depth0, depth_beta, opacity, feature, config


def _hidden_params(feature_dim: int, hidden_dim: int, *, seed: int = 7001) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_weight = torch.randn((hidden_dim, feature_dim), generator=generator, dtype=torch.float32) * 0.1
    hidden_bias = torch.randn((hidden_dim,), generator=generator, dtype=torch.float32) * 0.05
    output_weight = torch.randn((3, hidden_dim), generator=generator, dtype=torch.float32) * 0.1
    output_bias = torch.randn((3,), generator=generator, dtype=torch.float32) * 0.05
    return hidden_weight, hidden_bias, output_weight, output_bias


def _gather_sparse(feature_tfhw: Tensor, alpha_thw: Tensor, pixel_ids: Tensor) -> tuple[Tensor, Tensor]:
    feature_flat = feature_tfhw.permute(0, 2, 3, 1).contiguous().view(-1, feature_tfhw.shape[1])
    alpha_flat = alpha_thw.contiguous().view(-1)
    return feature_flat.index_select(0, pixel_ids.to(torch.long)), alpha_flat.index_select(0, pixel_ids.to(torch.long))


def _exact_gelu_grad(x: Tensor) -> Tensor:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + torch.erf(x * inv_sqrt2)) + x * inv_sqrt2pi * torch.exp(-0.5 * x * x)


def _hidden_sparse_vjp(
    feature_values: Tensor,
    alpha_values: Tensor,
    target_rgb_values: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    hidden_pre = feature_values @ hidden_weight.T + hidden_bias
    hidden = torch.nn.functional.gelu(hidden_pre)
    logits = hidden @ output_weight.T + output_bias
    splat_rgb = torch.sigmoid(logits)
    rgb = alpha_values[:, None] * splat_rgb
    diff = rgb - target_rgb_values
    loss = diff.square().mean()
    grad_rgb = (2.0 / float(target_rgb_values.numel())) * diff
    grad_logits = grad_rgb * alpha_values[:, None] * splat_rgb * (1.0 - splat_rgb)
    grad_alpha_values = (grad_rgb * splat_rgb).sum(dim=1).contiguous()
    grad_hidden = grad_logits @ output_weight
    grad_hidden_pre = grad_hidden * _exact_gelu_grad(hidden_pre)
    grad_feature_values = (grad_hidden_pre @ hidden_weight).contiguous()
    return loss, grad_feature_values, grad_alpha_values


def run_tiny_parity(feature_dim: int, hidden_dim: int) -> dict[str, Any]:
    ma, q_uvt, depth0, depth_beta, opacity, feature, config = _tiny_scene(feature_dim)
    pixel_ids = torch.tensor([0, 1, 5, 16, 63, 64, 80, 127], dtype=torch.int32)
    target_rgb_values = torch.linspace(0.05, 0.85, pixel_ids.numel() * 3, dtype=torch.float32).view(-1, 3)
    hidden_weight, hidden_bias, output_weight, output_bias = _hidden_params(feature_dim, hidden_dim)

    ma_ref = ma.clone().requires_grad_(True)
    q_ref = q_uvt.clone().requires_grad_(True)
    opacity_ref = opacity.clone().requires_grad_(True)
    feature_ref = feature.clone().requires_grad_(True)
    ref_feature, ref_alpha = brute_force_render_uvt_feature_tubes(
        ma_ref, q_ref, depth0, depth_beta, opacity_ref, feature_ref, config
    )
    sparse_feature_ref, sparse_alpha_ref = _gather_sparse(ref_feature, ref_alpha, pixel_ids)
    loss, _, _ = _hidden_sparse_vjp(
        sparse_feature_ref,
        sparse_alpha_ref,
        target_rgb_values,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )
    loss.backward()

    mps_inputs = [tensor.to("mps").contiguous() for tensor in (ma, q_uvt, depth0, depth_beta, opacity, feature)]
    pixel_ids_mps = pixel_ids.to("mps").contiguous()
    sparse_render = render_uvt_feature_sparse_pixels_with_bins(*mps_inputs, pixel_ids_mps, config)
    fused = direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins(
        *mps_inputs,
        pixel_ids_mps,
        target_rgb_values.to("mps").contiguous(),
        hidden_weight.to("mps").contiguous(),
        hidden_bias.to("mps").contiguous(),
        output_weight.to("mps").contiguous(),
        output_bias.to("mps").contiguous(),
        sparse_render.tile_counts,
        sparse_render.tile_tube_ids,
        sparse_render.tile_depths,
        sparse_render.tile_unstable,
        config,
    )
    errors = {
        "forward_feature": _max_err(sparse_render.feature_values, sparse_feature_ref),
        "forward_alpha": _max_err(sparse_render.alpha_values, sparse_alpha_ref),
        "loss": _max_err(fused.loss.reshape(()), loss.detach()),
        "ma": _max_err(fused.grad_ma, ma_ref.grad),
        "q": _max_err(fused.grad_q_uvt, q_ref.grad),
        "opacity": _max_err(fused.grad_opacity, opacity_ref.grad),
        "feature": _max_err(fused.grad_feature, feature_ref.grad),
    }
    return {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "sparse_count": int(pixel_ids.numel()),
        "loss_ref": float(loss.detach().item()),
        "loss_fused": float(fused.loss.detach().cpu().item()),
        "max_abs_errors": errors,
        "tile_overflow_sum": int(sparse_render.tile_overflow.sum().cpu().item()),
        "tile_unstable_sum": int(fused.tile_unstable.sum().cpu().item()),
        "pass": max(errors.values()) <= 1.0e-4 and int(sparse_render.tile_overflow.sum().cpu().item()) == 0,
    }


def _random_timing_scene(
    *,
    frames: int,
    size: int,
    tubes: int,
    feature_dim: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, UVTRenderConfig]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    center_u = torch.rand((tubes,), generator=generator) * float(size)
    center_v = torch.rand((tubes,), generator=generator) * float(size)
    center_t = (torch.rand((tubes,), generator=generator) - 0.5) * float(frames)
    ma = torch.stack((center_u, center_v, center_t), dim=-1).to(dtype=torch.float32)
    precision_uv = torch.full((tubes,), 0.18, dtype=torch.float32)
    precision_t = torch.full((tubes,), 0.12, dtype=torch.float32)
    q_uvt = torch.stack(
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
    config = UVTRenderConfig(height=size, width=size, frames=frames, tile_t=2, tile_capacity=128)
    return ma, q_uvt, depth0, depth_beta, opacity, feature, config


def _stratified_pixel_ids(frames: int, size: int, sparse_side: int) -> Tensor:
    if sparse_side <= 0:
        raise ValueError("sparse_side must be positive")
    ys = torch.clamp(((torch.arange(sparse_side, dtype=torch.float32) + 0.5) * size / sparse_side).long(), 0, size - 1)
    xs = torch.clamp(((torch.arange(sparse_side, dtype=torch.float32) + 0.5) * size / sparse_side).long(), 0, size - 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    per_frame = (yy * size + xx).reshape(-1)
    frame_offsets = torch.arange(frames, dtype=torch.long)[:, None] * (size * size)
    return (frame_offsets + per_frame[None, :]).reshape(-1).to(torch.int32)


def _mean(samples: list[float]) -> float:
    return sum(samples) / float(len(samples))


def run_timing_case(
    *,
    frames: int,
    size: int,
    tubes: int,
    feature_dim: int,
    hidden_dim: int,
    sparse_side: int,
    seed: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    ma, q_uvt, depth0, depth_beta, opacity, feature, config = _random_timing_scene(
        frames=frames,
        size=size,
        tubes=tubes,
        feature_dim=feature_dim,
        seed=seed,
    )
    mps_inputs = [tensor.to("mps").contiguous() for tensor in (ma, q_uvt, depth0, depth_beta, opacity, feature)]
    pixel_ids = _stratified_pixel_ids(frames, size, sparse_side).to("mps").contiguous()
    generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
    target_rgb_values = torch.rand((int(pixel_ids.numel()), 3), generator=generator, dtype=torch.float32).to("mps").contiguous()
    hidden_weight, hidden_bias, output_weight, output_bias = [
        tensor.to("mps").contiguous() for tensor in _hidden_params(feature_dim, hidden_dim, seed=seed + 2003)
    ]

    render_samples: list[float] = []
    baseline_prep_samples: list[float] = []
    baseline_backward_samples: list[float] = []
    fused_backward_samples: list[float] = []
    loss_error_samples: list[float] = []
    tile_overflow_sum = 0
    tile_unstable_sum = 0
    finite = True

    for iteration in range(warmup + repeat):
        _sync()
        t0 = time.perf_counter()
        sparse_render = render_uvt_feature_sparse_pixels_with_bins(*mps_inputs, pixel_ids, config)
        _sync()
        render_ms = (time.perf_counter() - t0) * 1000.0

        _sync()
        t1 = time.perf_counter()
        loss, grad_feature_values, grad_alpha_values = _hidden_sparse_vjp(
            sparse_render.feature_values,
            sparse_render.alpha_values,
            target_rgb_values,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )
        _sync()
        baseline_prep_ms = (time.perf_counter() - t1) * 1000.0

        _sync()
        t2 = time.perf_counter()
        baseline_grads = direct_atomic_feature_sparse_pixels_backward_cached_bins(
            *mps_inputs,
            pixel_ids,
            grad_feature_values,
            grad_alpha_values,
            sparse_render.tile_counts,
            sparse_render.tile_tube_ids,
            sparse_render.tile_depths,
            sparse_render.tile_unstable,
            config,
        )
        _sync()
        baseline_backward_ms = (time.perf_counter() - t2) * 1000.0

        _sync()
        t3 = time.perf_counter()
        fused = direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins(
            *mps_inputs,
            pixel_ids,
            target_rgb_values,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            sparse_render.tile_counts,
            sparse_render.tile_tube_ids,
            sparse_render.tile_depths,
            sparse_render.tile_unstable,
            config,
        )
        _sync()
        fused_backward_ms = (time.perf_counter() - t3) * 1000.0

        if iteration >= warmup:
            render_samples.append(render_ms)
            baseline_prep_samples.append(baseline_prep_ms)
            baseline_backward_samples.append(baseline_backward_ms)
            fused_backward_samples.append(fused_backward_ms)
            loss_error_samples.append(_max_err(fused.loss.reshape(()), loss.detach()))
            tile_overflow_sum = int(sparse_render.tile_overflow.sum().cpu().item())
            tile_unstable_sum = int(fused.tile_unstable.sum().cpu().item())
            finite = finite and bool(torch.isfinite(sparse_render.feature_values).all().cpu())
            finite = finite and bool(torch.isfinite(sparse_render.alpha_values).all().cpu())
            finite = finite and all(bool(torch.isfinite(grad).all().cpu()) for grad in baseline_grads[:4])
            finite = finite and all(
                bool(torch.isfinite(grad).all().cpu())
                for grad in (fused.grad_ma, fused.grad_q_uvt, fused.grad_opacity, fused.grad_feature, fused.loss)
            )

    render_ms = _mean(render_samples)
    baseline_prep_ms = _mean(baseline_prep_samples)
    baseline_backward_ms = _mean(baseline_backward_samples)
    fused_backward_ms = _mean(fused_backward_samples)
    return {
        "frames": frames,
        "size": size,
        "tubes": tubes,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "sparse_side": sparse_side,
        "sparse_count": int(pixel_ids.numel()),
        "warmup": warmup,
        "repeat": repeat,
        "render_ms_samples": render_samples,
        "baseline_prep_ms_samples": baseline_prep_samples,
        "baseline_backward_ms_samples": baseline_backward_samples,
        "fused_backward_ms_samples": fused_backward_samples,
        "loss_error_samples": loss_error_samples,
        "render_ms": render_ms,
        "baseline_prep_ms": baseline_prep_ms,
        "baseline_backward_ms": baseline_backward_ms,
        "fused_backward_ms": fused_backward_ms,
        "baseline_total_ms": render_ms + baseline_prep_ms + baseline_backward_ms,
        "fused_total_ms": render_ms + fused_backward_ms,
        "fused_vs_baseline_total_speedup": (render_ms + baseline_prep_ms + baseline_backward_ms)
        / max(render_ms + fused_backward_ms, 1.0e-9),
        "max_loss_error": max(loss_error_samples),
        "tile_overflow_sum": tile_overflow_sum,
        "tile_unstable_sum": tile_unstable_sum,
        "finite": finite,
        "pass": finite and tile_overflow_sum == 0 and max(loss_error_samples) <= 1.0e-4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dims", default="4,32")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--timing-frames", type=int, default=64)
    parser.add_argument("--timing-size", type=int, default=256)
    parser.add_argument("--timing-tubes", type=int, default=8192)
    parser.add_argument("--timing-feature-dim", type=int, default=32)
    parser.add_argument("--sparse-side", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--timing-warmup", type=int, default=1)
    parser.add_argument("--timing-repeat", type=int, default=3)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("sparse hidden sigmoid-MSE benchmark requires MPS")

    feature_dims = split_csv_ints(args.feature_dims)
    result: dict[str, Any] = {
        "gate": "star_uvt_sparse_hidden_sigmoid_mse_native",
        "tiny_parity": [run_tiny_parity(feature_dim, args.hidden_dim) for feature_dim in feature_dims],
    }
    if not args.skip_timing:
        result["timing"] = run_timing_case(
            frames=args.timing_frames,
            size=args.timing_size,
            tubes=args.timing_tubes,
            feature_dim=args.timing_feature_dim,
            hidden_dim=args.hidden_dim,
            sparse_side=args.sparse_side,
            seed=args.seed,
            warmup=args.timing_warmup,
            repeat=args.timing_repeat,
        )
    result["pass"] = all(row["pass"] for row in result["tiny_parity"]) and result.get("timing", {}).get("pass", True)

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json is not None:
        write_report_json(args.out_json, result)
    print(payload)


if __name__ == "__main__":
    main()
