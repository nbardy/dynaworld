"""Dense CPU-first prototype for feature-valued STAR/UVT tubes.

This is intentionally isolated from the RGB STAR UVT Metal renderer. It proves
the contract needed for F32 tubes:

    [N, F] tube features -> [T, F, H, W] feature image + alpha
        -> FeatureToColor -> RGB reconstruction loss.

The implementation is dense and meant only for tiny smoke tests.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

try:
    from .report_artifacts import write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import write_report_json
from colorize import FeatureToColor
from star_uvt_feature_tube_model import (
    FeatureScreenTimeTubeModel,
    FeatureTubeRenderConfig,
    _inv_softplus,
    _logit,
    colorize_and_compose,
    dense_render_feature_tubes,
    make_default_colorizer,
    make_uvt_grid,
    render_model_features,
)
from star_uvt_runtime import resolve_device as _resolve_device
from star_uvt_runtime import sync_device as _sync_device


def _target_rgb(config: FeatureTubeRenderConfig, *, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand((config.frames, 3, config.height, config.width), generator=generator).to(device)


def _zero_module_grads(*modules: nn.Module) -> None:
    for module in modules:
        for param in module.parameters():
            param.grad = None


def _grad_snapshot(module: nn.Module, *, prefix: str) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            out[f"{prefix}.{name}"] = param.grad.detach().cpu()
    return out


def _grad_norms(model: FeatureScreenTimeTubeModel, colorizer: FeatureToColor) -> dict[str, float]:
    snapshots = {
        **_grad_snapshot(model, prefix="model"),
        **_grad_snapshot(colorizer, prefix="colorizer"),
    }
    return {name: float(value.norm().item()) for name, value in snapshots.items()}


def _all_finite(*tensors: Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all().detach().cpu()) for tensor in tensors)


def _loss_full(
    model: FeatureScreenTimeTubeModel,
    colorizer: FeatureToColor,
    target_rgb: Tensor,
) -> tuple[Tensor, dict[str, float], tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]:
    device = target_rgb.device
    _sync_device(device)
    t0 = time.perf_counter()
    feature_image, alpha = render_model_features(model)
    _sync_device(device)
    t1 = time.perf_counter()
    rgb = colorize_and_compose(feature_image, alpha, colorizer)
    _sync_device(device)
    t2 = time.perf_counter()
    loss = torch.mean((rgb - target_rgb).square())
    _sync_device(device)
    t3 = time.perf_counter()
    timing = {
        "render_ms": (t1 - t0) * 1000.0,
        "colorize_compose_ms": (t2 - t1) * 1000.0,
        "loss_ms": (t3 - t2) * 1000.0,
    }
    finite = _all_finite(feature_image, alpha, rgb, loss)
    return loss, timing, tuple(feature_image.shape), tuple(alpha.shape), tuple(rgb.shape), finite


def _backward_full(
    model: FeatureScreenTimeTubeModel,
    colorizer: FeatureToColor,
    target_rgb: Tensor,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    _zero_module_grads(model, colorizer)
    loss, timing, feature_shape, alpha_shape, rgb_shape, finite = _loss_full(model, colorizer, target_rgb)
    device = target_rgb.device
    _sync_device(device)
    t0 = time.perf_counter()
    loss.backward()
    _sync_device(device)
    timing["backward_ms"] = (time.perf_counter() - t0) * 1000.0
    return float(loss.detach().cpu()), timing, {
        "feature_image_shape": list(feature_shape),
        "alpha_shape": list(alpha_shape),
        "rgb_shape": list(rgb_shape),
        "forward_outputs_finite": finite,
        "grad_norms": _grad_norms(model, colorizer),
    }


def _backward_chunked(
    model: FeatureScreenTimeTubeModel,
    colorizer: FeatureToColor,
    target_rgb: Tensor,
    *,
    chunk_size: int,
) -> tuple[float, dict[str, float]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    _zero_module_grads(model, colorizer)
    device = target_rgb.device
    total_loss = 0.0
    timing = {"render_ms": 0.0, "colorize_compose_ms": 0.0, "loss_ms": 0.0, "backward_ms": 0.0}
    denom = float(target_rgb.numel())
    for start in range(0, model.config.frames, chunk_size):
        end = min(model.config.frames, start + chunk_size)
        frame_indices = torch.arange(start, end, dtype=torch.int64, device=device)
        _sync_device(device)
        t0 = time.perf_counter()
        feature_image, alpha = render_model_features(model, frame_indices=frame_indices)
        _sync_device(device)
        t1 = time.perf_counter()
        rgb = colorize_and_compose(feature_image, alpha, colorizer)
        _sync_device(device)
        t2 = time.perf_counter()
        loss = (rgb - target_rgb[start:end]).square().sum() / denom
        _sync_device(device)
        t3 = time.perf_counter()
        loss.backward()
        _sync_device(device)
        t4 = time.perf_counter()
        total_loss += float(loss.detach().cpu())
        timing["render_ms"] += (t1 - t0) * 1000.0
        timing["colorize_compose_ms"] += (t2 - t1) * 1000.0
        timing["loss_ms"] += (t3 - t2) * 1000.0
        timing["backward_ms"] += (t4 - t3) * 1000.0
    return total_loss, timing


def _compare_gradients(
    full_model: FeatureScreenTimeTubeModel,
    full_colorizer: FeatureToColor,
    chunk_model: FeatureScreenTimeTubeModel,
    chunk_colorizer: FeatureToColor,
) -> dict[str, Any]:
    full = {
        **_grad_snapshot(full_model, prefix="model"),
        **_grad_snapshot(full_colorizer, prefix="colorizer"),
    }
    chunk = {
        **_grad_snapshot(chunk_model, prefix="model"),
        **_grad_snapshot(chunk_colorizer, prefix="colorizer"),
    }
    keys = sorted(set(full) | set(chunk))
    max_abs = 0.0
    max_rel = 0.0
    worst_key = ""
    missing: list[str] = []
    for key in keys:
        if key not in full or key not in chunk:
            missing.append(key)
            continue
        diff = (full[key] - chunk[key]).abs()
        abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        denom = float(full[key].abs().max().item()) if full[key].numel() else 0.0
        rel_diff = abs_diff / max(denom, 1.0e-8)
        if abs_diff > max_abs:
            max_abs = abs_diff
            worst_key = key
        max_rel = max(max_rel, rel_diff)
    return {
        "max_abs_grad_diff": max_abs,
        "max_rel_grad_diff": max_rel,
        "worst_grad_key": worst_key,
        "missing_grad_keys": missing,
        "compared_grad_key_count": len(keys) - len(missing),
    }


def _run_chunked_parity(
    model: FeatureScreenTimeTubeModel,
    colorizer: FeatureToColor,
    target_rgb: Tensor,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    full_model = copy.deepcopy(model)
    full_colorizer = copy.deepcopy(colorizer)
    chunk_model = copy.deepcopy(model)
    chunk_colorizer = copy.deepcopy(colorizer)
    full_loss, full_timing, _full_contract = _backward_full(full_model, full_colorizer, target_rgb)
    chunk_loss, chunk_timing = _backward_chunked(chunk_model, chunk_colorizer, target_rgb, chunk_size=chunk_size)
    grad_compare = _compare_gradients(full_model, full_colorizer, chunk_model, chunk_colorizer)
    return {
        "chunk_size": chunk_size,
        "full_loss": full_loss,
        "chunked_loss": chunk_loss,
        "loss_abs_diff": abs(full_loss - chunk_loss),
        "full_timing_ms": full_timing,
        "chunked_timing_ms": chunk_timing,
        **grad_compare,
        "pass": abs(full_loss - chunk_loss) <= 1.0e-6
        and grad_compare["max_abs_grad_diff"] <= 1.0e-5
        and not grad_compare["missing_grad_keys"],
    }


def _run_overfit(
    model: FeatureScreenTimeTubeModel,
    colorizer: FeatureToColor,
    target_rgb: Tensor,
    *,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=lr)
    losses: list[float] = []
    step_ms: list[float] = []
    for _step in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        feature_image, alpha = render_model_features(model)
        rgb = colorize_and_compose(feature_image, alpha, colorizer)
        loss = torch.mean((rgb - target_rgb).square())
        loss.backward()
        optimizer.step()
        _sync_device(target_rgb.device)
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        losses.append(float(loss.detach().cpu()))
    return {
        "steps": steps,
        "lr": lr,
        "start_loss": losses[0] if losses else None,
        "end_loss": losses[-1] if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "losses": losses,
        "mean_step_ms": sum(step_ms) / max(len(step_ms), 1),
        "max_step_ms": max(step_ms) if step_ms else None,
    }


def run_gate0_contract(
    *,
    frames: int,
    height: int,
    width: int,
    tube_count: int,
    feature_dim: int,
    steps: int,
    lr: float,
    seed: int,
    chunk_size: int,
    device_name: str,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    config = FeatureTubeRenderConfig(frames=frames, height=height, width=width, feature_dim=feature_dim)
    model = FeatureScreenTimeTubeModel(tube_count, config, seed=seed, device=device)
    colorizer = make_default_colorizer(config.feature_dim).to(device)
    target_rgb = _target_rgb(config, seed=seed + 101, device=device)

    full_loss, full_timing, contract = _backward_full(model, colorizer, target_rgb)
    grad_norms = contract["grad_norms"]
    contract.update(
        {
            "full_loss": full_loss,
            "full_timing_ms": full_timing,
            "loss_finite": bool(torch.isfinite(torch.tensor(full_loss))),
            "raw_feature_grad_seen": grad_norms.get("model.raw_feature", 0.0) > 0.0,
            "center_uv_grad_seen": grad_norms.get("model.center_uv", 0.0) > 0.0,
            "velocity_uv_grad_seen": grad_norms.get("model.velocity_uv", 0.0) > 0.0,
            "colorizer_grad_seen": any(key.startswith("colorizer.") and value > 0.0 for key, value in grad_norms.items()),
        }
    )

    parity: dict[str, Any] | None = None
    if 0 < chunk_size < frames:
        parity = _run_chunked_parity(
            FeatureScreenTimeTubeModel(tube_count, config, seed=seed, device=device),
            make_default_colorizer(config.feature_dim).to(device),
            target_rgb,
            chunk_size=chunk_size,
        )

    overfit = _run_overfit(
        FeatureScreenTimeTubeModel(tube_count, config, seed=seed, device=device),
        make_default_colorizer(config.feature_dim).to(device),
        target_rgb,
        steps=steps,
        lr=lr,
    )

    gate_pass = (
        contract["feature_image_shape"] == [frames, feature_dim, height, width]
        and contract["alpha_shape"] == [frames, height, width]
        and contract["rgb_shape"] == [frames, 3, height, width]
        and contract["forward_outputs_finite"]
        and contract["loss_finite"]
        and contract["raw_feature_grad_seen"]
        and contract["center_uv_grad_seen"]
        and contract["colorizer_grad_seen"]
        and overfit["loss_decreased"]
        and (parity is None or parity["pass"])
    )

    return {
        "gate": "star_uvt_feature_tubes_gate0_dense_contract",
        "pass": bool(gate_pass),
        "device": str(device),
        "config": {
            "frames": frames,
            "height": height,
            "width": width,
            "tube_count": tube_count,
            "feature_dim": feature_dim,
            "chunk_size": chunk_size,
            "seed": seed,
        },
        "contract": contract,
        "chunked_parity": parity,
        "tiny_overfit": overfit,
    }


def run_tiny_smoke() -> dict[str, float | bool]:
    config = FeatureTubeRenderConfig(frames=3, height=12, width=12, feature_dim=32)
    model = FeatureScreenTimeTubeModel(16, config, seed=7)
    colorizer = make_default_colorizer(config.feature_dim)
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=1.0e-2)
    generator = torch.Generator(device="cpu").manual_seed(11)
    target_rgb = torch.rand((config.frames, 3, config.height, config.width), generator=generator)

    losses: list[float] = []
    for _step in range(3):
        optimizer.zero_grad(set_to_none=True)
        feature_image, alpha = render_model_features(model)
        rgb = colorize_and_compose(feature_image, alpha, colorizer)
        loss = torch.mean((rgb - target_rgb).square())
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        optimizer.step()

    return {
        "start_loss": losses[0],
        "end_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "raw_feature_grad_seen": model.raw_feature.grad is not None,
        "center_uv_grad_seen": model.center_uv.grad is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run the tiny CPU gradient check")
    parser.add_argument("--gate0-benchmark", action="store_true", help="run the Gate 0 dense feature-tube contract")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--tubes", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(run_tiny_smoke(), sort_keys=True))
    if args.gate0_benchmark:
        result = run_gate0_contract(
            frames=args.frames,
            height=args.height,
            width=args.width,
            tube_count=args.tubes,
            feature_dim=args.feature_dim,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            chunk_size=args.chunk_size,
            device_name=args.device,
        )
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.out_json is not None:
            write_report_json(args.out_json, result)
        print(payload)


if __name__ == "__main__":
    main()
