from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn


from config_utils import load_config_file, path_or_none as _path_or_none
try:
    from .report_artifacts import ROOT as DYNAWORLD_ROOT, summary_stats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT as DYNAWORLD_ROOT, summary_stats, write_report_json, write_report_text
from research_experiments.star_uvt_feature_tubes.dense_feature_tube_prototype import (
    colorize_and_compose,
)
from star_uvt_colorizers import build_feature_colorizer
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    chunked_uvt_config,
    direct_logit_handoff_backward,
    render_uvt_feature_tubes,
    render_uvt_feature_tubes_autograd,
    shift_ma_for_frame_chunk,
)
from star_uvt_checkpoints import load_star_training_checkpoint as _load_training_checkpoint
from star_uvt_common import load_training_sequence as _load_training_sequence
from star_uvt_models import build_feature_tube_model
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from star_uvt_runtime import resolve_device as _resolve_device, sync_device as _sync_device
from star_uvt_tile_stats import _tile_load_stats
from star_uvt_feature_config import resolve_config


DEFAULT_CONFIG = (
    DYNAWORLD_ROOT
    / "src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc"
)
DEFAULT_OUT_BASE = DYNAWORLD_ROOT / "outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile"
BASELINE_BACKWARD_MODE = "gradcache_reduce_feature_grad_vec4"
HANDOFF_BACKWARD_MODE = "logit_handoff_reduce_vec4"


def _conv_colorizer(colorizer: nn.Module) -> nn.Conv2d:
    net = getattr(colorizer, "net", None)
    if not isinstance(net, nn.Conv2d):
        raise ValueError("logit handoff profile requires colorize.hidden_dim=null")
    if getattr(colorizer, "pre_norm", None) is not None:
        raise ValueError("logit handoff profile requires colorize.pre_norm=false")
    if getattr(colorizer, "activation", None) != "sigmoid":
        raise ValueError("logit handoff profile requires colorize.activation='sigmoid'")
    if getattr(colorizer, "view_condition", "none") != "none":
        raise ValueError("logit handoff profile requires colorize.view_condition='none'")
    return net


def _make_case(config_path: Path) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(config_path))
    device = _resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps":
        raise RuntimeError("STAR UVT logit handoff profile requires MPS")
    torch.manual_seed(int(cfg["train"]["seed"]))

    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    if feature_config.feature_dim > 64:
        raise ValueError("direct_logit_handoff_backward only supports feature_dim <= 64")
    sequence = _load_training_sequence(cfg, device)
    model = build_feature_tube_model(cfg, feature_config, device=device)
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_config.feature_dim, device=device)
    _conv_colorizer(colorizer)
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=float(cfg["train"]["lr"]))
    resume_checkpoint = _path_or_none(cfg["train"].get("resume_checkpoint"))
    resume_state: dict[str, Any] = {"path": None, "loaded": False, "optimizer_loaded": False, "steps": None}
    if resume_checkpoint is not None:
        resume_state = _load_training_checkpoint(
            resume_checkpoint,
            model=model,
            colorizer=colorizer,
            optimizer=optimizer,
            device=device,
            resume_optimizer=False,
        )
    return {
        "cfg": cfg,
        "config_path": config_path,
        "device": device,
        "feature_config": feature_config,
        "uvt_config": uvt_config,
        "target_rgb": sequence.frames.contiguous(),
        "model": model,
        "colorizer": colorizer,
        "resume_state": resume_state,
    }


def _chunk_inputs(
    model: Any,
    uvt_config: UVTRenderConfig,
    frame_start: int,
    chunk_frames: int,
) -> tuple[tuple[torch.Tensor, ...], UVTRenderConfig]:
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
    if chunk_frames == uvt_config.frames:
        return (ma, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature), uvt_config
    ma_chunk = shift_ma_for_frame_chunk(
        ma,
        global_frames=uvt_config.frames,
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    return (ma_chunk, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature), chunked_uvt_config(
        uvt_config,
        chunk_frames=chunk_frames,
    )


def _zero_grads(*modules: nn.Module) -> None:
    for module in modules:
        for param in module.parameters():
            param.grad = None


def _collect_grads(prefix: str, module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name}": param.grad.detach().cpu().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }


def _grad_comparison(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = sorted(set(a) | set(b))
    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    missing: list[str] = []
    for name in names:
        if name not in a or name not in b:
            missing.append(name)
            continue
        diff = (a[name] - b[name]).abs()
        abs_err = float(diff.max().item()) if diff.numel() else 0.0
        denom = float(torch.maximum(a[name].abs(), b[name].abs()).max().item()) if diff.numel() else 0.0
        rel_err = 0.0 if denom <= 1.0e-12 else abs_err / denom
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        rows.append(
            {
                "name": name,
                "max_abs_error": abs_err,
                "max_rel_error": rel_err,
                "baseline_norm": float(a[name].norm().item()),
                "handoff_norm": float(b[name].norm().item()),
            }
        )
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "missing": missing,
        "rows": rows,
    }


def _manual_colorizer_grads(
    conv: nn.Conv2d,
    feature_image: torch.Tensor,
    grad_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_weight = torch.einsum("tchw,tfhw->cf", grad_logits, feature_image).contiguous()
    grad_bias = grad_logits.sum(dim=(0, 2, 3)).contiguous()
    return grad_weight.view_as(conv.weight), grad_bias.view_as(conv.bias)


def _add_grad(param: torch.Tensor, grad: torch.Tensor) -> None:
    if param.grad is None:
        param.grad = grad.detach().clone()
    else:
        param.grad.add_(grad.detach())


def _run_autograd(case: dict[str, Any], *, chunk_size: int) -> dict[str, Any]:
    model = case["model"]
    colorizer = case["colorizer"]
    target_rgb = case["target_rgb"]
    uvt_config = case["uvt_config"]
    feature_config = case["feature_config"]
    device = case["device"]
    total_loss_elems = int(target_rgb.numel())
    _zero_grads(model, colorizer)
    phase = {"render_forward_ms": 0.0, "colorize_loss_ms": 0.0, "backward_ms": 0.0}
    loss_value = 0.0
    for frame_start in range(0, feature_config.frames, chunk_size):
        chunk_frames = min(chunk_size, feature_config.frames - frame_start)
        render_inputs, chunk_config = _chunk_inputs(model, uvt_config, frame_start, chunk_frames)
        _sync_device(device)
        t0 = time.perf_counter()
        render = render_uvt_feature_tubes_autograd(
            *render_inputs,
            chunk_config,
            backward_mode=BASELINE_BACKWARD_MODE,
        )
        _sync_device(device)
        t1 = time.perf_counter()
        rgb = colorize_and_compose(render.feature_image, render.alpha, colorizer)
        target_chunk = target_rgb[frame_start : frame_start + chunk_frames]
        loss = (rgb - target_chunk).square().sum() / float(total_loss_elems)
        loss_value += float(loss.detach().cpu().item())
        _sync_device(device)
        t2 = time.perf_counter()
        loss.backward()
        _sync_device(device)
        t3 = time.perf_counter()
        phase["render_forward_ms"] += (t1 - t0) * 1000.0
        phase["colorize_loss_ms"] += (t2 - t1) * 1000.0
        phase["backward_ms"] += (t3 - t2) * 1000.0
    return {
        "loss": loss_value,
        "timing_ms": {**phase, "total_ms": sum(phase.values())},
        "grads": {
            **_collect_grads("model", model),
            **_collect_grads("colorizer", colorizer),
        },
    }


def _run_logit_handoff(case: dict[str, Any], *, chunk_size: int, mode: str) -> dict[str, Any]:
    model = case["model"]
    colorizer = case["colorizer"]
    target_rgb = case["target_rgb"]
    uvt_config = case["uvt_config"]
    feature_config = case["feature_config"]
    device = case["device"]
    conv = _conv_colorizer(colorizer)
    total_loss_elems = int(target_rgb.numel())
    _zero_grads(model, colorizer)
    phase = {
        "render_forward_ms": 0.0,
        "logit_loss_vjp_ms": 0.0,
        "renderer_backward_ms": 0.0,
        "param_backward_ms": 0.0,
    }
    loss_value = 0.0
    tile_counts: list[torch.Tensor] = []
    tile_overflow: list[torch.Tensor] = []
    tile_unstable: list[torch.Tensor] = []
    finite = True
    for frame_start in range(0, feature_config.frames, chunk_size):
        chunk_frames = min(chunk_size, feature_config.frames - frame_start)
        render_inputs, chunk_config = _chunk_inputs(model, uvt_config, frame_start, chunk_frames)
        ma, q_uvt, _depth0, _depth_beta, opacity, feature = render_inputs
        _sync_device(device)
        t0 = time.perf_counter()
        render = render_uvt_feature_tubes(*render_inputs, chunk_config)
        _sync_device(device)
        t1 = time.perf_counter()

        weight_2d = conv.weight[:, :, 0, 0].contiguous()
        logits = torch.nn.functional.conv2d(render.feature_image, conv.weight, conv.bias)
        sigmoid_logits = torch.sigmoid(logits)
        target_chunk = target_rgb[frame_start : frame_start + chunk_frames]
        rgb = render.alpha.unsqueeze(1) * sigmoid_logits
        diff = rgb - target_chunk
        loss_value += float((diff.square().sum() / float(total_loss_elems)).detach().cpu().item())
        grad_rgb = (2.0 / float(total_loss_elems)) * diff
        grad_logits = grad_rgb * render.alpha.unsqueeze(1) * sigmoid_logits * (1.0 - sigmoid_logits)
        grad_alpha = (grad_rgb * sigmoid_logits).sum(dim=1)
        grad_weight, grad_bias = _manual_colorizer_grads(conv, render.feature_image, grad_logits)
        _add_grad(conv.weight, grad_weight)
        if conv.bias is not None:
            _add_grad(conv.bias, grad_bias)
        _sync_device(device)
        t2 = time.perf_counter()

        result = direct_logit_handoff_backward(
            *render_inputs,
            grad_logits.contiguous(),
            grad_alpha.contiguous(),
            weight_2d,
            chunk_config,
            backward_mode=mode,
        )
        _sync_device(device)
        t3 = time.perf_counter()
        torch.autograd.backward(
            (ma, q_uvt, opacity, feature),
            (result.grad_ma, result.grad_q_uvt, result.grad_opacity, result.grad_feature),
        )
        _sync_device(device)
        t4 = time.perf_counter()
        phase["render_forward_ms"] += (t1 - t0) * 1000.0
        phase["logit_loss_vjp_ms"] += (t2 - t1) * 1000.0
        phase["renderer_backward_ms"] += (t3 - t2) * 1000.0
        phase["param_backward_ms"] += (t4 - t3) * 1000.0
        tile_counts.append(render.tile_counts)
        tile_overflow.append(render.tile_overflow)
        tile_unstable.append(result.tile_unstable)
        finite = (
            finite
            and bool(torch.isfinite(render.feature_image).all().cpu())
            and bool(torch.isfinite(grad_logits).all().cpu())
            and bool(torch.isfinite(grad_alpha).all().cpu())
            and bool(torch.isfinite(result.grad_feature).all().cpu())
        )
    return {
        "loss": loss_value,
        "timing_ms": {**phase, "total_ms": sum(phase.values())},
        "grads": {
            **_collect_grads("model", model),
            **_collect_grads("colorizer", colorizer),
        },
        "tile_stats": _tile_load_stats(
            tile_counts=tile_counts,
            tile_overflow=tile_overflow,
            tile_unstable=tile_unstable,
            tile_capacity=int(case["cfg"]["feature_uvt"]["tile_capacity"]),
        ),
        "finite": finite,
    }


def profile(config_path: Path, *, warmup: int, repeat: int, mode: str) -> dict[str, Any]:
    case = _make_case(config_path)
    cfg = case["cfg"]
    chunk_size_cfg = cfg["train"]["frame_chunk_size"]
    chunk_size = case["feature_config"].frames if chunk_size_cfg is None else int(chunk_size_cfg)
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")
    chunk_size = min(chunk_size, case["feature_config"].frames)

    autograd_samples = {"render_forward_ms": [], "colorize_loss_ms": [], "backward_ms": [], "total_ms": []}
    handoff_samples = {
        "render_forward_ms": [],
        "logit_loss_vjp_ms": [],
        "renderer_backward_ms": [],
        "param_backward_ms": [],
        "total_ms": [],
    }
    comparison: dict[str, Any] | None = None
    last_handoff: dict[str, Any] | None = None
    loss_abs_error = 0.0
    for index in range(warmup + repeat):
        autograd = _run_autograd(case, chunk_size=chunk_size)
        handoff = _run_logit_handoff(case, chunk_size=chunk_size, mode=mode)
        if index >= warmup:
            for key, value in autograd["timing_ms"].items():
                autograd_samples[key].append(float(value))
            for key, value in handoff["timing_ms"].items():
                handoff_samples[key].append(float(value))
            loss_abs_error = max(loss_abs_error, abs(float(autograd["loss"]) - float(handoff["loss"])))
            if comparison is None:
                comparison = _grad_comparison(autograd["grads"], handoff["grads"])
            last_handoff = handoff

    if comparison is None or last_handoff is None:
        raise RuntimeError("repeat must be positive")
    autograd_stats = {key: summary_stats(values) for key, values in autograd_samples.items()}
    handoff_stats = {key: summary_stats(values) for key, values in handoff_samples.items()}
    autograd_total = autograd_stats["total_ms"]["mean"]
    handoff_total = handoff_stats["total_ms"]["mean"]
    pass_flag = (
        bool(last_handoff["finite"])
        and int(last_handoff["tile_stats"]["overflow_tile_count"]) == 0
        and comparison["max_abs_error"] <= 2.0e-4
        and loss_abs_error <= 1.0e-5
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate": "star_uvt_logit_handoff_rgb_vjp_profile",
        "config": str(config_path),
        "frames": int(case["feature_config"].frames),
        "size": int(case["feature_config"].height),
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "feature_dim": int(case["feature_config"].feature_dim),
        "frame_chunk_size": chunk_size,
        "tile_t": int(cfg["feature_uvt"]["tile_t"]),
        "tile_capacity": int(cfg["feature_uvt"]["tile_capacity"]),
        "alpha_threshold": float(cfg["feature_uvt"]["alpha_threshold"]),
        "resume_checkpoint": case["resume_state"]["path"],
        "resume_loaded": bool(case["resume_state"]["loaded"]),
        "resume_checkpoint_steps": case["resume_state"]["steps"],
        "baseline_backward_mode": BASELINE_BACKWARD_MODE,
        "handoff_backward_mode": mode,
        "scope": (
            "RGB reconstruction through a linear sigmoid FeatureToColor. "
            "This does not cover target-grid V-JEPA MSE or hidden64 frozen-probe VJP."
        ),
        "warmup": warmup,
        "repeat": repeat,
        "autograd_timing_ms": autograd_stats,
        "handoff_timing_ms": handoff_stats,
        "speedup_vs_autograd_total": 0.0 if handoff_total <= 0.0 else autograd_total / handoff_total,
        "loss_max_abs_error": loss_abs_error,
        "grad_comparison": comparison,
        "tile_stats": last_handoff["tile_stats"],
        "finite": bool(last_handoff["finite"]),
        "pass": pass_flag,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    auto = result["autograd_timing_ms"]
    handoff = result["handoff_timing_ms"]
    comp = result["grad_comparison"]
    tile = result["tile_stats"]
    lines = [
        "# STAR UVT Logit-Handoff RGB VJP Profile",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Scope",
        "",
        result["scope"],
        "It is the first trainer-style real-video gate for the logit-handoff reducer, but it is intentionally narrower than the current target-grid/frozen-probe objective.",
        "",
        "## Timing",
        "",
        "| path | total | render fwd | loss/VJP | renderer bwd | param bwd | colorize/loss bwd |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| autograd | "
        + " | ".join(
            [
                _fmt(auto["total_ms"]["mean"]),
                _fmt(auto["render_forward_ms"]["mean"]),
                _fmt(auto["colorize_loss_ms"]["mean"]),
                "",
                "",
                _fmt(auto["backward_ms"]["mean"]),
            ]
        )
        + " |",
        "| logit_handoff_reduce_vec4 | "
        + " | ".join(
            [
                _fmt(handoff["total_ms"]["mean"]),
                _fmt(handoff["render_forward_ms"]["mean"]),
                _fmt(handoff["logit_loss_vjp_ms"]["mean"]),
                _fmt(handoff["renderer_backward_ms"]["mean"]),
                _fmt(handoff["param_backward_ms"]["mean"]),
                "",
            ]
        )
        + " |",
        "",
        f"Total speedup versus autograd: `{result['speedup_vs_autograd_total']:.3f}x`.",
        "",
        "## Gradient Parity",
        "",
        f"- loss max abs error: `{result['loss_max_abs_error']:.3e}`",
        f"- grad max abs error: `{comp['max_abs_error']:.3e}`",
        f"- grad max rel error: `{comp['max_rel_error']:.3e}`",
        f"- missing grad names: `{comp['missing']}`",
        "",
        "## Tile State",
        "",
        f"- overflow tiles: `{tile['overflow_tile_count']}`",
        f"- unstable tiles: `{tile['unstable_tile_count']}`",
        f"- max/p95/cap: `{tile['max_tile_count']}/{tile['p95_tile_count']}/{tile['tile_capacity']}`",
        "",
        "## Decision",
        "",
        "- Passing parity means the logit-handoff reducer is viable for linear RGB-loss trainer paths.",
        "- Speed promotion still depends on the measured row; the 64f/512px checkpoint row is parity-clean but only near break-even.",
        "- It still does not solve V-JEPA target-grid feature loss or the hidden64 frozen RGB-probe objective; those need a generic image-space VJP/native loss bridge or a different probe shape.",
        "",
        f"Pass: `{result['pass']}`",
        "",
    ]
    write_report_text(path, "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", default=HANDOFF_BACKWARD_MODE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config if args.config.is_absolute() else DYNAWORLD_ROOT / args.config
    out_base = args.out_base if args.out_base.is_absolute() else DYNAWORLD_ROOT / args.out_base
    result = profile(config, warmup=args.warmup, repeat=args.repeat, mode=str(args.mode))
    write_report_json(out_base.with_suffix(".json"), result)
    write_markdown(out_base.with_suffix(".md"), result)
    print(json.dumps({"out_base": str(out_base), "pass": result["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
