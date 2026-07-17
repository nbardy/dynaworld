from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


from config_utils import load_config_file, path_or_none as _path_or_none
try:
    from .report_artifacts import ROOT as DYNAWORLD_ROOT, summary_stats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT as DYNAWORLD_ROOT, summary_stats, write_report_json, write_report_text
from star_uvt_checkpoints import load_star_training_checkpoint as _load_training_checkpoint
from star_uvt_colorizers import build_feature_colorizer
from star_uvt_common import (
    load_colorizer_init_checkpoint as _load_colorizer_init_checkpoint,
    load_training_sequence as _load_training_sequence,
)
from star_uvt_runtime import resolve_device as _resolve_device, sync_device as _sync_device
from star_uvt_models import build_feature_tube_model
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    chunked_uvt_config,
    render_uvt_feature_sparse_pixels_with_bins,
    shift_ma_for_frame_chunk,
)
from star_uvt_sparse_visual_sampling import (
    _sparse_visual_local_frame_ids_for_chunk,
    _sparse_visual_loss_sample_count,
    _sparse_visual_patch_phase_for_step,
    _sparse_visual_pixel_ids_for_chunk,
)
from star_uvt_sparse_visual_losses import (
    _gelu_grad_for_mode,
    _hidden64_vjp_options,
    _hidden64_colorizer_layers,
    _linear_colorizer_layer,
    _sparse_visual_loss_and_grad_pred_values,
)
from star_uvt_feature_config import resolve_config


DEFAULT_CONFIG = (
    DYNAWORLD_ROOT
    / "src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_manualvjp_from1500_lr001_5step_media.jsonc"
)


def _add_time(timings: dict[str, float], key: str, start: float, end: float) -> None:
    timings[key] = timings.get(key, 0.0) + (end - start) * 1000.0


def _make_case(config_path: Path) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(config_path))
    device = _resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps":
        raise RuntimeError("sparse visual loss VJP profile requires MPS")
    torch.manual_seed(int(cfg["train"]["seed"]))
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    sequence = _load_training_sequence(cfg, device)
    model = build_feature_tube_model(cfg, feature_config, device=device)
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_config.feature_dim, device=device)
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
            resume_colorizer=bool(cfg["train"]["resume_colorizer"]),
        )
    colorizer_init_checkpoint = _path_or_none(cfg["colorize"].get("init_checkpoint"))
    colorizer_init_state: dict[str, Any] = {"path": None, "loaded": False}
    if colorizer_init_checkpoint is not None:
        colorizer_init_state = _load_colorizer_init_checkpoint(
            colorizer_init_checkpoint,
            colorizer=colorizer,
            device=device,
        )
    vjp_mode = str(cfg["sparse_visual"].get("loss_vjp_mode"))
    if vjp_mode == "manual_linear":
        _linear_colorizer_layer(colorizer)
    else:
        _hidden64_colorizer_layers(colorizer)
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
        "colorizer_init_state": colorizer_init_state,
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


def _expected_loss_elems(case: dict[str, Any], *, chunk_size: int) -> int:
    cfg = case["cfg"]
    sparse_visual = cfg["sparse_visual"]
    feature_config = case["feature_config"]
    device = case["device"]
    total_samples = 0
    patch_phase = _sparse_visual_patch_phase_for_step(
        pixel_source=str(sparse_visual["pixel_source"]),
        global_step=int(cfg["train"].get("global_step_offset", 0)),
        patch_phase_shape=tuple(int(item) for item in sparse_visual.get("patch_phase_shape", (1, 1))),
    )
    for frame_start in range(0, feature_config.frames, chunk_size):
        chunk_frames = min(chunk_size, feature_config.frames - frame_start)
        pixel_ids = _sparse_visual_pixel_ids_for_chunk(
            pixel_source=str(sparse_visual["pixel_source"]),
            chunk_frames=chunk_frames,
            height=feature_config.height,
            width=feature_config.width,
            render_frames=feature_config.frames,
            frame_start=frame_start,
            sample_grid_shape=tuple(int(item) for item in sparse_visual["sample_grid_shape"]),
            patch_shape=tuple(int(item) for item in sparse_visual.get("patch_shape", (1, 1))),
            patch_phase=patch_phase,
            patch_phase_shape=tuple(int(item) for item in sparse_visual.get("patch_phase_shape", (1, 1))),
            device=device,
        )
        total_samples += _sparse_visual_loss_sample_count(
            int(pixel_ids.numel()),
            loss_basis=str(sparse_visual.get("loss_basis", "pixel")),
            patch_shape=tuple(int(item) for item in sparse_visual.get("patch_shape", (1, 1))),
        )
    return max(total_samples * 3, 1)


def _profile_chunk(
    case: dict[str, Any],
    *,
    frame_start: int,
    chunk_frames: int,
    total_loss_elems: int,
    patch_phase: tuple[int, int],
) -> dict[str, Any]:
    cfg = case["cfg"]
    device = case["device"]
    model = case["model"]
    colorizer: nn.Module = case["colorizer"]
    target_rgb = case["target_rgb"]
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    sparse_visual = cfg["sparse_visual"]
    sample_grid_shape = tuple(int(item) for item in sparse_visual["sample_grid_shape"])
    patch_shape = tuple(int(item) for item in sparse_visual.get("patch_shape", (1, 1)))
    patch_phase_shape = tuple(int(item) for item in sparse_visual.get("patch_phase_shape", (1, 1)))
    timings: dict[str, float] = {}
    _sync_device(device)
    t0 = time.perf_counter()
    pixel_ids = _sparse_visual_pixel_ids_for_chunk(
        pixel_source=str(sparse_visual["pixel_source"]),
        chunk_frames=chunk_frames,
        height=feature_config.height,
        width=feature_config.width,
        render_frames=feature_config.frames,
        frame_start=frame_start,
        sample_grid_shape=sample_grid_shape,
        patch_shape=patch_shape,
        patch_phase=patch_phase,
        patch_phase_shape=patch_phase_shape,
        device=device,
    )
    local_frame_ids = _sparse_visual_local_frame_ids_for_chunk(
        render_frames=feature_config.frames,
        frame_start=frame_start,
        chunk_frames=chunk_frames,
        sample_grid_shape=sample_grid_shape,
        device=device,
    )
    _sync_device(device)
    t1 = time.perf_counter()
    _add_time(timings, "pixel_id_ms", t0, t1)
    render_inputs, render_config = _chunk_inputs(model, uvt_config, frame_start, chunk_frames)
    render = render_uvt_feature_sparse_pixels_with_bins(*render_inputs, pixel_ids, render_config)
    feature_values = render.feature_values.detach()
    alpha_values = render.alpha_values.detach()
    _sync_device(device)
    t2 = time.perf_counter()
    _add_time(timings, "sparse_render_ms", t1, t2)
    vjp_mode = str(sparse_visual.get("loss_vjp_mode"))
    if vjp_mode == "manual_linear":
        conv = _linear_colorizer_layer(colorizer)
        x = feature_values
        alpha = alpha_values
        weight = conv.weight[:, :, 0, 0]
        bias = conv.bias
        logits = x.matmul(weight.t()) + bias
        _sync_device(device)
        t3 = time.perf_counter()
        _add_time(timings, "linear_fc_ms", t2, t3)
        rgb = torch.sigmoid(logits) if colorizer.activation == "sigmoid" else logits
        pred_values = alpha.unsqueeze(1).to(dtype=rgb.dtype) * rgb
        _sync_device(device)
        t4 = time.perf_counter()
        _add_time(timings, "activation_alpha_ms", t3, t4)
        target_rgb_chunk = target_rgb[frame_start : frame_start + chunk_frames]
        loss, grad_pred = _sparse_visual_loss_and_grad_pred_values(
            pred_values,
            None,
            total_loss_elems=total_loss_elems,
            loss_weight=float(sparse_visual["loss_weight"]),
            loss_basis=str(sparse_visual["loss_basis"]),
            sample_grid_shape=sample_grid_shape,
            patch_shape=patch_shape,
            target_rgb_chunk=target_rgb_chunk,
            local_frame_ids=local_frame_ids,
        )
        _sync_device(device)
        t5 = time.perf_counter()
        _add_time(timings, "target_area_loss_grad_pred_ms", t4, t5)
        grad_rgb = grad_pred * alpha.unsqueeze(1).to(dtype=grad_pred.dtype)
        grad_alpha = (grad_pred * rgb).sum(dim=1)
        grad_logits = grad_rgb * (rgb * (1.0 - rgb)) if colorizer.activation == "sigmoid" else grad_rgb
        _sync_device(device)
        t6 = time.perf_counter()
        _add_time(timings, "rgb_alpha_logit_grad_ms", t5, t6)
        grad_weight = grad_logits.t().matmul(x)
        grad_bias = grad_logits.sum(dim=0)
        grad_feature = grad_logits.matmul(weight)
        _sync_device(device)
        t7 = time.perf_counter()
        _add_time(timings, "linear_param_feature_grad_ms", t6, t7)
        finite = all(
            bool(torch.isfinite(tensor).all().cpu())
            for tensor in (
                feature_values,
                alpha_values,
                pred_values,
                grad_pred,
                grad_logits,
                grad_alpha,
                grad_feature,
                grad_weight,
                grad_bias,
            )
        )
        timings["loss_vjp_ms"] = sum(
            value for key, value in timings.items() if key not in {"pixel_id_ms", "sparse_render_ms"}
        )
        timings["total_profiled_ms"] = timings["pixel_id_ms"] + timings["sparse_render_ms"] + timings["loss_vjp_ms"]
        return {
            "frame_start": frame_start,
            "chunk_frames": chunk_frames,
            "pixel_count": int(pixel_ids.numel()),
            "loss_sample_count": _sparse_visual_loss_sample_count(
                int(pixel_ids.numel()),
                loss_basis=str(sparse_visual["loss_basis"]),
                patch_shape=patch_shape,
            ),
            "loss": float(loss.detach().cpu().item()),
            "finite": finite,
            "timing_ms": timings,
            "feature_shape": list(feature_values.shape),
            "hidden_shape": None,
        }

    conv1, _gelu, conv2 = _hidden64_colorizer_layers(colorizer)
    accumulate_colorizer_grads, gelu_grad_mode = _hidden64_vjp_options(str(sparse_visual.get("loss_vjp_mode")))
    x = feature_values
    alpha = alpha_values
    w1 = conv1.weight[:, :, 0, 0]
    b1 = conv1.bias
    w2 = conv2.weight[:, :, 0, 0]
    b2 = conv2.bias
    hidden_pre = x.matmul(w1.t()) + b1
    _sync_device(device)
    t3 = time.perf_counter()
    _add_time(timings, "fc1_ms", t2, t3)
    hidden = F.gelu(hidden_pre)
    _sync_device(device)
    t4 = time.perf_counter()
    _add_time(timings, "gelu_ms", t3, t4)
    logits = hidden.matmul(w2.t()) + b2
    _sync_device(device)
    t5 = time.perf_counter()
    _add_time(timings, "fc2_ms", t4, t5)
    rgb = torch.sigmoid(logits) if colorizer.activation == "sigmoid" else logits
    pred_values = alpha.unsqueeze(1).to(dtype=rgb.dtype) * rgb
    _sync_device(device)
    t6 = time.perf_counter()
    _add_time(timings, "activation_alpha_ms", t5, t6)
    target_rgb_chunk = target_rgb[frame_start : frame_start + chunk_frames]
    loss, grad_pred = _sparse_visual_loss_and_grad_pred_values(
        pred_values,
        None,
        total_loss_elems=total_loss_elems,
        loss_weight=float(sparse_visual["loss_weight"]),
        loss_basis=str(sparse_visual["loss_basis"]),
        sample_grid_shape=sample_grid_shape,
        patch_shape=patch_shape,
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=local_frame_ids,
    )
    _sync_device(device)
    t7 = time.perf_counter()
    _add_time(timings, "target_area_loss_grad_pred_ms", t6, t7)
    grad_rgb = grad_pred * alpha.unsqueeze(1).to(dtype=grad_pred.dtype)
    grad_alpha = (grad_pred * rgb).sum(dim=1)
    grad_logits = grad_rgb * (rgb * (1.0 - rgb)) if colorizer.activation == "sigmoid" else grad_rgb
    _sync_device(device)
    t8 = time.perf_counter()
    _add_time(timings, "rgb_alpha_logit_grad_ms", t7, t8)
    if accumulate_colorizer_grads:
        grad_w2 = grad_logits.t().matmul(hidden)
        grad_b2 = grad_logits.sum(dim=0)
    else:
        grad_w2 = torch.empty((0,), device=device)
        grad_b2 = torch.empty((0,), device=device)
    grad_hidden = grad_logits.matmul(w2)
    _sync_device(device)
    t9 = time.perf_counter()
    _add_time(timings, "conv2_param_hidden_grad_ms", t8, t9)
    grad_hidden_pre = grad_hidden * _gelu_grad_for_mode(hidden_pre, gelu_grad_mode)
    _sync_device(device)
    t10 = time.perf_counter()
    _add_time(timings, "gelu_back_ms", t9, t10)
    if accumulate_colorizer_grads:
        grad_w1 = grad_hidden_pre.t().matmul(x)
        grad_b1 = grad_hidden_pre.sum(dim=0)
    else:
        grad_w1 = torch.empty((0,), device=device)
        grad_b1 = torch.empty((0,), device=device)
    _sync_device(device)
    t11 = time.perf_counter()
    _add_time(timings, "conv1_param_grad_ms", t10, t11)
    grad_feature = grad_hidden_pre.matmul(w1)
    _sync_device(device)
    t12 = time.perf_counter()
    _add_time(timings, "conv1_feature_grad_ms", t11, t12)
    finite = all(
        bool(torch.isfinite(tensor).all().cpu())
        for tensor in (
            feature_values,
            alpha_values,
            pred_values,
            grad_pred,
            grad_logits,
            grad_alpha,
            grad_feature,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
        )
    )
    timings["loss_vjp_ms"] = sum(value for key, value in timings.items() if key not in {"pixel_id_ms", "sparse_render_ms"})
    timings["total_profiled_ms"] = timings["pixel_id_ms"] + timings["sparse_render_ms"] + timings["loss_vjp_ms"]
    return {
        "frame_start": frame_start,
        "chunk_frames": chunk_frames,
        "pixel_count": int(pixel_ids.numel()),
        "loss_sample_count": _sparse_visual_loss_sample_count(
            int(pixel_ids.numel()),
            loss_basis=str(sparse_visual["loss_basis"]),
            patch_shape=patch_shape,
        ),
        "loss": float(loss.detach().cpu().item()),
        "finite": finite,
        "timing_ms": timings,
        "feature_shape": list(feature_values.shape),
        "hidden_shape": list(hidden.shape),
    }


def profile(config_path: Path, *, max_chunks: int | None, warmup: int, repeat: int) -> dict[str, Any]:
    case = _make_case(config_path)
    cfg = case["cfg"]
    sparse_visual = cfg["sparse_visual"]
    if not str(sparse_visual.get("loss_vjp_mode")).startswith("manual_hidden") and str(
        sparse_visual.get("loss_vjp_mode")
    ) != "manual_linear":
        raise ValueError("profile target should use a manual sparse_visual.loss_vjp_mode")
    chunk_size = int(cfg["train"]["frame_chunk_size"])
    feature_config = case["feature_config"]
    chunk_starts = list(range(0, feature_config.frames, chunk_size))
    if max_chunks is not None:
        chunk_starts = chunk_starts[: int(max_chunks)]
    total_loss_elems = _expected_loss_elems(case, chunk_size=chunk_size)
    patch_phase = _sparse_visual_patch_phase_for_step(
        pixel_source=str(sparse_visual["pixel_source"]),
        global_step=int(cfg["train"].get("global_step_offset", 0)),
        patch_phase_shape=tuple(int(item) for item in sparse_visual.get("patch_phase_shape", (1, 1))),
    )
    repeated_rows: list[dict[str, Any]] = []
    for iteration in range(warmup + repeat):
        rows = []
        for frame_start in chunk_starts:
            rows.append(
                _profile_chunk(
                    case,
                    frame_start=frame_start,
                    chunk_frames=min(chunk_size, feature_config.frames - frame_start),
                    total_loss_elems=total_loss_elems,
                    patch_phase=patch_phase,
                )
            )
        if iteration >= warmup:
            repeated_rows.extend(rows)
    if not repeated_rows:
        raise RuntimeError("repeat must be positive")
    timing_keys = sorted({key for row in repeated_rows for key in row["timing_ms"]})
    timing_stats = {
        key: summary_stats([float(row["timing_ms"].get(key, 0.0)) for row in repeated_rows])
        for key in timing_keys
    }
    per_sample = {
        key: float(stats["mean"]) / max(float(repeated_rows[0]["pixel_count"]), 1.0)
        for key, stats in timing_stats.items()
    }
    chunks_per_step = int((feature_config.frames + chunk_size - 1) // chunk_size)
    profiled_chunks = len(chunk_starts)
    extrapolated_step_ms = {
        key: float(stats["mean"]) * float(chunks_per_step)
        for key, stats in timing_stats.items()
    }
    return {
        "config_path": str(config_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frames": feature_config.frames,
        "size": feature_config.height,
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "feature_dim": feature_config.feature_dim,
        "chunk_size": chunk_size,
        "profiled_chunks": profiled_chunks,
        "chunks_per_step": chunks_per_step,
        "repeat": repeat,
        "warmup": warmup,
        "total_loss_elems": total_loss_elems,
        "sparse_visual": {
            "pixel_source": str(sparse_visual["pixel_source"]),
            "loss_basis": str(sparse_visual["loss_basis"]),
            "loss_vjp_mode": str(sparse_visual["loss_vjp_mode"]),
            "sample_grid_shape": list(sparse_visual["sample_grid_shape"]),
            "patch_shape": list(sparse_visual.get("patch_shape", (1, 1))),
            "patch_phase": list(patch_phase),
            "accumulate_colorizer_grads": (
                None
                if str(sparse_visual.get("loss_vjp_mode")) == "manual_linear"
                else _hidden64_vjp_options(str(sparse_visual.get("loss_vjp_mode")))[0]
            ),
            "gelu_grad_mode": (
                None
                if str(sparse_visual.get("loss_vjp_mode")) == "manual_linear"
                else _hidden64_vjp_options(str(sparse_visual.get("loss_vjp_mode")))[1]
            ),
        },
        "rows": repeated_rows,
        "timing_stats_ms": timing_stats,
        "timing_per_pixel_ms": per_sample,
        "extrapolated_step_ms": extrapolated_step_ms,
        "finite": all(bool(row["finite"]) for row in repeated_rows),
    }


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    timing = payload["timing_stats_ms"]
    extrapolated = payload["extrapolated_step_ms"]
    vjp_mode = str(payload["sparse_visual"]["loss_vjp_mode"])
    loss_vjp_mean = float(timing["loss_vjp_ms"]["mean"])
    preferred_phase_keys = [
        "fc1_ms",
        "linear_fc_ms",
        "gelu_ms",
        "fc2_ms",
        "activation_alpha_ms",
        "target_area_loss_grad_pred_ms",
        "rgb_alpha_logit_grad_ms",
        "conv2_param_hidden_grad_ms",
        "gelu_back_ms",
        "conv1_param_grad_ms",
        "conv1_feature_grad_ms",
        "linear_param_feature_grad_ms",
    ]
    phase_keys = [key for key in preferred_phase_keys if key in timing]
    lines = [
        "# STAR UVT Sparse Visual Loss VJP Profile",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        f"Config: `{payload['config_path']}`",
        "",
        f"Profiled `{payload['profiled_chunks']}` of `{payload['chunks_per_step']}` chunks, repeat `{payload['repeat']}` after warmup `{payload['warmup']}`.",
        "",
        "| Phase | Mean/chunk ms | Share of loss VJP | Extrapolated full-step ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in phase_keys:
        mean = float(timing[key]["mean"])
        share = 0.0 if loss_vjp_mean <= 0.0 else 100.0 * mean / loss_vjp_mean
        lines.append(f"| `{key}` | {_fmt(mean)} | {share:.1f}% | {_fmt(float(extrapolated[key]))} |")
    lines.extend(
        [
            "",
            "| Aggregate | Mean/chunk ms | Extrapolated full-step ms |",
            "| --- | ---: | ---: |",
            f"| Sparse render | {_fmt(float(timing['sparse_render_ms']['mean']))} | {_fmt(float(extrapolated['sparse_render_ms']))} |",
            f"| Loss VJP | {_fmt(float(timing['loss_vjp_ms']['mean']))} | {_fmt(float(extrapolated['loss_vjp_ms']))} |",
            f"| Pixel id build | {_fmt(float(timing['pixel_id_ms']['mean']))} | {_fmt(float(extrapolated['pixel_id_ms']))} |",
            "",
            "## Read",
            "",
            f"This profiler isolates the `{vjp_mode}` sparse visual loss VJP before",
            "the sparse-pixel Metal backward. It times the current Python/Torch",
            "loss-side work that a native fused RGB/loss/gradient or visibility/prefix",
            "path should replace.",
            "",
            "The next implementation gate should target the largest loss-VJP phases first",
            "and avoid materializing full per-pixel hidden/colorizer intermediates in",
            "Python when dense visual support is used.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out-base", type=Path, default=None)
    args = parser.parse_args()
    out_base = args.out_base or Path("outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile")
    payload = profile(args.config, max_chunks=args.max_chunks, warmup=args.warmup, repeat=args.repeat)
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    write_report_json(json_path, payload)
    write_markdown(payload, md_path)
    print(json.dumps({"out_json": str(json_path), "out_md": str(md_path), "profiled_chunks": payload["profiled_chunks"]}))


if __name__ == "__main__":
    main()
