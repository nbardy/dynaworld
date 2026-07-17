from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


try:
    from .report_artifacts import load_report_json, summary_stats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import load_report_json, summary_stats, write_report_json, write_report_text
from config_utils import load_config_file, path_or_none as _path_or_none
from star_uvt_colorizers import build_feature_colorizer
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    chunked_uvt_config,
    direct_atomic_feature_backward,
    render_uvt_feature_tubes,
    shift_ma_for_frame_chunk,
)
from star_uvt_checkpoints import (
    load_feature_to_rgb_probe as _load_feature_to_rgb_probe,
    load_star_training_checkpoint as _load_training_checkpoint,
)
from star_uvt_common import load_training_sequence as _load_training_sequence
from star_uvt_feature_targets import (
    _adapt_render_to_feature_target,
    _adapt_rgb_to_grid,
    _load_cached_feature_target,
)
from star_uvt_feature_losses import _feature_target_loss
from star_uvt_runtime import resolve_device as _resolve_device
from star_uvt_runtime import sync_device as _sync_device
from star_uvt_models import build_feature_tube_model
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from star_uvt_render_modes import backward_mode_for_feature_render_mode
from star_uvt_schedules import (
    _feature_target_enabled,
    _feature_target_weight_schedule,
    _feature_target_weights_for_step,
)
from star_uvt_tile_stats import _tile_load_stats
from star_uvt_feature_config import resolve_config


def _chunk_render_inputs(
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


def _load_reference_timing(cfg: dict[str, Any]) -> dict[str, Any] | None:
    path = _path_or_none(cfg["output"].get("out_json"))
    if path is None or not path.exists():
        return None
    data = load_report_json(path)
    timing = data.get("mean_timing_ms", {})
    return {
        "path": str(path),
        "pass": data.get("pass"),
        "global_steps": [data.get("start_global_step"), data.get("end_global_step")],
        "feature_loss": [data.get("start_feature_target_loss"), data.get("end_feature_target_loss")],
        "rgb_probe_psnr": [data.get("start_rgb_probe_psnr"), data.get("end_rgb_probe_psnr")],
        "mean_timing_ms": timing,
        "tile_overflow_sum": data.get("tile_overflow_sum"),
        "max_tile_count": data.get("tile_stats", {}).get("max_tile_count"),
        "p95_tile_count": data.get("tile_stats", {}).get("p95_tile_count"),
    }


def _load_case(config_path: Path) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(config_path))
    if not _feature_target_enabled(cfg):
        raise ValueError("wholegraph profile requires feature_target.enabled=true")
    if str(cfg["feature_target"]["materialization"]) != "target_grid":
        raise ValueError("wholegraph profile currently targets materialization=target_grid")

    device = _resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps":
        raise RuntimeError("STAR UVT feature wholegraph profile requires MPS")
    torch.manual_seed(int(cfg["train"]["seed"]))
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    sequence = _load_training_sequence(cfg, device)
    target_rgb = sequence.frames.contiguous()
    _sync_device(device)
    target_t0 = time.perf_counter()
    target_feature = _load_cached_feature_target(
        cfg=cfg,
        sequence_data=sequence,
        device=device,
        frames=feature_config.frames,
        height=feature_config.height,
        width=feature_config.width,
        feature_dim=feature_config.feature_dim,
    )
    _sync_device(device)
    feature_target_load_ms = (time.perf_counter() - target_t0) * 1000.0
    rgb_probe, rgb_probe_meta = _load_feature_to_rgb_probe(
        cfg,
        device=device,
        feature_dim=feature_config.feature_dim,
    )
    rgb_probe_target = None
    if rgb_probe is not None:
        rgb_probe_target = _adapt_rgb_to_grid(
            target_rgb,
            target_shape=(
                int(target_feature.source.shape[0]),
                int(target_feature.source.shape[2]),
                int(target_feature.source.shape[3]),
            ),
            mode=str(cfg["feature_target"]["rgb_probe_target_rgb_adapter"]),
        ).detach()

    model = build_feature_tube_model(cfg, feature_config, device=device)
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_config.feature_dim, device=device)
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=float(cfg["train"]["lr"]))
    resume_checkpoint = _path_or_none(cfg["train"].get("resume_checkpoint"))
    resume_state = {"path": None, "loaded": False, "optimizer_loaded": False, "steps": None}
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
        "feature_config": feature_config,
        "uvt_config": uvt_config,
        "target_rgb": target_rgb,
        "target_feature": target_feature,
        "feature_target_load_ms": feature_target_load_ms,
        "rgb_probe": rgb_probe,
        "rgb_probe_meta": rgb_probe_meta,
        "rgb_probe_target": rgb_probe_target,
        "model": model,
        "device": device,
        "resume_state": resume_state,
        "reference": _load_reference_timing(cfg),
    }


def profile_config(config_path: Path, *, warmup: int, repeat: int, global_step: int | None) -> dict[str, Any]:
    case = _load_case(config_path)
    cfg = case["cfg"]
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    target_rgb = case["target_rgb"]
    target_feature = case["target_feature"]
    rgb_probe = case["rgb_probe"]
    rgb_probe_target = case["rgb_probe_target"]
    model = case["model"]
    device = case["device"]
    feature_dim = feature_config.feature_dim
    render_mode = str(cfg["feature_uvt"]["render_mode"])
    backward_mode = backward_mode_for_feature_render_mode(render_mode, feature_dim)
    chunk_size = cfg["train"]["frame_chunk_size"]
    chunk_size = feature_config.frames if chunk_size is None else min(int(chunk_size), feature_config.frames)
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")

    selected_global_step = int(cfg["train"]["global_step_offset"] if global_step is None else global_step)
    stage = _feature_target_weights_for_step(_feature_target_weight_schedule(cfg), selected_global_step)
    feature_loss_weight = float(stage.loss_weight)
    rgb_probe_loss_weight = float(stage.rgb_probe_loss_weight)
    if rgb_probe_loss_weight > 0.0 and (rgb_probe is None or rgb_probe_target is None):
        raise ValueError("rgb_probe_loss_weight > 0 requires a loaded rgb_probe and target")

    total_feature_loss_elems = int(target_feature.numel)
    total_rgb_probe_loss_elems = 0 if rgb_probe_target is None else int(rgb_probe_target.numel())
    feature_loss_type = str(cfg["feature_target"]["loss_type"])
    samples = {
        "render_forward_ms": [],
        "target_grid_prep_ms": [],
        "feature_loss_forward_ms": [],
        "rgb_probe_loss_forward_ms": [],
        "image_loss_backward_ms": [],
        "renderer_backward_ms": [],
        "manual_total_ms": [],
        "weighted_loss": [],
    }
    finite = True
    alpha_grad_missing_count = 0
    tile_counts: list[torch.Tensor] = []
    tile_overflow: list[torch.Tensor] = []
    tile_unstable: list[torch.Tensor] = []
    grad_norm_accum = {
        "ma": 0.0,
        "q_uvt": 0.0,
        "opacity": 0.0,
        "feature": 0.0,
        "feature_image": 0.0,
        "alpha": 0.0,
    }

    phase_keys = tuple(key for key in samples if key.endswith("_ms") and key != "manual_total_ms")
    for iteration in range(warmup + repeat):
        phase = {key: 0.0 for key in phase_keys}
        weighted_loss_value = 0.0
        tile_counts_iter: list[torch.Tensor] = []
        tile_overflow_iter: list[torch.Tensor] = []
        tile_unstable_iter: list[torch.Tensor] = []
        for frame_start in range(0, feature_config.frames, chunk_size):
            chunk_frames = min(chunk_size, feature_config.frames - frame_start)
            render_inputs, chunk_config = _chunk_render_inputs(model, uvt_config, frame_start, chunk_frames)

            _sync_device(device)
            t0 = time.perf_counter()
            render = render_uvt_feature_tubes(*render_inputs, chunk_config)
            _sync_device(device)
            t1 = time.perf_counter()

            feature_probe = render.feature_image.detach().requires_grad_(True)
            alpha_probe = render.alpha.detach().requires_grad_(True)
            target_t0 = time.perf_counter()
            target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
            rendered_target_grid = _adapt_render_to_feature_target(
                feature_probe,
                target_shape=tuple(int(item) for item in target_feature_chunk.shape),
                mode=target_feature.grid_mode,
            )
            _sync_device(device)
            target_t1 = time.perf_counter()

            loss = feature_probe.new_zeros(())
            feature_loss_t0 = time.perf_counter()
            if feature_loss_weight > 0.0:
                feature_loss = _feature_target_loss(
                    rendered_target_grid,
                    target_feature_chunk,
                    feature_loss_type,
                ) / float(total_feature_loss_elems)
                loss = loss + feature_loss_weight * feature_loss
            _sync_device(device)
            feature_loss_t1 = time.perf_counter()

            probe_t0 = time.perf_counter()
            if rgb_probe_loss_weight > 0.0:
                target_start = frame_start * int(rgb_probe_target.shape[0]) // feature_config.frames
                target_end = (frame_start + chunk_frames) * int(rgb_probe_target.shape[0]) // feature_config.frames
                target_rgb_probe_chunk = rgb_probe_target[target_start:target_end]
                rgb_probe_pred = rgb_probe(rendered_target_grid)
                rgb_probe_loss = (
                    (rgb_probe_pred - target_rgb_probe_chunk).square().sum()
                    / float(total_rgb_probe_loss_elems)
                )
                loss = loss + rgb_probe_loss_weight * rgb_probe_loss
            _sync_device(device)
            probe_t1 = time.perf_counter()

            weighted_loss_value += float(loss.detach().cpu().item())
            _sync_device(device)
            bwd_t0 = time.perf_counter()
            loss.backward()
            _sync_device(device)
            bwd_t1 = time.perf_counter()

            grad_feature_image = feature_probe.grad
            if grad_feature_image is None:
                raise RuntimeError("feature/probe loss did not produce feature-image gradients")
            grad_alpha = alpha_probe.grad
            if grad_alpha is None:
                alpha_grad_missing_count += 1
                grad_alpha = torch.zeros_like(alpha_probe)

            grads = direct_atomic_feature_backward(
                *render_inputs,
                grad_feature_image.contiguous(),
                grad_alpha.contiguous(),
                chunk_config,
                backward_mode=backward_mode,
            )
            _sync_device(device)
            bwd_t2 = time.perf_counter()

            phase["render_forward_ms"] += (t1 - t0) * 1000.0
            phase["target_grid_prep_ms"] += (target_t1 - target_t0) * 1000.0
            phase["feature_loss_forward_ms"] += (feature_loss_t1 - feature_loss_t0) * 1000.0
            phase["rgb_probe_loss_forward_ms"] += (probe_t1 - probe_t0) * 1000.0
            phase["image_loss_backward_ms"] += (bwd_t1 - bwd_t0) * 1000.0
            phase["renderer_backward_ms"] += (bwd_t2 - bwd_t1) * 1000.0

            finite = (
                finite
                and bool(torch.isfinite(render.feature_image).all().cpu())
                and bool(torch.isfinite(grad_feature_image).all().cpu())
                and bool(torch.isfinite(grad_alpha).all().cpu())
                and all(bool(torch.isfinite(grad).all().cpu()) for grad in grads[:4])
            )
            if iteration >= warmup:
                grad_norm_accum["ma"] += float(grads[0].norm().detach().cpu().item())
                grad_norm_accum["q_uvt"] += float(grads[1].norm().detach().cpu().item())
                grad_norm_accum["opacity"] += float(grads[2].norm().detach().cpu().item())
                grad_norm_accum["feature"] += float(grads[3].norm().detach().cpu().item())
                grad_norm_accum["feature_image"] += float(grad_feature_image.norm().detach().cpu().item())
                grad_norm_accum["alpha"] += float(grad_alpha.norm().detach().cpu().item())
            tile_counts_iter.append(render.tile_counts)
            tile_overflow_iter.append(render.tile_overflow)
            tile_unstable_iter.append(grads[-1])
        if iteration >= warmup:
            for key, value in phase.items():
                samples[key].append(value)
            samples["manual_total_ms"].append(sum(phase.values()))
            samples["weighted_loss"].append(weighted_loss_value)
            tile_counts = tile_counts_iter
            tile_overflow = tile_overflow_iter
            tile_unstable = tile_unstable_iter

    phase_stats = {key: summary_stats(values) for key, values in samples.items()}
    mean_timing = {key: value["mean"] for key, value in phase_stats.items() if key.endswith("_ms")}
    loss_forward = (
        mean_timing["target_grid_prep_ms"]
        + mean_timing["feature_loss_forward_ms"]
        + mean_timing["rgb_probe_loss_forward_ms"]
    )
    backward_total = mean_timing["image_loss_backward_ms"] + mean_timing["renderer_backward_ms"]
    manual_total = mean_timing["manual_total_ms"]
    tile_stats = _tile_load_stats(
        tile_counts=tile_counts,
        tile_overflow=tile_overflow,
        tile_unstable=tile_unstable,
        tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
    )
    grad_sample_count = max(repeat * ((feature_config.frames + chunk_size - 1) // chunk_size), 1)
    return {
        "config": str(config_path),
        "reference": case["reference"],
        "resume_checkpoint": case["resume_state"]["path"],
        "resume_loaded": bool(case["resume_state"]["loaded"]),
        "resume_checkpoint_steps": case["resume_state"]["steps"],
        "global_step": selected_global_step,
        "feature_loss_weight": feature_loss_weight,
        "rgb_probe_loss_weight": rgb_probe_loss_weight,
        "render_mode": render_mode,
        "backward_mode": backward_mode,
        "frames": feature_config.frames,
        "size": feature_config.height,
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "feature_dim": feature_dim,
        "frame_chunk_size": chunk_size,
        "tile_t": int(cfg["feature_uvt"]["tile_t"]),
        "tile_capacity": int(cfg["feature_uvt"]["tile_capacity"]),
        "feature_target_load_ms": case["feature_target_load_ms"],
        "feature_target": case["target_feature"].meta,
        "rgb_probe": case["rgb_probe_meta"],
        "warmup": warmup,
        "repeat": repeat,
        "phase_stats": phase_stats,
        "mean_timing_ms": {
            **mean_timing,
            "loss_forward_total_ms": loss_forward,
            "backward_total_ms": backward_total,
        },
        "share": {
            "render_forward_of_manual_total": 0.0 if manual_total <= 0.0 else mean_timing["render_forward_ms"] / manual_total,
            "loss_forward_of_manual_total": 0.0 if manual_total <= 0.0 else loss_forward / manual_total,
            "image_loss_backward_of_backward": 0.0 if backward_total <= 0.0 else mean_timing["image_loss_backward_ms"] / backward_total,
            "renderer_backward_of_backward": 0.0 if backward_total <= 0.0 else mean_timing["renderer_backward_ms"] / backward_total,
            "renderer_backward_of_manual_total": 0.0 if manual_total <= 0.0 else mean_timing["renderer_backward_ms"] / manual_total,
        },
        "mean_grad_norms": {
            key: value / float(grad_sample_count)
            for key, value in grad_norm_accum.items()
        },
        "alpha_grad_missing_count": alpha_grad_missing_count,
        "tile_stats": tile_stats,
        "tile_overflow_sum": int(tile_stats["overflow_tile_count"]),
        "finite": finite,
        "pass": finite and int(tile_stats["overflow_tile_count"]) == 0 and bool(case["resume_state"]["loaded"]),
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = result["rows"]
    lines = [
        "# STAR UVT Feature1 Whole-Graph Profile",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "This diagnostic profiles the current target-grid plus frozen RGB-probe objective by",
        "detaching the rendered feature image, backpropagating only the image-space",
        "feature/probe losses, then calling the STAR UVT Metal feature backward manually.",
        "It excludes optimizer time and checkpoint/media work, so compare it to trainer",
        "timings as a split diagnostic rather than an end-to-end replacement.",
        "",
        "## Rows",
        "",
        "| config | global step | trainer step ms | manual total | render fwd | target prep | feature loss fwd | probe fwd | image-loss bwd | renderer bwd | renderer share of bwd | overflow | max/p95/cap | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        timing = row["mean_timing_ms"]
        ref_timing = (row.get("reference") or {}).get("mean_timing_ms", {})
        tile = row["tile_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(row["config"]).name,
                    str(row["global_step"]),
                    _fmt(ref_timing.get("step_ms"), 1),
                    _fmt(timing["manual_total_ms"], 1),
                    _fmt(timing["render_forward_ms"], 1),
                    _fmt(timing["target_grid_prep_ms"], 1),
                    _fmt(timing["feature_loss_forward_ms"], 1),
                    _fmt(timing["rgb_probe_loss_forward_ms"], 1),
                    _fmt(timing["image_loss_backward_ms"], 1),
                    _fmt(timing["renderer_backward_ms"], 1),
                    f"{100.0 * row['share']['renderer_backward_of_backward']:.1f}%",
                    str(row["tile_overflow_sum"]),
                    f"{tile['max_tile_count']}/{tile['p95_tile_count']:.0f}/{row['tile_capacity']}",
                    "yes" if row["pass"] else "no",
                ]
            )
            + " |"
        )

    if len(rows) >= 2:
        first = rows[0]
        last = rows[-1]
        first_timing = first["mean_timing_ms"]
        last_timing = last["mean_timing_ms"]
        lines.extend(
            [
                "",
                "## Delta",
                "",
                (
                    f"Last minus first manual total: "
                    f"{last_timing['manual_total_ms'] - first_timing['manual_total_ms']:.1f}ms. "
                    f"Render delta: {last_timing['render_forward_ms'] - first_timing['render_forward_ms']:.1f}ms. "
                    f"Image-loss backward delta: "
                    f"{last_timing['image_loss_backward_ms'] - first_timing['image_loss_backward_ms']:.1f}ms. "
                    f"Renderer-backward delta: "
                    f"{last_timing['renderer_backward_ms'] - first_timing['renderer_backward_ms']:.1f}ms."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `target prep` is target-grid slicing plus render-to-target-grid interpolation.",
            "- `image-loss bwd` is backward through target-grid MSE and the frozen RGB probe to the rendered feature image.",
            "- `renderer bwd` is the Metal STAR UVT feature backward using those image gradients.",
            "- A missing alpha gradient is expected for this objective because the target-grid/probe losses consume the composited feature image, not the separate alpha output.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--global-step", type=int, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be nonnegative and repeat must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("STAR UVT feature wholegraph profile requires MPS")

    rows = [
        profile_config(
            config_path,
            warmup=args.warmup,
            repeat=args.repeat,
            global_step=args.global_step,
        )
        for config_path in args.config
    ]
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "gate": "star_uvt_feature1_wholegraph_profile",
        "rows": rows,
        "pass": all(bool(row["pass"]) for row in rows),
    }
    write_report_json(args.out_json, result)
    if args.out_md is not None:
        _write_markdown(args.out_md, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
