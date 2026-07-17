from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import split_csv_strings, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import split_csv_strings, write_report_json, write_report_text
from config_utils import load_config_file
from research_experiments.star_uvt_feature_tubes.dense_feature_tube_prototype import (
    colorize_and_compose,
)
from research_project.trainer_harness.data import load_video_target
from star_uvt_colorizers import build_feature_colorizer
from star_uvt_models import build_feature_tube_model
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    chunked_uvt_config,
    direct_atomic_feature_backward,
    render_uvt_feature_tubes,
    shift_ma_for_frame_chunk,
)
from star_uvt_render_modes import backward_mode_for_feature_render_mode
from star_uvt_runtime import resolve_device as _resolve_device, sync_device as _sync_device
from star_uvt_tile_stats import _tile_load_stats
from star_uvt_feature_config import resolve_config


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _load_case(
    config_path: Path,
    *,
    colorize_pre_norm: bool | None,
) -> tuple[dict[str, Any], Any, UVTRenderConfig, torch.Tensor, Any, Any, torch.device]:
    cfg = resolve_config(load_config_file(config_path))
    if colorize_pre_norm is not None:
        cfg["colorize"]["pre_norm"] = bool(colorize_pre_norm)
    device = _resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps":
        raise RuntimeError("STAR UVT feature backward breakdown currently requires MPS")
    torch.manual_seed(int(cfg["train"]["seed"]))
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    target_thwc = load_video_target(
        Path(cfg["data"]["video_path"]),
        target_size=feature_config.height,
        max_frames=feature_config.frames,
        device=device,
        start_seconds=cfg["data"]["start_seconds"],
        fps=cfg["data"]["fps"],
        duration_seconds=cfg["data"]["duration_seconds"],
        image_crop_mode=str(cfg["data"]["image_crop_mode"]),
    )
    target_rgb = target_thwc.permute(0, 3, 1, 2).contiguous()
    model = build_feature_tube_model(cfg, feature_config, device=device)
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_config.feature_dim, device=device)
    return cfg, feature_config, uvt_config, target_rgb, model, colorizer, device


def _chunk_render_inputs(model: Any, uvt_config: UVTRenderConfig, frame_start: int, chunk_frames: int) -> tuple[Any, UVTRenderConfig]:
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


def run_breakdown(
    config_path: Path,
    *,
    mode: str,
    colorize_pre_norm: bool | None,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    cfg, feature_config, uvt_config, target_rgb, model, colorizer, device = _load_case(
        config_path,
        colorize_pre_norm=colorize_pre_norm,
    )
    feature_dim = feature_config.feature_dim
    backward_mode = backward_mode_for_feature_render_mode(mode, feature_dim)
    chunk_size = cfg["train"]["frame_chunk_size"]
    chunk_size = feature_config.frames if chunk_size is None else min(int(chunk_size), feature_config.frames)
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")
    total_loss_elems = target_rgb.numel()
    render_samples: list[float] = []
    color_forward_samples: list[float] = []
    color_backward_samples: list[float] = []
    renderer_backward_samples: list[float] = []
    loss_samples: list[float] = []
    finite = True

    tile_counts: list[torch.Tensor] = []
    tile_overflow: list[torch.Tensor] = []
    tile_unstable: list[torch.Tensor] = []
    for iteration in range(warmup + repeat):
        colorizer.zero_grad(set_to_none=True)
        render_ms = 0.0
        color_forward_ms = 0.0
        color_backward_ms = 0.0
        renderer_backward_ms = 0.0
        loss_value = 0.0
        finite_this_iter = True
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
            target_chunk = target_rgb[frame_start : frame_start + chunk_frames]
            rgb = colorize_and_compose(feature_probe, alpha_probe, colorizer)
            loss = (rgb - target_chunk).square().sum() / float(total_loss_elems)
            _sync_device(device)
            t2 = time.perf_counter()
            loss.backward()
            _sync_device(device)
            t3 = time.perf_counter()

            grad_feature = feature_probe.grad
            grad_alpha = alpha_probe.grad
            if grad_feature is None or grad_alpha is None:
                raise RuntimeError("colorizer/loss backward did not produce feature/alpha image gradients")
            grads = direct_atomic_feature_backward(
                *render_inputs,
                grad_feature.contiguous(),
                grad_alpha.contiguous(),
                chunk_config,
                backward_mode=backward_mode,
            )
            _sync_device(device)
            t4 = time.perf_counter()

            render_ms += (t1 - t0) * 1000.0
            color_forward_ms += (t2 - t1) * 1000.0
            color_backward_ms += (t3 - t2) * 1000.0
            renderer_backward_ms += (t4 - t3) * 1000.0
            loss_value += float(loss.detach().cpu().item())
            finite_this_iter = (
                finite_this_iter
                and bool(torch.isfinite(render.feature_image).all().cpu())
                and bool(torch.isfinite(render.alpha).all().cpu())
                and bool(torch.isfinite(grad_feature).all().cpu())
                and bool(torch.isfinite(grad_alpha).all().cpu())
                and all(bool(torch.isfinite(grad).all().cpu()) for grad in grads[:4])
            )
            tile_counts_iter.append(render.tile_counts)
            tile_overflow_iter.append(render.tile_overflow)
            tile_unstable_iter.append(grads[-1])
        if iteration >= warmup:
            render_samples.append(render_ms)
            color_forward_samples.append(color_forward_ms)
            color_backward_samples.append(color_backward_ms)
            renderer_backward_samples.append(renderer_backward_ms)
            loss_samples.append(loss_value)
            finite = finite and finite_this_iter
            tile_counts = tile_counts_iter
            tile_overflow = tile_overflow_iter
            tile_unstable = tile_unstable_iter

    mean_render = _mean(render_samples)
    mean_color_forward = _mean(color_forward_samples)
    mean_color_backward = _mean(color_backward_samples)
    mean_renderer_backward = _mean(renderer_backward_samples)
    mean_total = mean_render + mean_color_forward + mean_color_backward + mean_renderer_backward
    backward_total = mean_color_backward + mean_renderer_backward
    tile_stats = _tile_load_stats(
        tile_counts=tile_counts,
        tile_overflow=tile_overflow,
        tile_unstable=tile_unstable,
        tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
    )
    return {
        "config": str(config_path),
        "mode": mode,
        "backward_mode": backward_mode,
        "frames": feature_config.frames,
        "size": feature_config.height,
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "feature_dim": feature_dim,
        "frame_chunk_size": chunk_size,
        "tile_t": int(cfg["feature_uvt"]["tile_t"]),
        "tile_capacity": int(cfg["feature_uvt"]["tile_capacity"]),
        "alpha_threshold": float(cfg["feature_uvt"]["alpha_threshold"]),
        "colorize_pre_norm": bool(cfg["colorize"]["pre_norm"]),
        "colorize_pre_norm_overridden": colorize_pre_norm is not None,
        "warmup": warmup,
        "repeat": repeat,
        "loss_samples": loss_samples,
        "mean_loss": _mean(loss_samples),
        "render_forward_ms_samples": render_samples,
        "colorize_loss_forward_ms_samples": color_forward_samples,
        "colorize_loss_backward_ms_samples": color_backward_samples,
        "renderer_backward_ms_samples": renderer_backward_samples,
        "mean_timing_ms": {
            "render_forward": mean_render,
            "colorize_loss_forward": mean_color_forward,
            "colorize_loss_backward": mean_color_backward,
            "renderer_backward": mean_renderer_backward,
            "backward_total": backward_total,
            "manual_total": mean_total,
        },
        "share": {
            "colorize_loss_backward_of_backward": 0.0 if backward_total <= 0.0 else mean_color_backward / backward_total,
            "renderer_backward_of_backward": 0.0 if backward_total <= 0.0 else mean_renderer_backward / backward_total,
            "colorize_loss_forward_of_total": 0.0 if mean_total <= 0.0 else mean_color_forward / mean_total,
            "renderer_backward_of_total": 0.0 if mean_total <= 0.0 else mean_renderer_backward / mean_total,
        },
        "tile_stats": tile_stats,
        "finite": finite,
        "pass": finite and int(tile_stats["overflow_tile_count"]) == 0,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = result["rows"]
    lines = [
        "# STAR UVT Feature First-Class Backward Breakdown",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "| config | mode | pre-norm | size | tubes | chunk | total ms | render fwd | color/loss fwd | color/loss bwd | renderer bwd | renderer bwd share | overflow | max tile | p95 tile | pass |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        timing = row["mean_timing_ms"]
        tile = row["tile_stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(row["config"]).name,
                    row["mode"],
                    "yes" if row["colorize_pre_norm"] else "no",
                    str(row["size"]),
                    str(row["tubes"]),
                    str(row["frame_chunk_size"]),
                    f"{timing['manual_total']:.1f}",
                    f"{timing['render_forward']:.1f}",
                    f"{timing['colorize_loss_forward']:.1f}",
                    f"{timing['colorize_loss_backward']:.1f}",
                    f"{timing['renderer_backward']:.1f}",
                    f"{100.0 * row['share']['renderer_backward_of_backward']:.1f}%",
                    str(tile["overflow_tile_count"]),
                    str(tile["max_tile_count"]),
                    f"{tile['p95_tile_count']:.0f}",
                    "yes" if row["pass"] else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- `colorize_loss_backward` is the image-space backward that produces `grad_feature_image` and `grad_alpha` from the real `FeatureToColor` graph.",
            "- `renderer_backward` is the STAR UVT Metal feature backward called manually with those image gradients.",
            "- `manual_total` excludes optimizer time and media/logging; it is a diagnostic split, not a replacement for full trainer timing.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--modes", default=None, help="comma-separated feature_direct_* modes; default uses each config mode")
    parser.add_argument(
        "--colorize-pre-norm",
        choices=("config", "true", "false"),
        default="config",
        help="override colorize.pre_norm for timing A/B; default keeps the config value",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("STAR UVT feature backward breakdown requires MPS")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be nonnegative and repeat must be positive")

    colorize_pre_norm = None
    if args.colorize_pre_norm == "true":
        colorize_pre_norm = True
    elif args.colorize_pre_norm == "false":
        colorize_pre_norm = False

    rows: list[dict[str, Any]] = []
    for config_path in args.config:
        cfg = resolve_config(load_config_file(config_path))
        modes = [str(cfg["feature_uvt"]["render_mode"])]
        if args.modes is not None:
            modes = list(split_csv_strings(args.modes))
        for mode in modes:
            rows.append(
                run_breakdown(
                    config_path,
                    mode=mode,
                    colorize_pre_norm=colorize_pre_norm,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            )

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "gate": "star_uvt_feature_firstclass_backward_breakdown",
        "rows": rows,
        "pass": all(bool(row["pass"]) for row in rows),
    }
    write_report_json(args.out_json, result)
    if args.out_md is not None:
        _write_markdown(args.out_md, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
