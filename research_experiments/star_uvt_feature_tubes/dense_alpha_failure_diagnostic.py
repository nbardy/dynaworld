from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from checkpoint_utils import load_checkpoint_mapping
from config_utils import load_config_file
try:
    from .report_artifacts import split_csv_floats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import split_csv_floats, write_report_json, write_report_text
from research_project.trainer_harness.data import load_video_target
from star_uvt_colorizers import build_feature_colorizer
from star_uvt_models import build_feature_tube_model
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    chunked_uvt_config,
    render_uvt_feature_tubes,
    shift_ma_for_frame_chunk,
)
from star_uvt_runtime import resolve_device as _resolve_device, sync_device as _sync_device
from star_uvt_feature_config import resolve_config


ALPHA_THRESHOLDS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
POSTHOC_ALPHA_GAINS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
POSTHOC_ALPHA_FLOORS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
TUBE_OPACITY_MAX = 0.99


def _psnr(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(float(mse))


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _load_checkpoint(path: Path, *, model: torch.nn.Module, colorizer: torch.nn.Module, device: torch.device) -> None:
    payload = load_checkpoint_mapping(path, map_location=device)
    model_state = payload.get("model")
    colorizer_state = payload.get("colorizer")
    if not isinstance(model_state, dict) or not isinstance(colorizer_state, dict):
        raise ValueError(f"Checkpoint {path} must contain model and colorizer states")
    model.load_state_dict(model_state)
    colorizer.load_state_dict(colorizer_state)


def _parse_float_list(raw: str | None) -> tuple[float, ...]:
    if raw is None or raw.strip() == "":
        return ()
    return split_csv_floats(raw)


def _opacity_with_logit_bias(opacity: torch.Tensor, bias: float) -> torch.Tensor:
    if abs(float(bias)) < 1.0e-12:
        return opacity
    normalized = torch.clamp(opacity / TUBE_OPACITY_MAX, min=1.0e-6, max=1.0 - 1.0e-6)
    return torch.sigmoid(torch.logit(normalized) + float(bias)) * TUBE_OPACITY_MAX


def _make_case(config_path: Path) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(config_path))
    device = _resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps":
        raise RuntimeError("STAR UVT dense alpha diagnostic currently requires MPS")
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
    checkpoint = Path(cfg["output"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint from config output.checkpoint: {checkpoint}")
    _load_checkpoint(checkpoint, model=model, colorizer=colorizer, device=device)
    model.eval()
    colorizer.eval()
    return {
        "cfg": cfg,
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "device": device,
        "feature_config": feature_config,
        "uvt_config": uvt_config,
        "target_rgb": target_rgb,
        "model": model,
        "colorizer": colorizer,
    }


def _render_chunk(
    case: dict[str, Any],
    frame_start: int,
    chunk_frames: int,
    *,
    raw_opacity_bias: float = 0.0,
) -> Any:
    model = case["model"]
    uvt_config = case["uvt_config"]
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
    opacity = _opacity_with_logit_bias(opacity, raw_opacity_bias)
    if chunk_frames == int(uvt_config.frames):
        return render_uvt_feature_tubes(
            ma,
            q_uvt,
            depth0.detach(),
            depth_beta.detach(),
            opacity,
            feature,
            uvt_config,
        )
    ma_chunk = shift_ma_for_frame_chunk(
        ma,
        global_frames=uvt_config.frames,
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    return render_uvt_feature_tubes(
        ma_chunk,
        q_uvt,
        depth0.detach(),
        depth_beta.detach(),
        opacity,
        feature,
        chunked_uvt_config(uvt_config, chunk_frames=chunk_frames),
    )


def _render_bias_summary(case: dict[str, Any], *, raw_opacity_bias: float) -> dict[str, Any]:
    cfg = case["cfg"]
    device = case["device"]
    target_rgb = case["target_rgb"]
    colorizer = case["colorizer"]
    frames = int(case["feature_config"].frames)
    chunk_size = cfg["train"]["frame_chunk_size"]
    chunk_size = frames if chunk_size is None else min(int(chunk_size), frames)

    sse = 0.0
    alpha_sum = 0.0
    alpha_sq_sum = 0.0
    alpha_max = 0.0
    alpha_pixels = 0
    threshold_counts = {threshold: 0 for threshold in ALPHA_THRESHOLDS}
    step_ms: list[float] = []

    _sync_device(device)
    started = time.perf_counter()
    with torch.no_grad():
        for frame_start in range(0, frames, chunk_size):
            chunk_frames = min(chunk_size, frames - frame_start)
            chunk_t0 = time.perf_counter()
            render = _render_chunk(case, frame_start, chunk_frames, raw_opacity_bias=raw_opacity_bias)
            splat_rgb = colorizer(render.feature_image)
            alpha = render.alpha.to(dtype=splat_rgb.dtype)
            target = case["target_rgb"][frame_start : frame_start + chunk_frames].to(dtype=splat_rgb.dtype)
            composite = alpha.unsqueeze(1) * splat_rgb
            sse += float((composite - target).square().sum().detach().cpu().item())
            alpha_sum += float(alpha.sum().detach().cpu().item())
            alpha_sq_sum += float(alpha.square().sum().detach().cpu().item())
            alpha_max = max(alpha_max, float(alpha.max().detach().cpu().item()))
            alpha_pixels += int(alpha.numel())
            for threshold in ALPHA_THRESHOLDS:
                threshold_counts[threshold] += int((alpha > threshold).sum().detach().cpu().item())
            _sync_device(device)
            step_ms.append((time.perf_counter() - chunk_t0) * 1000.0)
    _sync_device(device)

    mse = sse / float(target_rgb.numel())
    alpha_mean = alpha_sum / float(alpha_pixels)
    alpha_var = max(alpha_sq_sum / float(alpha_pixels) - alpha_mean * alpha_mean, 0.0)
    return {
        "bias": float(raw_opacity_bias),
        "mse": mse,
        "psnr": _psnr(mse),
        "alpha_mean": alpha_mean,
        "alpha_std": math.sqrt(alpha_var),
        "alpha_max": alpha_max,
        "alpha_thresholds": {
            str(threshold): {
                "pixel_fraction": threshold_counts[threshold] / float(alpha_pixels),
            }
            for threshold in ALPHA_THRESHOLDS
        },
        "render_wall_ms": (time.perf_counter() - started) * 1000.0,
        "chunk_ms_mean": _mean(step_ms),
        "chunk_ms_max": max(step_ms) if step_ms else 0.0,
    }


def _analyze_case(label: str, config_path: Path, *, raw_opacity_biases: tuple[float, ...] = ()) -> dict[str, Any]:
    case = _make_case(config_path)
    cfg = case["cfg"]
    device = case["device"]
    target_rgb = case["target_rgb"]
    colorizer = case["colorizer"]
    frames = int(case["feature_config"].frames)
    chunk_size = cfg["train"]["frame_chunk_size"]
    chunk_size = frames if chunk_size is None else min(int(chunk_size), frames)

    normal_sse = 0.0
    forced_alpha_sse = 0.0
    target_background_sse = 0.0
    black_baseline_sse = 0.0
    color_mean_sum = 0.0
    color_sq_sum = 0.0
    feature_abs_sum = 0.0
    alpha_sum = 0.0
    alpha_sq_sum = 0.0
    alpha_max = 0.0
    alpha_pixels = 0
    rgb_values = 0
    threshold_counts = {threshold: 0 for threshold in ALPHA_THRESHOLDS}
    threshold_residual_sse = {threshold: 0.0 for threshold in ALPHA_THRESHOLDS}
    threshold_forced_sse = {threshold: 0.0 for threshold in ALPHA_THRESHOLDS}
    threshold_elems = {threshold: 0 for threshold in ALPHA_THRESHOLDS}
    gain_sse = {gain: 0.0 for gain in POSTHOC_ALPHA_GAINS}
    floor_sse = {floor: 0.0 for floor in POSTHOC_ALPHA_FLOORS}
    step_ms: list[float] = []

    _sync_device(device)
    started = time.perf_counter()
    with torch.no_grad():
        for frame_start in range(0, frames, chunk_size):
            chunk_frames = min(chunk_size, frames - frame_start)
            chunk_t0 = time.perf_counter()
            render = _render_chunk(case, frame_start, chunk_frames)
            splat_rgb = colorizer(render.feature_image)
            alpha = render.alpha.to(dtype=splat_rgb.dtype)
            target = target_rgb[frame_start : frame_start + chunk_frames].to(dtype=splat_rgb.dtype)
            composite = alpha.unsqueeze(1) * splat_rgb
            target_background = alpha.unsqueeze(1) * splat_rgb + (1.0 - alpha.unsqueeze(1)) * target
            normal_residual = (composite - target).square()
            forced_residual = (splat_rgb - target).square()
            target_background_residual = (target_background - target).square()
            normal_sse += float(normal_residual.sum().detach().cpu().item())
            forced_alpha_sse += float(forced_residual.sum().detach().cpu().item())
            target_background_sse += float(target_background_residual.sum().detach().cpu().item())
            black_baseline_sse += float(target.square().sum().detach().cpu().item())
            for gain in POSTHOC_ALPHA_GAINS:
                gained_alpha = torch.clamp(alpha * float(gain), min=0.0, max=1.0)
                gained_composite = gained_alpha.unsqueeze(1) * splat_rgb
                gain_sse[gain] += float((gained_composite - target).square().sum().detach().cpu().item())
            for floor in POSTHOC_ALPHA_FLOORS:
                floored_alpha = torch.clamp(alpha, min=float(floor), max=1.0)
                floored_composite = floored_alpha.unsqueeze(1) * splat_rgb
                floor_sse[floor] += float((floored_composite - target).square().sum().detach().cpu().item())
            color_mean_sum += float(splat_rgb.sum().detach().cpu().item())
            color_sq_sum += float(splat_rgb.square().sum().detach().cpu().item())
            feature_abs_sum += float(render.feature_image.abs().sum().detach().cpu().item())
            alpha_sum += float(alpha.sum().detach().cpu().item())
            alpha_sq_sum += float(alpha.square().sum().detach().cpu().item())
            alpha_max = max(alpha_max, float(alpha.max().detach().cpu().item()))
            alpha_pixels += int(alpha.numel())
            rgb_values += int(splat_rgb.numel())
            for threshold in ALPHA_THRESHOLDS:
                mask = alpha > threshold
                count = int(mask.sum().detach().cpu().item())
                threshold_counts[threshold] += count
                if count:
                    mask_rgb = mask.unsqueeze(1)
                    threshold_residual_sse[threshold] += float(
                        normal_residual.masked_select(mask_rgb).sum().detach().cpu().item()
                    )
                    threshold_forced_sse[threshold] += float(
                        forced_residual.masked_select(mask_rgb).sum().detach().cpu().item()
                    )
                    threshold_elems[threshold] += count * 3
            _sync_device(device)
            step_ms.append((time.perf_counter() - chunk_t0) * 1000.0)
    _sync_device(device)
    wall_ms = (time.perf_counter() - started) * 1000.0

    total_elems = int(target_rgb.numel())
    normal_mse = normal_sse / float(total_elems)
    forced_mse = forced_alpha_sse / float(total_elems)
    target_background_mse = target_background_sse / float(total_elems)
    black_mse = black_baseline_sse / float(total_elems)
    alpha_mean = alpha_sum / float(alpha_pixels)
    alpha_var = max(alpha_sq_sum / float(alpha_pixels) - alpha_mean * alpha_mean, 0.0)
    color_mean = color_mean_sum / float(rgb_values)
    color_var = max(color_sq_sum / float(rgb_values) - color_mean * color_mean, 0.0)
    gain_sweep = {
        str(gain): {
            "mse": sse / float(total_elems),
            "psnr": _psnr(sse / float(total_elems)),
        }
        for gain, sse in gain_sse.items()
    }
    floor_sweep = {
        str(floor): {
            "mse": sse / float(total_elems),
            "psnr": _psnr(sse / float(total_elems)),
        }
        for floor, sse in floor_sse.items()
    }
    best_gain_key, best_gain = max(gain_sweep.items(), key=lambda item: item[1]["psnr"])
    best_floor_key, best_floor = max(floor_sweep.items(), key=lambda item: item[1]["psnr"])

    thresholds = {}
    for threshold in ALPHA_THRESHOLDS:
        elems = threshold_elems[threshold]
        thresholds[str(threshold)] = {
            "pixel_fraction": threshold_counts[threshold] / float(alpha_pixels),
            "normal_residual_share": threshold_residual_sse[threshold] / normal_sse if normal_sse > 0.0 else 0.0,
            "normal_masked_psnr": None if elems == 0 else _psnr(threshold_residual_sse[threshold] / float(elems)),
            "forced_alpha_masked_psnr": None if elems == 0 else _psnr(threshold_forced_sse[threshold] / float(elems)),
        }

    raw_bias_sweep = {
        str(bias): _render_bias_summary(case, raw_opacity_bias=bias)
        for bias in raw_opacity_biases
    }
    best_raw_bias_key = None
    best_raw_bias = None
    if raw_bias_sweep:
        best_raw_bias_key, best_raw_bias = max(raw_bias_sweep.items(), key=lambda item: item[1]["psnr"])

    return {
        "label": label,
        "config_path": case["config_path"],
        "checkpoint": case["checkpoint"],
        "frames": frames,
        "size": int(case["feature_config"].height),
        "chunk_size": chunk_size,
        "normal_black_psnr": _psnr(normal_mse),
        "forced_alpha_1_psnr": _psnr(forced_mse),
        "target_background_oracle_psnr": _psnr(target_background_mse),
        "best_alpha_gain": float(best_gain_key),
        "best_alpha_gain_psnr": float(best_gain["psnr"]),
        "best_alpha_floor": float(best_floor_key),
        "best_alpha_floor_psnr": float(best_floor["psnr"]),
        "best_raw_opacity_bias": None if best_raw_bias_key is None else float(best_raw_bias_key),
        "best_raw_opacity_bias_psnr": None if best_raw_bias is None else float(best_raw_bias["psnr"]),
        "black_baseline_psnr": _psnr(black_mse),
        "normal_mse": normal_mse,
        "forced_alpha_1_mse": forced_mse,
        "target_background_oracle_mse": target_background_mse,
        "alpha_mean": alpha_mean,
        "alpha_std": math.sqrt(alpha_var),
        "alpha_max": alpha_max,
        "colorizer_rgb_mean": color_mean,
        "colorizer_rgb_std": math.sqrt(color_var),
        "feature_abs_mean": feature_abs_sum / float(rgb_values / 3.0 * int(case["feature_config"].feature_dim)),
        "alpha_thresholds": thresholds,
        "posthoc_alpha_gain_sweep": gain_sweep,
        "posthoc_alpha_floor_sweep": floor_sweep,
        "raw_opacity_bias_sweep": raw_bias_sweep,
        "render_wall_ms": wall_ms,
        "chunk_ms_mean": _mean(step_ms),
        "chunk_ms_max": max(step_ms) if step_ms else 0.0,
    }


def _parse_case(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path)
    path = Path(raw)
    return path.stem, path


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = [
        [
            "label",
            "normal PSNR",
            "forced-alpha PSNR",
            "target-bg oracle PSNR",
            "best gain PSNR",
            "best floor PSNR",
            "best raw-opacity PSNR",
            "alpha mean",
            "alpha>0.1",
            "alpha>0.5",
            "high-alpha residual share",
        ]
    ]
    for case in result["cases"]:
        thresholds = case["alpha_thresholds"]
        rows.append(
            [
                case["label"],
                f"{case['normal_black_psnr']:.3f}",
                f"{case['forced_alpha_1_psnr']:.3f}",
                f"{case['target_background_oracle_psnr']:.3f}",
                f"{case['best_alpha_gain_psnr']:.3f} @ {case['best_alpha_gain']:g}x",
                f"{case['best_alpha_floor_psnr']:.3f} @ {case['best_alpha_floor']:g}",
                (
                    "n/a"
                    if case["best_raw_opacity_bias_psnr"] is None
                    else f"{case['best_raw_opacity_bias_psnr']:.3f} @ {case['best_raw_opacity_bias']:+g}"
                ),
                f"{case['alpha_mean']:.4f}",
                f"{thresholds['0.1']['pixel_fraction']:.3f}",
                f"{thresholds['0.5']['pixel_fraction']:.3f}",
                f"{thresholds['0.1']['normal_residual_share']:.3f}",
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    lines = [
        "# STAR UVT Dense Alpha Failure Diagnostic",
        "",
        f"Date: {result['date']}",
        "",
        "## Purpose",
        "",
        "Disentangle the rejected STAR UVT dense visual gates into alpha/coverage",
        "failure versus feature-to-RGB content failure. `normal PSNR` is the actual",
        "black-background composite. `forced-alpha PSNR` ignores alpha and compares",
        "the colorizer output directly to RGB. `target-bg oracle PSNR` composites the",
        "prediction over the target frame, removing black-background holes and leaving",
        "only alpha-weighted color error. `best gain PSNR` clamps post-render alpha",
        "after multiplying by a scalar. `best floor PSNR` clamps post-render alpha",
        "to a minimum floor; the `1.0` floor equals forced-alpha behavior.",
        "`best raw-opacity PSNR` rerenders after adding a logit-space bias to each",
        "tube opacity, so it tests support expansion before raster accumulation.",
        "",
        "## Results",
        "",
        fmt(rows[0]),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows[1:])
    lines.extend(
        [
            "",
            "## Read",
            "",
            result["read"],
            "",
            "## Inputs",
            "",
        ]
    )
    for case in result["cases"]:
        lines.extend(
            [
                f"- `{case['label']}` config: `{case['config_path']}`",
                f"- `{case['label']}` checkpoint: `{case['checkpoint']}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_report_text(path, "\n".join(lines) + "\n")


def _make_read(cases: Iterable[dict[str, Any]]) -> str:
    reads: list[str] = []
    for case in cases:
        normal = case["normal_black_psnr"]
        forced = case["forced_alpha_1_psnr"]
        oracle = case["target_background_oracle_psnr"]
        best_gain = case["best_alpha_gain_psnr"]
        best_floor = case["best_alpha_floor_psnr"]
        best_raw_bias = case["best_raw_opacity_bias_psnr"]
        alpha01 = case["alpha_thresholds"]["0.1"]["pixel_fraction"]
        if forced > normal + 1.0 and oracle > normal + 3.0:
            raw_bias_clause = (
                ""
                if best_raw_bias is None
                else f" Raw-opacity bias reaches {best_raw_bias:.3f} PSNR."
            )
            reads.append(
                f"`{case['label']}` is strongly coverage/visibility limited: forced alpha improves "
                f"{normal:.3f}->{forced:.3f} PSNR and target-background oracle reaches {oracle:.3f}, "
                f"with alpha>0.1 on {alpha01:.1%} of pixels. Posthoc alpha gain reaches "
                f"{best_gain:.3f} PSNR and alpha floor reaches {best_floor:.3f} PSNR.{raw_bias_clause}"
            )
        elif forced <= normal + 1.0 and oracle > normal + 3.0:
            reads.append(
                f"`{case['label']}` has black-background coverage loss, but forced-alpha RGB is still weak "
                f"({forced:.3f} PSNR), so output features/colors are not sufficient either."
            )
        else:
            reads.append(
                f"`{case['label']}` is not rescued by alpha forcing ({normal:.3f}->{forced:.3f} PSNR); "
                "the dense feature/color field itself remains a blocker."
            )
    return " ".join(reads)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="label=config.jsonc or config.jsonc")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--date", default="2026-05-20")
    parser.add_argument(
        "--raw-opacity-biases",
        default="",
        help="Optional comma-separated logit-space opacity biases to rerender, for example --raw-opacity-biases=-2,-1,0,1,2,3,4.",
    )
    args = parser.parse_args()

    raw_opacity_biases = _parse_float_list(args.raw_opacity_biases)
    cases = [
        _analyze_case(label, path, raw_opacity_biases=raw_opacity_biases)
        for label, path in (_parse_case(raw) for raw in args.case)
    ]
    result = {
        "date": args.date,
        "raw_opacity_biases": list(raw_opacity_biases),
        "cases": cases,
    }
    result["read"] = _make_read(cases)
    write_report_json(Path(args.out_json), result)
    _write_markdown(Path(args.out_md), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
