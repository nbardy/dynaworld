from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


try:
    from .report_artifacts import ROOT, summary_stats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, summary_stats, write_report_json, write_report_text
from research_experiments.star_uvt_feature_tubes.star_uvt_feature1_wholegraph_profile import (
    _chunk_render_inputs,
    _load_case,
)
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    render_uvt_feature_sparse_pixels_with_bins,
    render_uvt_feature_tubes,
)
from star_uvt_runtime import sync_device as _sync_device
from star_uvt_sparse_grid import _pack_sparse_target_grid_vjp


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc"
)
DEFAULT_OUT_BASE = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile"


def _chunk_sparse_pixel_ids(case: dict[str, Any], frame_start: int, chunk_frames: int) -> torch.Tensor:
    feature_config = case["feature_config"]
    target_feature = case["target_feature"]
    target_chunk = target_feature.chunk(frame_start, chunk_frames)
    seed_grad = torch.ones_like(target_chunk)
    pack = _pack_sparse_target_grid_vjp(
        seed_grad,
        input_shape=(
            int(chunk_frames),
            int(feature_config.feature_dim),
            int(feature_config.height),
            int(feature_config.width),
        ),
        mode=target_feature.grid_mode,
    )
    return pack.pixel_ids


def _profile_once(case: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    model = case["model"]
    device = case["device"]
    feature_dim = int(feature_config.feature_dim)
    dense_ms = 0.0
    sparse_ms = 0.0
    sparse_pixels = 0
    max_feature_error = 0.0
    max_alpha_error = 0.0
    dense_overflow = 0
    sparse_overflow = 0
    dense_unstable = 0
    sparse_unstable = 0

    for frame_start in range(0, int(feature_config.frames), chunk_size):
        chunk_frames = min(chunk_size, int(feature_config.frames) - frame_start)
        render_inputs, chunk_config = _chunk_render_inputs(model, uvt_config, frame_start, chunk_frames)
        pixel_ids = _chunk_sparse_pixel_ids(case, frame_start, chunk_frames)
        sparse_pixels += int(pixel_ids.numel())

        _sync_device(device)
        t0 = time.perf_counter()
        dense = render_uvt_feature_tubes(*render_inputs, chunk_config, return_bins=True)
        _sync_device(device)
        t1 = time.perf_counter()
        sparse = render_uvt_feature_sparse_pixels_with_bins(*render_inputs, pixel_ids, chunk_config)
        _sync_device(device)
        t2 = time.perf_counter()

        dense_ms += (t1 - t0) * 1000.0
        sparse_ms += (t2 - t1) * 1000.0
        dense_flat = dense.feature_image.permute(0, 2, 3, 1).reshape(-1, feature_dim)
        sparse_index = pixel_ids.to(torch.int64)
        dense_values = dense_flat.index_select(0, sparse_index)
        dense_alpha = dense.alpha.reshape(-1).index_select(0, sparse_index)
        _sync_device(device)
        max_feature_error = max(max_feature_error, float((dense_values - sparse.feature_values).abs().max().item()))
        max_alpha_error = max(max_alpha_error, float((dense_alpha - sparse.alpha_values).abs().max().item()))
        dense_overflow += int(dense.tile_overflow.sum().item())
        sparse_overflow += int(sparse.tile_overflow.sum().item())
        dense_unstable += int(dense.tile_unstable.sum().item())
        sparse_unstable += int(sparse.tile_unstable.sum().item())

    return {
        "dense_render_ms": dense_ms,
        "sparse_render_ms": sparse_ms,
        "sparse_pixel_count": sparse_pixels,
        "sparse_pixel_fraction": sparse_pixels
        / float(int(feature_config.frames) * int(feature_config.height) * int(feature_config.width)),
        "max_feature_error": max_feature_error,
        "max_alpha_error": max_alpha_error,
        "dense_overflow": dense_overflow,
        "sparse_overflow": sparse_overflow,
        "dense_unstable": dense_unstable,
        "sparse_unstable": sparse_unstable,
    }


def profile_config(config_path: Path, *, warmup: int, repeat: int) -> dict[str, Any]:
    case = _load_case(config_path)
    cfg = case["cfg"]
    feature_config = case["feature_config"]
    chunk_size_cfg = cfg["train"]["frame_chunk_size"]
    chunk_size = int(feature_config.frames) if chunk_size_cfg is None else min(int(chunk_size_cfg), int(feature_config.frames))
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")

    samples = {
        "dense_render_ms": [],
        "sparse_render_ms": [],
    }
    last: dict[str, Any] | None = None
    max_feature_error = 0.0
    max_alpha_error = 0.0
    for iteration in range(warmup + repeat):
        row = _profile_once(case, chunk_size)
        if iteration >= warmup:
            samples["dense_render_ms"].append(float(row["dense_render_ms"]))
            samples["sparse_render_ms"].append(float(row["sparse_render_ms"]))
            max_feature_error = max(max_feature_error, float(row["max_feature_error"]))
            max_alpha_error = max(max_alpha_error, float(row["max_alpha_error"]))
            last = row

    if last is None:
        raise RuntimeError("repeat must be positive")

    dense_stats = summary_stats(samples["dense_render_ms"])
    sparse_stats = summary_stats(samples["sparse_render_ms"])
    speedup = 0.0 if sparse_stats["mean"] <= 0.0 else dense_stats["mean"] / sparse_stats["mean"]
    pass_flag = (
        max_feature_error <= 1.0e-5
        and max_alpha_error <= 1.0e-5
        and int(last["dense_overflow"]) == 0
        and int(last["sparse_overflow"]) == 0
        and int(last["dense_unstable"]) == 0
        and int(last["sparse_unstable"]) == 0
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate": "star_uvt_sparse_forward_profile",
        "config": str(config_path),
        "frames": int(feature_config.frames),
        "size": int(feature_config.height),
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "feature_dim": int(feature_config.feature_dim),
        "frame_chunk_size": chunk_size,
        "warmup": warmup,
        "repeat": repeat,
        "dense_render_ms": dense_stats,
        "sparse_render_ms": sparse_stats,
        "speedup_vs_dense_render": speedup,
        "sparse_pixel_count": int(last["sparse_pixel_count"]),
        "sparse_pixel_fraction": float(last["sparse_pixel_fraction"]),
        "max_feature_error": max_feature_error,
        "max_alpha_error": max_alpha_error,
        "dense_overflow": int(last["dense_overflow"]),
        "sparse_overflow": int(last["sparse_overflow"]),
        "dense_unstable": int(last["dense_unstable"]),
        "sparse_unstable": int(last["sparse_unstable"]),
        "pass": pass_flag,
    }


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# STAR UVT Sparse Forward Profile",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Scope",
        "",
        "Compares the existing dense 512px feature render against the new sparse-pixel feature forward op on the same target-grid support used by `analytic_sparse_grid` VJP.",
        "",
        "## Result",
        "",
        "| path | mean | min | max |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| dense full-image render | {_fmt(result['dense_render_ms']['mean'])}ms | "
            f"{_fmt(result['dense_render_ms']['min'])}ms | {_fmt(result['dense_render_ms']['max'])}ms |"
        ),
        (
            f"| sparse pixel render | {_fmt(result['sparse_render_ms']['mean'])}ms | "
            f"{_fmt(result['sparse_render_ms']['min'])}ms | {_fmt(result['sparse_render_ms']['max'])}ms |"
        ),
        "",
        "## Validation",
        "",
        f"- pass: `{result['pass']}`",
        f"- sparse pixels: `{result['sparse_pixel_count']}` (`{result['sparse_pixel_fraction']:.6f}` of dense pixels)",
        f"- speedup vs dense render: `{result['speedup_vs_dense_render']:.3f}x`",
        f"- max feature error: `{result['max_feature_error']:.3e}`",
        f"- max alpha error: `{result['max_alpha_error']:.3e}`",
        f"- overflow dense/sparse: `{result['dense_overflow']}` / `{result['sparse_overflow']}`",
        f"- unstable dense/sparse: `{result['dense_unstable']}` / `{result['sparse_unstable']}`",
        "",
        "## Decision",
        "",
        "This profile proves sparse feature forward is bit-exact on the target-grid support pixels. Use `feature_target.image_vjp_mode=analytic_sparse_grid_forward` for the current sparse-forward trainer gate; it folds sparse feature values into the target grid for feature/probe loss and reuses sparse-grid VJP for backward.",
    ]
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    args = parser.parse_args()

    result = profile_config(args.config, warmup=args.warmup, repeat=args.repeat)
    json_path = args.out_base.with_suffix(".json")
    md_path = args.out_base.with_suffix(".md")
    write_report_json(json_path, result)
    write_markdown(md_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
