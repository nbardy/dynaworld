from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


try:
    from .report_artifacts import ROOT, distribution_stats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, distribution_stats, write_report_json, write_report_text
from research_experiments.star_uvt_feature_tubes.star_uvt_feature1_wholegraph_profile import (
    _chunk_render_inputs,
    _load_case,
)
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    render_uvt_feature_sparse_pixels_with_bins,
)
from star_uvt_schedules import (
    _feature_target_weight_schedule,
    _feature_target_weights_for_step,
)
from star_uvt_common import target_grid_slice_for_render_chunk as _target_grid_slice_for_render_chunk
from star_uvt_feature_losses import (
    _manual_rgb_probe_loss_and_grid_grad,
    _manual_sparse_target_grid_loss_and_vjp,
)
from star_uvt_runtime import sync_device as _sync_device
from star_uvt_sparse_grid import (
    _sparse_target_grid_pixel_ids,
    _target_grid_sparse_vjp_plan_device,
)


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforwardvjp.jsonc"
)
DEFAULT_OUT_BASE = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_target_vjp_profile"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _target_context(case: dict[str, Any]) -> dict[str, Any]:
    cfg = case["cfg"]
    stage = _feature_target_weights_for_step(
        _feature_target_weight_schedule(cfg),
        int(cfg["train"]["global_step_offset"]),
    )
    rgb_probe_target = case["rgb_probe_target"]
    return {
        "feature_loss_weight": float(stage.loss_weight),
        "rgb_probe_loss_weight": float(stage.rgb_probe_loss_weight),
        "total_feature_loss_elems": int(case["target_feature"].numel),
        "total_rgb_probe_loss_elems": 0 if rgb_probe_target is None else int(rgb_probe_target.numel()),
    }


def _load_sparse_forward_chunks(case: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    model = case["model"]
    device = case["device"]
    target_feature = case["target_feature"]
    chunks: list[dict[str, Any]] = []
    render_ms = 0.0
    total_sparse_pixels = 0
    total_dense_pixels = 0
    max_tile_count = 0
    tile_overflow_sum = 0
    for frame_start in range(0, int(feature_config.frames), chunk_size):
        chunk_frames = min(chunk_size, int(feature_config.frames) - frame_start)
        target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
        input_shape = (
            int(chunk_frames),
            int(feature_config.feature_dim),
            int(feature_config.height),
            int(feature_config.width),
        )
        target_shape = tuple(int(item) for item in target_feature_chunk.shape)
        pixel_ids = _sparse_target_grid_pixel_ids(
            input_shape=input_shape,
            target_shape=target_shape,
            mode=target_feature.grid_mode,
            device=device,
        )
        render_inputs, render_config = _chunk_render_inputs(model, uvt_config, frame_start, chunk_frames)
        _sync_device(device)
        t0 = time.perf_counter()
        render = render_uvt_feature_sparse_pixels_with_bins(*render_inputs, pixel_ids, render_config)
        _sync_device(device)
        render_ms += (time.perf_counter() - t0) * 1000.0
        chunks.append(
            {
                "frame_start": int(frame_start),
                "chunk_frames": int(chunk_frames),
                "input_shape": input_shape,
                "target_shape": target_shape,
                "pixel_ids": pixel_ids,
                "feature_values": render.feature_values.detach(),
            }
        )
        total_sparse_pixels += int(pixel_ids.numel())
        total_dense_pixels += int(chunk_frames) * int(feature_config.height) * int(feature_config.width)
        max_tile_count = max(max_tile_count, int(render.tile_counts.max().item()))
        tile_overflow_sum += int(render.tile_overflow.sum().item())
    return {
        "chunks": chunks,
        "render_ms": render_ms,
        "total_sparse_pixels": total_sparse_pixels,
        "total_dense_pixels": total_dense_pixels,
        "max_tile_count": max_tile_count,
        "tile_overflow_sum": tile_overflow_sum,
    }


def _cat_target_chunks(case: dict[str, Any], chunks: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor | None]:
    target_feature = case["target_feature"]
    rgb_probe_target = case["rgb_probe_target"]
    feature_config = case["feature_config"]
    target_chunks = []
    probe_chunks = []
    for chunk in chunks:
        frame_start = int(chunk["frame_start"])
        chunk_frames = int(chunk["chunk_frames"])
        target_chunks.append(target_feature.chunk(frame_start, chunk_frames))
        if rgb_probe_target is not None:
            target_start, target_frames = _target_grid_slice_for_render_chunk(
                target_frames=int(rgb_probe_target.shape[0]),
                render_frames=int(feature_config.frames),
                frame_start=frame_start,
                chunk_frames=chunk_frames,
            )
            probe_chunks.append(rgb_probe_target[target_start : target_start + target_frames])
    return torch.cat(target_chunks, dim=0).contiguous(), None if not probe_chunks else torch.cat(probe_chunks, dim=0).contiguous()


def _batched_values_to_target_grid(
    feature_values: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
) -> torch.Tensor:
    if feature_values.ndim != 3:
        raise ValueError("feature_values must have shape [chunks,sparse_pixels,feature_dim]")
    batch = int(feature_values.shape[0])
    feature_dim = int(feature_values.shape[2])
    plan = _target_grid_sparse_vjp_plan_device(
        input_shape,
        target_shape,
        mode,
        device=feature_values.device,
        dtype=feature_values.dtype,
    )
    source_values = feature_values.index_select(1, plan.inverse)
    weighted = source_values * plan.weights.view(1, -1, 1)
    target_cells = int(target_shape[0]) * int(target_shape[2]) * int(target_shape[3])
    target_flat = torch.zeros((batch, target_cells, feature_dim), device=feature_values.device, dtype=feature_values.dtype)
    scatter_ids = plan.target_flat_ids.view(1, -1, 1).expand(batch, -1, feature_dim)
    target_flat.scatter_add_(1, scatter_ids, weighted)
    return (
        target_flat.reshape(batch * int(target_shape[0]), int(target_shape[2]), int(target_shape[3]), feature_dim)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _batched_pack_sparse_vjp(
    grad_target_grid: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
) -> list[dict[str, torch.Tensor]]:
    batch = int(grad_target_grid.shape[0]) // int(target_shape[0])
    feature_dim = int(grad_target_grid.shape[1])
    plan = _target_grid_sparse_vjp_plan_device(
        input_shape,
        target_shape,
        mode,
        device=grad_target_grid.device,
        dtype=grad_target_grid.dtype,
    )
    target_values = (
        grad_target_grid.reshape(batch, int(target_shape[0]), feature_dim, int(target_shape[2]), int(target_shape[3]))
        .permute(0, 1, 3, 4, 2)
        .reshape(batch, -1, feature_dim)
    )
    weighted = target_values.index_select(1, plan.target_flat_ids) * plan.weights.view(1, -1, 1)
    if plan.has_duplicate_pixels:
        grad_values = torch.zeros(
            (batch, int(plan.unique_pixel_ids.numel()), feature_dim),
            device=grad_target_grid.device,
            dtype=grad_target_grid.dtype,
        )
        scatter_ids = plan.inverse.view(1, -1, 1).expand(batch, -1, feature_dim)
        grad_values.scatter_add_(1, scatter_ids, weighted)
        pixel_ids = plan.unique_pixel_ids
    else:
        grad_values = weighted.contiguous()
        pixel_ids = plan.source_pixel_ids
    zero_alpha = torch.zeros((int(pixel_ids.numel()),), device=grad_target_grid.device, dtype=grad_target_grid.dtype)
    return [
        {
            "pixel_ids": pixel_ids.contiguous(),
            "grad_feature_values": grad_values[index].contiguous(),
            "grad_alpha_values": zero_alpha.contiguous(),
        }
        for index in range(batch)
    ]


def _profile_per_chunk(case: dict[str, Any], chunks: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    feature_config = case["feature_config"]
    device = case["device"]
    packs = []
    loss = 0.0
    feature_loss_value = 0.0
    rgb_probe_loss_value = 0.0
    feature_target_ms = 0.0
    rgb_probe_loss_ms = 0.0
    image_vjp_ms = 0.0
    for chunk in chunks:
        result = _manual_sparse_target_grid_loss_and_vjp(
            chunk["feature_values"],
            target_feature=case["target_feature"],
            rgb_probe=case["rgb_probe"],
            rgb_probe_target=case["rgb_probe_target"],
            feature_config=feature_config,
            frame_start=int(chunk["frame_start"]),
            chunk_frames=int(chunk["chunk_frames"]),
            feature_loss_type="mse",
            feature_loss_weight=float(context["feature_loss_weight"]),
            rgb_probe_loss_weight=float(context["rgb_probe_loss_weight"]),
            total_feature_loss_elems=int(context["total_feature_loss_elems"]),
            total_rgb_probe_loss_elems=int(context["total_rgb_probe_loss_elems"]),
            device=device,
        )
        loss += float(result.loss.detach().cpu().item())
        feature_loss_value += float(result.feature_target_loss)
        rgb_probe_loss_value += float(result.rgb_probe_loss)
        feature_target_ms += float(result.feature_target_ms)
        rgb_probe_loss_ms += float(result.rgb_probe_loss_ms)
        image_vjp_ms += float(result.image_vjp_ms)
        if result.sparse_pack is None:
            raise RuntimeError("per-chunk sparse target-grid VJP did not produce sparse_pack")
        packs.append(
            {
                "pixel_ids": result.sparse_pack.pixel_ids.detach(),
                "grad_feature_values": result.sparse_pack.grad_feature_values.detach(),
                "grad_alpha_values": result.sparse_pack.grad_alpha_values.detach(),
            }
        )
    return {
        "loss": loss,
        "feature_target_loss": feature_loss_value,
        "rgb_probe_loss": rgb_probe_loss_value,
        "packs": packs,
        "timing_ms": {
            "feature_target_ms": feature_target_ms,
            "rgb_probe_loss_ms": rgb_probe_loss_ms,
            "image_vjp_ms": image_vjp_ms,
            "total_loss_vjp_ms": feature_target_ms + rgb_probe_loss_ms + image_vjp_ms,
        },
    }


def _profile_batched(case: dict[str, Any], chunks: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("chunks must be non-empty")
    first_shape = chunks[0]["input_shape"]
    first_target_shape = chunks[0]["target_shape"]
    if any(chunk["input_shape"] != first_shape or chunk["target_shape"] != first_target_shape for chunk in chunks):
        raise ValueError("batched profile currently requires equal chunk and target shapes")
    feature_values = torch.stack([chunk["feature_values"] for chunk in chunks], dim=0).contiguous()
    target_feature, rgb_probe_target = _cat_target_chunks(case, chunks)
    rgb_probe = case["rgb_probe"]
    device = case["device"]

    _sync_device(device)
    target_t0 = time.perf_counter()
    rendered_target_grid = _batched_values_to_target_grid(
        feature_values,
        input_shape=first_shape,
        target_shape=first_target_shape,
        mode=case["target_feature"].grid_mode,
    )
    _sync_device(device)
    feature_target_ms = (time.perf_counter() - target_t0) * 1000.0

    loss = rendered_target_grid.new_zeros(())
    grad_target_grid = torch.zeros_like(rendered_target_grid)
    feature_loss_value = 0.0
    rgb_probe_loss_value = 0.0
    if float(context["feature_loss_weight"]) > 0.0:
        _sync_device(device)
        feature_t0 = time.perf_counter()
        diff = rendered_target_grid - target_feature
        feature_loss = diff.square().sum() / float(context["total_feature_loss_elems"])
        feature_loss_value = float(feature_loss.detach().cpu().item())
        loss = loss + float(context["feature_loss_weight"]) * feature_loss
        grad_target_grid = grad_target_grid + (
            2.0 * float(context["feature_loss_weight"]) / float(context["total_feature_loss_elems"])
        ) * diff
        _sync_device(device)
        feature_target_ms += (time.perf_counter() - feature_t0) * 1000.0

    rgb_probe_loss_ms = 0.0
    if rgb_probe is not None and float(context["rgb_probe_loss_weight"]) > 0.0:
        if rgb_probe_target is None:
            raise RuntimeError("RGB probe target is missing")
        _sync_device(device)
        probe_t0 = time.perf_counter()
        rgb_probe_loss, probe_grad_grid = _manual_rgb_probe_loss_and_grid_grad(
            rgb_probe,
            rendered_target_grid,
            rgb_probe_target,
            total_rgb_probe_loss_elems=int(context["total_rgb_probe_loss_elems"]),
            loss_weight=float(context["rgb_probe_loss_weight"]),
        )
        rgb_probe_loss_value = float(rgb_probe_loss.detach().cpu().item())
        loss = loss + float(context["rgb_probe_loss_weight"]) * rgb_probe_loss
        grad_target_grid = grad_target_grid + probe_grad_grid
        _sync_device(device)
        rgb_probe_loss_ms = (time.perf_counter() - probe_t0) * 1000.0

    _sync_device(device)
    vjp_t0 = time.perf_counter()
    packs = _batched_pack_sparse_vjp(
        grad_target_grid,
        input_shape=first_shape,
        target_shape=first_target_shape,
        mode=case["target_feature"].grid_mode,
    )
    _sync_device(device)
    image_vjp_ms = (time.perf_counter() - vjp_t0) * 1000.0
    return {
        "loss": float(loss.detach().cpu().item()),
        "feature_target_loss": feature_loss_value,
        "rgb_probe_loss": rgb_probe_loss_value,
        "packs": packs,
        "timing_ms": {
            "feature_target_ms": feature_target_ms,
            "rgb_probe_loss_ms": rgb_probe_loss_ms,
            "image_vjp_ms": image_vjp_ms,
            "total_loss_vjp_ms": feature_target_ms + rgb_probe_loss_ms + image_vjp_ms,
        },
    }


def _compare_packs(per_chunk: list[dict[str, torch.Tensor]], batched: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    if len(per_chunk) != len(batched):
        return {"pass": False, "error": f"pack count mismatch {len(per_chunk)} != {len(batched)}"}
    max_feature_error = 0.0
    max_alpha_error = 0.0
    pixel_mismatches = 0
    for expected, actual in zip(per_chunk, batched, strict=True):
        expected_pixels = expected["pixel_ids"].to(torch.int64)
        actual_pixels = actual["pixel_ids"].to(torch.int64)
        if expected_pixels.shape != actual_pixels.shape or not bool(torch.equal(expected_pixels.cpu(), actual_pixels.cpu())):
            pixel_mismatches += 1
            continue
        feature_error = (expected["grad_feature_values"] - actual["grad_feature_values"]).abs()
        alpha_error = (expected["grad_alpha_values"] - actual["grad_alpha_values"]).abs()
        max_feature_error = max(max_feature_error, float(feature_error.max().item()) if feature_error.numel() else 0.0)
        max_alpha_error = max(max_alpha_error, float(alpha_error.max().item()) if alpha_error.numel() else 0.0)
    return {
        "pass": pixel_mismatches == 0 and max_feature_error <= 1.0e-7 and max_alpha_error <= 1.0e-7,
        "pixel_mismatches": pixel_mismatches,
        "max_feature_grad_error": max_feature_error,
        "max_alpha_grad_error": max_alpha_error,
    }


def profile(config_path: Path, *, warmup: int, repeat: int) -> dict[str, Any]:
    case = _load_case(config_path)
    cfg = case["cfg"]
    feature_config = case["feature_config"]
    chunk_size_cfg = cfg["train"]["frame_chunk_size"]
    chunk_size = int(feature_config.frames) if chunk_size_cfg is None else min(int(chunk_size_cfg), int(feature_config.frames))
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")
    context = _target_context(case)
    sparse = _load_sparse_forward_chunks(case, chunk_size)
    chunks = sparse["chunks"]
    per_samples: dict[str, list[float]] = {
        "feature_target_ms": [],
        "rgb_probe_loss_ms": [],
        "image_vjp_ms": [],
        "total_loss_vjp_ms": [],
    }
    batched_samples: dict[str, list[float]] = {
        "feature_target_ms": [],
        "rgb_probe_loss_ms": [],
        "image_vjp_ms": [],
        "total_loss_vjp_ms": [],
    }
    comparison: dict[str, Any] | None = None
    loss_error = 0.0
    for iteration in range(warmup + repeat):
        per = _profile_per_chunk(case, chunks, context)
        batched = _profile_batched(case, chunks, context)
        if iteration >= warmup:
            for key, values in per_samples.items():
                values.append(float(per["timing_ms"][key]))
            for key, values in batched_samples.items():
                values.append(float(batched["timing_ms"][key]))
            comparison = _compare_packs(per["packs"], batched["packs"])
            loss_error = max(loss_error, abs(float(per["loss"]) - float(batched["loss"])))
    if comparison is None:
        raise RuntimeError("repeat must be positive")
    per_stats = {key: distribution_stats(values) for key, values in per_samples.items()}
    batched_stats = {key: distribution_stats(values) for key, values in batched_samples.items()}
    speedup = per_stats["total_loss_vjp_ms"]["mean"] / batched_stats["total_loss_vjp_ms"]["mean"]
    pass_flag = (
        bool(comparison["pass"])
        and loss_error <= 1.0e-7
        and int(sparse["tile_overflow_sum"]) == 0
        and float(speedup) > 1.0
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate": "star_uvt_sparse_forward_batched_target_vjp_profile",
        "config": str(config_path),
        "frames": int(feature_config.frames),
        "size": int(feature_config.height),
        "feature_dim": int(feature_config.feature_dim),
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "warmup": int(warmup),
        "repeat": int(repeat),
        "feature_loss_weight": float(context["feature_loss_weight"]),
        "rgb_probe_loss_weight": float(context["rgb_probe_loss_weight"]),
        "sparse_forward_render_ms": float(sparse["render_ms"]),
        "sparse_pixel_count": int(sparse["total_sparse_pixels"]),
        "sparse_pixel_fraction": float(sparse["total_sparse_pixels"]) / float(sparse["total_dense_pixels"]),
        "max_tile_count": int(sparse["max_tile_count"]),
        "tile_overflow_sum": int(sparse["tile_overflow_sum"]),
        "per_chunk_timing_ms": per_stats,
        "batched_timing_ms": batched_stats,
        "loss_error": float(loss_error),
        "pack_comparison": comparison,
        "speedup_total_loss_vjp": float(speedup),
        "pass": pass_flag,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    per = result["per_chunk_timing_ms"]
    batched = result["batched_timing_ms"]
    rows = []
    for key in ("feature_target_ms", "rgb_probe_loss_ms", "image_vjp_ms", "total_loss_vjp_ms"):
        rows.append(
            "| "
            + " | ".join(
                (
                    key,
                    _fmt(per[key]["mean"]),
                    _fmt(batched[key]["mean"]),
                    _fmt(per[key]["mean"] / batched[key]["mean"] if batched[key]["mean"] else 0.0),
                    _fmt(batched[key]["min"]),
                    _fmt(batched[key]["max"]),
                )
            )
            + " |"
        )
    lines = [
        "# STAR UVT Sparse-Forward Batched Target VJP Profile",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Scope",
        "",
        "Profiles the remaining Torch-side target-grid/frozen-probe loss and sparse VJP work after sparse feature forward.",
        "The batched path stacks all frame chunks and computes the target-grid feature loss, hidden64 frozen RGB-probe VJP,",
        "and sparse target-grid VJP in one batched MPS path. This is a preflight for native target-grid/probe loss+VJP, not a new renderer.",
        "",
        "## Result",
        "",
        "| metric | per-chunk mean | batched mean | speedup | batched min | batched max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Validation",
        "",
        f"- pass: `{result['pass']}`",
        f"- chunks: `{result['chunk_count']}` x `{result['chunk_size']}` frames",
        f"- sparse pixels: `{result['sparse_pixel_count']}` (`{result['sparse_pixel_fraction']:.6f}` of dense)",
        f"- sparse render preload: `{result['sparse_forward_render_ms']:.3f}ms`",
        f"- tile overflow / max tile: `{result['tile_overflow_sum']}` / `{result['max_tile_count']}`",
        f"- loss error: `{result['loss_error']:.3e}`",
        f"- max feature grad error: `{result['pack_comparison']['max_feature_grad_error']:.3e}`",
        f"- max alpha grad error: `{result['pack_comparison']['max_alpha_grad_error']:.3e}`",
        f"- pixel mismatches: `{result['pack_comparison']['pixel_mismatches']}`",
        "",
        "## Decision",
        "",
    ]
    if result["pass"]:
        lines.extend(
            [
                "The batched target-grid/probe VJP is a valid implementation candidate.",
                "If the trainer can batch this work across chunks without changing optimizer semantics, it should be benchmarked against the repeat-3 sparse-forward distribution before spending on a lower-level native kernel.",
            ]
        )
    else:
        lines.extend(
            [
                "Do not promote this path. Treat the timings and validation fields above as the reason it failed, and keep the native target-grid/probe loss+VJP or scalar fixedbin/tile-slot shader as the next speed gate.",
            ]
        )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    args = parser.parse_args()
    result = profile(args.config, warmup=args.warmup, repeat=args.repeat)
    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    write_report_json(args.out_base.with_suffix(".json"), result)
    write_markdown(args.out_base.with_suffix(".md"), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
