from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from direct_feature_kernel_benchmark import _random_timing_scene, _sync
try:
    from .report_artifacts import split_csv_ints, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import split_csv_ints, write_report_json, write_report_text
from torch_gsplat_bridge_star_uvt.feature_rasterize import render_uvt_feature_tubes


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _fmt_ms(value: float) -> str:
    return f"{value:.1f}"


def _fmt_float(value: float) -> str:
    return f"{value:.3f}"


def run_budget_case(
    *,
    frames: int,
    size: int,
    tubes: int,
    feature_dim: int,
    seed: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    ma, q, depth0, depth_beta, opacity, feature, config = _random_timing_scene(
        frames=frames,
        height=size,
        width=size,
        tubes=tubes,
        feature_dim=feature_dim,
        seed=seed,
    )
    mps_inputs = [tensor.to("mps").contiguous() for tensor in (ma, q, depth0, depth_beta, opacity, feature)]
    forward_samples: list[float] = []
    metal = None
    for iteration in range(warmup + repeat):
        _sync()
        t0 = time.perf_counter()
        metal = render_uvt_feature_tubes(*mps_inputs, config, return_bins=True)
        _sync()
        forward_ms = (time.perf_counter() - t0) * 1000.0
        if iteration >= warmup:
            forward_samples.append(forward_ms)
    assert metal is not None

    counts = metal.tile_counts.detach().cpu().to(torch.int64)
    overflows = metal.tile_overflow.detach().cpu().to(torch.int64)
    capacity = int(config.tile_capacity)
    clipped = torch.clamp(counts, max=capacity)
    clipped_list = [int(value) for value in clipped.tolist()]
    count_list = [int(value) for value in counts.tolist()]
    tile_pixels = int(config.tile_x * config.tile_y * config.tile_t)
    tile_count = int(counts.numel())
    occupied_slots = int(clipped.sum().item())
    overflow_tiles = int((overflows > 0).sum().item())
    overflow_sum = int(overflows.sum().item())
    direct_slot_pixel_visits = occupied_slots * tile_pixels
    direct_feature_atomic_writes = direct_slot_pixel_visits * int(feature_dim)
    tile_slot_feature_atomic_writes = occupied_slots * int(feature_dim)
    prefix_pair_visits = int((clipped * (clipped + 1) // 2).sum().item()) * tile_pixels
    dense_weight_bytes_f32 = direct_slot_pixel_visits * 4
    dense_weight_bytes_f16 = direct_slot_pixel_visits * 2
    dense_per_channel_weight_bytes_f32 = direct_feature_atomic_writes * 4
    feature_image_bytes_f32 = int(frames) * int(size) * int(size) * int(feature_dim) * 4
    tile_slot_feature_grad_bytes = tile_slot_feature_atomic_writes * 4

    return {
        "frames": frames,
        "size": size,
        "tubes": tubes,
        "feature_dim": feature_dim,
        "tile_capacity": capacity,
        "tile_pixels": tile_pixels,
        "tile_count": tile_count,
        "forward_ms": sum(forward_samples) / float(len(forward_samples)),
        "forward_ms_samples": forward_samples,
        "occupied_slots": occupied_slots,
        "mean_slots_per_tile": float(occupied_slots) / float(max(tile_count, 1)),
        "p95_slots_per_tile": _percentile(clipped_list, 95.0),
        "max_slots_per_tile": int(max(clipped_list) if clipped_list else 0),
        "raw_p95_slots_per_tile": _percentile(count_list, 95.0),
        "raw_max_slots_per_tile": int(max(count_list) if count_list else 0),
        "overflow_tiles": overflow_tiles,
        "overflow_sum": overflow_sum,
        "direct_slot_pixel_visits": direct_slot_pixel_visits,
        "direct_feature_atomic_writes": direct_feature_atomic_writes,
        "tile_slot_feature_atomic_writes": tile_slot_feature_atomic_writes,
        "atomic_write_reduction_x": (
            float(direct_feature_atomic_writes) / float(tile_slot_feature_atomic_writes)
            if tile_slot_feature_atomic_writes
            else 0.0
        ),
        "prefix_pair_visits": prefix_pair_visits,
        "prefix_recompute_x_vs_direct_slot_pixel": (
            float(prefix_pair_visits) / float(direct_slot_pixel_visits) if direct_slot_pixel_visits else 0.0
        ),
        "feature_image_gib_f32": float(feature_image_bytes_f32) / (1024.0**3),
        "dense_weight_tape_gib_f32": float(dense_weight_bytes_f32) / (1024.0**3),
        "dense_weight_tape_gib_f16": float(dense_weight_bytes_f16) / (1024.0**3),
        "dense_weight_tape_vs_feature_image": (
            float(dense_weight_bytes_f32) / float(feature_image_bytes_f32) if feature_image_bytes_f32 else 0.0
        ),
        "dense_per_channel_weight_tape_gib_f32": float(dense_per_channel_weight_bytes_f32) / (1024.0**3),
        "tile_slot_feature_grad_mib_f32": float(tile_slot_feature_grad_bytes) / (1024.0**2),
        "finite": bool(torch.isfinite(metal.feature_image).all().cpu())
        and bool(torch.isfinite(metal.alpha).all().cpu()),
    }


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = (
        "size",
        "forward_ms",
        "tile_count",
        "occupied_slots",
        "mean_slots_per_tile",
        "p95_slots_per_tile",
        "max_slots_per_tile",
        "overflow_sum",
        "direct_feature_atomic_writes_m",
        "tile_slot_feature_atomic_writes_m",
        "atomic_write_reduction_x",
        "prefix_recompute_x_vs_direct_slot_pixel",
        "feature_image_gib_f32",
        "dense_weight_tape_gib_f32",
        "dense_weight_tape_vs_feature_image",
        "dense_per_channel_weight_tape_gib_f32",
        "tile_slot_feature_grad_mib_f32",
    )
    lines = [
        "# STAR UVT Feature Tile-Slot Accumulator Budget",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This is a fixedbin/tile-slot feasibility gate. It uses the current forward",
        "bins to estimate the work and memory shape for replacing per-pixel",
        "feature-gradient atomics with one feature-gradient atomic per tile slot",
        "and channel.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        display = {
            **row,
            "forward_ms": _fmt_ms(float(row["forward_ms"])),
            "mean_slots_per_tile": _fmt_float(float(row["mean_slots_per_tile"])),
            "p95_slots_per_tile": _fmt_float(float(row["p95_slots_per_tile"])),
            "atomic_write_reduction_x": _fmt_float(float(row["atomic_write_reduction_x"])),
            "prefix_recompute_x_vs_direct_slot_pixel": _fmt_float(
                float(row["prefix_recompute_x_vs_direct_slot_pixel"])
            ),
            "feature_image_gib_f32": _fmt_float(float(row["feature_image_gib_f32"])),
            "dense_weight_tape_gib_f32": _fmt_float(float(row["dense_weight_tape_gib_f32"])),
            "dense_weight_tape_vs_feature_image": _fmt_float(float(row["dense_weight_tape_vs_feature_image"])),
            "dense_per_channel_weight_tape_gib_f32": _fmt_float(
                float(row["dense_per_channel_weight_tape_gib_f32"])
            ),
            "tile_slot_feature_grad_mib_f32": _fmt_float(float(row["tile_slot_feature_grad_mib_f32"])),
            "direct_feature_atomic_writes_m": _fmt_float(float(row["direct_feature_atomic_writes"]) / 1.0e6),
            "tile_slot_feature_atomic_writes_m": _fmt_float(
                float(row["tile_slot_feature_atomic_writes"]) / 1.0e6
            ),
        }
        lines.append("| " + " | ".join(str(display.get(key, "")) for key in columns) + " |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `atomic_write_reduction_x` is the theoretical write-count reduction if",
            "  feature gradients are accumulated per tile slot and channel instead of",
            "  per pixel, slot, and channel.",
            "- `prefix_recompute_x_vs_direct_slot_pixel` is the naive cost multiplier",
            "  if each tile-slot accumulator recomputes transmittance prefixes instead",
            "  of reusing a compact prefix/weight structure.",
            "- `dense_weight_tape_gib_f32` is the memory needed to store one f32",
            "  contribution weight per tile slot and tile pixel. It excludes feature",
            "  channels, so multiplying it by feature dimension would be the wrong",
            "  storage design.",
            "- `dense_per_channel_weight_tape_gib_f32` shows that wrong design: a",
            "  per-feature-channel tape is far larger than the already-dense feature",
            "  image.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="128,256,512")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--tubes", type=int, default=32768)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("tile-slot accumulator budget requires MPS")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir or Path("outputs/benchmarks") / f"{timestamp}_star_uvt_feature_tile_slot_budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in split_csv_ints(args.sizes):
        print(f"[tile-slot-budget] size={size}")
        rows.append(
            run_budget_case(
                frames=args.frames,
                size=size,
                tubes=args.tubes,
                feature_dim=args.feature_dim,
                seed=args.seed,
                warmup=args.warmup,
                repeat=args.repeat,
            )
        )
    payload = {
        "args": vars(args) | {"out_dir": str(out_dir)},
        "rows": rows,
    }
    write_report_json(out_dir / "summary.json", payload)
    write_markdown(rows, out_dir / "summary.md")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
