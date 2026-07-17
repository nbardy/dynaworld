#!/usr/bin/env python3
"""Compare compact STAR UVT target-area visual VJP gates from trainer JSONs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import load_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import load_report_json, write_report_text


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else float("nan")


def _timing_mean(row: dict[str, Any], key: str, *, skip_first: bool = False, limit: int = 5) -> float:
    timings = row.get("step_timings_ms") or []
    selected = timings[:limit]
    if skip_first:
        selected = selected[1:]
    return _mean([float(item[key]) for item in selected if key in item])


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": row.get("pass"),
        "colorizer": f"{_fmt(row.get('colorizer_grad_required'), 0)}/{_fmt(row.get('colorizer_grad_seen'), 0)}",
        "loss": f"{_fmt(row.get('start_loss'), 6)} -> {_fmt(row.get('end_loss'), 6)}",
        "feature": f"{_fmt(row.get('start_feature_target_loss'), 6)} -> {_fmt(row.get('end_feature_target_loss'), 6)}",
        "probe": f"{_fmt(row.get('start_rgb_probe_psnr'), 3)} -> {_fmt(row.get('end_rgb_probe_psnr'), 3)}",
        "sparse": f"{_fmt(row.get('start_sparse_visual_loss'), 6)} -> {_fmt(row.get('end_sparse_visual_loss'), 6)}",
    }


def _timings(row: dict[str, Any]) -> dict[str, float]:
    return {
        "step": _timing_mean(row, "step_ms"),
        "step_no_first": _timing_mean(row, "step_ms", skip_first=True),
        "backward": _timing_mean(row, "backward_ms"),
        "backward_no_first": _timing_mean(row, "backward_ms", skip_first=True),
        "visual_render": _timing_mean(row, "sparse_visual_render_ms"),
        "visual_loss": _timing_mean(row, "sparse_visual_loss_ms"),
        "visual_backward": _timing_mean(row, "sparse_visual_backward_ms"),
    }


def _row(label: str, path: Path) -> dict[str, Any]:
    data = load_report_json(path)
    return {
        "label": label,
        "path": path,
        "mode": data.get("sparse_visual_loss_vjp_mode") or "autograd",
        "quality": _quality(data),
        "timings": _timings(data),
        "tile_overflow_sum": data.get("tile_overflow_sum"),
        "sparse_visual_fraction": data.get("mean_sparse_visual_pixel_fraction"),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# STAR UVT Compact Target-Area Visual VJP Gate",
        "",
        "Date: 2026-05-20",
        "",
        "## Purpose",
        "",
        "Compare the current compact target-area visual route against three colorizer/VJP follow-ups:",
        "native target-area star-only vec4 W^T, manual hidden64 target-area VJP with colorizer gradients,",
        "and native hidden64 target-area vec4 W^T with returned colorizer gradients.",
        "The gate is intentionally compact (`patch_shape=[2,2]`, 6.25% visual support) because the full-cell8",
        "target-area path was already too slow for the single-video overfit route.",
        "",
        "## Result",
        "",
        "Keep compact autograd as the practical visual route. Manual hidden64 proves colorizer gradients can be",
        "computed without PyTorch autograd, but it is slower and destabilizes the first 5-step quality gate.",
        "Native star-only proves compact native shape plumbing, but it freezes the colorizer and is slower than",
        "the compact autograd keeper. Native colorizer-gradient vec4 W^T proves the missing parameter-gradient",
        "return path, but it is slower again. The next native port must reduce the colorizer-gradient atomic",
        "envelope or change support/objective; returning the gradients alone is not enough.",
        "",
        "## Timing",
        "",
        "All timing columns below use the first five recorded trainer steps so the 50-step compact-autograd",
        "keeper can be compared directly to the 5-step diagnostics. `No-first` drops the warm first step.",
        "",
        "| Route | Mode | Pass | Colorizer grad req/seen | Mean step ms | No-first step ms | Mean backward ms | No-first backward ms | Visual render ms | Visual loss ms | Visual backward ms | Overflow | Support |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        q = item["quality"]
        t = item["timings"]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["label"],
                    str(item["mode"]),
                    _fmt(q["pass"], 0),
                    q["colorizer"],
                    _fmt(t["step"]),
                    _fmt(t["step_no_first"]),
                    _fmt(t["backward"]),
                    _fmt(t["backward_no_first"]),
                    _fmt(t["visual_render"]),
                    _fmt(t["visual_loss"]),
                    _fmt(t["visual_backward"]),
                    _fmt(item["tile_overflow_sum"], 0),
                    _fmt(float(item["sparse_visual_fraction"]) * 100.0 if item["sparse_visual_fraction"] is not None else None, 2) + "%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Quality",
            "",
            "| Route | Weighted loss | Feature loss | RGB-probe PSNR | Sparse visual loss |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in rows:
        q = item["quality"]
        lines.append(
            f"| {item['label']} | {q['loss']} | {q['feature']} | {q['probe']} | {q['sparse']} |"
        )
    lines.extend(
        [
            "",
            "## Source JSONs",
            "",
        ]
    )
    for item in rows:
        lines.append(f"- `{item['path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autograd",
        type=Path,
        default=Path(
            "outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.json"
        ),
    )
    parser.add_argument(
        "--native-star-only",
        type=Path,
        default=Path(
            "outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_nativehidden_vec4wt_from1500_lr001_5step_diagnostic.json"
        ),
    )
    parser.add_argument(
        "--manual-hidden64",
        type=Path,
        default=Path(
            "outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_manualhidden64_from1500_lr001_5step_diagnostic.json"
        ),
    )
    parser.add_argument(
        "--native-colorizer",
        type=Path,
        default=Path(
            "outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_nativecolorizer_vec4wt_from1500_lr001_5step_diagnostic.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/benchmarks/2026-05-20_star_uvt_compact_visual_vjp_gate.md"),
    )
    args = parser.parse_args()

    rows = [
        _row("compact autograd", args.autograd),
        _row("compact native star-only vec4 W^T", args.native_star_only),
        _row("compact manual hidden64", args.manual_hidden64),
        _row("compact native colorizer vec4 W^T", args.native_colorizer),
    ]
    text = _markdown(rows)
    write_report_text(args.out, text)
    print(args.out)


if __name__ == "__main__":
    main()
