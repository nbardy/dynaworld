from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import load_report_json, read_report_csv, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import load_report_json, read_report_csv, write_report_json, write_report_text


def _fmt(value: Any, *, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _rgb_star_row(path: Path) -> dict[str, Any]:
    data = load_report_json(path)
    uvt = data["uvt"]
    return {
        "kind": "rgb_star_direct_atomic",
        "path": str(path),
        "target_source": data.get("video_path"),
        "frames": int(data["frames"]),
        "size": int(data["height"]),
        "tubes": int(uvt["tube_count"]),
        "feature_dim": 3,
        "steps": int(data["steps"]),
        "mode": str(uvt.get("sample_emission_mode")),
        "decoder": "rgb_direct",
        "pre_norm": None,
        "pass": bool(float(uvt["final_loss"]) < float(uvt["initial_loss"])),
        "start_loss": float(uvt["initial_loss"]),
        "end_loss": float(uvt["final_loss"]),
        "start_psnr": None,
        "end_psnr": float(uvt["final_psnr"]),
        "mean_step_ms": float(uvt["wall_clock_ms"]) / float(data["steps"]),
        "backward_ms": None,
        "render_ms": float(uvt["render_ms"]),
        "tile_overflow_sum": None,
        "media": [value for value in (data.get("contact_sheet"), data.get("side_by_side_video")) if value],
    }


def _feature_star_row(path: Path) -> dict[str, Any]:
    data = load_report_json(path)
    mean = data.get("mean_timing_ms", {})
    return {
        "kind": "feature_star",
        "path": str(path),
        "target_source": data.get("target_source"),
        "frames": int(data["frames"]),
        "size": int(data["size"]),
        "tubes": int(data["tubes"]),
        "feature_dim": int(data["feature_dim"]),
        "steps": int(data["steps"]),
        "mode": str(data.get("requested_render_mode") or data.get("effective_render_mode")),
        "decoder": (
            f"{data.get('colorize_activation') or 'sigmoid'}"
            f"/hidden={data.get('colorize_hidden_dim')}"
            f"/pre_norm={data.get('colorize_pre_norm')}"
            f"/gain={data.get('colorize_weight_init_gain')}"
        ),
        "pre_norm": bool(data["colorize_pre_norm"]),
        "pass": bool(data.get("pass")),
        "start_loss": float(data["start_loss"]),
        "end_loss": float(data["end_loss"]),
        "start_psnr": float(data["start_psnr"]),
        "end_psnr": float(data["end_psnr"]),
        "mean_step_ms": float(mean["step_ms"]),
        "backward_ms": float(mean["backward_ms"]),
        "render_ms": float(mean["render_forward_ms"]),
        "tile_overflow_sum": int(data.get("tile_overflow_sum", 0)),
        "media": [value for value in (data.get("contact_sheet"), data.get("side_by_side_video")) if value],
    }


def _read_speed_refs(path: Path, *, size: int, frames: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in read_report_csv(path):
        if row.get("status") != "ok":
            continue
        if int(row.get("resolution") or -1) != size or int(row.get("frames") or -1) != frames:
            continue
        family = row.get("family")
        if family not in {"star_uvt_kernel", "dynamic_gsplat_rgb_raster", "feature_f32_raster"}:
            continue
        rows.append(
            {
                "family": family,
                "renderer": row.get("renderer"),
                "resolution": int(row["resolution"]),
                "frames": int(row["frames"]),
                "primitives": int(row["primitives"]),
                "feature_dim": int(row["feature_dim"]),
                "total_ms": None if not row.get("total_ms") else float(row["total_ms"]),
                "forward_ms": None if not row.get("forward_ms") else float(row["forward_ms"]),
                "backward_ms": None if not row.get("backward_ms") else float(row["backward_ms"]),
                "overflow_tiles": None if not row.get("overflow_tiles") else int(float(row["overflow_tiles"])),
                "artifact": row.get("artifact"),
                "note": row.get("note"),
            }
        )
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row["family"])
        if family not in best or float(row["total_ms"] or 1e30) < float(best[family]["total_ms"] or 1e30):
            best[family] = row
    return [best[key] for key in sorted(best)]


def _write_markdown(
    path: Path,
    *,
    quality_rows: list[dict[str, Any]],
    speed_refs: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    lines = [
        "# STAR UVT Gate 4 Quality Bracket",
        "",
        "Same-clip quality rows use `test_data/test_video_384_128_6fps.mp4`, 64 frames, 512px center crop, 8192 tubes, and 20 optimizer steps.",
        "Projected dynamic/F32 rows are speed references only; they are synthetic raster benchmarks, not same-source quality gates.",
        "",
        "## Verdict",
        "",
        f"- Feature rows meet RGB STAR 20-step PSNR target: `{_fmt(comparison['feature_meets_rgb_psnr'])}`.",
        f"- RGB STAR best PSNR: `{_fmt(comparison['rgb_best_psnr'])}`.",
        f"- Best feature PSNR: `{_fmt(comparison['feature_best_psnr'])}`.",
        f"- Fastest feature step: `{_fmt(comparison['feature_fastest_step_ms'])}ms`.",
        "",
        "## Same-Clip Quality Rows",
        "",
        "| kind | mode | decoder | pass | loss | PSNR | step ms | backward ms | render/fwd ms | overflow | artifact |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in quality_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["kind"]),
                    str(row["mode"]),
                    str(row["decoder"]),
                    _fmt(row["pass"]),
                    f"{_fmt(row['start_loss'], precision=5)} -> {_fmt(row['end_loss'], precision=5)}",
                    f"{_fmt(row['start_psnr'])} -> {_fmt(row['end_psnr'])}",
                    _fmt(row["mean_step_ms"], precision=1),
                    _fmt(row["backward_ms"], precision=1),
                    _fmt(row["render_ms"], precision=1),
                    _fmt(row["tile_overflow_sum"], precision=0),
                    f"`{row['path']}`",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Speed-Only References",
            "",
            "| family | renderer | res | primitives | F | total ms | fwd ms | bwd ms | overflow | artifact |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in speed_refs:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["family"]),
                    str(row["renderer"]),
                    _fmt(row["resolution"], precision=0),
                    _fmt(row["primitives"], precision=0),
                    _fmt(row["feature_dim"], precision=0),
                    _fmt(row["total_ms"], precision=1),
                    _fmt(row["forward_ms"], precision=1),
                    _fmt(row["backward_ms"], precision=1),
                    _fmt(row["overflow_tiles"], precision=0),
                    f"`{row['artifact']}`",
                )
            )
            + " |"
    )
    lines.append("")
    write_report_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-star-json", type=Path, action="append", required=True)
    parser.add_argument("--feature-json", type=Path, action="append", required=True)
    parser.add_argument("--renderer-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    quality_rows = [_rgb_star_row(path) for path in args.rgb_star_json]
    quality_rows.extend(_feature_star_row(path) for path in args.feature_json)
    rgb_rows = [row for row in quality_rows if row["kind"] == "rgb_star_direct_atomic" and row["pass"]]
    feature_rows = [row for row in quality_rows if row["kind"] == "feature_star" and row["pass"]]
    if not rgb_rows:
        raise SystemExit("No passing RGB STAR rows.")
    if not feature_rows:
        raise SystemExit("No passing feature STAR rows.")
    size = int(rgb_rows[0]["size"])
    frames = int(rgb_rows[0]["frames"])
    rgb_best_psnr = max(float(row["end_psnr"]) for row in rgb_rows)
    feature_best_psnr = max(float(row["end_psnr"]) for row in feature_rows)
    comparison = {
        "rgb_best_psnr": rgb_best_psnr,
        "feature_best_psnr": feature_best_psnr,
        "feature_meets_rgb_psnr": bool(feature_best_psnr >= rgb_best_psnr),
        "feature_fastest_step_ms": min(float(row["mean_step_ms"]) for row in feature_rows),
    }
    speed_refs = _read_speed_refs(args.renderer_csv, size=size, frames=frames)
    payload = {
        "quality_rows": quality_rows,
        "speed_refs": speed_refs,
        "comparison": comparison,
    }
    write_report_json(args.out_json, payload)
    _write_markdown(args.out_md, quality_rows=quality_rows, speed_refs=speed_refs, comparison=comparison)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), **comparison}, sort_keys=True))


if __name__ == "__main__":
    main()
