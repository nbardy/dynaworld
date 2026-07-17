from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import (
        ROOT,
        load_research_json,
        load_research_jsonl,
        read_research_csv,
        write_research_csv,
        write_research_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        load_research_json,
        load_research_jsonl,
        read_research_csv,
        write_research_csv,
        write_research_text,
    )

STAR_DIRS = (
    ROOT / "outputs/benchmarks/2026-05-18_star_uvt_scale_128_64f_32768_top",
    ROOT / "outputs/benchmarks/2026-05-18_star_uvt_scale_256_64f_32768_top",
    ROOT / "outputs/benchmarks/2026-05-18_star_uvt_scale_512_64f_32768_top",
)
RGB_JSONL = ROOT / "outputs/benchmarks/2026-05-18_fastmac_rgb_dynamic_B64_G32768_res128_256_512.jsonl"
FEATURE_JSONLS = (
    ROOT / "outputs/benchmarks/2026-05-18_feature_f32_B64_G32768_res128.jsonl",
    ROOT / "outputs/benchmarks/2026-05-18_feature_f32_B64_G32768_res256.jsonl",
    ROOT / "outputs/benchmarks/2026-05-18_feature_f32_B64_G32768_res512.jsonl",
)
STAR_FEATURE_DIRECT_GLOB = "2026-05-18_star_uvt_feature_direct_metal*.json"
STAR_FEATURE_FIRSTCLASS_GLOB = "*_star_uvt_feature_firstclass_testvideo_*.json"
CURRENT_STAR_KERNEL_GLOB = "*_star_uvt_backward_*/summary.csv"


def _available_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path for path in paths if path.exists())


def _latest_star_feature_summary() -> Path:
    candidates = sorted((ROOT / "outputs/benchmarks").glob("*_star_uvt_feature_firstclass_scale_summary.json"))
    if not candidates:
        return ROOT / "outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_summary.json"
    return candidates[-1]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _command_arg(wrapper: dict[str, Any], name: str) -> str | None:
    command = wrapper.get("command")
    if not isinstance(command, list):
        return None
    for index, value in enumerate(command):
        if value == name and index + 1 < len(command):
            return str(command[index + 1])
    return None


def _read_star_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    historical_paths = tuple(directory / "summary.csv" for directory in _available_paths(STAR_DIRS))
    current_paths = tuple(sorted((ROOT / "outputs/benchmarks").glob(CURRENT_STAR_KERNEL_GLOB)))
    for csv_path in historical_paths + current_paths:
        for row in read_research_csv(csv_path):
            if not row.get("target_size"):
                # The per-case matrix preserves unsupported/OOM rows; omit them
                # here because this resolution-indexed aggregation has no key.
                continue
            mode = row.get("mode") or ""
            reducer = row.get("reducer") or ""
            variant = row.get("variant") or "star_uvt_v0"
            total_ms = _float(row.get("sample_plus_reduce_ms") or row.get("wall_ms") or row.get("fused_median_ms"))
            backward_ms = _float(row.get("sample_backward_ms") or row.get("backward_ms"))
            rows.append(
                {
                    "family": "star_uvt_kernel",
                    "renderer": f"{variant}/{mode}+{reducer}",
                    "resolution": int(row.get("target_size") or 0),
                    "frames": int(row.get("frames") or 0),
                    "primitives": int(row.get("tube_count") or 0),
                    "feature_dim": 3,
                    "status": row["status"],
                    "steps": None,
                    "total_ms": total_ms,
                    "forward_ms": _float(row.get("forward_ms")),
                    "backward_ms": backward_ms,
                    "sample_count": int(float(row["sample_count"])) if row.get("sample_count") else None,
                    "mean_pairs_per_tile": None,
                    "overflow_tiles": int(row.get("overflow_tile_count") or row.get("fused_overflow_tile_count") or 0),
                    "artifact": str(csv_path.relative_to(ROOT)),
                    "note": row.get("sample_unit") or row.get("error") or "",
                }
            )
    return rows


def _read_rgb_rows() -> list[dict[str, Any]]:
    if not RGB_JSONL.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in load_research_jsonl(RGB_JSONL):
        rows.append(
            {
                "family": "dynamic_gsplat_rgb_raster",
                "renderer": row.get("variant"),
                "resolution": int(row["resolution"]),
                "frames": int(row["batch_size_requested"]),
                "primitives": int(row["gaussians_requested"]),
                "feature_dim": 3,
                "status": row["status"],
                "steps": None,
                "total_ms": _float(row.get("median_ms") or row.get("total_median_ms")),
                "forward_ms": _float(row.get("forward_ms") or row.get("forward_median_ms")),
                "backward_ms": _float(row.get("backward_ms") or row.get("backward_median_ms")),
                "sample_count": None,
                "mean_pairs_per_tile": _float(row.get("profile_mean_pairs_per_tile")),
                "overflow_tiles": int(row["profile_overflow_tile_count"]) if row.get("profile_overflow_tile_count") is not None else None,
                "artifact": str(RGB_JSONL.relative_to(ROOT)),
                "note": "projected raster synthetic B=frames",
            }
        )
    return rows


def _read_feature_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _available_paths(FEATURE_JSONLS):
        for wrapper in load_research_jsonl(path):
            result = wrapper.get("result") if isinstance(wrapper.get("result"), dict) else {}
            status = str(wrapper.get("status"))
            height = result.get("height") or wrapper.get("height") or _command_arg(wrapper, "--height")
            batch = result.get("batch_size") or wrapper.get("batch_size") or _command_arg(wrapper, "--batch-size")
            gaussians = result.get("gaussians") or wrapper.get("gaussians") or _command_arg(wrapper, "--gaussians")
            feature_dim = result.get("feature_dim") or wrapper.get("feature_dim") or _command_arg(wrapper, "--feature-dim")
            rows.append(
                {
                    "family": "feature_f32_raster",
                    "renderer": wrapper.get("variant"),
                    "resolution": int(height) if height is not None else None,
                    "frames": int(batch) if batch is not None else 64,
                    "primitives": int(gaussians) if gaussians is not None else 32768,
                    "feature_dim": int(feature_dim) if feature_dim is not None else 32,
                    "status": status,
                    "steps": None,
                    "total_ms": _float(result.get("median_ms")),
                    "forward_ms": _float(result.get("forward_ms")),
                    "backward_ms": _float(result.get("backward_ms")),
                    "sample_count": None,
                    "mean_pairs_per_tile": _float(result.get("profile_mean_pairs_per_tile")),
                    "overflow_tiles": int(result["profile_overflow_tile_count"]) if result.get("profile_overflow_tile_count") is not None else None,
                    "artifact": str(path.relative_to(ROOT)),
                    "note": wrapper.get("error") or wrapper.get("stderr") or "projected F32 raster synthetic B=frames",
                }
            )
    return rows


def _read_star_feature_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_research_json(path)
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows") or []:
        requested_mode = str(row.get("requested_render_mode") or "feature_direct_atomic")
        effective_mode = str(row.get("effective_render_mode") or requested_mode)
        renderer = str(row.get("variant"))
        if requested_mode != "feature_direct_atomic":
            renderer = f"{renderer}/{requested_mode}"
            if effective_mode != requested_mode:
                renderer = f"{renderer}->{effective_mode}"
        rows.append(
            {
                "family": "star_uvt_feature_firstclass",
                "renderer": renderer,
                "resolution": int(row["size"]),
                "frames": int(row["frames"]),
                "primitives": int(row["tubes"]),
                "feature_dim": int(row["feature_dim"]),
                "status": "ok" if bool(row.get("pass")) else "nonpassing",
                "steps": int(row["steps"]),
                "total_ms": _float(row.get("mean_step_ms")),
                "forward_ms": _float(row.get("mean_forward_ms")),
                "backward_ms": _float(row.get("mean_backward_ms")),
                "sample_count": None,
                "mean_pairs_per_tile": _float(row.get("mean_active_tile_count")),
                "overflow_tiles": int(row.get("tile_overflow_sum", 0)),
                "artifact": str(Path(str(row.get("source_json"))).as_posix()),
                "note": (
                    f"real-video feature tubes cap={row.get('tile_capacity')} "
                    f"chunk={row.get('frame_chunk_size')} fixedbin={row.get('fixedbin_eligible')} "
                    f"mode={requested_mode}->{effective_mode} "
                    f"loss={row.get('start_loss'):.5f}->{row.get('end_loss'):.5f}"
                    if row.get("start_loss") is not None and row.get("end_loss") is not None
                    else "real-video feature tubes"
                ),
            }
        )
    return rows


def _read_star_feature_individual_rows(existing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_artifacts = {str(row.get("artifact")) for row in existing_rows}
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "outputs/benchmarks").glob(STAR_FEATURE_FIRSTCLASS_GLOB)):
        artifact = str(path.relative_to(ROOT))
        if artifact in existing_artifacts:
            continue
        try:
            payload = load_research_json(path)
        except ValueError:
            continue
        timing = payload.get("mean_timing_ms")
        if not isinstance(timing, dict):
            continue
        requested_mode = str(payload.get("requested_render_mode") or "feature_direct_atomic")
        effective_mode = str(payload.get("effective_render_mode") or requested_mode)
        renderer = str(payload.get("variant") or "default")
        if requested_mode != "feature_direct_atomic":
            renderer = f"{renderer}/{requested_mode}"
            if effective_mode != requested_mode:
                renderer = f"{renderer}->{effective_mode}"
        start_loss = payload.get("start_loss")
        end_loss = payload.get("end_loss")
        if start_loss is not None and end_loss is not None:
            loss_note = f"loss={float(start_loss):.5f}->{float(end_loss):.5f}"
        else:
            loss_note = "real-video feature tubes"
        rows.append(
            {
                "family": "star_uvt_feature_firstclass",
                "renderer": renderer,
                "resolution": int(payload["size"]),
                "frames": int(payload["frames"]),
                "primitives": int(payload.get("tube_count") or payload["tubes"]),
                "feature_dim": int(payload["feature_dim"]),
                "status": "ok" if bool(payload.get("pass")) else "nonpassing",
                "steps": int(payload["steps"]),
                "total_ms": _float(timing.get("step_ms")),
                "forward_ms": _float(timing.get("render_forward_ms")),
                "backward_ms": _float(timing.get("backward_ms")),
                "sample_count": None,
                "mean_pairs_per_tile": _float(payload.get("mean_active_tile_count")),
                "overflow_tiles": int(payload.get("tile_overflow_sum", 0)),
                "artifact": artifact,
                "note": (
                    f"real-video feature tubes cap={payload.get('tile_capacity')} "
                    f"chunk={payload.get('frame_chunk_size')} fixedbin={payload.get('fixedbin_eligible')} "
                    f"mode={requested_mode}->{effective_mode} {loss_note}"
                ),
            }
        )
    return rows


def _read_star_feature_direct_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "outputs/benchmarks").glob(STAR_FEATURE_DIRECT_GLOB)):
        try:
            payload = load_research_json(path)
        except ValueError:
            continue
        timing = payload.get("timing")
        if not isinstance(timing, dict):
            continue
        rows.append(
            {
                "family": "star_uvt_feature_direct_kernel",
                "renderer": str(payload.get("backward_mode") or timing.get("backward_mode") or "unknown"),
                "resolution": int(timing["size"]),
                "frames": int(timing["frames"]),
                "primitives": int(timing["tubes"]),
                "feature_dim": int(timing["feature_dim"]),
                "status": "ok" if bool(payload.get("pass")) and bool(timing.get("finite", True)) else "nonpassing",
                "steps": None,
                "total_ms": _float(timing.get("total_ms")),
                "forward_ms": _float(timing.get("forward_ms")),
                "backward_ms": _float(timing.get("backward_ms")),
                "sample_count": None,
                "mean_pairs_per_tile": None,
                "overflow_tiles": int(timing.get("tile_overflow_sum", 0)),
                "artifact": str(path.relative_to(ROOT)),
                "note": (
                    "synthetic STAR UVT feature direct kernel "
                    f"warmup={timing.get('warmup')} repeat={timing.get('repeat')} "
                    f"skip_feature_grad={timing.get('feature_grad_skipped')} "
                    f"skip_colorizer_grad={timing.get('colorizer_grad_skipped')} "
                    f"fused_first3_sigmoid_mse={timing.get('fused_first3_sigmoid_mse')} "
                    f"linear_sigmoid_mse={timing.get('linear_sigmoid_mse')} "
                    f"logit_handoff={timing.get('logit_handoff')} "
                    f"handoff_prep_ms={timing.get('handoff_prep_ms')}"
                ),
            }
        )
    return rows


def _fmt_ms(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def _fmt_table_value(key: str, value: Any) -> str:
    if value is None:
        return ""
    if key.endswith("_ms"):
        return _fmt_ms(value)
    if key == "mean_pairs_per_tile":
        return f"{float(value):.1f}"
    return str(value)


def _display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def _artifact_status(path: Path) -> str:
    label = _display_path(path)
    return f"`{label}`" if path.exists() else f"`{label}` (missing; omitted)"


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(label for label, _key in columns) + " |")
    lines.append("| " + " | ".join("---" for _label, _key in columns) + " |")
    for row in rows:
        values = []
        for _label, key in columns:
            values.append(_fmt_table_value(key, row.get(key)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _best_ok(rows: list[dict[str, Any]], *, family: str) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["family"] == family and row["status"] == "ok" and row["total_ms"] is not None]
    best: dict[int, dict[str, Any]] = {}
    for row in candidates:
        resolution = int(row["resolution"])
        if resolution not in best or float(row["total_ms"]) < float(best[resolution]["total_ms"]):
            best[resolution] = row
    return [best[key] for key in sorted(best)]


def _matched_rows(rows: list[dict[str, Any]], *, resolution: int, frames: int, primitives: int) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in rows
            if int(row.get("resolution") or -1) == resolution
            and int(row.get("frames") or -1) == frames
            and int(row.get("primitives") or -1) == primitives
        ],
        key=lambda row: (row["family"], row["status"] != "ok", row["total_ms"] or 1e30),
    )


def write_report(
    rows: list[dict[str, Any]],
    md_path: Path,
    csv_path: Path,
    *,
    star_feature_summary: Path,
    report_date: str,
) -> None:
    fields = [
        "family",
        "renderer",
        "resolution",
        "frames",
        "primitives",
        "feature_dim",
        "status",
        "steps",
        "total_ms",
        "forward_ms",
        "backward_ms",
        "sample_count",
        "mean_pairs_per_tile",
        "overflow_tiles",
        "artifact",
        "note",
    ]
    write_research_csv(csv_path, rows, fieldnames=fields)

    star = sorted([row for row in rows if row["family"] == "star_uvt_kernel"], key=lambda row: (row["resolution"], row["total_ms"] or 1e30))
    rgb = sorted([row for row in rows if row["family"] == "dynamic_gsplat_rgb_raster"], key=lambda row: (row["resolution"], row["total_ms"] or 1e30))
    feature = sorted([row for row in rows if row["family"] == "feature_f32_raster"], key=lambda row: (row["resolution"] or 0, row["status"] != "ok", row["total_ms"] or 1e30))
    star_feature = sorted(
        [row for row in rows if row["family"] == "star_uvt_feature_firstclass"],
        key=lambda row: (row["resolution"], row["status"] != "ok", row["total_ms"] or 1e30),
    )
    star_feature_direct = sorted(
        [row for row in rows if row["family"] == "star_uvt_feature_direct_kernel"],
        key=lambda row: (row["resolution"], row["status"] != "ok", row["total_ms"] or 1e30),
    )
    best = (
        _best_ok(rows, family="star_uvt_kernel")
        + _best_ok(rows, family="star_uvt_feature_firstclass")
        + _best_ok(rows, family="star_uvt_feature_direct_kernel")
        + _best_ok(rows, family="dynamic_gsplat_rgb_raster")
        + _best_ok(rows, family="feature_f32_raster")
    )
    columns = [
        ("family", "family"),
        ("renderer", "renderer"),
        ("res", "resolution"),
        ("frames", "frames"),
        ("G/tubes", "primitives"),
        ("F", "feature_dim"),
        ("status", "status"),
        ("steps", "steps"),
        ("total ms", "total_ms"),
        ("fwd ms", "forward_ms"),
        ("bwd ms", "backward_ms"),
        ("mean pairs/tile", "mean_pairs_per_tile"),
        ("overflow tiles", "overflow_tiles"),
    ]
    matched_256 = _matched_rows(rows, resolution=256, frames=64, primitives=32768)
    matched_512_star_feature = sorted(
        [
            row
            for row in rows
            if row["family"] == "star_uvt_feature_firstclass"
            and int(row.get("resolution") or -1) == 512
            and int(row.get("frames") or -1) == 64
        ],
        key=lambda row: (int(row.get("primitives") or 0), row["status"] != "ok", row["total_ms"] or 1e30),
    )
    md = [
        "# Renderer Scaling Matrix",
        "",
        f"Date: {report_date}",
        "",
        "All rows use 64 frame/batch slots and 32,768 primitives/tubes unless a row explicitly reports otherwise.",
        "Dynamic/feature rows are projected-raster synthetic benchmarks; STAR UVT rows are screen-space direct backward kernel probes.",
        "Direct-kernel rows named `skip_feature_grad`, `fused_first3_sigmoid_mse`, `linear_sigmoid_mse`, `logit_handoff`, or `gradcache_reduce_feature_grad_vec4` are diagnostics/prototypes: skip is not trainable, fused first3 is a narrow RGB handoff, linear sigmoid MSE is a generalized in-tile handoff, logit handoff is an image-space-prep handoff benchmark, and vec4 reduce is trainer-selectable but only promoted for the no-pre-norm 512px feature diagnostic, not as a quality baseline.",
        "",
        "## Best Rows By Family And Resolution",
        "",
        _markdown_table(best, columns),
        "",
        "## Matched 64f/256px/32768 Rows",
        "",
        "This is the apples-to-apples speed table for the current 32768 primitive/tube question. STAR feature rows include real-video first-class trainer steps and synthetic direct-kernel probes; dynamic RGB/F32 rows are projected-raster synthetic rows; STAR RGB rows are direct kernel probes.",
        "",
        _markdown_table(matched_256, columns),
        "",
        "## Matched 64f/512px STAR Feature Rows",
        "",
        "These are first-class real-video feature-overfit rows. They are not directly comparable to the synthetic 32768-primitive projected-raster rows because the current valid 512px STAR feature rows use 2048, 4096, or 8192 tubes.",
        "",
        _markdown_table(matched_512_star_feature, columns),
        "",
        "## STAR UVT Kernel Rows",
        "",
        _markdown_table(star, columns),
        "",
        "## Dynamic GSplat RGB Raster Rows",
        "",
        _markdown_table(rgb, columns),
        "",
        "## STAR UVT F32 Feature First-Class Rows",
        "",
        _markdown_table(star_feature, columns),
        "",
        "## STAR UVT F32 Feature Direct Kernel Rows",
        "",
        _markdown_table(star_feature_direct, columns),
        "",
        "## F32 Feature Raster Rows",
        "",
        _markdown_table(feature, columns),
        "",
        "## Artifacts",
        "",
        f"- CSV: `{_display_path(csv_path)}`",
        "- RGB JSONL: " + _artifact_status(RGB_JSONL),
        "- STAR feature summary: " + _artifact_status(star_feature_summary),
        f"- STAR feature direct glob: `outputs/benchmarks/{STAR_FEATURE_DIRECT_GLOB}`",
        f"- Current STAR kernel glob: `outputs/benchmarks/{CURRENT_STAR_KERNEL_GLOB}`",
        "- STAR summaries: " + ", ".join(_artifact_status(path / "summary.csv") for path in STAR_DIRS),
        "- Feature JSONL: " + ", ".join(_artifact_status(path) for path in FEATURE_JSONLS),
    ]
    write_research_text(md_path, "\n".join(md) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-md", type=Path, default=ROOT / "outputs/benchmarks/2026-05-18_renderer_scaling_report.md")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "outputs/benchmarks/2026-05-18_renderer_scaling_report.csv")
    parser.add_argument("--star-feature-summary", type=Path, default=_latest_star_feature_summary())
    parser.add_argument("--report-date", default=date.today().isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    star_feature_rows = _read_star_feature_rows(args.star_feature_summary)
    rows = (
        _read_star_rows()
        + star_feature_rows
        + _read_star_feature_individual_rows(star_feature_rows)
        + _read_star_feature_direct_rows()
        + _read_rgb_rows()
        + _read_feature_rows()
    )
    write_report(
        rows,
        args.out_md,
        args.out_csv,
        star_feature_summary=args.star_feature_summary,
        report_date=str(args.report_date),
    )
    print(json.dumps({"rows": len(rows), "out_md": str(args.out_md), "out_csv": str(args.out_csv)}, sort_keys=True))


if __name__ == "__main__":
    main()
