from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import (
        ROOT,
        fmt_cell as _fmt,
        fmt_pair as _pair,
        load_report_json,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        fmt_cell as _fmt,
        fmt_pair as _pair,
        load_report_json,
        write_report_json,
        write_report_text,
    )

ROWS = [
    (
        "600->800 feature0.25/probe40",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.json",
    ),
    (
        "800->1000 scheduled balance",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.json",
    ),
    (
        "1000->1100 feature0.5/probe40",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.json",
    ),
    (
        "1100->1200 recover schedule",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.json",
    ),
    (
        "1200->1250 feature0.75/probe40",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.json",
    ),
    (
        "1250->1300 feature1/probe40",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.json",
    ),
    (
        "1300->1400 feature1/probe40",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json",
    ),
    (
        "1300->1400 feature1/probe40 timing repeat",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat.json",
    ),
]


def _row(label: str, path: str) -> dict[str, Any]:
    data = load_report_json(path)
    timing = data.get("mean_timing_ms", {})
    tile = data.get("tile_stats", {})
    return {
        "label": label,
        "path": path,
        "pass": bool(data.get("pass")),
        "global_steps": [data.get("start_global_step"), data.get("end_global_step")],
        "feature_loss": [data.get("start_feature_target_loss"), data.get("end_feature_target_loss")],
        "rgb_probe_loss": [data.get("start_rgb_probe_loss"), data.get("end_rgb_probe_loss")],
        "rgb_probe_psnr": [data.get("start_rgb_probe_psnr"), data.get("end_rgb_probe_psnr")],
        "step_ms": timing.get("step_ms"),
        "backward_ms": timing.get("backward_ms"),
        "render_forward_ms": timing.get("render_forward_ms"),
        "rgb_probe_loss_ms": timing.get("rgb_probe_loss_ms"),
        "feature_target_ms": timing.get("feature_target_ms"),
        "tile_overflow_sum": data.get("tile_overflow_sum"),
        "max_tile_count": tile.get("max_tile_count"),
        "p95_tile_count": tile.get("p95_tile_count"),
        "tile_capacity": data.get("tile_capacity"),
        "checkpoint": data.get("checkpoint"),
        "rgb_probe_contact_sheet": data.get("rgb_probe_contact_sheet"),
        "rgb_probe_side_by_side_video": data.get("rgb_probe_side_by_side_video"),
    }


def build_report() -> dict[str, Any]:
    rows = [_row(label, path) for label, path in ROWS]
    fast_feature1 = next(row for row in rows if row["label"] == "1250->1300 feature1/probe40")
    extended = next(row for row in rows if row["label"] == "1300->1400 feature1/probe40")
    repeat = next(row for row in rows if row["label"] == "1300->1400 feature1/probe40 timing repeat")
    extended_delta_ms = extended["step_ms"] - fast_feature1["step_ms"]
    repeat_delta_ms = repeat["step_ms"] - fast_feature1["step_ms"]
    repeat_vs_extended_ms = repeat["step_ms"] - extended["step_ms"]
    return {
        "gate": "star_uvt_feature1_continuation_chain",
        "report_date": "2026-05-19",
        "rows": rows,
        "conclusion": {
            "extended_feature1_balance_passes": extended["pass"],
            "repeat_feature1_balance_passes": repeat["pass"],
            "extended_global_steps": extended["global_steps"],
            "repeat_global_steps": repeat["global_steps"],
            "extended_improves_feature_loss": extended["feature_loss"][1] < extended["feature_loss"][0],
            "repeat_improves_feature_loss": repeat["feature_loss"][1] < repeat["feature_loss"][0],
            "extended_improves_probe_psnr": extended["rgb_probe_psnr"][1] > extended["rgb_probe_psnr"][0],
            "repeat_improves_probe_psnr": repeat["rgb_probe_psnr"][1] > repeat["rgb_probe_psnr"][0],
            "extended_tile_overflow_sum": extended["tile_overflow_sum"],
            "repeat_tile_overflow_sum": repeat["tile_overflow_sum"],
            "repeat_max_tile_count": repeat["max_tile_count"],
            "repeat_p95_tile_count": repeat["p95_tile_count"],
            "repeat_tile_capacity": repeat["tile_capacity"],
            "extended_step_ms_delta_vs_1250_1300": extended_delta_ms,
            "repeat_step_ms_delta_vs_1250_1300": repeat_delta_ms,
            "repeat_step_ms_delta_vs_first_1300_1400": repeat_vs_extended_ms,
            "repeat_reproduces_slow_timing": repeat["step_ms"] > 1600.0 and repeat_delta_ms > 400.0,
            "read": (
                "The feature1/probe40 balance objective remains quality-positive "
                "through 1400 global steps, and the timing-control repeat "
                "reproduces the slower 1300->1400 regime. The tile stats stay "
                "comfortably below capacity, so this timing regression should "
                "be treated as whole-graph/MPS timing or objective-cost variance "
                "before blaming overflow."
            ),
            "next": (
                "Stop treating the 1300->1400 slowdown as a one-off row. The "
                "next implementation gate should either profile the whole graph "
                "around render/probe/backward timing, or shift to native "
                "VJP/dataset-scale work because the oracle gap is now larger "
                "than the local objective-plumbing gap."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STAR UVT Feature1 Continuation Chain",
        "",
        f"Date: {report['report_date']}",
        "",
        "## Answer",
        "",
        report["conclusion"]["read"],
        "",
        report["conclusion"]["next"],
        "",
        "## Rows",
        "",
        "| row | pass | global | feature loss | probe PSNR | step ms | backward ms | render ms | probe ms | target ms | overflow | max/p95/cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        tile = f"{row['max_tile_count']}/{row['p95_tile_count']}/{row['tile_capacity']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    str(row["pass"]),
                    _pair(row["global_steps"], 0),
                    _pair(row["feature_loss"], 6),
                    _pair(row["rgb_probe_psnr"], 3),
                    _fmt(row["step_ms"], 1),
                    _fmt(row["backward_ms"], 1),
                    _fmt(row["render_forward_ms"], 1),
                    _fmt(row["rgb_probe_loss_ms"], 1),
                    _fmt(row["feature_target_ms"], 1),
                    str(row["tile_overflow_sum"]),
                    tile,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Latest Artifacts",
            "",
        ]
    )
    latest = report["rows"][-1]
    lines.extend(
        [
            f"- result: `{latest['path']}`",
            f"- checkpoint: `{latest['checkpoint']}`",
            f"- media: `{latest['rgb_probe_contact_sheet']}`, `{latest['rgb_probe_side_by_side_video']}`",
            "",
        ]
    )
    write_report_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md",
    )
    args = parser.parse_args()
    report = build_report()
    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    write_report_json(out_json, report)
    write_markdown(report, out_md)
    print(json.dumps({"out_md": str(out_md), "rows": len(report["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
