from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


try:
    from .report_artifacts import (
        ROOT,
        fmt_cell as _fmt,
        load_report_json,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        fmt_cell as _fmt,
        load_report_json,
        write_report_json,
        write_report_text,
    )

INPUTS = [
    (
        "lr005_resume100",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json",
    ),
    (
        "lr001_resume100_configlr",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_checkpoint_media.json",
    ),
]


def _max_positive_delta(row: dict[str, Any]) -> dict[str, Any] | None:
    losses = [float(value) for value in row["losses"]]
    features = [float(value) for value in row["feature_target_losses"]]
    probes = [float(value) for value in row["rgb_probe_losses"]]
    steps = [int(value) for value in row["step_global_steps"]]
    if len(losses) < 2:
        return None
    best_i = max(range(1, len(losses)), key=lambda i: losses[i] - losses[i - 1])
    return {
        "from_step": steps[best_i - 1],
        "to_step": steps[best_i],
        "loss_delta": losses[best_i] - losses[best_i - 1],
        "feature_target_loss_delta": features[best_i] - features[best_i - 1],
        "rgb_probe_loss_delta": probes[best_i] - probes[best_i - 1],
        "loss_after": losses[best_i],
        "final_loss_delta_from_spike_after": losses[-1] - losses[best_i],
    }


def _delta_at(row: dict[str, Any], step: int) -> dict[str, float] | None:
    steps = [int(value) for value in row["step_global_steps"]]
    if step not in steps:
        return None
    i = steps.index(step)
    if i == 0:
        return None
    return {
        "loss_delta": float(row["losses"][i]) - float(row["losses"][i - 1]),
        "feature_target_loss_delta": float(row["feature_target_losses"][i])
        - float(row["feature_target_losses"][i - 1]),
        "rgb_probe_loss_delta": float(row["rgb_probe_losses"][i]) - float(row["rgb_probe_losses"][i - 1]),
    }


def _row_summary(label: str, path: str) -> dict[str, Any]:
    row = load_report_json(path)
    return {
        "label": label,
        "source": path,
        "pass": bool(row["pass"]),
        "configured_lr": float(row["lr"]),
        "resume_optimizer_lrs_loaded": row.get("resume_optimizer_lrs_loaded"),
        "optimizer_lrs": row.get("optimizer_lrs"),
        "start_loss": float(row["start_loss"]),
        "end_loss": float(row["end_loss"]),
        "loss_delta": float(row["end_loss"]) - float(row["start_loss"]),
        "start_feature_target_loss": float(row["start_feature_target_loss"]),
        "end_feature_target_loss": float(row["end_feature_target_loss"]),
        "feature_target_loss_delta": float(row["end_feature_target_loss"]) - float(row["start_feature_target_loss"]),
        "start_rgb_probe_psnr": float(row["start_rgb_probe_psnr"]),
        "end_rgb_probe_psnr": float(row["end_rgb_probe_psnr"]),
        "rgb_probe_psnr_delta": float(row["end_rgb_probe_psnr"]) - float(row["start_rgb_probe_psnr"]),
        "mean_step_ms": float(row["mean_timing_ms"]["step_ms"]),
        "mean_render_ms": float(row["mean_timing_ms"]["render_forward_ms"]),
        "mean_backward_ms": float(row["mean_timing_ms"]["backward_ms"]),
        "tile_overflow_sum": int(row["tile_overflow_sum"]),
        "tile_max_p95_cap": [
            int(row["tile_stats"]["max_tile_count"]),
            int(row["tile_stats"]["p95_tile_count"]),
            int(row["tile_stats"]["tile_capacity"]),
        ],
        "delta_at_1318": _delta_at(row, 1318),
        "max_positive_delta": _max_positive_delta(row),
        "checkpoint": row["checkpoint"],
        "rgb_probe_contact_sheet": row["rgb_probe_contact_sheet"],
        "rgb_probe_side_by_side_video": row["rgb_probe_side_by_side_video"],
    }


def build_report() -> dict[str, Any]:
    rows = [_row_summary(label, path) for label, path in INPUTS]
    by_label = {row["label"]: row for row in rows}
    return {
        "gate": "star_uvt_feature1_lr001_100step_continuation",
        "report_date": "2026-05-19",
        "rows": rows,
        "conclusion": {
            "read": (
                "The effective-lr001 100-step continuation passes and avoids the "
                "early 1318 spike, but it is not a clean dominance over the older "
                "lr005 1300->1400 row. It improves final probe PSNR and mean timing, "
                "while lr005 still ends with better feature loss and slightly better "
                "weighted loss. Both schedules have transient objective jumps; lr001 "
                "moves the largest jump later and recovers by the final step."
            ),
            "next": (
                "Treat effective lr=0.001 as the safer probe/visual continuation "
                "from the 1300 checkpoint, not as the final quality schedule. The "
                "next quality gate should use a real LR schedule or checkpoint "
                "selection around transient jumps; the speed gate remains native "
                "VJP/scalar fixedbin."
            ),
            "lr001_minus_lr005": {
                "end_loss": by_label["lr001_resume100_configlr"]["end_loss"]
                - by_label["lr005_resume100"]["end_loss"],
                "end_feature_target_loss": by_label["lr001_resume100_configlr"]["end_feature_target_loss"]
                - by_label["lr005_resume100"]["end_feature_target_loss"],
                "end_rgb_probe_psnr": by_label["lr001_resume100_configlr"]["end_rgb_probe_psnr"]
                - by_label["lr005_resume100"]["end_rgb_probe_psnr"],
                "mean_step_ms": by_label["lr001_resume100_configlr"]["mean_step_ms"]
                - by_label["lr005_resume100"]["mean_step_ms"],
                "mean_backward_ms": by_label["lr001_resume100_configlr"]["mean_backward_ms"]
                - by_label["lr005_resume100"]["mean_backward_ms"],
            },
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STAR UVT Feature1 LR001 100-Step Continuation",
        "",
        f"Date: {report['report_date']}",
        "",
        "## Answer",
        "",
        report["conclusion"]["read"],
        "",
        report["conclusion"]["next"],
        "",
        "## Comparison",
        "",
        "| label | pass | cfg lr | loaded/effective lrs | end loss | d loss | end feature | d feature | end probe PSNR | d probe PSNR | d loss at 1318 | largest jump | mean step ms | mean render ms | mean backward ms | tile max/p95/cap |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        d1318 = row["delta_at_1318"] or {}
        jump = row["max_positive_delta"] or {}
        lr_pair = f"{row['resume_optimizer_lrs_loaded']} -> {row['optimizer_lrs']}"
        jump_text = (
            ""
            if not jump
            else f"{jump['from_step']}->{jump['to_step']} {jump['loss_delta']:.6f}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    str(row["pass"]).lower(),
                    _fmt(row["configured_lr"], 4),
                    lr_pair,
                    _fmt(row["end_loss"], 6),
                    _fmt(row["loss_delta"], 6),
                    _fmt(row["end_feature_target_loss"], 6),
                    _fmt(row["feature_target_loss_delta"], 6),
                    _fmt(row["end_rgb_probe_psnr"], 3),
                    _fmt(row["rgb_probe_psnr_delta"], 3),
                    _fmt(d1318.get("loss_delta"), 6),
                    jump_text,
                    _fmt(row["mean_step_ms"], 1),
                    _fmt(row["mean_render_ms"], 1),
                    _fmt(row["mean_backward_ms"], 1),
                    "/".join(str(value) for value in row["tile_max_p95_cap"]),
                ]
            )
            + " |"
        )
    diff = report["conclusion"]["lr001_minus_lr005"]
    lines.extend(
        [
            "",
            "## LR001 Minus LR005",
            "",
            f"- end weighted loss: `{diff['end_loss']:.6f}`",
            f"- end feature loss: `{diff['end_feature_target_loss']:.6f}`",
            f"- end probe PSNR: `{diff['end_rgb_probe_psnr']:.3f}`",
            f"- mean step: `{diff['mean_step_ms']:.1f}ms`",
            f"- mean backward: `{diff['mean_backward_ms']:.1f}ms`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for row in report["rows"]:
        lines.append(f"- `{row['label']}` JSON: `{row['source']}`")
        if row["checkpoint"]:
            lines.append(f"- `{row['label']}` checkpoint: `{row['checkpoint']}`")
        if row["rgb_probe_contact_sheet"]:
            lines.append(f"- `{row['label']}` contact sheet: `{row['rgb_probe_contact_sheet']}`")
        if row["rgb_probe_side_by_side_video"]:
            lines.append(f"- `{row['label']}` video: `{row['rgb_probe_side_by_side_video']}`")
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md",
    )
    args = parser.parse_args()
    report = build_report()
    out_json = ROOT / args.out_json
    write_report_json(out_json, report)
    write_markdown(report, ROOT / args.out_md)


if __name__ == "__main__":
    main()
