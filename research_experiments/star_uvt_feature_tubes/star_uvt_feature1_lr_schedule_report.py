from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


try:
    from .report_artifacts import (
        ROOT,
        fmt_cell as _fmt,
        load_report_json,
        mean_timing_without_first,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        fmt_cell as _fmt,
        load_report_json,
        mean_timing_without_first,
        write_report_json,
        write_report_text,
    )

INPUTS = [
    (
        "static_lr005_resume100",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json",
    ),
    (
        "static_lr001_resume100",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_checkpoint_media.json",
    ),
    (
        "scheduled_lr001_to_lr00025_resume100",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_to_lr00025_resume100_from1300_trace.json",
    ),
    (
        "scheduled_lr001_to_lr00025_late_trace88",
        "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_to_lr00025_resume88_from1300_late_spike_trace.json",
    ),
]


def _max_positive_delta(row: dict[str, Any]) -> dict[str, Any] | None:
    losses = [float(value) for value in row["losses"]]
    steps = [int(value) for value in row["step_global_steps"]]
    if len(losses) < 2:
        return None
    best_i = max(range(1, len(losses)), key=lambda i: losses[i] - losses[i - 1])
    return {
        "from_step": steps[best_i - 1],
        "to_step": steps[best_i],
        "loss_delta": losses[best_i] - losses[best_i - 1],
        "feature_target_loss_delta": float(row["feature_target_losses"][best_i])
        - float(row["feature_target_losses"][best_i - 1]),
        "rgb_probe_loss_delta": float(row["rgb_probe_losses"][best_i])
        - float(row["rgb_probe_losses"][best_i - 1]),
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


def _step_row(row: dict[str, Any], step: int) -> dict[str, Any] | None:
    steps = [int(value) for value in row["step_global_steps"]]
    if step not in steps:
        return None
    i = steps.index(step)
    rgb_probe_loss = float(row["rgb_probe_losses"][i])
    return {
        "step": step,
        "loss": float(row["losses"][i]),
        "feature_target_loss": float(row["feature_target_losses"][i]),
        "rgb_probe_psnr": -10.0 * math.log10(rgb_probe_loss),
        "lr": (row.get("step_lrs") or [None] * len(steps))[i],
        "loss_delta": None if i == 0 else float(row["losses"][i]) - float(row["losses"][i - 1]),
    }


def _window(row: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
    return [step for value in range(start, end + 1) if (step := _step_row(row, value)) is not None]


def _trace_by_step(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(trace["global_step"]): trace for trace in row.get("chunk_traces") or []}


def _chunk_delta_summary(row: dict[str, Any], before_step: int, after_step: int) -> dict[str, Any] | None:
    traces = _trace_by_step(row)
    if before_step not in traces or after_step not in traces:
        return None
    before = traces[before_step]
    after = traces[after_step]
    before_chunks = {int(chunk["frame_start"]): chunk for chunk in before.get("chunks", [])}
    after_chunks = {int(chunk["frame_start"]): chunk for chunk in after.get("chunks", [])}
    deltas = []
    for frame_start, after_chunk in after_chunks.items():
        before_chunk = before_chunks.get(frame_start)
        if before_chunk is None:
            continue
        deltas.append(
            {
                "frame_start": frame_start,
                "weighted_loss_delta": float(after_chunk["weighted_loss"]) - float(before_chunk["weighted_loss"]),
                "feature_target_loss_delta": float(after_chunk["feature_target_loss"])
                - float(before_chunk["feature_target_loss"]),
                "rgb_probe_loss_delta": float(after_chunk["rgb_probe_loss"]) - float(before_chunk["rgb_probe_loss"]),
            }
        )
    if not deltas:
        return None
    largest = max(deltas, key=lambda item: item["weighted_loss_delta"])
    return {
        "from_step": before_step,
        "to_step": after_step,
        "chunk_count": len(deltas),
        "positive_chunk_count": sum(1 for item in deltas if item["weighted_loss_delta"] > 0.0),
        "negative_chunk_count": sum(1 for item in deltas if item["weighted_loss_delta"] < 0.0),
        "weighted_loss_delta_sum": sum(item["weighted_loss_delta"] for item in deltas),
        "feature_target_loss_delta_sum": sum(item["feature_target_loss_delta"] for item in deltas),
        "rgb_probe_loss_delta_sum": sum(item["rgb_probe_loss_delta"] for item in deltas),
        "largest_positive_chunk": largest,
    }


def _wandb_dirs_for_source(path: str) -> list[str]:
    matches = []
    for debug_log in (ROOT / "wandb").glob("offline-run-20260519_*/logs/debug.log"):
        try:
            text = debug_log.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path in text:
            matches.append(str(debug_log.parents[1].relative_to(ROOT)))
    return sorted(matches)


def _lr_read(row: dict[str, Any]) -> str:
    schedule = row.get("optimizer_lr_schedule")
    if schedule:
        return " -> ".join(f"{stage['lr']} until {stage['end_step']}" for stage in schedule)
    step_lrs = row.get("step_lrs") or []
    if step_lrs:
        unique = sorted({float(value) for value in step_lrs})
        return ", ".join(str(value) for value in unique)
    loaded = row.get("resume_optimizer_lrs_loaded")
    effective = row.get("optimizer_lrs")
    if loaded is not None or effective is not None:
        return f"{loaded} -> {effective}"
    return str(row.get("lr"))


def _row_summary(label: str, path: str) -> dict[str, Any]:
    row = load_report_json(path)
    return {
        "label": label,
        "source": path,
        "wandb_dirs": _wandb_dirs_for_source(path),
        "pass": bool(row["pass"]),
        "steps": int(row["steps"]),
        "start_global_step": int(row["start_global_step"]),
        "end_global_step": int(row["end_global_step"]),
        "lr_read": _lr_read(row),
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
        "mean_no_first_step_ms": mean_timing_without_first(row, "step_ms"),
        "mean_no_first_render_ms": mean_timing_without_first(row, "render_forward_ms"),
        "mean_no_first_backward_ms": mean_timing_without_first(row, "backward_ms"),
        "tile_overflow_sum": int(row["tile_overflow_sum"]),
        "tile_unstable_sum": int(row["tile_unstable_sum"]),
        "tile_max_p95_cap": [
            int(row["tile_stats"]["max_tile_count"]),
            int(row["tile_stats"]["p95_tile_count"]),
            int(row["tile_stats"]["tile_capacity"]),
        ],
        "delta_at_1318": _delta_at(row, 1318),
        "delta_at_1378": _delta_at(row, 1378),
        "delta_at_1386": _delta_at(row, 1386),
        "max_positive_delta": _max_positive_delta(row),
        "window_1374_1381": _window(row, 1374, 1381),
        "window_1383_1388": _window(row, 1383, 1388),
        "chunk_delta_1378_vs_1377": _chunk_delta_summary(row, 1377, 1378),
        "chunk_delta_1386_vs_1385": _chunk_delta_summary(row, 1385, 1386),
        "checkpoint": row["checkpoint"],
        "rgb_probe_contact_sheet": row["rgb_probe_contact_sheet"],
        "rgb_probe_side_by_side_video": row["rgb_probe_side_by_side_video"],
    }


def build_report() -> dict[str, Any]:
    rows = [_row_summary(label, path) for label, path in INPUTS]
    comparable = [row for row in rows if row["steps"] == 100 and row["pass"]]
    best_weighted = min(comparable, key=lambda row: row["end_loss"])
    best_feature = min(comparable, key=lambda row: row["end_feature_target_loss"])
    best_probe = max(comparable, key=lambda row: row["end_rgb_probe_psnr"])
    scheduled = next(row for row in rows if row["label"] == "scheduled_lr001_to_lr00025_resume100")
    static_lr001 = next(row for row in rows if row["label"] == "static_lr001_resume100")
    return {
        "gate": "star_uvt_feature1_lr_schedule_gate",
        "report_date": "2026-05-19",
        "rows": rows,
        "conclusion": {
            "best_weighted_label": best_weighted["label"],
            "best_feature_label": best_feature["label"],
            "best_probe_label": best_probe["label"],
            "scheduled_minus_static_lr001": {
                "end_loss": scheduled["end_loss"] - static_lr001["end_loss"],
                "end_feature_target_loss": scheduled["end_feature_target_loss"]
                - static_lr001["end_feature_target_loss"],
                "end_rgb_probe_psnr": scheduled["end_rgb_probe_psnr"] - static_lr001["end_rgb_probe_psnr"],
                "mean_step_ms": scheduled["mean_step_ms"] - static_lr001["mean_step_ms"],
                "mean_backward_ms": scheduled["mean_backward_ms"] - static_lr001["mean_backward_ms"],
            },
            "read": (
                "The lr=0.001->0.00025 schedule plumbing works and the run passes, "
                "but it is not the quality fix. It removes the earlier 1377->1378 "
                "jump, then a comparable jump reappears at 1385->1386. End weighted "
                "loss, feature MSE, probe PSNR, and timing are all worse than the "
                "static effective-lr001 100-step row. The 88-step late-trace row "
                "is expected to fail the quality pass bit because it intentionally "
                "stops just after the spike; its role is chunk-level attribution."
            ),
            "next": (
                "Do not promote this schedule. Treat static effective-lr001 as the "
                "current safer media/checkpoint path, and move the next quality gate "
                "to checkpoint selection or a schedule keyed to measured transient "
                "recovery. The speed gate is unchanged: native VJP/scalar fixedbin "
                "feature gradients, not LR tuning, is what can remove the 800ms-class "
                "backward."
            ),
        },
    }


def _delta_text(delta: dict[str, Any] | None) -> str:
    if not delta:
        return ""
    return _fmt(delta["loss_delta"], 6)


def _jump_text(row: dict[str, Any]) -> str:
    jump = row["max_positive_delta"]
    if not jump:
        return ""
    return f"{jump['from_step']}->{jump['to_step']} {jump['loss_delta']:.6f}"


def _chunk_text(delta: dict[str, Any] | None) -> str:
    if not delta:
        return ""
    chunk = delta["largest_positive_chunk"]
    return (
        f"{delta['positive_chunk_count']}/{delta['negative_chunk_count']} chunks, "
        f"sum {delta['weighted_loss_delta_sum']:.6f}, "
        f"max frame {chunk['frame_start']} {chunk['weighted_loss_delta']:.6f}"
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STAR UVT Feature1 LR Schedule Gate",
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
        "| label | pass | steps | global steps | lr read | end loss | d loss | end feature | d feature | end probe PSNR | d probe PSNR | d@1318 | d@1378 | d@1386 | largest jump | mean step ms | mean backward ms | tile max/p95/cap |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    str(row["pass"]).lower(),
                    str(row["steps"]),
                    f"{row['start_global_step']}->{row['end_global_step']}",
                    row["lr_read"],
                    _fmt(row["end_loss"], 6),
                    _fmt(row["loss_delta"], 6),
                    _fmt(row["end_feature_target_loss"], 6),
                    _fmt(row["feature_target_loss_delta"], 6),
                    _fmt(row["end_rgb_probe_psnr"], 3),
                    _fmt(row["rgb_probe_psnr_delta"], 3),
                    _delta_text(row["delta_at_1318"]),
                    _delta_text(row["delta_at_1378"]),
                    _delta_text(row["delta_at_1386"]),
                    _jump_text(row),
                    _fmt(row["mean_step_ms"], 1),
                    _fmt(row["mean_backward_ms"], 1),
                    "/".join(str(value) for value in row["tile_max_p95_cap"]),
                ]
            )
            + " |"
        )
    diff = report["conclusion"]["scheduled_minus_static_lr001"]
    lines.extend(
        [
            "",
            "## Scheduled Minus Static LR001",
            "",
            f"- end weighted loss: `{diff['end_loss']:.6f}`",
            f"- end feature loss: `{diff['end_feature_target_loss']:.6f}`",
            f"- end probe PSNR: `{diff['end_rgb_probe_psnr']:.3f}`",
            f"- mean step: `{diff['mean_step_ms']:.1f}ms`",
            f"- mean backward: `{diff['mean_backward_ms']:.1f}ms`",
            "",
            "## Step Windows",
            "",
        ]
    )
    for row in report["rows"]:
        if row["window_1374_1381"] or row["window_1383_1388"]:
            lines.extend(["", f"### {row['label']}", ""])
        if row["window_1374_1381"]:
            lines.append("1374-1381:")
            for item in row["window_1374_1381"]:
                lines.append(
                    f"- {item['step']}: loss `{item['loss']:.6f}`, d `{_fmt(item['loss_delta'], 6)}`, "
                    f"feature `{item['feature_target_loss']:.6f}`, probe `{item['rgb_probe_psnr']:.3f}`, lr `{item['lr']}`"
                )
        if row["window_1383_1388"]:
            lines.append("1383-1388:")
            for item in row["window_1383_1388"]:
                lines.append(
                    f"- {item['step']}: loss `{item['loss']:.6f}`, d `{_fmt(item['loss_delta'], 6)}`, "
                    f"feature `{item['feature_target_loss']:.6f}`, probe `{item['rgb_probe_psnr']:.3f}`, lr `{item['lr']}`"
                )
        if row["chunk_delta_1378_vs_1377"] or row["chunk_delta_1386_vs_1385"]:
            lines.append("Chunk deltas:")
            if row["chunk_delta_1378_vs_1377"]:
                lines.append(f"- 1377->1378: {_chunk_text(row['chunk_delta_1378_vs_1377'])}")
            if row["chunk_delta_1386_vs_1385"]:
                lines.append(f"- 1385->1386: {_chunk_text(row['chunk_delta_1386_vs_1385'])}")
    lines.extend(["", "## Artifacts", ""])
    for row in report["rows"]:
        lines.append(f"- `{row['label']}` JSON: `{row['source']}`")
        for wandb_dir in row["wandb_dirs"]:
            lines.append(f"- `{row['label']}` W&B offline: `{wandb_dir}`")
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
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md",
    )
    args = parser.parse_args()
    report = build_report()
    out_json = ROOT / args.out_json
    write_report_json(out_json, report)
    write_markdown(report, ROOT / args.out_md)


if __name__ == "__main__":
    main()
