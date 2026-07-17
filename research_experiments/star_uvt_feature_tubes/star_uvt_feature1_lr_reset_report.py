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
        "lr005_resumeopt",
        "outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace20_from1300.json",
    ),
    (
        "lr001_resumeopt_configlr",
        "outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_resumeopt_chunktrace20_from1300.json",
    ),
    (
        "lr001_resetopt",
        "outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_resetopt_chunktrace20_from1300.json",
    ),
]


def _trace_by_step(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(trace["global_step"]): trace for trace in row.get("chunk_traces") or []}


def _chunk_delta_summary(row: dict[str, Any]) -> dict[str, Any] | None:
    traces = _trace_by_step(row)
    if 1317 not in traces or 1318 not in traces:
        return None
    before = traces[1317]
    after = traces[1318]
    chunks_before = {int(chunk["frame_start"]): chunk for chunk in before.get("chunks", [])}
    chunks_after = {int(chunk["frame_start"]): chunk for chunk in after.get("chunks", [])}
    deltas = []
    for frame_start, chunk_after in chunks_after.items():
        chunk_before = chunks_before.get(frame_start)
        if chunk_before is None:
            continue
        deltas.append(float(chunk_after["weighted_loss"]) - float(chunk_before["weighted_loss"]))
    positive = sum(1 for value in deltas if value > 0.0)
    negative = sum(1 for value in deltas if value < 0.0)
    return {
        "weighted_loss_delta": float(after["loss"]) - float(before["loss"]),
        "feature_target_loss_delta": float(after["feature_target_loss"]) - float(before["feature_target_loss"]),
        "rgb_probe_loss_delta": float(after["rgb_probe_loss"]) - float(before["rgb_probe_loss"]),
        "positive_chunk_count": positive,
        "negative_chunk_count": negative,
        "chunk_count": len(deltas),
        "spike_avoided": float(after["loss"]) <= float(before["loss"]),
    }


def _row_summary(label: str, path: str) -> dict[str, Any]:
    row = load_report_json(path)
    trace_delta = _chunk_delta_summary(row)
    return {
        "label": label,
        "source": path,
        "pass": bool(row["pass"]),
        "resume_optimizer": bool(row["resume_optimizer"]),
        "resume_optimizer_loaded": bool(row["resume_optimizer_loaded"]),
        "resume_optimizer_lrs_loaded": row.get("resume_optimizer_lrs_loaded"),
        "optimizer_lrs": row.get("optimizer_lrs"),
        "configured_lr": float(row["lr"]),
        "start_loss": float(row["start_loss"]),
        "end_loss": float(row["end_loss"]),
        "loss_delta": float(row["end_loss"]) - float(row["start_loss"]),
        "start_feature_target_loss": float(row["start_feature_target_loss"]),
        "end_feature_target_loss": float(row["end_feature_target_loss"]),
        "feature_target_loss_delta": float(row["end_feature_target_loss"]) - float(row["start_feature_target_loss"]),
        "start_rgb_probe_loss": float(row["start_rgb_probe_loss"]),
        "end_rgb_probe_loss": float(row["end_rgb_probe_loss"]),
        "rgb_probe_loss_delta": float(row["end_rgb_probe_loss"]) - float(row["start_rgb_probe_loss"]),
        "start_rgb_probe_psnr": float(row["start_rgb_probe_psnr"]),
        "end_rgb_probe_psnr": float(row["end_rgb_probe_psnr"]),
        "rgb_probe_psnr_delta": float(row["end_rgb_probe_psnr"]) - float(row["start_rgb_probe_psnr"]),
        "mean_no_first_step_ms": mean_timing_without_first(row, "step_ms"),
        "mean_no_first_render_ms": mean_timing_without_first(row, "render_forward_ms"),
        "mean_no_first_backward_ms": mean_timing_without_first(row, "backward_ms"),
        "tile_overflow_sum": int(row["tile_overflow_sum"]),
        "tile_max_p95_cap": [
            int(row["tile_stats"]["max_tile_count"]),
            int(row["tile_stats"]["p95_tile_count"]),
            int(row["tile_stats"]["tile_capacity"]),
        ],
        "trace_delta_1318_vs_1317": trace_delta,
    }


def build_report() -> dict[str, Any]:
    rows = [_row_summary(label, path) for label, path in INPUTS]
    passed_rows = [row for row in rows if row["pass"]]
    best_weighted = min(passed_rows, key=lambda row: row["end_loss"]) if passed_rows else None
    best_feature = min(passed_rows, key=lambda row: row["end_feature_target_loss"]) if passed_rows else None
    best_probe = max(passed_rows, key=lambda row: row["end_rgb_probe_psnr"]) if passed_rows else None
    return {
        "gate": "star_uvt_feature1_lr001_checkpoint_gate",
        "report_date": "2026-05-19",
        "rows": rows,
        "conclusion": {
            "best_weighted_label": None if best_weighted is None else best_weighted["label"],
            "best_feature_label": None if best_feature is None else best_feature["label"],
            "best_probe_label": None if best_probe is None else best_probe["label"],
            "read": (
                "The 1318 loss jump was not a tile overflow or single-frame-chunk "
                "failure. It is schedule-state sensitive: the original lr=0.005 "
                "optimizer continuation fails, while lr=0.001 continuations avoid "
                "the spike. The trainer also needed to re-apply config LR after "
                "optimizer.load_state_dict, because the checkpoint optimizer carried "
                "lr=0.005."
            ),
            "next": (
                "Use the 1300-step checkpoint with effective lr=0.001 for the next "
                "quality continuation gate. Retaining optimizer moments gives the "
                "best weighted/probe result in this 20-step diagnostic; reset "
                "optimizer gives the slightly lower feature MSE but weaker weighted "
                "objective. This is a schedule fix, not the renderer-speed fix."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STAR UVT Feature1 LR Reset Gate",
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
        "| label | pass | cfg lr | loaded opt lrs | effective opt lrs | end loss | d loss | end feature | d feature | end probe PSNR | d probe PSNR | spike d loss | chunks +/- | no-first step ms | no-first render ms | no-first backward ms | tile max/p95/cap |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        trace = row["trace_delta_1318_vs_1317"] or {}
        loaded_lrs = row["resume_optimizer_lrs_loaded"]
        effective_lrs = row["optimizer_lrs"]
        chunks = (
            ""
            if not trace
            else f"{trace['positive_chunk_count']}/{trace['negative_chunk_count']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    str(row["pass"]).lower(),
                    _fmt(row["configured_lr"], 4),
                    "" if loaded_lrs is None else str(loaded_lrs),
                    "" if effective_lrs is None else str(effective_lrs),
                    _fmt(row["end_loss"], 6),
                    _fmt(row["loss_delta"], 6),
                    _fmt(row["end_feature_target_loss"], 6),
                    _fmt(row["feature_target_loss_delta"], 6),
                    _fmt(row["end_rgb_probe_psnr"], 3),
                    _fmt(row["rgb_probe_psnr_delta"], 3),
                    _fmt(trace.get("weighted_loss_delta"), 6),
                    chunks,
                    _fmt(row["mean_no_first_step_ms"], 1),
                    _fmt(row["mean_no_first_render_ms"], 1),
                    _fmt(row["mean_no_first_backward_ms"], 1),
                    "/".join(str(value) for value in row["tile_max_p95_cap"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for row in report["rows"]:
        lines.append(f"- `{row['label']}`: `{row['source']}`")
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md",
    )
    args = parser.parse_args()
    report = build_report()
    out_json = ROOT / args.out_json
    write_report_json(out_json, report)
    write_markdown(report, ROOT / args.out_md)


if __name__ == "__main__":
    main()
