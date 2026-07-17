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

TRACE_JSON = "outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace20_from1300.json"


def _trace_by_step(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(trace["global_step"]): trace for trace in row["chunk_traces"]}


def _chunk_by_frame(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(chunk["frame_start"]): chunk for chunk in trace["chunks"]}


def _range_delta(deltas: list[dict[str, Any]], start: int, end: int) -> float:
    return sum(float(item["weighted_loss_delta"]) for item in deltas if start <= int(item["frame_start"]) < end)


def build_report() -> dict[str, Any]:
    row = load_report_json(TRACE_JSON)
    by_step = _trace_by_step(row)
    pre = by_step[1317]
    spike = by_step[1318]
    post = by_step[1319]
    pre_chunks = _chunk_by_frame(pre)
    spike_chunks = _chunk_by_frame(spike)
    post_chunks = _chunk_by_frame(post)
    deltas = []
    for frame_start, chunk in spike_chunks.items():
        previous = pre_chunks[frame_start]
        next_chunk = post_chunks[frame_start]
        deltas.append(
            {
                "frame_start": frame_start,
                "weighted_loss_delta": float(chunk["weighted_loss"]) - float(previous["weighted_loss"]),
                "feature_loss_delta": float(chunk["feature_target_loss"]) - float(previous["feature_target_loss"]),
                "rgb_probe_loss_delta": float(chunk["rgb_probe_loss"]) - float(previous["rgb_probe_loss"]),
                "post_weighted_loss_delta_vs_1317": (
                    float(next_chunk["weighted_loss"]) - float(previous["weighted_loss"])
                ),
                "weighted_loss_1317": float(previous["weighted_loss"]),
                "weighted_loss_1318": float(chunk["weighted_loss"]),
                "weighted_loss_1319": float(next_chunk["weighted_loss"]),
            }
        )
    deltas_sorted = sorted(deltas, key=lambda item: item["weighted_loss_delta"], reverse=True)
    positive_count = sum(1 for item in deltas if item["weighted_loss_delta"] > 0.0)
    negative_count = sum(1 for item in deltas if item["weighted_loss_delta"] < 0.0)
    total_delta = sum(float(item["weighted_loss_delta"]) for item in deltas)
    quarters = [
        {"range": [0, 16], "weighted_loss_delta": _range_delta(deltas, 0, 16)},
        {"range": [16, 32], "weighted_loss_delta": _range_delta(deltas, 16, 32)},
        {"range": [32, 48], "weighted_loss_delta": _range_delta(deltas, 32, 48)},
        {"range": [48, 64], "weighted_loss_delta": _range_delta(deltas, 48, 64)},
    ]
    return {
        "gate": "star_uvt_feature1_chunktrace_spike_localization",
        "report_date": "2026-05-19",
        "source": TRACE_JSON,
        "pass": bool(row["pass"]),
        "trace_global_steps": row["chunk_trace_global_steps"],
        "step_summary": [
            {
                "global_step": trace["global_step"],
                "loss": trace["loss"],
                "feature_target_loss": trace["feature_target_loss"],
                "rgb_probe_loss": trace["rgb_probe_loss"],
                "step_ms": trace["timing_ms"]["step_ms"],
                "render_forward_ms": trace["timing_ms"]["render_forward_ms"],
                "backward_ms": trace["timing_ms"]["backward_ms"],
            }
            for trace in (pre, spike, post)
        ],
        "spike_delta": {
            "weighted_loss": float(spike["loss"]) - float(pre["loss"]),
            "feature_target_loss": float(spike["feature_target_loss"]) - float(pre["feature_target_loss"]),
            "rgb_probe_loss": float(spike["rgb_probe_loss"]) - float(pre["rgb_probe_loss"]),
            "positive_chunk_count": positive_count,
            "negative_chunk_count": negative_count,
            "chunk_count": len(deltas),
            "sum_chunk_weighted_delta": total_delta,
            "quarters": quarters,
        },
        "top_chunk_deltas": deltas_sorted[:10],
        "conclusion": {
            "localized_to_one_chunk": positive_count <= 3,
            "mostly_distributed": positive_count >= 16,
            "first_quarter_share": 0.0 if total_delta == 0.0 else quarters[0]["weighted_loss_delta"] / total_delta,
            "read": (
                "The global-step 1318 objective spike is distributed across most "
                "frame chunks, not localized to a single bad chunk. 27 of 32 chunks "
                "increase weighted loss versus step 1317, with the largest share in "
                "the earliest quarter of the clip. The spike persists into 1319, "
                "so it looks like an optimizer/objective-state jump rather than a "
                "single transient render timing outlier."
            ),
            "next": (
                "Continuing this schedule should checkpoint or lower LR before the "
                "spike region, or move to native VJP/scalar fixedbin since the "
                "speed path remains renderer-backward dominated and not tile-limited."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STAR UVT Feature1 Chunk Trace Spike Localization",
        "",
        f"Date: {report['report_date']}",
        "",
        "## Answer",
        "",
        report["conclusion"]["read"],
        "",
        report["conclusion"]["next"],
        "",
        "## Step Summary",
        "",
        "| global step | loss | feature loss | probe loss | step ms | render ms | backward ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["step_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["global_step"]),
                    _fmt(row["loss"], 6),
                    _fmt(row["feature_target_loss"], 6),
                    _fmt(row["rgb_probe_loss"], 6),
                    _fmt(row["step_ms"], 1),
                    _fmt(row["render_forward_ms"], 1),
                    _fmt(row["backward_ms"], 1),
                ]
            )
            + " |"
        )
    spike = report["spike_delta"]
    lines.extend(
        [
            "",
            "## Spike Delta",
            "",
            f"- weighted loss delta 1318-1317: `{spike['weighted_loss']:.6f}`",
            f"- feature loss delta 1318-1317: `{spike['feature_target_loss']:.6f}`",
            f"- probe loss delta 1318-1317: `{spike['rgb_probe_loss']:.6f}`",
            f"- positive/negative chunks: `{spike['positive_chunk_count']}/{spike['negative_chunk_count']}` out of `{spike['chunk_count']}`",
            "",
            "| frame range | weighted-loss delta | share of spike |",
            "|---:|---:|---:|",
        ]
    )
    total = float(spike["weighted_loss"])
    for item in spike["quarters"]:
        share = 0.0 if total == 0.0 else float(item["weighted_loss_delta"]) / total
        lines.append(
            f"| {item['range'][0]}-{item['range'][1]} | {item['weighted_loss_delta']:.6f} | {100.0 * share:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Top Chunk Deltas",
            "",
            "| frame start | weighted delta | feature delta | probe delta | weighted 1317 | weighted 1318 | weighted 1319 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["top_chunk_deltas"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["frame_start"]),
                    _fmt(item["weighted_loss_delta"], 6),
                    _fmt(item["feature_loss_delta"], 6),
                    _fmt(item["rgb_probe_loss_delta"], 6),
                    _fmt(item["weighted_loss_1317"], 6),
                    _fmt(item["weighted_loss_1318"], 6),
                    _fmt(item["weighted_loss_1319"], 6),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifact",
            "",
            f"- source trace JSON: `{report['source']}`",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md",
    )
    args = parser.parse_args()
    report = build_report()
    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    write_report_json(out_json, report)
    write_markdown(report, out_md)
    print(json.dumps({"out_md": str(out_md), "top_chunks": len(report["top_chunk_deltas"])}, sort_keys=True))


if __name__ == "__main__":
    main()
