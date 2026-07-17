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
        "1250->1270 trace20",
        "outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace20_from1250.json",
    ),
    (
        "1300->1320 trace20",
        "outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace20_from1300.json",
    ),
]

TIMING_KEYS = (
    "step_ms",
    "render_forward_ms",
    "colorize_loss_ms",
    "backward_ms",
    "optimizer_ms",
    "feature_target_ms",
    "rgb_probe_loss_ms",
)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _timing_summary(row: dict[str, Any], key: str) -> dict[str, float | None]:
    values = [float(step[key]) for step in row["step_timings_ms"]]
    no_first = values[1:]
    return {
        "mean": _mean(values),
        "mean_after_first": _mean(no_first),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
    }


def _loss_spikes(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = [float(value) for value in row[key]]
    steps = row["step_global_steps"]
    spikes = []
    for index in range(1, len(values)):
        if values[index] > values[index - 1]:
            spikes.append(
                {
                    "index": index,
                    "global_step": steps[index],
                    "previous": values[index - 1],
                    "current": values[index],
                    "delta": values[index] - values[index - 1],
                }
            )
    return spikes


def _row(label: str, path: str) -> dict[str, Any]:
    data = load_report_json(path)
    timing = {key: _timing_summary(data, key) for key in TIMING_KEYS}
    tile = data["tile_stats"]
    return {
        "label": label,
        "path": path,
        "pass": bool(data["pass"]),
        "loss_decreased": bool(data["loss_decreased"]),
        "feature_target_loss_decreased": bool(data["feature_target_loss_decreased"]),
        "rgb_probe_loss_decreased": bool(data["rgb_probe_loss_decreased"]),
        "global_steps": [data["start_global_step"], data["end_global_step"]],
        "feature_loss": [data["start_feature_target_loss"], data["end_feature_target_loss"]],
        "rgb_probe_psnr": [data["start_rgb_probe_psnr"], data["end_rgb_probe_psnr"]],
        "timing": timing,
        "loss_spikes": _loss_spikes(data, "losses"),
        "feature_loss_spikes": _loss_spikes(data, "feature_target_losses"),
        "rgb_probe_loss_spikes": _loss_spikes(data, "rgb_probe_losses"),
        "tile_overflow_sum": int(data["tile_overflow_sum"]),
        "max_tile_count": int(tile["max_tile_count"]),
        "p95_tile_count": float(tile["p95_tile_count"]),
        "tile_capacity": int(data["tile_capacity"]),
        "offline_wandb_note": "see local wandb/offline-run directory from command output",
    }


def build_report() -> dict[str, Any]:
    rows = [_row(label, path) for label, path in ROWS]
    first, second = rows
    deltas = {
        key: {
            "mean": second["timing"][key]["mean"] - first["timing"][key]["mean"],
            "mean_after_first": (
                second["timing"][key]["mean_after_first"]
                - first["timing"][key]["mean_after_first"]
            ),
        }
        for key in TIMING_KEYS
    }
    return {
        "gate": "star_uvt_feature1_trainer_timing_trace",
        "report_date": "2026-05-19",
        "rows": rows,
        "deltas_second_minus_first_ms": deltas,
        "conclusion": {
            "trainer_trace_reproduces_slowdown": deltas["step_ms"]["mean_after_first"] > 100.0,
            "slowdown_not_first_step_only": deltas["step_ms"]["mean_after_first"] > 100.0,
            "tile_overflow_explains_slowdown": False,
            "quality_spike_global_steps": [
                spike["global_step"]
                for spike in second["loss_spikes"]
            ],
            "read": (
                "The short end-to-end trainer trace reproduces a real 1300-source "
                "slowdown even after dropping the first optimizer/warmup step. The "
                "slowdown is spread across render forward, the combined target/probe "
                "loss region, and backward, and both rows remain zero-overflow. The "
                "1300-source trace also exposes a late loss/probe spike at global "
                "step 1318, so this gate is both a timing trace and an objective "
                "stability warning."
            ),
            "next": (
                "Do not chase tile capacity for this issue. The next implementation "
                "gate is either native VJP/scalar fixedbin work to reduce renderer "
                "backward, or a narrower trainer autograd/MPS trace around the "
                "late-spike step if continuing this exact schedule."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STAR UVT Feature1 Trainer Timing Trace",
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
        "| row | pass | global | feature loss | probe PSNR | step mean | step mean no-first | render no-first | loss-region no-first | backward no-first | max step | overflow | max/p95/cap | loss spikes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        timing = row["timing"]
        loss_spikes = ",".join(str(item["global_step"]) for item in row["loss_spikes"])
        tile = f"{row['max_tile_count']}/{row['p95_tile_count']:.0f}/{row['tile_capacity']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    str(row["pass"]),
                    _pair(row["global_steps"], 0),
                    _pair(row["feature_loss"], 6),
                    _pair(row["rgb_probe_psnr"], 3),
                    _fmt(timing["step_ms"]["mean"], 1),
                    _fmt(timing["step_ms"]["mean_after_first"], 1),
                    _fmt(timing["render_forward_ms"]["mean_after_first"], 1),
                    _fmt(timing["colorize_loss_ms"]["mean_after_first"], 1),
                    _fmt(timing["backward_ms"]["mean_after_first"], 1),
                    _fmt(timing["step_ms"]["max"], 1),
                    str(row["tile_overflow_sum"]),
                    tile,
                    loss_spikes,
                ]
            )
            + " |"
        )
    delta = report["deltas_second_minus_first_ms"]
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| metric | mean delta | mean no-first delta |",
            "|---|---:|---:|",
        ]
    )
    for key in TIMING_KEYS:
        lines.append(
            f"| {key} | {_fmt(delta[key]['mean'], 1)} | {_fmt(delta[key]['mean_after_first'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for row in report["rows"]:
        lines.append(f"- {row['label']}: `{row['path']}`")
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.json",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md",
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
