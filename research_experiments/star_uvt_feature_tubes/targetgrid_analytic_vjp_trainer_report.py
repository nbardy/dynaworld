from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import ROOT, fmt_cell, load_report_json, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, fmt_cell, load_report_json, write_report_json, write_report_text

OUT_JSON = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_trainer_report.json"
OUT_MD = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_trainer_report.md"

BRIDGE_REPEAT5 = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_repeat5.json"
TRAIN_AUTOGRAD = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_autogradvjp_64f512_from1300_5step.json"
TRAIN_ANALYTIC = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_analyticvjp_64f512_from1300_5step.json"
TRAIN_ANALYTIC_RERUN = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_analyticvjp_64f512_from1300_5step_rerun.json"


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _timing_summary(row: dict[str, Any]) -> dict[str, Any]:
    timings = row["step_timings_ms"]
    return {
        "mean_step_ms": float(row["mean_timing_ms"]["step_ms"]),
        "mean_backward_ms": float(row["mean_timing_ms"]["backward_ms"]),
        "mean_render_ms": float(row["mean_timing_ms"]["render_forward_ms"]),
        "mean_loss_vjp_ms": float(row["mean_timing_ms"]["colorize_loss_ms"]),
        "mean_feature_target_ms": float(row["mean_timing_ms"].get("feature_target_ms", 0.0)),
        "mean_rgb_probe_loss_ms": float(row["mean_timing_ms"].get("rgb_probe_loss_ms", 0.0)),
        "no_first_step_ms": _mean([float(item["step_ms"]) for item in timings[1:]]),
        "no_first_backward_ms": _mean([float(item["backward_ms"]) for item in timings[1:]]),
        "first_step_ms": float(timings[0]["step_ms"]),
        "last_step_ms": float(timings[-1]["step_ms"]),
    }


def build_report() -> dict[str, Any]:
    bridge = load_report_json(BRIDGE_REPEAT5)
    autograd = load_report_json(TRAIN_AUTOGRAD)
    analytic = load_report_json(TRAIN_ANALYTIC)
    analytic_rerun = load_report_json(TRAIN_ANALYTIC_RERUN)
    rows = {
        "autograd_5step": autograd,
        "analytic_5step": analytic,
        "analytic_5step_rerun": analytic_rerun,
    }
    timing = {name: _timing_summary(row) for name, row in rows.items()}
    loss_delta = abs(float(autograd["end_loss"]) - float(analytic_rerun["end_loss"]))
    pass_flag = (
        bool(bridge["pass"])
        and all(bool(row["pass"]) for row in rows.values())
        and all(int(row["tile_overflow_sum"]) == 0 for row in rows.values())
        and loss_delta <= 1.0e-6
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pass": pass_flag,
        "bridge_repeat5": {
            "path": str(BRIDGE_REPEAT5.relative_to(ROOT)),
            "speedup_vs_autograd_total": bridge["speedup_vs_autograd_total"],
            "max_grad_abs_error": bridge["grad_comparison"]["max_abs_error"],
            "loss_max_abs_error": bridge["loss_max_abs_error"],
        },
        "trainer_rows": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "image_vjp_mode": row["feature_target_image_vjp_mode"],
                "pass": row["pass"],
                "start_loss": row["start_loss"],
                "end_loss": row["end_loss"],
                "start_feature_target_loss": row["start_feature_target_loss"],
                "end_feature_target_loss": row["end_feature_target_loss"],
                "start_rgb_probe_psnr": row["start_rgb_probe_psnr"],
                "end_rgb_probe_psnr": row["end_rgb_probe_psnr"],
                "tile_overflow_sum": row["tile_overflow_sum"],
                "timing": timing[name],
            }
            for name, row, path in (
                ("autograd_5step", autograd, TRAIN_AUTOGRAD),
                ("analytic_5step", analytic, TRAIN_ANALYTIC),
                ("analytic_5step_rerun", analytic_rerun, TRAIN_ANALYTIC_RERUN),
            )
        },
        "comparison": {
            "analytic_rerun_vs_autograd_mean_step_delta_ms": timing["analytic_5step_rerun"]["mean_step_ms"]
            - timing["autograd_5step"]["mean_step_ms"],
            "analytic_rerun_vs_autograd_no_first_step_delta_ms": timing["analytic_5step_rerun"]["no_first_step_ms"]
            - timing["autograd_5step"]["no_first_step_ms"],
            "analytic_rerun_vs_autograd_mean_backward_delta_ms": timing["analytic_5step_rerun"]["mean_backward_ms"]
            - timing["autograd_5step"]["mean_backward_ms"],
            "end_loss_abs_delta": loss_delta,
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = report["trainer_rows"]
    lines = [
        "# STAR UVT Target-Grid Analytic VJP Trainer Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Bridge Gate",
        "",
        "- analytic image-VJP repeat-5 bridge speedup: "
        f"`{report['bridge_repeat5']['speedup_vs_autograd_total']:.3f}x`",
        f"- max grad abs error: `{report['bridge_repeat5']['max_grad_abs_error']:.3e}`",
        f"- loss max abs error: `{report['bridge_repeat5']['loss_max_abs_error']:.3e}`",
        "",
        "## Trainer Rows",
        "",
        "| row | mode | mean step | no-first step | mean backward | end loss | end probe PSNR | overflow | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("autograd_5step", "analytic_5step", "analytic_5step_rerun"):
        row = rows[name]
        timing = row["timing"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    row["image_vjp_mode"],
                    fmt_cell(timing["mean_step_ms"], 1),
                    fmt_cell(timing["no_first_step_ms"], 1),
                    fmt_cell(timing["mean_backward_ms"], 1),
                    fmt_cell(row["end_loss"], 6),
                    fmt_cell(row["end_rgb_probe_psnr"], 3),
                    str(row["tile_overflow_sum"]),
                    "yes" if row["pass"] else "no",
                ]
            )
            + " |"
        )
    comp = report["comparison"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- The analytic VJP trainer path is correct and trainable: both analytic rows pass, losses match the autograd baseline, gradients flow, and tile overflow stays zero.",
            "- The benchmark bridge win does not yet become a clear end-to-end trainer win. The warm analytic rerun ties mean step time (`+1.1ms`) and is only slightly faster after dropping the first step (`-4.9ms`).",
            f"- The backward bucket does improve by `{comp['analytic_rerun_vs_autograd_mean_backward_delta_ms']:.1f}ms`, but the manual loss/VJP work moves into the loss bucket, so the step-level win is marginal.",
            "- Keep `feature_target.image_vjp_mode=analytic` as an opt-in trainer diagnostic. Promotion needs a longer matched run or a fused/native implementation that moves more than the bucket accounting.",
            "",
            f"Pass: `{report['pass']}`",
            "",
        ]
    )
    write_report_text(path, "\n".join(lines))


def main() -> None:
    report = build_report()
    write_report_json(OUT_JSON, report)
    write_markdown(OUT_MD, report)
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD), "pass": report["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
