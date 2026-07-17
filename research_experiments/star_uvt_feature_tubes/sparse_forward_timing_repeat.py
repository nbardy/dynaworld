from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import (
        ROOT,
        distribution_stats,
        load_optional_report_json,
        mean_timing_without_first,
        run_star_uvt_feature_trainer_subprocess,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        distribution_stats,
        load_optional_report_json,
        mean_timing_without_first,
        run_star_uvt_feature_trainer_subprocess,
        write_report_json,
        write_report_text,
    )
from config_utils import load_config_file


BASE_CONFIG = (
    ROOT
    / "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforwardvjp.jsonc"
)
DEFAULT_OUT_BASE = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_512_repeat_timing"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _repeat_config(
    base: dict[str, Any],
    *,
    repeat_index: int,
    out_json: Path,
    image_vjp_mode: str | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    if image_vjp_mode is not None:
        cfg["feature_target"]["image_vjp_mode"] = image_vjp_mode
    cfg["output"]["out_json"] = str(out_json)
    cfg["output"]["checkpoint"] = None
    cfg["output"]["contact_sheet"] = None
    cfg["output"]["side_by_side_video"] = None
    cfg["output"]["rgb_probe_contact_sheet"] = None
    cfg["output"]["rgb_probe_side_by_side_video"] = None
    cfg["logging"]["wandb_mode"] = "offline"
    mode = str(cfg["feature_target"]["image_vjp_mode"])
    cfg["logging"]["wandb_run_name"] = f"star-uvt-{mode}-512-repeat-{repeat_index:02d}"
    tags = list(cfg["logging"].get("wandb_tags", []))
    cfg["logging"]["wandb_tags"] = [*tags, "repeat_timing", f"repeat_{repeat_index:02d}"]
    return cfg


def _row_from_run(
    *,
    repeat_index: int,
    row: dict[str, Any] | None,
    status: str,
    error: str,
    elapsed_sec: float,
    config_path: Path,
    json_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    timing = row.get("mean_timing_ms", {}) if isinstance(row, dict) else {}
    steps = row.get("step_timings_ms", []) if isinstance(row, dict) else []
    tile_stats = row.get("tile_stats", {}) if isinstance(row, dict) else {}
    return {
        "repeat": repeat_index,
        "status": status,
        "error": error,
        "elapsed_sec": round(elapsed_sec, 3),
        "pass": row.get("pass") if isinstance(row, dict) else None,
        "mean_step_ms": timing.get("step_ms"),
        "no_first_step_ms": mean_timing_without_first(row, "step_ms") if isinstance(row, dict) else None,
        "last_step_ms": steps[-1].get("step_ms") if steps else None,
        "mean_backward_ms": timing.get("backward_ms"),
        "no_first_backward_ms": mean_timing_without_first(row, "backward_ms") if isinstance(row, dict) else None,
        "mean_render_forward_ms": timing.get("render_forward_ms"),
        "no_first_render_forward_ms": mean_timing_without_first(row, "render_forward_ms") if isinstance(row, dict) else None,
        "mean_feature_target_ms": timing.get("feature_target_ms"),
        "mean_rgb_probe_loss_ms": timing.get("rgb_probe_loss_ms"),
        "start_loss": row.get("start_loss") if isinstance(row, dict) else None,
        "end_loss": row.get("end_loss") if isinstance(row, dict) else None,
        "start_feature_target_loss": row.get("start_feature_target_loss") if isinstance(row, dict) else None,
        "end_feature_target_loss": row.get("end_feature_target_loss") if isinstance(row, dict) else None,
        "start_rgb_probe_psnr": row.get("start_rgb_probe_psnr") if isinstance(row, dict) else None,
        "end_rgb_probe_psnr": row.get("end_rgb_probe_psnr") if isinstance(row, dict) else None,
        "tile_overflow_sum": row.get("tile_overflow_sum") if isinstance(row, dict) else None,
        "tile_unstable_sum": row.get("tile_unstable_sum") if isinstance(row, dict) else None,
        "max_tile_count": tile_stats.get("max_tile_count"),
        "p95_tile_count": tile_stats.get("p95_tile_count"),
        "config_path": str(config_path),
        "json_path": str(json_path),
        "log_path": str(log_path),
    }


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "mean_step_ms",
        "no_first_step_ms",
        "last_step_ms",
        "mean_backward_ms",
        "no_first_backward_ms",
        "mean_render_forward_ms",
        "no_first_render_forward_ms",
        "mean_feature_target_ms",
        "mean_rgb_probe_loss_ms",
    )
    out: dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        out[key] = distribution_stats(values)
    return out


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    columns = (
        "repeat",
        "status",
        "pass",
        "mean_step_ms",
        "no_first_step_ms",
        "last_step_ms",
        "mean_backward_ms",
        "no_first_backward_ms",
        "mean_render_forward_ms",
        "end_loss",
        "end_feature_target_loss",
        "end_rgb_probe_psnr",
        "tile_overflow_sum",
        "max_tile_count",
        "json_path",
    )
    rows = result["rows"]
    summary = result["summary_stats"]
    lines = [
        "# STAR UVT Sparse-Forward 512px Repeat Timing",
        "",
        f"Generated: {result['generated_at']}",
        f"Image VJP mode: `{result['feature_target_image_vjp_mode']}`",
        "",
        "Runs the selected 64f/512px target-grid/frozen-probe sparse-forward trainer",
        "multiple times from the same 1300-step checkpoint/config. This is a timing",
        "stability gate, not a new quality objective.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in columns) + " |")
    lines.extend(
        [
            "",
            "## Timing Summary",
            "",
            "| metric | mean | min | max | stdev |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in (
        "mean_step_ms",
        "no_first_step_ms",
        "last_step_ms",
        "mean_backward_ms",
        "no_first_backward_ms",
        "mean_render_forward_ms",
        "no_first_render_forward_ms",
    ):
        stat = summary[key]
        lines.append(
            "| "
            + " | ".join(
                (
                    key,
                    _fmt(stat["mean"]),
                    _fmt(stat["min"]),
                    _fmt(stat["max"]),
                    _fmt(stat["stdev"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Decision:",
            "",
            "Use these repeat stats as the comparison surface for the next native target-grid/probe loss+VJP or fixedbin/tile-slot gate. A candidate must preserve the same loss/probe movement, keep zero overflow, and beat the repeat distribution rather than one cherry-picked run.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--image-vjp-mode", default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    base = load_config_file(args.base_config)
    out_base = args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_base.parent / f"{out_base.name}_work"
    config_dir = work_dir / "configs"
    log_dir = work_dir / "logs"
    for directory in (config_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        json_path = out_base.parent / f"{out_base.name}_repeat{repeat_index:02d}.json"
        config_path = config_dir / f"repeat{repeat_index:02d}.json"
        log_path = log_dir / f"repeat{repeat_index:02d}.log"
        cfg = _repeat_config(
            base,
            repeat_index=repeat_index,
            out_json=json_path,
            image_vjp_mode=args.image_vjp_mode,
        )
        write_report_json(config_path, cfg)
        print(f"[sparse-forward-repeat] repeat={repeat_index}")
        if args.dry_run:
            rows.append(
                _row_from_run(
                    repeat_index=repeat_index,
                    row=None,
                    status="dry-run",
                    error="",
                    elapsed_sec=0.0,
                    config_path=config_path,
                    json_path=json_path,
                    log_path=log_path,
                )
            )
            continue
        if json_path.exists():
            json_path.unlink()
        result = run_star_uvt_feature_trainer_subprocess(
            config_path=config_path,
            log_path=log_path,
            python=args.python,
            timeout_sec=args.timeout_sec,
            tmp_dir=log_path.parent.parent / "tmp",
        )
        rows.append(
            _row_from_run(
                repeat_index=repeat_index,
                row=load_optional_report_json(json_path),
                status=result.status,
                error=result.error,
                elapsed_sec=result.elapsed_sec,
                config_path=config_path,
                json_path=json_path,
                log_path=log_path,
            )
        )

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(args.base_config),
        "feature_target_image_vjp_mode": str(
            args.image_vjp_mode if args.image_vjp_mode is not None else base["feature_target"]["image_vjp_mode"]
        ),
        "repeat": args.repeat,
        "rows": rows,
        "summary_stats": _summary_stats(rows),
    }
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    write_report_json(json_path, result)
    _write_markdown(md_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
