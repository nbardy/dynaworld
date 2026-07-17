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
        load_optional_report_json,
        mean_timing_without_first,
        run_star_uvt_feature_trainer_subprocess,
        split_csv_strings,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        load_optional_report_json,
        mean_timing_without_first,
        run_star_uvt_feature_trainer_subprocess,
        split_csv_strings,
        write_report_json,
        write_report_text,
    )
from config_utils import load_config_file
from star_uvt_render_modes import (
    FEATURE_RENDER_MODE_ORDER,
    backward_mode_for_feature_render_mode,
)


BASE_CONFIG = (
    ROOT
    / "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_analyticvjp.jsonc"
)
DEFAULT_MODES = FEATURE_RENDER_MODE_ORDER


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _case_config(base: dict[str, Any], *, mode: str, out_json: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["feature_uvt"]["render_mode"] = mode
    cfg["output"]["out_json"] = str(out_json)
    cfg["output"]["checkpoint"] = None
    cfg["output"]["contact_sheet"] = None
    cfg["output"]["side_by_side_video"] = None
    cfg["output"]["rgb_probe_contact_sheet"] = None
    cfg["output"]["rgb_probe_side_by_side_video"] = None
    cfg["logging"]["wandb_mode"] = "offline"
    cfg["logging"]["wandb_run_name"] = f"star-uvt-targetgrid-render-mode-matrix-{mode}"
    tags = [tag for tag in cfg["logging"]["wandb_tags"] if not str(tag).startswith("feature_direct_")]
    cfg["logging"]["wandb_tags"] = [*tags, mode, "render_mode_matrix"]
    return cfg


def _run_case(
    *,
    base: dict[str, Any],
    mode: str,
    config_path: Path,
    out_json: Path,
    log_path: Path,
    python: str,
    timeout_sec: int,
) -> dict[str, Any]:
    cfg = _case_config(base, mode=mode, out_json=out_json)
    write_report_json(config_path, cfg)
    if out_json.exists():
        out_json.unlink()

    result = run_star_uvt_feature_trainer_subprocess(
        config_path=config_path,
        log_path=log_path,
        python=python,
        timeout_sec=timeout_sec,
        tmp_dir=log_path.parent.parent / "tmp",
    )
    row = load_optional_report_json(out_json)
    feature_dim = int(base["feature_uvt"]["feature_dim"])
    timing = row.get("mean_timing_ms", {}) if isinstance(row, dict) else {}
    tile_stats = row.get("tile_stats", {}) if isinstance(row, dict) else {}
    return {
        "mode": mode,
        "status": result.status,
        "error": result.error,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "pass": row.get("pass") if isinstance(row, dict) else None,
        "kernel_backward_mode_expected": backward_mode_for_feature_render_mode(mode, feature_dim),
        "kernel_backward_mode_reported": row.get("kernel_backward_mode") if isinstance(row, dict) else None,
        "effective_render_mode": row.get("effective_render_mode") if isinstance(row, dict) else None,
        "fixedbin_alias": row.get("requested_fixedbin_is_direct_atomic_alias") if isinstance(row, dict) else None,
        "image_vjp_mode": row.get("feature_target_image_vjp_mode") if isinstance(row, dict) else None,
        "mean_step_ms": timing.get("step_ms"),
        "no_first_step_ms": mean_timing_without_first(row, "step_ms") if isinstance(row, dict) else None,
        "render_forward_ms": timing.get("render_forward_ms"),
        "backward_ms": timing.get("backward_ms"),
        "colorize_loss_ms": timing.get("colorize_loss_ms"),
        "feature_target_ms": timing.get("feature_target_ms"),
        "rgb_probe_loss_ms": timing.get("rgb_probe_loss_ms"),
        "start_loss": row.get("start_loss") if isinstance(row, dict) else None,
        "end_loss": row.get("end_loss") if isinstance(row, dict) else None,
        "start_feature_target_loss": row.get("start_feature_target_loss") if isinstance(row, dict) else None,
        "end_feature_target_loss": row.get("end_feature_target_loss") if isinstance(row, dict) else None,
        "start_rgb_probe_psnr": row.get("start_rgb_probe_psnr") if isinstance(row, dict) else None,
        "end_rgb_probe_psnr": row.get("end_rgb_probe_psnr") if isinstance(row, dict) else None,
        "tile_overflow_sum": row.get("tile_overflow_sum") if isinstance(row, dict) else None,
        "tile_unstable_sum": row.get("tile_unstable_sum") if isinstance(row, dict) else None,
        "fixedbin_eligible": tile_stats.get("fixedbin_eligible"),
        "max_tile_count": tile_stats.get("max_tile_count"),
        "p95_tile_count": tile_stats.get("p95_tile_count"),
        "json_path": str(out_json),
        "log_path": str(log_path),
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = (
        "mode",
        "status",
        "pass",
        "kernel_backward_mode_reported",
        "effective_render_mode",
        "fixedbin_alias",
        "mean_step_ms",
        "no_first_step_ms",
        "render_forward_ms",
        "backward_ms",
        "feature_target_ms",
        "rgb_probe_loss_ms",
        "end_loss",
        "end_feature_target_loss",
        "end_rgb_probe_psnr",
        "tile_overflow_sum",
        "fixedbin_eligible",
        "max_tile_count",
        "json_path",
    )
    lines = [
        "# STAR UVT Target-Grid Trainer Render-Mode Matrix",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Matched 5-step trainer runs from the same 1300-step checkpoint/config.",
        "This is the end-to-end check for current STAR feature renderer modes,",
        "including the Python loss/VJP path and optimizer overhead.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in columns) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `feature_direct_fixedbin` is currently an eligibility/request label; the actual kernel is `direct_atomic`.",
            "- `no_first_step_ms` drops the first local optimizer/warmup step and is the fairest short-run timing.",
            "- Promotion requires both timing and identical quality movement from the same checkpoint.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--out-base", type=Path, default=ROOT / "outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config_file(args.base_config)
    out_base = args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_base.parent / f"{out_base.name}_work"
    config_dir = work_dir / "configs"
    log_dir = work_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mode in split_csv_strings(args.modes):
        out_json = out_base.parent / f"{out_base.name}_{mode}.json"
        config_path = config_dir / f"{mode}.json"
        log_path = log_dir / f"{mode}.log"
        print(f"[render-mode-matrix] mode={mode}")
        if args.dry_run:
            rows.append(
                {
                    "mode": mode,
                    "status": "dry_run",
                    "kernel_backward_mode_expected": backward_mode_for_feature_render_mode(
                        mode,
                        int(base["feature_uvt"]["feature_dim"]),
                    ),
                    "json_path": str(out_json),
                    "log_path": str(log_path),
                }
            )
            continue
        rows.append(
            _run_case(
                base=base,
                mode=mode,
                config_path=config_path,
                out_json=out_json,
                log_path=log_path,
                python=args.python,
                timeout_sec=int(args.timeout_sec),
            )
        )

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(args.base_config),
        "rows": rows,
        "pass": all(row.get("status") in {"ok", "dry_run"} and row.get("pass", True) is not False for row in rows),
    }
    write_report_json(out_base.with_suffix(".json"), summary)
    _write_markdown(rows, out_base.with_suffix(".md"))


if __name__ == "__main__":
    main()
