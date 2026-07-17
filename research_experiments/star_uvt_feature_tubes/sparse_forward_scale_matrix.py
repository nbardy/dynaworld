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
        split_csv_ints,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        load_optional_report_json,
        mean_timing_without_first,
        run_star_uvt_feature_trainer_subprocess,
        split_csv_ints,
        write_report_json,
        write_report_text,
    )
from config_utils import load_config_file
from research_experiments.star_uvt_feature_tubes.star_uvt_sparse_forward_profile import (
    profile_config,
)


BASE_CONFIG = (
    ROOT
    / "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforwardvjp.jsonc"
)
DEFAULT_OUT_BASE = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_scale_128_256_512"


def _split_sizes(value: str) -> tuple[int, ...]:
    sizes = split_csv_ints(value)
    if not sizes:
        raise ValueError("--sizes must contain at least one integer")
    for size in sizes:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
    return sizes


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _size_config(base: dict[str, Any], *, size: int, out_json: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["data"]["target_size"] = int(size)
    cfg["feature_target"]["image_vjp_mode"] = "analytic_sparse_grid_forward"
    cfg["output"]["out_json"] = str(out_json)
    cfg["output"]["checkpoint"] = None
    cfg["output"]["contact_sheet"] = None
    cfg["output"]["side_by_side_video"] = None
    cfg["output"]["rgb_probe_contact_sheet"] = None
    cfg["output"]["rgb_probe_side_by_side_video"] = None
    cfg["logging"]["wandb_mode"] = "offline"
    cfg["logging"]["wandb_run_name"] = f"star-uvt-sparseforward-scale-{size}px-from1300-5step"
    tags = [tag for tag in cfg["logging"]["wandb_tags"] if not str(tag).endswith("px")]
    cfg["logging"]["wandb_tags"] = [*tags, f"{size}px", "sparse_forward_scale"]
    return cfg


def _row_from_outputs(
    *,
    size: int,
    profile: dict[str, Any] | None,
    trainer: dict[str, Any] | None,
    trainer_status: str,
    trainer_error: str,
    elapsed_sec: float,
    config_path: Path,
    trainer_json_path: Path,
    profile_json_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    timing = trainer.get("mean_timing_ms", {}) if isinstance(trainer, dict) else {}
    steps = trainer.get("step_timings_ms", []) if isinstance(trainer, dict) else []
    tile_stats = trainer.get("tile_stats", {}) if isinstance(trainer, dict) else {}
    return {
        "size": size,
        "profile_pass": profile.get("pass") if isinstance(profile, dict) else None,
        "dense_forward_ms": profile.get("dense_render_ms", {}).get("mean") if isinstance(profile, dict) else None,
        "sparse_forward_ms": profile.get("sparse_render_ms", {}).get("mean") if isinstance(profile, dict) else None,
        "forward_speedup": profile.get("speedup_vs_dense_render") if isinstance(profile, dict) else None,
        "feature_error": profile.get("max_feature_error") if isinstance(profile, dict) else None,
        "alpha_error": profile.get("max_alpha_error") if isinstance(profile, dict) else None,
        "sparse_pixels": profile.get("sparse_pixel_count") if isinstance(profile, dict) else None,
        "sparse_pixel_fraction": profile.get("sparse_pixel_fraction") if isinstance(profile, dict) else None,
        "trainer_status": trainer_status,
        "trainer_error": trainer_error,
        "elapsed_sec": round(elapsed_sec, 3),
        "trainer_pass": trainer.get("pass") if isinstance(trainer, dict) else None,
        "mean_step_ms": timing.get("step_ms"),
        "no_first_step_ms": mean_timing_without_first(trainer, "step_ms") if isinstance(trainer, dict) else None,
        "last_step_ms": steps[-1].get("step_ms") if steps else None,
        "mean_backward_ms": timing.get("backward_ms"),
        "no_first_backward_ms": mean_timing_without_first(trainer, "backward_ms") if isinstance(trainer, dict) else None,
        "mean_render_forward_ms": timing.get("render_forward_ms"),
        "no_first_render_forward_ms": mean_timing_without_first(trainer, "render_forward_ms") if isinstance(trainer, dict) else None,
        "mean_feature_target_ms": timing.get("feature_target_ms"),
        "mean_rgb_probe_loss_ms": timing.get("rgb_probe_loss_ms"),
        "start_loss": trainer.get("start_loss") if isinstance(trainer, dict) else None,
        "end_loss": trainer.get("end_loss") if isinstance(trainer, dict) else None,
        "start_feature_target_loss": trainer.get("start_feature_target_loss") if isinstance(trainer, dict) else None,
        "end_feature_target_loss": trainer.get("end_feature_target_loss") if isinstance(trainer, dict) else None,
        "start_rgb_probe_psnr": trainer.get("start_rgb_probe_psnr") if isinstance(trainer, dict) else None,
        "end_rgb_probe_psnr": trainer.get("end_rgb_probe_psnr") if isinstance(trainer, dict) else None,
        "tile_overflow_sum": trainer.get("tile_overflow_sum") if isinstance(trainer, dict) else None,
        "tile_unstable_sum": trainer.get("tile_unstable_sum") if isinstance(trainer, dict) else None,
        "max_tile_count": tile_stats.get("max_tile_count"),
        "p95_tile_count": tile_stats.get("p95_tile_count"),
        "config_path": str(config_path),
        "profile_json_path": str(profile_json_path),
        "trainer_json_path": str(trainer_json_path),
        "log_path": str(log_path),
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = (
        "size",
        "profile_pass",
        "dense_forward_ms",
        "sparse_forward_ms",
        "forward_speedup",
        "sparse_pixels",
        "trainer_pass",
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
    )
    lines = [
        "# STAR UVT Sparse-Forward Scale Matrix",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Runs the selected target-grid/frozen-probe sparse-forward route at multiple render sizes.",
        "Each row uses the same 64-frame, 8192-tube, F32 from-1300 checkpoint and",
        "`feature_target.image_vjp_mode=analytic_sparse_grid_forward`.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in columns) + " |")
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"- {row['size']}px config: `{row['config_path']}`",
                f"- {row['size']}px profile JSON: `{row['profile_json_path']}`",
                f"- {row['size']}px trainer JSON: `{row['trainer_json_path']}`",
                f"- {row['size']}px log: `{row['log_path']}`",
            ]
        )
    lines.extend(
        [
            "",
            "Decision:",
            "",
            "Use this table to reason about render-size scaling for the selected sparse-forward path. "
            "Promotion still requires zero overflow, matching loss/probe movement, and a clear end-to-end timing win.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--sizes", default="128,256,512")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config_file(args.base_config)
    sizes = _split_sizes(args.sizes)
    out_base = args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_base.parent / f"{out_base.name}_work"
    config_dir = work_dir / "configs"
    log_dir = work_dir / "logs"
    profile_dir = work_dir / "profiles"
    for directory in (config_dir, log_dir, profile_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for size in sizes:
        trainer_json_path = out_base.parent / f"{out_base.name}_{size}px_trainer.json"
        profile_json_path = profile_dir / f"{size}px_profile.json"
        config_path = config_dir / f"{size}px_sparseforward.json"
        log_path = log_dir / f"{size}px_trainer.log"
        cfg = _size_config(base, size=size, out_json=trainer_json_path)
        write_report_json(config_path, cfg)
        print(f"[sparse-forward-scale] size={size}px")
        if args.dry_run:
            rows.append(
                _row_from_outputs(
                    size=size,
                    profile=None,
                    trainer=None,
                    trainer_status="dry-run",
                    trainer_error="",
                    elapsed_sec=0.0,
                    config_path=config_path,
                    trainer_json_path=trainer_json_path,
                    profile_json_path=profile_json_path,
                    log_path=log_path,
                )
            )
            continue

        profile = profile_config(config_path, warmup=args.profile_warmup, repeat=args.profile_repeat)
        write_report_json(profile_json_path, profile)
        if trainer_json_path.exists():
            trainer_json_path.unlink()
        result = run_star_uvt_feature_trainer_subprocess(
            config_path=config_path,
            log_path=log_path,
            python=args.python,
            timeout_sec=args.timeout_sec,
            tmp_dir=log_path.parent.parent / "tmp",
        )
        trainer = load_optional_report_json(trainer_json_path)
        rows.append(
            _row_from_outputs(
                size=size,
                profile=profile,
                trainer=trainer,
                trainer_status=result.status,
                trainer_error=result.error,
                elapsed_sec=result.elapsed_sec,
                config_path=config_path,
                trainer_json_path=trainer_json_path,
                profile_json_path=profile_json_path,
                log_path=log_path,
            )
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(args.base_config),
        "sizes": list(sizes),
        "profile_warmup": args.profile_warmup,
        "profile_repeat": args.profile_repeat,
        "rows": rows,
    }
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    write_report_json(json_path, summary)
    _write_markdown(rows, md_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
