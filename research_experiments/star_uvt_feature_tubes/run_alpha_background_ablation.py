from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


try:
    from .report_artifacts import ROOT as DYNAWORLD_ROOT, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT as DYNAWORLD_ROOT, write_report_json, write_report_text
from config_utils import load_config_file
from pipeline.diagnostics import eval_metric_payload, temporal_similarity_payload
from rendering import resize_images
from train_devices import sync_torch_device as _sync_device
from train_logging import finish_wandb_run
from trainer_registry import instantiate_trainer_for_config, run_config_dict


DYNAMIC_BASE_CONFIG = DYNAWORLD_ROOT / "src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc"
STAR_BASE_CONFIG = DYNAWORLD_ROOT / "src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc"
DEFAULT_OUTPUT_ROOT = DYNAWORLD_ROOT / "outputs/benchmarks/2026-05-21_alpha_background_ablation"

STRATEGIES = {
    "random_rgb_after_colorizer": {
        "dynamic_background": {
            "train_mode": "random_rgb",
            "eval_mode": "black",
            "fixed_rgb": [0.0, 0.0, 0.0],
            "sample_scope": "step",
            "feature_train_mode": "none",
            "feature_eval_mode": "none",
            "feature_sample_scope": "step",
        },
        "star_alpha_background": {
            "train_strategy": "random_rgb_after_colorizer",
            "eval_strategy": "fixed_black_after_colorizer",
            "sample_scope": "step",
        },
    },
    "random_feature_before_colorizer": {
        "dynamic_background": {
            "train_mode": "none",
            "eval_mode": "none",
            "fixed_rgb": [0.0, 0.0, 0.0],
            "sample_scope": "step",
            "feature_train_mode": "random_feature",
            "feature_eval_mode": "fixed_zero",
            "feature_sample_scope": "step",
        },
        "star_alpha_background": {
            "train_strategy": "random_feature_before_colorizer",
            "eval_strategy": "fixed_zero_feature_before_colorizer",
            "sample_scope": "step",
        },
    },
}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _psnr_from_mse(mse: float | None) -> float | None:
    if mse is None:
        return None
    return float(-10.0 * math.log10(max(float(mse), 1.0e-12)))


def _metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _patch_dynamic_config(
    base: dict[str, Any],
    *,
    strategy_name: str,
    steps: int,
    run_dir: Path,
    render_size: int | None,
    frames: int | None,
    dynamic_video_path: Path | None,
    dynamic_gaussians: int | None,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["train"]["steps"] = int(steps)
    cfg["train"]["profile_timing"] = True
    cfg["train"]["profile_timing_sync"] = True
    cfg["train"]["profile_timing_log_every"] = 1
    cfg["logging"]["wandb_enabled"] = False
    cfg["logging"]["wandb_mode"] = "offline"
    cfg["logging"]["wandb_run_name"] = f"dynamic-gsplat-alpha-bg-{strategy_name}-{steps}step"
    cfg["logging"]["log_initial_media"] = False
    cfg["logging"]["feature_pca_log"] = False
    cfg["logging"]["image_log_every"] = int(steps) + 1
    cfg["logging"]["video_log_every"] = int(steps) + 1
    cfg["logging"]["always_log_last_step"] = False
    cfg["logging"]["log_every"] = max(1, min(10, int(steps)))
    cfg["losses"]["background"] = deepcopy(STRATEGIES[strategy_name]["dynamic_background"])
    cfg["export"] = {"enabled": False, "output_root": str(run_dir / "browser_export"), "id": None}
    if render_size is not None:
        cfg["model"]["size"] = int(render_size)
        cfg["render"]["render_size"] = int(render_size)
    if frames is not None:
        cfg["data"]["max_frames"] = int(frames)
        cfg["model"]["train_frame_count"] = int(frames)
        cfg["train"]["temporal_microbatch_size"] = min(int(frames), int(cfg["train"]["temporal_microbatch_size"]))
    if dynamic_video_path is not None:
        cfg["data"]["video_path"] = str(dynamic_video_path)
    if dynamic_gaussians is not None:
        tokens = int(cfg["model"]["tokens"])
        if int(dynamic_gaussians) % tokens != 0:
            raise ValueError(f"--dynamic-gaussians must be divisible by model.tokens={tokens}.")
        cfg["model"]["gaussians_per_token"] = int(dynamic_gaussians) // tokens
    return cfg


def _patch_star_config(
    base: dict[str, Any],
    *,
    strategy_name: str,
    steps: int,
    run_dir: Path,
    render_size: int | None,
    frames: int | None,
    star_video_path: Path | None,
    star_tubes: int | None,
    star_frame_chunk_size: int | None,
    star_tile_capacity: int | None,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["train"]["steps"] = int(steps)
    cfg["train"]["seed"] = 41
    cfg["train"]["require_loss_decrease"] = False
    cfg["logging"]["wandb_enabled"] = False
    cfg["logging"]["wandb_mode"] = "offline"
    cfg["logging"]["wandb_run_name"] = f"star-uvt-alpha-bg-{strategy_name}-{steps}step"
    cfg["alpha_background"] = deepcopy(STRATEGIES[strategy_name]["star_alpha_background"])
    cfg["output"]["out_json"] = str(run_dir / "result.json")
    cfg["output"]["contact_sheet"] = str(run_dir / "contact.png")
    cfg["output"]["contact_sheet_frames"] = int(cfg["data"]["max_frames"])
    cfg["output"]["side_by_side_video"] = None
    cfg["output"]["rgb_probe_contact_sheet"] = None
    cfg["output"]["rgb_probe_side_by_side_video"] = None
    if render_size is not None:
        cfg["data"]["target_size"] = int(render_size)
    if frames is not None:
        cfg["data"]["max_frames"] = int(frames)
        cfg["output"]["contact_sheet_frames"] = int(frames)
        cfg["train"]["frame_chunk_size"] = min(int(frames), int(cfg["train"]["frame_chunk_size"]))
    if star_video_path is not None:
        cfg["data"]["video_path"] = str(star_video_path)
    if star_tubes is not None:
        cfg["feature_uvt"]["tube_count"] = int(star_tubes)
    if star_frame_chunk_size is not None:
        cfg["train"]["frame_chunk_size"] = int(star_frame_chunk_size)
    if star_tile_capacity is not None:
        cfg["feature_uvt"]["tile_capacity"] = int(star_tile_capacity)
    return cfg


def _summarize_dynamic_timings(step_timings: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in step_timings for key in row})
    return {
        key: sum(float(row[key]) for row in step_timings if key in row) / max(sum(key in row for row in step_timings), 1)
        for key in keys
    }


def _warmed_timing_summary(step_timings: list[dict[str, float]]) -> dict[str, float]:
    if len(step_timings) <= 1:
        return _summarize_dynamic_timings(step_timings)
    return _summarize_dynamic_timings(step_timings[1:])


def _run_dynamic_variant(*, cfg: dict[str, Any], strategy_name: str, run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    write_report_json(config_path, cfg)
    torch.manual_seed(41)
    trainer = instantiate_trainer_for_config(cfg, config_path)
    try:
        _sync_device(trainer.device)
        initial = trainer.initial_step_result()
        _sync_device(trainer.device)
        step_timings: list[dict[str, float]] = []
        recon_losses: list[float] = []
        total_losses: list[float] = []
        for step in range(1, int(trainer.train_cfg["steps"]) + 1):
            started = time.perf_counter()
            result = trainer.step(keep_preview=False)
            _sync_device(trainer.device)
            wall_ms = (time.perf_counter() - started) * 1000.0
            recon_losses.append(float(result.recon_loss.item()))
            total_losses.append(float(result.loss.item()))
            timing_row = {"step": float(step), "wall_ms": float(wall_ms)}
            timing_row.update({f"{key}_ms": float(value) * 1000.0 for key, value in trainer.last_timing_terms.items()})
            step_timings.append(timing_row)

        rendered = trainer.render_full_sequence(trainer.sequence_data)
        gt_sequence = resize_images(trainer.sequence_data.frames, int(trainer.render_size)).detach().cpu()
        metrics = eval_metric_payload(rendered.rgb_sequence, gt_sequence, trainer.loss_cfg)
        metrics.update(temporal_similarity_payload(rendered.rgb_sequence, gt_sequence, trainer.loss_cfg))
        alpha_metrics: dict[str, float | None] = {
            "alpha_mean": None,
            "alpha_min": None,
            "alpha_max": None,
        }
        if rendered.alpha_sequence is not None:
            alpha = rendered.alpha_sequence.float()
            alpha_metrics = {
                "alpha_mean": float(alpha.mean().item()),
                "alpha_min": float(alpha.min().item()),
                "alpha_max": float(alpha.max().item()),
            }
        row = {
            "renderer": "dynamic_gsplat",
            "strategy": strategy_name,
            "status": "ok",
            "config": str(config_path),
            "steps": int(trainer.train_cfg["steps"]),
            "frames": int(trainer.model_cfg["train_frame_count"]),
            "render_size": int(trainer.render_size),
            "feature_dim": int(trainer.feature_dim),
            "effective_gaussians": int(trainer.effective_gaussians),
            "renderer_mode": str(trainer.renderer_mode),
            "background": deepcopy(cfg["losses"]["background"]),
            "start_recon_loss": float(initial.recon_loss.item()),
            "end_recon_loss": recon_losses[-1] if recon_losses else None,
            "loss_decreased": bool(recon_losses and recon_losses[-1] < float(initial.recon_loss.item())),
            "recon_losses": recon_losses,
            "total_losses": total_losses,
            "eval": metrics,
            "eval_loss": metrics.get("Eval/Loss"),
            "eval_l1": metrics.get("Eval/L1"),
            "eval_mse": metrics.get("Eval/MSE"),
            "eval_psnr": metrics.get("Eval/PSNR"),
            "eval_ssim": metrics.get("Eval/SSIM"),
            **alpha_metrics,
            "mean_timing_ms": _summarize_dynamic_timings(step_timings),
            "warm_mean_timing_ms": _warmed_timing_summary(step_timings),
            "step_timings_ms": step_timings,
        }
        write_report_json(run_dir / "result.json", row)
        return row
    finally:
        trainer.close_sequence_prefetch()
        finish_wandb_run(trainer.wandb_run)


def _run_star_variant(*, cfg: dict[str, Any], strategy_name: str, run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    write_report_json(config_path, cfg)
    torch.manual_seed(41)
    stdout_path = run_dir / "trainer_stdout.log"
    with stdout_path.open("w", encoding="utf-8") as stdout:
        with redirect_stdout(stdout):
            row = run_config_dict(cfg, config_path)
    row.update(
        {
            "renderer": "star_uvt",
            "strategy": strategy_name,
            "status": "ok",
            "config": str(config_path),
            "trainer_stdout": str(stdout_path),
            "eval_loss": row.get("final_full_rgb_loss"),
            "eval_mse": row.get("final_full_rgb_loss"),
            "eval_psnr": row.get("final_full_rgb_psnr"),
        }
    )
    row["warm_mean_timing_ms"] = _warmed_timing_summary(row.get("step_timings_ms") or [])
    write_report_json(run_dir / "result.json", row)
    return row


def _error_row(*, renderer: str, strategy_name: str, run_dir: Path, exc: BaseException) -> dict[str, Any]:
    row = {
        "renderer": renderer,
        "strategy": strategy_name,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    write_report_json(run_dir / "result.json", row)
    return row


def _write_summary_markdown(rows: list[dict[str, Any]], output_root: Path) -> None:
    frame_values = sorted({int(row["frames"]) for row in rows if row.get("frames") is not None})
    size_values = sorted(
        {
            int(row.get("render_size") or row.get("size"))
            for row in rows
            if row.get("render_size") is not None or row.get("size") is not None
        }
    )
    shape_note = (
        f"{frame_values[0]}-frame, {size_values[0]}px"
        if len(frame_values) == 1 and len(size_values) == 1
        else f"frames={frame_values}, resolutions={size_values}"
    )
    header = (
        "| renderer | strategy | status | steps | frames | res | splats/tubes | "
        "train start | train end | eval loss | eval PSNR | alpha mean | mean step ms | forward ms | backward ms | artifact |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = [header]
    for row in rows:
        timing = row.get("warm_mean_timing_ms") or row.get("mean_timing_ms") or {}
        render_size = row.get("render_size") or row.get("size")
        if row.get("renderer") == "star_uvt":
            step_ms = timing.get("step_ms")
            forward_ms = timing.get("render_forward_ms")
            backward_ms = timing.get("backward_ms")
            train_start = row.get("start_loss")
            train_end = row.get("end_loss")
            count = row.get("tube_count") or row.get("tubes")
        else:
            step_ms = timing.get("wall_ms")
            forward_ms = timing.get("forward_decode_ms")
            backward_ms = timing.get("backward_ms")
            train_start = row.get("start_recon_loss")
            train_end = row.get("end_recon_loss")
            count = row.get("effective_gaussians")
        artifact = row.get("contact_sheet") or str(output_root / row["renderer"] / row["strategy"] / "result.json")
        lines.append(
            "| "
            + " | ".join(
                [
                    _metric(row.get("renderer")),
                    _metric(row.get("strategy")),
                    _metric(row.get("status")),
                    _metric(row.get("steps")),
                    _metric(row.get("frames")),
                    _metric(render_size),
                    _metric(count),
                    _metric(train_start),
                    _metric(train_end),
                    _metric(row.get("eval_loss")),
                    _metric(row.get("eval_psnr")),
                    _metric(row.get("alpha_mean")),
                    _metric(step_ms),
                    _metric(forward_ms),
                    _metric(backward_ms),
                    artifact,
                ]
            )
            + " |\n"
        )
    lines.append(
        "\nNotes:\n"
        "- `random_rgb_after_colorizer` matches the current post-colorizer random RGB training idea, with fixed black eval for this matched ablation.\n"
        "- `random_feature_before_colorizer` injects random feature background before the colorizer during train and fixed-zero feature background at eval, so empty/partial-alpha pixels still exercise the colorizer and alpha path.\n"
        f"- This is a matched {shape_note} ablation. Treat the ordering as shape-specific until repeated at the scale you intend to train.\n"
    )
    write_report_text(output_root / "summary.md", "".join(lines))


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    timing = row.get("warm_mean_timing_ms") or row.get("mean_timing_ms") or {}
    return {
        "renderer": row.get("renderer"),
        "strategy": row.get("strategy"),
        "status": row.get("status"),
        "steps": row.get("steps"),
        "frames": row.get("frames"),
        "render_size": row.get("render_size") or row.get("size"),
        "count": row.get("effective_gaussians") or row.get("tubes"),
        "train_start": row.get("start_recon_loss") if row.get("renderer") != "star_uvt" else row.get("start_loss"),
        "train_end": row.get("end_recon_loss") if row.get("renderer") != "star_uvt" else row.get("end_loss"),
        "eval_loss": row.get("eval_loss"),
        "eval_psnr": row.get("eval_psnr"),
        "alpha_mean": row.get("alpha_mean"),
        "warm_step_ms": timing.get("wall_ms") or timing.get("step_ms"),
        "forward_ms": timing.get("forward_decode_ms") or timing.get("render_forward_ms"),
        "backward_ms": timing.get("backward_ms"),
        "artifact": row.get("contact_sheet") or row.get("config"),
    }


def run_ablation(
    *,
    steps: int,
    output_root: Path,
    render_size: int | None = None,
    frames: int | None = None,
    dynamic_video_path: Path | None = None,
    star_video_path: Path | None = None,
    dynamic_gaussians: int | None = None,
    star_tubes: int | None = None,
    star_frame_chunk_size: int | None = None,
    star_tile_capacity: int | None = None,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    dynamic_base = load_config_file(DYNAMIC_BASE_CONFIG)
    star_base = load_config_file(STAR_BASE_CONFIG)
    rows: list[dict[str, Any]] = []
    for renderer in ("dynamic_gsplat", "star_uvt"):
        for strategy_name in STRATEGIES:
            run_dir = output_root / renderer / strategy_name
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                if renderer == "dynamic_gsplat":
                    cfg = _patch_dynamic_config(
                        dynamic_base,
                        strategy_name=strategy_name,
                        steps=steps,
                        run_dir=run_dir,
                        render_size=render_size,
                        frames=frames,
                        dynamic_video_path=dynamic_video_path,
                        dynamic_gaussians=dynamic_gaussians,
                    )
                    row = _run_dynamic_variant(cfg=cfg, strategy_name=strategy_name, run_dir=run_dir)
                else:
                    cfg = _patch_star_config(
                        star_base,
                        strategy_name=strategy_name,
                        steps=steps,
                        run_dir=run_dir,
                        render_size=render_size,
                        frames=frames,
                        star_video_path=star_video_path,
                        star_tubes=star_tubes,
                        star_frame_chunk_size=star_frame_chunk_size,
                        star_tile_capacity=star_tile_capacity,
                    )
                    row = _run_star_variant(cfg=cfg, strategy_name=strategy_name, run_dir=run_dir)
            except Exception as exc:  # Keep partial ablation evidence instead of losing the whole run.
                row = _error_row(renderer=renderer, strategy_name=strategy_name, run_dir=run_dir, exc=exc)
            rows.append(row)
            write_report_json(output_root / "summary.json", {"rows": rows})
            _write_summary_markdown(rows, output_root)
    write_report_json(output_root / "summary.json", {"rows": rows})
    _write_summary_markdown(rows, output_root)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--render-size", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--dynamic-video-path", type=Path, default=None)
    parser.add_argument("--star-video-path", type=Path, default=None)
    parser.add_argument("--dynamic-gaussians", type=int, default=None)
    parser.add_argument("--star-tubes", type=int, default=None)
    parser.add_argument("--star-frame-chunk-size", type=int, default=None)
    parser.add_argument("--star-tile-capacity", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_ablation(
        steps=int(args.steps),
        output_root=args.output_root,
        render_size=args.render_size,
        frames=args.frames,
        dynamic_video_path=args.dynamic_video_path,
        star_video_path=args.star_video_path,
        dynamic_gaussians=args.dynamic_gaussians,
        star_tubes=args.star_tubes,
        star_frame_chunk_size=args.star_frame_chunk_size,
        star_tile_capacity=args.star_tile_capacity,
    )
    print(
        json.dumps(
            {
                "summary": str(args.output_root / "summary.md"),
                "summary_json": str(args.output_root / "summary.json"),
                "rows": [_compact_row(row) for row in rows],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
