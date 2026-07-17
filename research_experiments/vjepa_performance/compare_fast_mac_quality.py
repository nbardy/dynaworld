from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from vjepa_benchmark_common import (
    ROOT,
    apply_video_benchmark_shape,
    effective_splat_count,
    parse_nonempty_csv,
    quiet_training_logging,
    seed_everything,
    set_total_splat_count,
    sync_torch_device as sync,
)

from config_utils import load_config_file
from pipeline.diagnostics import eval_metric_payload, temporal_similarity_payload
from rendering import resize_images
from train_artifacts import append_jsonl, write_jsonl
from train_logging import finish_wandb_run, set_default_wandb_mode
from trainer_registry import instantiate_trainer_for_config


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
)


def prepare_config(
    base_cfg: dict[str, Any],
    *,
    variant: str,
    variant_mode: str,
    steps: int,
    render_size: int | None,
    clip_length: int | None,
    splat_count: int | None,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    apply_video_benchmark_shape(cfg, render_size=render_size, clip_length=clip_length, steps=steps)
    set_total_splat_count(cfg, splat_count)
    cfg["render"].setdefault("fast_mac", {})
    if variant_mode == "rgb":
        cfg["render"]["fast_mac"]["rgb_variant"] = variant
    elif variant_mode == "feature":
        cfg["render"]["fast_mac"]["feature_variant"] = variant
    else:
        raise ValueError(f"unknown variant_mode={variant_mode!r}")
    quiet_training_logging(cfg)
    cfg["logging"]["wandb_run_name"] = (
        f"quality-parity-{cfg['model']['variant']}-{variant}-"
        f"r{cfg['render']['render_size']}-f{cfg['model']['train_frame_count']}-"
        f"g{effective_splat_count(cfg)}"
    )
    return cfg


def eval_metrics(trainer) -> dict[str, float]:
    sequences = trainer.eval_sequences or [trainer.sequence_data]
    metric_payloads: list[dict[str, float]] = []
    with torch.no_grad():
        for sequence_data in sequences:
            rendered = trainer.render_full_sequence(sequence_data)
            gt_sequence = resize_images(sequence_data.frames, trainer.render_size).detach().cpu()
            metric_payloads.append(
                {
                    **eval_metric_payload(rendered.rgb_sequence, gt_sequence, trainer.loss_cfg),
                    **temporal_similarity_payload(rendered.rgb_sequence, gt_sequence, trainer.loss_cfg),
                    **rendered.temporal_metrics,
                }
            )
    keys = sorted({key for metrics in metric_payloads for key in metrics})
    return {key: float(sum(metrics[key] for metrics in metric_payloads if key in metrics) / len(metric_payloads)) for key in keys}


def run_variant(
    base_cfg: dict[str, Any],
    *,
    config_path: Path,
    variant: str,
    variant_mode: str,
    steps: int,
    render_size: int | None,
    clip_length: int | None,
    splat_count: int | None,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    cfg = prepare_config(
        base_cfg,
        variant=variant,
        variant_mode=variant_mode,
        steps=steps,
        render_size=render_size,
        clip_length=clip_length,
        splat_count=splat_count,
    )
    trainer = instantiate_trainer_for_config(cfg, config_path)
    train_losses: list[float] = []
    start = time.perf_counter()
    try:
        for _ in range(steps):
            result = trainer.step(keep_preview=False)
            sync(trainer.device)
            train_losses.append(float(result.loss.detach().cpu()))
        metrics = eval_metrics(trainer)
        sync(trainer.device)
    finally:
        finish_wandb_run()
    elapsed = time.perf_counter() - start
    return {
        "variant": variant,
        "variant_mode": variant_mode,
        "seed": int(seed),
        "model_variant": str(cfg["model"]["variant"]),
        "render_size": int(cfg["render"]["render_size"]),
        "clip_length": int(cfg["model"]["train_frame_count"]),
        "gaussians": effective_splat_count(cfg),
        "steps": int(steps),
        "elapsed_sec": float(elapsed),
        "steps_per_sec": float(steps / elapsed) if elapsed > 0 else 0.0,
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-seed v5/v6_refined trainer quality parity checks.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", type=parse_nonempty_csv, default=parse_nonempty_csv("v5,v6_refined"))
    parser.add_argument("--variant-mode", choices=("rgb", "feature"), default="rgb")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=None)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--splat-count", type=int, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    set_default_wandb_mode("disabled", silent=None)
    base_cfg = load_config_file(args.config)
    write_jsonl(args.output_jsonl, ())
    for variant in args.variants:
        row = run_variant(
            base_cfg,
            config_path=args.config,
            variant=variant,
            variant_mode=args.variant_mode,
            steps=args.steps,
            render_size=args.render_size,
            clip_length=args.clip_length,
            splat_count=args.splat_count,
            seed=args.seed,
        )
        append_jsonl(args.output_jsonl, row)
        print(
            f"{variant:10s} steps/s={row['steps_per_sec']:.3f} "
            f"loss={row['last_train_loss']:.6f} "
            f"eval_loss={row['metrics']['Eval/Loss']:.6f} "
            f"psnr={row['metrics']['Eval/PSNR']:.3f} "
            f"ssim={row['metrics']['Eval/SSIM']:.3f}"
        )
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
