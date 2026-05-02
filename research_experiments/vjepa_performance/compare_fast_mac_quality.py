from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import wandb


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src" / "train"))

from config_utils import load_config_file  # noqa: E402
from pipeline.diagnostics import eval_metric_payload, temporal_similarity_payload  # noqa: E402
from rendering import resize_images  # noqa: E402
from train_video_token_implicit_dynamic import trainer_class_for_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
)


def parse_csv_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


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
    if render_size is not None:
        cfg["model"]["size"] = int(render_size)
        cfg["render"]["render_size"] = int(render_size)
    if clip_length is not None:
        cfg["model"]["train_frame_count"] = int(clip_length)
    if splat_count is not None:
        tokens = int(cfg["model"]["tokens"])
        if int(splat_count) % tokens != 0:
            raise ValueError(f"splat_count={splat_count} must be divisible by model.tokens={tokens}.")
        cfg["model"]["gaussians_per_token"] = int(splat_count) // tokens
    cfg["render"].setdefault("fast_mac", {})
    if variant_mode == "rgb":
        cfg["render"]["fast_mac"]["rgb_variant"] = variant
    elif variant_mode == "feature":
        cfg["render"]["fast_mac"]["feature_variant"] = variant
    else:
        raise ValueError(f"unknown variant_mode={variant_mode!r}")
    cfg["train"]["steps"] = int(steps)
    cfg["logging"]["log_every"] = 1_000_000
    cfg["logging"]["image_log_every"] = 1_000_000
    cfg["logging"]["video_log_every"] = 1_000_000
    cfg["logging"]["always_log_last_step"] = False
    cfg["logging"]["wandb_run_name"] = (
        f"quality-parity-{cfg['model']['variant']}-{variant}-"
        f"r{cfg['render']['render_size']}-f{cfg['model']['train_frame_count']}-"
        f"g{int(cfg['model']['tokens']) * int(cfg['model']['gaussians_per_token'])}"
    )
    return cfg


def sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    variant: str,
    variant_mode: str,
    steps: int,
    render_size: int | None,
    clip_length: int | None,
    splat_count: int | None,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    cfg = prepare_config(
        base_cfg,
        variant=variant,
        variant_mode=variant_mode,
        steps=steps,
        render_size=render_size,
        clip_length=clip_length,
        splat_count=splat_count,
    )
    trainer = trainer_class_for_config(cfg)(cfg)
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
        wandb.finish()
    elapsed = time.perf_counter() - start
    return {
        "variant": variant,
        "variant_mode": variant_mode,
        "seed": int(seed),
        "model_variant": str(cfg["model"]["variant"]),
        "render_size": int(cfg["render"]["render_size"]),
        "clip_length": int(cfg["model"]["train_frame_count"]),
        "gaussians": int(cfg["model"]["tokens"]) * int(cfg["model"]["gaussians_per_token"]),
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
    parser.add_argument("--variants", type=parse_csv_strings, default=parse_csv_strings("v5,v6_refined"))
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

    os.environ.setdefault("WANDB_MODE", "disabled")
    base_cfg = load_config_file(args.config)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for variant in args.variants:
            row = run_variant(
                base_cfg,
                variant=variant,
                variant_mode=args.variant_mode,
                steps=args.steps,
                render_size=args.render_size,
                clip_length=args.clip_length,
                splat_count=args.splat_count,
                seed=args.seed,
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
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
