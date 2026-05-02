from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

import torch
import wandb


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "train"))

from config_utils import load_config_file  # noqa: E402
from pipeline.losses import build_bank_rate_loss  # noqa: E402
from train_multicam_precomputed_feature_implicit_dynamic import (  # noqa: E402
    MulticamPrecomputedFeatureImplicitTrainer,
)


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc"
)


def sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def timed(name: str, device: torch.device, fn: Callable):
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return name, time.perf_counter() - start, value


def tensor_mib(tensor: torch.Tensor) -> float:
    return float(tensor.numel() * tensor.element_size()) / (1024.0 * 1024.0)


def summarize(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.6f}s"
    return (
        f"mean={statistics.mean(values):.6f}s "
        f"median={statistics.median(values):.6f}s "
        f"min={min(values):.6f}s max={max(values):.6f}s"
    )


def prepare_config(args: argparse.Namespace) -> dict:
    cfg = deepcopy(load_config_file(args.config))
    cfg.setdefault("features", {})
    cfg["features"]["force_rebake"] = bool(args.force_rebake)
    if args.video_feature_token_stride is not None:
        cfg["model"]["video_feature_token_stride"] = int(args.video_feature_token_stride)
    if args.no_camera_refine_with_decode_time:
        cfg["model"]["camera_refine_with_decode_time"] = False
    cfg["logging"]["wandb_run_name"] = f"vjepa-perf-benchmark-{Path(args.config).stem}"
    cfg["logging"]["image_log_every"] = 1_000_000
    cfg["logging"]["video_log_every"] = 1_000_000
    cfg["logging"]["always_log_last_step"] = False
    cfg["train"]["steps"] = int(args.steps)
    return cfg


def print_feature_summary(trainer: MulticamPrecomputedFeatureImplicitTrainer) -> None:
    features = trainer.model_input_for_clip(
        trainer.sequence_data,
        trainer.sequence_data.frames.unsqueeze(0),
        trainer.sequence_data.frame_times.reshape(1, -1),
    )
    if not hasattr(trainer.model, "video_encoder"):
        print(f"feature_payload: disabled backend={trainer.model_cfg['video_encoder_backend']}")
        if torch.is_tensor(features):
            print(f"  model_input: shape={tuple(features.shape)} dtype={features.dtype} size={tensor_mib(features):.2f} MiB")
        return
    print("feature_payload:")
    for name, value in features.items():
        if torch.is_tensor(value):
            print(f"  {name}: shape={tuple(value.shape)} dtype={value.dtype} size={tensor_mib(value):.2f} MiB")
    name, elapsed, projected = timed(
        "adapter_project_features",
        trainer.device,
        lambda: _project_video_features(trainer, features),
    )
    print(f"{name}: {elapsed:.6f}s")
    print(f"projected_features: shape={tuple(projected.shape)} dtype={projected.dtype} size={tensor_mib(projected):.2f} MiB")


def _project_video_features(trainer: MulticamPrecomputedFeatureImplicitTrainer, features):
    with trainer.autocast_context():
        return trainer.model.video_encoder(features)


def benchmark_step(trainer: MulticamPrecomputedFeatureImplicitTrainer) -> dict[str, float]:
    timings: dict[str, float] = {}
    trainer.optimizer.zero_grad(set_to_none=True)

    name, elapsed, sampled = timed("sample_clip", trainer.device, trainer.sample_multicam_clip)
    timings[name] = elapsed
    sequence_data, clip_indices, clip_frames, clip_times, views = sampled

    name, elapsed, model_input = timed(
        "feature_load_or_memory_hit",
        trainer.device,
        lambda: trainer.model_input_for_clip(sequence_data, clip_frames, clip_times),
    )
    timings[name] = elapsed

    name, elapsed, decoded = timed(
        "model_forward_decode",
        trainer.device,
        lambda: trainer.forward_clip(model_input, clip_times),
    )
    timings[name] = elapsed

    name, elapsed, bank_payload = timed(
        "bank_and_rig_losses",
        trainer.device,
        lambda: (
            build_bank_rate_loss(decoded, trainer.loss_cfg),
            trainer.rig_regularization_loss(),
        ),
    )
    timings[name] = elapsed
    (bank_rate_loss, _bank_rate_terms), rig_loss = bank_payload

    recon_loss = trainer.multicam_bundle.train_frames.new_zeros(())
    background = trainer.rgb_objective.sample_background(
        phase="train",
        like=trainer.multicam_bundle.train_frames[int(views[0]), clip_indices],
        frame_count=len(clip_indices),
    )
    render_times = []
    loss_times = []
    for view in views:
        name, elapsed, rendered = timed(
            f"render_view_{int(view)}",
            trainer.device,
            lambda view=view: trainer.render_view_clip(
                decoded,
                view=int(view),
                clip_indices=clip_indices,
                phase="train",
                background=background,
            ),
        )
        timings[name] = elapsed
        render_times.append(elapsed)

        name, elapsed, view_loss = timed(
            f"recon_loss_view_{int(view)}",
            trainer.device,
            lambda rendered=rendered: trainer.rgb_objective.reconstruction_loss(rendered),
        )
        timings[name] = elapsed
        loss_times.append(elapsed)
        recon_loss = recon_loss + view_loss

    timings["render_views_total"] = sum(render_times)
    timings["recon_losses_total"] = sum(loss_times)
    recon_loss = recon_loss / float(max(len(views), 1))
    loss = recon_loss + bank_rate_loss + rig_loss

    name, elapsed, _ = timed("backward", trainer.device, lambda: loss.backward())
    timings[name] = elapsed
    name, elapsed, _ = timed("optimizer_step", trainer.device, trainer.optimizer.step)
    timings[name] = elapsed
    timings["loss"] = float(loss.detach().cpu())
    timings["recon_loss"] = float(recon_loss.detach().cpu())
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--force-rebake", action="store_true")
    parser.add_argument("--video-feature-token-stride", type=int, default=None)
    parser.add_argument("--no-camera-refine-with-decode-time", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    init_start = time.perf_counter()
    trainer = MulticamPrecomputedFeatureImplicitTrainer(prepare_config(args))
    sync(trainer.device)
    init_elapsed = time.perf_counter() - init_start
    print(f"trainer_init_total: {init_elapsed:.6f}s")
    print_feature_summary(trainer)

    rows = []
    try:
        for index in range(int(args.warmup)):
            row = benchmark_step(trainer)
            print(f"warmup_step_{index + 1}: loss={row['loss']:.6f}")
        for index in range(int(args.steps)):
            row = benchmark_step(trainer)
            rows.append(row)
            phase_text = " ".join(
                f"{key}={value:.6f}s"
                for key, value in row.items()
                if key not in {"loss", "recon_loss"}
            )
            print(f"bench_step_{index + 1}: loss={row['loss']:.6f} recon={row['recon_loss']:.6f} {phase_text}")
    finally:
        wandb.finish()

    keys = [key for key in rows[0] if key not in {"loss", "recon_loss"}] if rows else []
    print("summary:")
    for key in keys:
        print(f"  {key}: {summarize([row[key] for row in rows])}")
    if rows:
        total_step = [
            sum(value for key, value in row.items() if key not in {"loss", "recon_loss"})
            for row in rows
        ]
        print(f"  measured_step_total: {summarize(total_step)}")
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "config": str(args.config),
                "steps": int(args.steps),
                "warmup": int(args.warmup),
                "device": str(trainer.device),
                "model_variant": str(trainer.model_cfg["variant"]),
                "video_encoder_backend": str(trainer.model_cfg["video_encoder_backend"]),
                "train_frame_count": int(trainer.model_cfg["train_frame_count"]),
                "train_views_per_step": int(trainer.train_cfg["train_views_per_step"]),
                "timings": {key: [row[key] for row in rows] for key in keys},
                "measured_step_total": total_step,
                "summary": {key: summarize([row[key] for row in rows]) for key in keys},
                "measured_step_total_summary": summarize(total_step),
            }
            args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
