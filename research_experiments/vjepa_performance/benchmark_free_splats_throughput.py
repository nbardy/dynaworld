from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch
import wandb


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src" / "train"))

from config_utils import load_config_file  # noqa: E402
from pipeline.losses import build_bank_rate_loss, build_camera_loss  # noqa: E402
from pipeline.render import gaussian_sequence_slice  # noqa: E402
from train_video_token_implicit_dynamic import trainer_class_for_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
)


def sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def timed(name: str, device: torch.device, fn: Callable[[], Any]) -> tuple[str, float, Any]:
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return name, time.perf_counter() - start, value


def parse_int_csv(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if min(items) < 1:
        raise argparse.ArgumentTypeError("all values must be >= 1")
    return items


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def prepare_config(
    base_cfg: dict[str, Any],
    *,
    render_size: int,
    clip_length: int,
    splat_count: int | None,
    source_frame_count: int | None,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["model"]["size"] = int(render_size)
    cfg["model"]["train_frame_count"] = int(clip_length)
    if splat_count is not None:
        tokens = int(cfg["model"]["tokens"])
        if int(splat_count) % tokens != 0:
            raise ValueError(f"splat_count={splat_count} must be divisible by model.tokens={tokens}.")
        cfg["model"]["gaussians_per_token"] = int(splat_count) // tokens
    cfg["render"]["render_size"] = int(render_size)
    cfg["train"]["steps"] = 1
    cfg["logging"]["log_every"] = 1_000_000
    cfg["logging"]["image_log_every"] = 1_000_000
    cfg["logging"]["video_log_every"] = 1_000_000
    cfg["logging"]["always_log_last_step"] = False
    cfg["logging"]["wandb_run_name"] = (
        f"splat-throughput-{cfg['model']['variant']}-r{render_size}-f{clip_length}-g"
        f"{cfg['model']['tokens'] * cfg['model']['gaussians_per_token']}"
        + ("-single-source-frame" if source_frame_count == 1 else "")
    )
    if source_frame_count is not None:
        cfg["data"]["max_frames"] = int(source_frame_count)
        cfg["model"]["free_frame_count"] = int(source_frame_count)
    return cfg


def recon_backward_timed(trainer, clip_frames, decoded, regularizer_loss) -> tuple[torch.Tensor, dict[str, float]]:
    timings: dict[str, float] = {}
    recon_loss = clip_frames.new_tensor(0.0)
    if decoded.cameras is None:
        raise ValueError("Implicit-camera video decode must include cameras.")

    frame_count = len(decoded.cameras)
    chunk_size = trainer.temporal_recon_chunk_size(frame_count)
    name, elapsed, train_background = timed(
        "background_sample",
        trainer.device,
        lambda: trainer.rgb_objective.sample_background(
            phase="train",
            like=clip_frames,
            frame_count=frame_count,
        ),
    )
    timings[name] = elapsed

    render_total = 0.0
    loss_total = 0.0
    backward_total = 0.0
    for chunk_start in range(0, frame_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, frame_count)
        chunk_sequence = gaussian_sequence_slice(decoded, chunk_start, chunk_end)
        chunk_indices = torch.arange(chunk_start, chunk_end, device=clip_frames.device)
        chunk_times = chunk_indices.to(dtype=torch.float32) / float(max(frame_count - 1, 1))
        target = trainer.make_target_view(
            view_id="bench_train_clip",
            frames=clip_frames[0, chunk_start:chunk_end],
            frame_indices=chunk_indices,
            frame_times=chunk_times,
            cameras=tuple(decoded.cameras[chunk_start:chunk_end]),
            role="train",
        )

        name, elapsed, rendered_chunk = timed(
            "render_and_compose",
            trainer.device,
            lambda: trainer.rgb_objective.render_view(
                chunk_sequence,
                target,
                phase="train",
                background=train_background,
            ),
        )
        render_total += elapsed

        name, elapsed, chunk_losses = timed(
            "recon_loss_compute",
            trainer.device,
            lambda: trainer.rgb_objective.reconstruction_loss_per_image(rendered_chunk),
        )
        loss_total += elapsed
        chunk_recon_loss = chunk_losses.sum() / frame_count
        recon_loss = recon_loss + chunk_recon_loss.detach()

        is_last_chunk = chunk_end == frame_count
        backward_loss = chunk_recon_loss + (regularizer_loss if is_last_chunk else 0.0)
        name, elapsed, _ = timed(
            "backward",
            trainer.device,
            lambda backward_loss=backward_loss, is_last_chunk=is_last_chunk: backward_loss.backward(
                retain_graph=not is_last_chunk
            ),
        )
        backward_total += elapsed

    timings["render_and_compose"] = render_total
    timings["recon_loss"] = loss_total
    timings["backward"] = backward_total
    return recon_loss, timings


def benchmark_step(trainer) -> dict[str, float]:
    timings: dict[str, float] = {}
    trainer.optimizer.zero_grad(set_to_none=True)

    name, elapsed, sampled = timed("sample_clip", trainer.device, trainer.sample_clip)
    timings[name] = elapsed
    sequence_data, clip_frames, clip_times = sampled

    name, elapsed, model_input = timed(
        "model_input",
        trainer.device,
        lambda: trainer.model_input_for_clip(sequence_data, clip_frames, clip_times),
    )
    timings[name] = elapsed

    name, elapsed, decoded = timed(
        "forward_decode",
        trainer.device,
        lambda: trainer.forward_clip(model_input, clip_times),
    )
    timings[name] = elapsed

    if decoded.camera_state is None:
        raise ValueError("Implicit-camera video decode must include camera_state.")
    name, elapsed, regularizers = timed(
        "regularizers",
        trainer.device,
        lambda: (
            build_camera_loss(clip_times, decoded.camera_state, trainer.loss_cfg),
            build_bank_rate_loss(decoded, trainer.loss_cfg),
        ),
    )
    timings[name] = elapsed
    (camera_loss, _camera_motion, _camera_temporal, _camera_global), (bank_rate_loss, _bank_terms) = regularizers

    recon_loss, recon_timings = recon_backward_timed(
        trainer,
        clip_frames,
        decoded,
        camera_loss + bank_rate_loss,
    )
    timings.update(recon_timings)

    name, elapsed, _ = timed("optimizer_step", trainer.device, trainer.optimizer.step)
    timings[name] = elapsed
    timings["loss"] = float((recon_loss + camera_loss.detach() + bank_rate_loss.detach()).detach().cpu())
    timings["recon_loss_value"] = float(recon_loss.detach().cpu())
    return timings


def run_case(
    base_cfg: dict[str, Any],
    *,
    render_size: int,
    clip_length: int,
    splat_count: int | None,
    source_frame_count: int | None,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    cfg = prepare_config(
        base_cfg,
        render_size=render_size,
        clip_length=clip_length,
        splat_count=splat_count,
        source_frame_count=source_frame_count,
    )
    trainer = trainer_class_for_config(cfg)(cfg)
    rows: list[dict[str, float]] = []
    try:
        for _ in range(warmup):
            benchmark_step(trainer)
        for _ in range(steps):
            rows.append(benchmark_step(trainer))
    finally:
        wandb.finish()

    timing_keys = [key for key in rows[0] if key not in {"loss", "recon_loss_value"}] if rows else []
    step_totals = [sum(row[key] for key in timing_keys) for row in rows]
    elapsed_total = float(sum(step_totals))
    frames = int(clip_length) * int(steps)
    result = {
        "render_size": int(render_size),
        "clip_length": int(clip_length),
        "source_frame_count": "all" if source_frame_count is None else int(source_frame_count),
        "model_variant": str(trainer.model_cfg["variant"]),
        "device": str(trainer.device),
        "renderer": str(trainer.renderer_mode),
        "gaussians": int(trainer.effective_gaussians),
        "warmup": int(warmup),
        "steps": int(steps),
        "frames": frames,
        "elapsed_total_sec": elapsed_total,
        "steps_per_sec": float(steps / elapsed_total) if elapsed_total > 0 else 0.0,
        "frames_per_sec": float(frames / elapsed_total) if elapsed_total > 0 else 0.0,
        "ms_per_frame": float(1000.0 * elapsed_total / frames) if frames > 0 else 0.0,
        "last_loss": rows[-1]["loss"] if rows else None,
        "last_recon_loss": rows[-1]["recon_loss_value"] if rows else None,
        "timings": {key: summarize([row[key] for row in rows]) for key in timing_keys},
    }
    return result


def print_result(result: dict[str, Any]) -> None:
    label = (
        f"variant={result['model_variant']} size={result['render_size']} "
        f"clip={result['clip_length']} splats={result['gaussians']} "
        f"source_frames={result['source_frame_count']}"
    )
    print(
        f"CASE {label}: "
        f"steps/s={result['steps_per_sec']:.3f} "
        f"frames/s={result['frames_per_sec']:.2f} "
        f"ms/frame={result['ms_per_frame']:.2f} "
        f"elapsed={result['elapsed_total_sec']:.3f}s "
        f"loss={result['last_loss']:.6f}"
    )
    for key, stats in result["timings"].items():
        print(
            f"  {key}: mean={stats['mean']:.6f}s median={stats['median']:.6f}s "
            f"min={stats['min']:.6f}s max={stats['max']:.6f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct free-splats frame throughput.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--render-sizes", type=parse_int_csv, default=parse_int_csv("64,128,256"))
    parser.add_argument("--clip-lengths", type=parse_int_csv, default=parse_int_csv("1,4,16"))
    parser.add_argument(
        "--splat-counts",
        type=parse_int_csv,
        default=None,
        help="Optional total splat counts. Each must divide model.tokens.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--include-single-source-frame", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if args.steps < 1 or args.warmup < 0:
        raise SystemExit("--steps must be >= 1 and --warmup must be >= 0")

    os.environ.setdefault("WANDB_MODE", "disabled")
    base_cfg = load_config_file(args.config)

    results = []
    source_frame_cases: list[int | None] = [None]
    if args.include_single_source_frame:
        source_frame_cases.append(1)
    splat_counts = args.splat_counts or [
        int(base_cfg["model"]["tokens"]) * int(base_cfg["model"]["gaussians_per_token"])
    ]
    for source_frame_count in source_frame_cases:
        for splat_count in splat_counts:
            for render_size in args.render_sizes:
                for clip_length in args.clip_lengths:
                    if source_frame_count is not None and clip_length > source_frame_count:
                        continue
                    result = run_case(
                        base_cfg,
                        render_size=render_size,
                        clip_length=clip_length,
                        splat_count=splat_count,
                        source_frame_count=source_frame_count,
                        warmup=args.warmup,
                        steps=args.steps,
                    )
                    print_result(result)
                    results.append(result)

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
