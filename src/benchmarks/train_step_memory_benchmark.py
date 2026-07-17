from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import benchmark_bootstrap

from config_utils import load_config_file
from benchmark_compare import seed_everything
from benchmark_memory import clear_device_cache, run_with_memory_sampling
from train_artifacts import write_json
from train_devices import sync_torch_device as sync_device
from train_logging import finish_wandb_run, set_default_wandb_mode
from trainer_registry import instantiate_trainer_for_config

set_default_wandb_mode("disabled", silent=True)


def _float_tensor(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _step_payload(result, elapsed_ms: float) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "elapsed_ms": float(elapsed_ms),
        "loss": float(_float_tensor(result.loss) or 0.0),
        "recon_loss": float(_float_tensor(result.recon_loss) or 0.0),
        "bank_rate_loss": float(_float_tensor(result.bank_rate_loss) or 0.0),
        "sequence_frame_count": int(result.sequence_frame_count),
    }
    for key, value in result.bank_rate_terms.items():
        scalar = _float_tensor(value)
        if scalar is not None:
            payload[f"bank_rate_terms.{key}"] = float(scalar)
    return payload


def _time_step(trainer, *, keep_preview: bool) -> dict[str, float | int]:
    sync_device(trainer.device)
    start = time.perf_counter()
    result = trainer.step(keep_preview=keep_preview)
    sync_device(trainer.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return _step_payload(result, elapsed_ms)


def _summarize(samples: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    keys = sorted(key for key in samples[0] if isinstance(samples[0][key], float | int))
    for key in keys:
        values = [float(sample[key]) for sample in samples if key in sample]
        if not values:
            continue
        out[key] = {
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark whole trainer.step() wall time and sampled memory.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--keep-preview", action="store_true")
    parser.add_argument("--memory-sample-interval-ms", type=float, default=1.0)
    parser.add_argument("--memory-clear-cache", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(int(args.seed))
    trainer = instantiate_trainer_for_config(load_config_file(args.config), args.config)
    for _ in range(max(0, int(args.warmup))):
        _time_step(trainer, keep_preview=bool(args.keep_preview))
    samples = []
    for _ in range(max(1, int(args.iters))):
        samples.append(
            run_with_memory_sampling(
                trainer.device,
                interval_ms=float(args.memory_sample_interval_ms),
                clear_cache=bool(args.memory_clear_cache),
                fn=lambda: _time_step(trainer, keep_preview=bool(args.keep_preview)),
            )
        )
    summary = _summarize(samples)
    payload: dict[str, Any] = {
        "config": str(args.config),
        "trainer": type(trainer).__name__,
        "renderer_mode": trainer.renderer_mode,
        "seed": int(args.seed),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "keep_preview": bool(args.keep_preview),
        "memory_sample_interval_ms": float(args.memory_sample_interval_ms),
        "memory_clear_cache": bool(args.memory_clear_cache),
        "camera_swap_mode": str(trainer.train_cfg.get("camera_swap_mode", "none")),
        "camera_swap_pairs_per_step": int(trainer.train_cfg.get("camera_swap_pairs_per_step", 0)),
        "train_views_per_step": int(trainer.train_cfg.get("train_views_per_step", 0)),
        "samples": samples,
        "summary": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output is not None:
        write_json(args.json_output, payload)
        print(f"wrote {args.json_output}")
    clear_device_cache(trainer.device)

    finish_wandb_run()


if __name__ == "__main__":
    main()
