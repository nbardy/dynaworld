from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "src" / "train"


def ensure_repo_paths(*, chdir: bool = True) -> None:
    if chdir:
        os.chdir(ROOT)
    for path in (TRAIN_ROOT, ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_repo_paths()

from train_cli import parse_csv_ints, parse_csv_strings  # noqa: E402
from train_devices import sync_torch_device  # noqa: E402


def parse_positive_int_csv(value: str) -> list[int]:
    items = parse_csv_ints(value)
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if min(items) < 1:
        raise argparse.ArgumentTypeError("all values must be >= 1")
    return items


def parse_nonempty_csv(value: str) -> list[str]:
    items = parse_csv_strings(value)
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    return items


def effective_splat_count(cfg: dict[str, Any]) -> int:
    return int(cfg["model"]["tokens"]) * int(cfg["model"]["gaussians_per_token"])


def set_total_splat_count(cfg: dict[str, Any], splat_count: int | None) -> None:
    if splat_count is None:
        return
    tokens = int(cfg["model"]["tokens"])
    if int(splat_count) % tokens != 0:
        raise ValueError(f"splat_count={splat_count} must be divisible by model.tokens={tokens}.")
    cfg["model"]["gaussians_per_token"] = int(splat_count) // tokens


def quiet_training_logging(cfg: dict[str, Any], *, log_every: int | None = 1_000_000) -> None:
    logging_cfg = cfg.setdefault("logging", {})
    if log_every is not None:
        logging_cfg["log_every"] = int(log_every)
    logging_cfg["image_log_every"] = 1_000_000
    logging_cfg["video_log_every"] = 1_000_000
    logging_cfg["always_log_last_step"] = False


def apply_video_benchmark_shape(
    cfg: dict[str, Any],
    *,
    render_size: int | None = None,
    clip_length: int | None = None,
    steps: int | None = None,
) -> None:
    if render_size is not None:
        cfg["model"]["size"] = int(render_size)
        cfg["render"]["render_size"] = int(render_size)
    if clip_length is not None:
        cfg["model"]["train_frame_count"] = int(clip_length)
    if steps is not None:
        cfg["train"]["steps"] = int(steps)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timed(name: str, device: torch.device, fn: Callable[[], Any]) -> tuple[str, float, Any]:
    sync_torch_device(device)
    start = time.perf_counter()
    value = fn()
    sync_torch_device(device)
    return name, time.perf_counter() - start, value


def timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def format_timing_stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.6f}s"
    stats = timing_stats(values)
    return (
        f"mean={stats['mean']:.6f}s "
        f"median={stats['median']:.6f}s "
        f"min={stats['min']:.6f}s max={stats['max']:.6f}s"
    )
