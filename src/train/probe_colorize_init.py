"""Probe the colorize-MLP output distribution at init.

Builds a model + colorize from a train config, runs ONE forward + rasterize +
colorize end-to-end (no training, no W&B, no GT load), and prints metrics on
the post-MLP RGB image. Tight-iteration tool for diagnosing gray-collapse and
testing alternative colorize-MLP initializations.

Run:
    PYTHONPATH=src/train uv run python src/train/probe_colorize_init.py \
        src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc \
        --seeds 0,1,2

Per-seed metrics are printed; the gray-collapse signal is when
`ColorizeOut/PerPixelChroma/Mean` is near zero AND `ColorizeOut/SpatialStdMean`
is near zero. The pre-sigmoid logit fractions tell you whether the sigmoid is
engaged or stuck at 0.5.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from config_utils import load_config_file
from init_diagnostics import format_colorize_init_summary, post_colorize_image_diagnostics
from model_factories import build_colorizer, build_model_from_config
from pipeline.render import colorize_view_dirs_for_features, render_clip_sequence
from train_cli import parse_csv_ints
from train_devices import resolve_torch_device
from trainer_registry import resolve_config_for_arch


def _build_model_and_colorize(config_path: Path, seed: int) -> tuple[torch.nn.Module, torch.nn.Module | None, dict[str, Any]]:
    """Build the trainer's model + optional colorize for a given config and seed."""
    torch.manual_seed(seed)
    config = load_config_file(config_path)

    resolved = resolve_config_for_arch(config, config_path)
    model = build_model_from_config(resolved).eval()

    colorizer = build_colorizer(resolved.get("colorize"), feature_dim=int(resolved["model"]["feature_dim"]))
    colorize_module = None if colorizer.module is None else colorizer.module.eval()
    return model, colorize_module, resolved


@torch.no_grad()
def _run_one_forward(
    model: torch.nn.Module,
    colorize_module: torch.nn.Module | None,
    resolved: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one forward + rasterize + colorize. Returns (rgb_image, pre_sigmoid_logits or None)."""
    model = model.to(device)
    if colorize_module is not None:
        colorize_module = colorize_module.to(device)

    train_frame_count = int(resolved["model"]["train_frame_count"])
    decode_times = torch.linspace(0.0, 1.0, train_frame_count, device=device).reshape(1, -1)
    decoded = model(video=None, decode_times=decode_times)
    if decoded.cameras is None:
        raise RuntimeError("Decoded output missing cameras; this probe expects implicit-camera models.")

    rasterized = render_clip_sequence(
        resolved,
        decoded,
        decoded.cameras,
        renderer_mode=str(resolved["render"]["renderer"]).lower(),
        dense_grid=None,
    )
    rendered_features = rasterized.features

    if colorize_module is None:
        # F=3 path: rendered_features IS the RGB image, already in [0, 1].
        return rendered_features, None

    view_dirs = colorize_view_dirs_for_features(resolved, colorize_module, rendered_features, decoded.cameras)
    rgb, pre_sigmoid_logits = colorize_module.forward_with_logits(rendered_features, view_dirs=view_dirs)
    return rgb, pre_sigmoid_logits


def probe_colorize_one_seed(config_path: Path, seed: int, device: torch.device) -> dict[str, Any]:
    model, colorize_module, resolved = _build_model_and_colorize(config_path, seed)
    rgb, logits = _run_one_forward(model, colorize_module, resolved, device)
    metrics = post_colorize_image_diagnostics(rgb, logits=logits)
    return {
        "seed": seed,
        "feature_dim": int(resolved["model"]["feature_dim"]),
        "render_size": int(resolved["render"]["render_size"]),
        "train_frame_count": int(resolved["model"]["train_frame_count"]),
        "colorize_present": colorize_module is not None,
        "metrics": metrics,
    }


def aggregate_across_seeds(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean / std across seeds for each metric."""
    if not results:
        return {}
    keys = sorted(results[0]["metrics"].keys())
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [r["metrics"][key] for r in results if key in r["metrics"]]
        finite = [v for v in values if v == v]  # filter NaN
        if not finite:
            continue
        aggregated[key] = {
            "mean": statistics.fmean(finite),
            "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
            "n": len(finite),
        }
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the colorize-MLP output distribution at init.")
    parser.add_argument("config", type=Path, help="Path to a JSONC train config.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seed list, e.g. '0,1,2'. Reports mean/std across seeds when more than one.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/mps/cuda.")
    parser.add_argument("--json", action="store_true", help="Print all metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    device = resolve_torch_device(args.device.lower(), auto_cuda=True)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Seeds: {seeds}")

    results = [probe_colorize_one_seed(args.config, seed=seed, device=device) for seed in seeds]

    if args.json:
        print(json.dumps({"results": results, "aggregated": aggregate_across_seeds(results)}, indent=2))
        return

    for r in results:
        print(
            f"\n[seed={r['seed']}] feature_dim={r['feature_dim']}, "
            f"colorize={r['colorize_present']}, "
            f"render_size={r['render_size']}, frames={r['train_frame_count']}"
        )
        print(format_colorize_init_summary(r["metrics"]))

    if len(results) > 1:
        print("\n--- Aggregated across seeds (mean ± std) ---")
        aggregated = aggregate_across_seeds(results)
        important = (
            "ColorizeOut/PerPixelChroma/Mean",
            "ColorizeOut/SpatialStdMean",
            "ColorizeOut/Range",
            "ColorizeOut/R/Std",
            "ColorizeOut/G/Std",
            "ColorizeOut/B/Std",
            "ColorizeOut/Logit/AbsGt0p5Frac",
            "ColorizeOut/Logit/AbsGt2Frac",
        )
        for key in important:
            if key in aggregated:
                a = aggregated[key]
                print(f"  {key:50s} {a['mean']:.4g} ± {a['std']:.4g}  (n={a['n']})")


if __name__ == "__main__":
    main()
