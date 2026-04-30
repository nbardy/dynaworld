"""Sweep colorize-MLP init strategies on the rendered F-channel feature buffer.

Renders the rasterized feature map ONCE per seed (the heavy step ~3 s on MPS),
then iterates over a hardcoded matrix of colorize cells, applying each one to
the cached features and computing post-MLP metrics. Each cell is just a
fresh FeatureToColor + forward pass through a 1x1 conv -- sub-millisecond.

Run:
    PYTHONPATH=src/train uv run python src/train/probe_colorize_matrix.py \
        src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc \
        --seeds 0,1,2

Reads a base config to determine feature_dim, render size, frames, etc., then
overrides only the colorize cell. The base config's `colorize` section (if any)
is ignored -- this script defines its own matrix.

Output: a table of mean values per cell (across seeds) for the key gray-collapse
metrics. Higher is better for PerPixelChroma/SpatialStd/Range; Logit/AbsGt2Frac
near 0 means sigmoid is stuck (gray collapse); too far past 0.5 saturates sigmoid.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from config_utils import load_config_file
from init_diagnostics import post_colorize_image_diagnostics


# (label, pre_norm, weight_init, gain, hidden_dim)
CELLS: list[tuple[str, bool, str, float, int | None]] = [
    # Controls
    ("default-kaiming",       False, "kaiming",    1.0,  None),
    ("LN+kaiming",            True,  "kaiming",    1.0,  None),
    ("LN+kaiming-g2",         True,  "kaiming",    2.0,  None),
    ("LN+kaiming-g4",         True,  "kaiming",    4.0,  None),
    ("LN+kaiming-g4-hid16",   True,  "kaiming",    4.0,  16),
    ("LN+kaiming-g6-hid16",   True,  "kaiming",    6.0,  16),
    # Orthogonal gain sweep, no LN, no hidden
    ("orth-g2",               False, "orthogonal", 2.0,  None),
    ("orth-g3",               False, "orthogonal", 3.0,  None),
    ("orth-g5",               False, "orthogonal", 5.0,  None),
    ("orth-g7",               False, "orthogonal", 7.0,  None),
    ("orth-g10",              False, "orthogonal", 10.0, None),
    # Orthogonal gain sweep, with LN, no hidden
    ("LN+orth-g2",            True,  "orthogonal", 2.0,  None),
    ("LN+orth-g3",            True,  "orthogonal", 3.0,  None),
    ("LN+orth-g5",            True,  "orthogonal", 5.0,  None),
    ("LN+orth-g7",            True,  "orthogonal", 7.0,  None),
    ("LN+orth-g10",           True,  "orthogonal", 10.0, None),
    # The "everything" combos: LN + orthogonal + hidden layer
    ("LN+orth-g2-hid8",       True,  "orthogonal", 2.0,  8),
    ("LN+orth-g3-hid8",       True,  "orthogonal", 3.0,  8),
    ("LN+orth-g3-hid16",      True,  "orthogonal", 3.0,  16),
    ("LN+orth-g3-hid32",      True,  "orthogonal", 3.0,  32),
    ("LN+orth-g5-hid16",      True,  "orthogonal", 5.0,  16),
    ("LN+orth-g7-hid16",      True,  "orthogonal", 7.0,  16),
    # Hidden + orth without LN (capacity test, no normalization)
    ("orth-g5-hid16",         False, "orthogonal", 5.0,  16),
    ("orth-g7-hid32",         False, "orthogonal", 7.0,  32),
]


@torch.no_grad()
def render_features_once(config_path: Path, seed: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Build the model under `seed`, run forward + raster, return cached features and feature_dim."""
    torch.manual_seed(seed)
    config = load_config_file(config_path)

    import train_video_token_implicit_dynamic as trainer
    from train_video_token_implicit_dynamic import render_clip_sequence

    resolved = trainer.resolve_config(config)
    feature_dim = int(resolved["model"]["feature_dim"])
    if feature_dim == 3:
        raise ValueError(
            "Matrix probe assumes feature_dim != 3 (F=3 path doesn't go through FeatureToColor)."
        )
    model = trainer.build_model_from_config(resolved).eval().to(device)
    train_frame_count = int(resolved["model"]["train_frame_count"])
    decode_times = torch.linspace(0.0, 1.0, train_frame_count, device=device).reshape(1, -1)
    decoded = model(video=None, decode_times=decode_times)
    if decoded.cameras is None:
        raise RuntimeError("Decoded output missing cameras; matrix probe expects implicit-camera models.")
    render_cfg = resolved["render"]
    rendered, _alpha_unused = render_clip_sequence(
        resolved,
        decoded,
        decoded.cameras,
        renderer_mode=str(render_cfg["renderer"]).lower(),
        dense_grid=None,  # type: ignore[arg-type]
    )
    return rendered.detach(), feature_dim


@torch.no_grad()
def metrics_for_cell(
    features: torch.Tensor,
    feature_dim: int,
    *,
    pre_norm: bool,
    weight_init: str,
    gain: float,
    hidden_dim: int | None,
    cell_seed: int,
    device: torch.device,
) -> dict[str, float]:
    from colorize import FeatureToColor

    torch.manual_seed(cell_seed)  # consistent colorize init across runs of the same cell
    colorize = (
        FeatureToColor(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            activation="sigmoid",
            pre_norm=pre_norm,
            weight_init=weight_init,
            weight_init_gain=gain,
        )
        .eval()
        .to(device)
    )
    rgb, logits = colorize.forward_with_logits(features)
    return post_colorize_image_diagnostics(rgb, logits=logits)


def aggregate_means(per_seed_metrics: list[dict[str, float]], keys: list[str]) -> list[float]:
    out = []
    for key in keys:
        vals = [m[key] for m in per_seed_metrics if key in m and m[key] == m[key]]
        out.append(statistics.fmean(vals) if vals else float("nan"))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep colorize-MLP init strategies.")
    parser.add_argument("config", type=Path, help="Base config (used for model + raster; colorize section ignored).")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/mps/cuda.")
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = _resolve_device(args.device)
    print(f"device={device}, config={args.config}, seeds={seeds}")

    seed_features: dict[int, torch.Tensor] = {}
    feature_dim: int | None = None
    for seed in seeds:
        feat, fd = render_features_once(args.config, seed, device)
        seed_features[seed] = feat
        feature_dim = fd
        print(f"  rendered features for seed={seed} shape={tuple(feat.shape)}")
    assert feature_dim is not None

    headers = ["cell", "PxChroma", "SpatStd", "Range", "R/Std", "G/Std", "B/Std", "L>0.5", "L>2", "L>4"]
    keys = [
        "ColorizeOut/PerPixelChroma/Mean",
        "ColorizeOut/SpatialStdMean",
        "ColorizeOut/Range",
        "ColorizeOut/R/Std",
        "ColorizeOut/G/Std",
        "ColorizeOut/B/Std",
        "ColorizeOut/Logit/AbsGt0p5Frac",
        "ColorizeOut/Logit/AbsGt2Frac",
        "ColorizeOut/Logit/AbsGt4Frac",
    ]

    print()
    print("  " + f"{headers[0]:<22s}" + " | ".join(f"{h:>9s}" for h in headers[1:]))
    print("  " + "-" * 24 + "-" * (12 * (len(headers) - 1)))

    for cell in CELLS:
        label, pre_norm, weight_init, gain, hidden_dim = cell
        per_seed_metrics: list[dict[str, float]] = []
        for seed in seeds:
            metrics = metrics_for_cell(
                seed_features[seed],
                feature_dim,
                pre_norm=pre_norm,
                weight_init=weight_init,
                gain=gain,
                hidden_dim=hidden_dim,
                cell_seed=seed,
                device=device,
            )
            per_seed_metrics.append(metrics)
        means = aggregate_means(per_seed_metrics, keys)
        cells = " | ".join(f"{m:>9.3g}" for m in means)
        print(f"  {label:<22s}{cells}")


if __name__ == "__main__":
    main()
