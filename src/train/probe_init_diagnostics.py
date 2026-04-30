from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config_utils import load_config_file
from init_diagnostics import (
    decoded_gaussian_init_diagnostics,
    format_init_diagnostic_summary,
    gaussian_head_raw_outputs,
    infer_valid_ranges_from_config,
    raw_head_output_diagnostics,
)


GAUSSIAN_FIELDS = ("xyz", "scales", "quats", "opacities", "rgbs")
VIDEO_TOKEN_ARCHES = {
    "tokengs",
    "tokengs_video_implicit_camera",
    "tokengs_video_known_camera",
}
PRECOMPUTED_FEATURE_ARCHES = {
    "precomputed_feature_implicit_camera",
    "wan_vace_feature_implicit_camera",
    "ltx_feature_implicit_camera",
}


def _decoded_mapping_from_head(gaussian_heads: Any, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return dict(zip(GAUSSIAN_FIELDS, gaussian_heads(tokens), strict=True))

def _video_splat_tokens(model: torch.nn.Module) -> torch.Tensor:
    if not hasattr(model, "query_tokens"):
        raise AttributeError(f"Could not find query_tokens on {model.__class__.__name__}.")
    tokens = model.query_tokens(1)
    if int(getattr(model, "total_tokens", tokens.shape[1])) == int(getattr(model, "num_tokens", tokens.shape[1])) + 2:
        return tokens[:, 2:, :]
    return tokens


def _decoded_from_time_only_model(model: torch.nn.Module, frame_count: int):
    decode_times = torch.linspace(0.0, 1.0, int(frame_count)).reshape(1, -1)
    with torch.no_grad():
        return model(video=None, decode_times=decode_times)


def _require_single_gaussian_head(model: torch.nn.Module):
    if hasattr(model, "gaussian_heads"):
        return model.gaussian_heads
    if hasattr(model, "static_gaussian_heads") or hasattr(model, "dynamic_gaussian_heads"):
        raise ValueError(
            "Static/dynamic Gaussian-bank init probing needs separate static/dynamic reports; "
            "disable model.static_tokens/model.dynamic_tokens or extend this probe for split heads."
        )
    raise AttributeError(f"Could not find gaussian_heads on {model.__class__.__name__}.")


def _probe_target_from_config(config: dict[str, Any]):
    arch = config.get("arch")
    if arch in VIDEO_TOKEN_ARCHES:
        import train_video_token_implicit_dynamic as trainer

        resolved = trainer.resolve_config(config)
        model = trainer.build_model_from_config(resolved).eval()
        variant = str(resolved["model"]["variant"]).lower()
        if variant in {
            "free_splats",
            "free_gaussian_bank",
            "free_linear_splats",
            "free_linear_time_splats",
            "linear_free_splats",
        }:
            return {
                "arch": arch,
                "model_name": model.__class__.__name__,
                "config": resolved,
                "decoded": _decoded_from_time_only_model(model, resolved["model"]["train_frame_count"]),
                "token_count": int(resolved["model"]["tokens"]),
                "gaussians_per_token": int(resolved["model"]["gaussians_per_token"]),
            }
        tokens = _video_splat_tokens(model)
    elif arch in PRECOMPUTED_FEATURE_ARCHES:
        import train_precomputed_feature_implicit_dynamic as trainer
        import train_video_token_implicit_dynamic as video_trainer

        resolved = trainer.PrecomputedFeatureImplicitTrainer.resolve_config(config)
        if resolved["model"]["video_feature_channels"] is None:
            raise ValueError(
                "Precomputed-feature init probing needs model.video_feature_channels. "
                "Run after feature prebake/inference or set the cached layer channel counts in the config."
            )
        model = video_trainer.build_model_from_config(resolved).eval()
        tokens = _video_splat_tokens(model)
    else:
        raise ValueError(f"Unsupported arch={arch!r}.")

    model_cfg = resolved["model"]
    return {
        "arch": arch,
        "model_name": model.__class__.__name__,
        "config": resolved,
        "gaussian_heads": _require_single_gaussian_head(model),
        "tokens": tokens,
        "token_count": int(model_cfg["tokens"]),
        "gaussians_per_token": int(model_cfg["gaussians_per_token"]),
    }


def probe_config(path: Path, *, seed: int, bins: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    target = _probe_target_from_config(load_config_file(path))
    if "decoded" in target:
        decoded = target["decoded"]
        raw_outputs = {}
    else:
        decoded = _decoded_mapping_from_head(target["gaussian_heads"], target["tokens"])
        raw_outputs = gaussian_head_raw_outputs(target["gaussian_heads"], target["tokens"])
    metrics = decoded_gaussian_init_diagnostics(
        decoded,
        token_count=target["token_count"],
        gaussians_per_token=target["gaussians_per_token"],
        valid_ranges=infer_valid_ranges_from_config(target["config"]),
        bins=bins,
    )
    if raw_outputs:
        metrics.update(raw_head_output_diagnostics(raw_outputs, bins=bins))
    return {
        "path": str(path),
        "arch": target["arch"],
        "model_name": target["model_name"],
        "token_count": target["token_count"],
        "gaussians_per_token": target["gaussians_per_token"],
        "gaussian_count": target["token_count"] * target["gaussians_per_token"],
        "seed": seed,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe random-init Gaussian head health for a train config.")
    parser.add_argument("config", type=Path, help="Path to a JSONC train config.")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed used before model construction.")
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins for normalized entropy metrics.")
    parser.add_argument("--json", action="store_true", help="Print the full metric payload as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = probe_config(args.config, seed=args.seed, bins=args.bins)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(
        f"{result['model_name']} arch={result['arch']} "
        f"tokens={result['token_count']} splits={result['gaussians_per_token']} "
        f"gaussians={result['gaussian_count']} seed={result['seed']}"
    )
    print(format_init_diagnostic_summary(result["metrics"]))


if __name__ == "__main__":
    main()
