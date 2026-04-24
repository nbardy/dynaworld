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


def _decoded_mapping_from_head(gaussian_heads: Any, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return dict(zip(GAUSSIAN_FIELDS, gaussian_heads(tokens), strict=True))


def _image_implicit_splat_tokens(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "splat_tokens"):
        return model.splat_tokens
    if hasattr(model, "tokens"):
        # Joint implicit-camera models reserve token 0 for global camera and token 1 for path camera.
        return model.tokens[:, 2:, :]
    raise AttributeError(f"Could not find splat tokens on {model.__class__.__name__}.")


def _probe_target_from_config(config: dict[str, Any]):
    arch = config.get("arch")
    if arch == "tokengs_prebaked_camera":
        import dynamicTokenGS as trainer

        resolved = trainer.resolve_config(config)
        model = trainer.build_model_from_config(resolved["model"]).eval()
        tokens = model.tokens
    elif arch == "tokengs":
        import train_video_token_implicit_dynamic as trainer

        resolved = trainer.resolve_config(config)
        model = trainer.build_model_from_config(resolved).eval()
        tokens = model.query_tokens(1)[:, 2:, :]
    elif arch == "tokengs_image_implicit_camera":
        import train_camera_implicit_dynamic as trainer

        resolved = trainer.resolve_config(config)
        if hasattr(trainer, "build_model_from_config"):
            model = trainer.build_model_from_config(resolved["model"]).eval()
        else:
            from gs_models import DynamicTokenGSImplicitCamera

            model = DynamicTokenGSImplicitCamera(
                num_tokens=resolved["model"]["tokens"],
                gaussians_per_token=resolved["model"]["gaussians_per_token"],
            ).eval()
        tokens = _image_implicit_splat_tokens(model)
    else:
        raise ValueError(f"Unsupported arch={arch!r}.")

    model_cfg = resolved["model"]
    return {
        "arch": arch,
        "model_name": model.__class__.__name__,
        "config": resolved,
        "gaussian_heads": model.gaussian_heads,
        "tokens": tokens,
        "token_count": int(model_cfg["tokens"]),
        "gaussians_per_token": int(model_cfg["gaussians_per_token"]),
    }


def probe_config(path: Path, *, seed: int, bins: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    target = _probe_target_from_config(load_config_file(path))
    decoded = _decoded_mapping_from_head(target["gaussian_heads"], target["tokens"])
    raw_outputs = gaussian_head_raw_outputs(target["gaussian_heads"], target["tokens"])
    metrics = decoded_gaussian_init_diagnostics(
        decoded,
        token_count=target["token_count"],
        gaussians_per_token=target["gaussians_per_token"],
        valid_ranges=infer_valid_ranges_from_config(target["config"]),
        bins=bins,
    )
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
