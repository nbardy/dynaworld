from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from config_utils import load_config_file, serialize_config_value
from renderers.fast_mac import (
    FastMacRendererConfig,
    _ensure_fast_mac_v5_softmax_gs_on_path,
    _make_v5_softmax_gs_config,
    project_for_fast_mac_batch,
)
from rendering import _camera_scalar_vector, _camera_to_world_batch, _resolve_camera_projection_mode
from trainer_registry import instantiate_trainer_for_config


def _parse_int_csv(value: str) -> list[int]:
    values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("K values must be >= 1")
    return values


def _parse_views(value: str) -> list[str]:
    views = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if not views:
        raise argparse.ArgumentTypeError("expected at least one view, e.g. train0 or heldout0")
    for view in views:
        if not (view.startswith("train") or view.startswith("heldout")):
            raise argparse.ArgumentTypeError(f"unknown view selector {view!r}")
        int(view.replace("train", "").replace("heldout", ""))
    return views


def _configure_for_diagnostic(config: dict[str, Any], *, train_steps: int) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg.setdefault("logging", {})["wandb_enabled"] = False
    cfg["logging"]["log_initial_media"] = False
    cfg["logging"]["always_log_last_step"] = False
    cfg.setdefault("train", {})["steps"] = int(train_steps)
    cfg["train"]["profile_timing"] = False
    cfg["train"]["profile_timing_sync"] = False
    return cfg


def _view_cameras(trainer: Any, view: str, clip_indices: torch.Tensor):
    if view.startswith("train"):
        index = int(view.removeprefix("train"))
        cameras = trainer.camera_rig.cameras_for_view(index, clip_indices)
        label = f"train{index}_{trainer.multicam_bundle.train_camera_names[index]}"
        return label, cameras
    index = int(view.removeprefix("heldout"))
    cameras = trainer.camera_rig.heldout_cameras_for(index, clip_indices)
    names = trainer.multicam_bundle.heldout_camera_names or []
    name = names[index] if index < len(names) else f"heldout_{index}"
    return f"heldout{index}_{name}", cameras


def _project_for_view(cfg: dict[str, Any], decoded: Any, cameras: tuple[Any, ...]):
    if decoded.rgbs.shape[-1] != 3:
        raise ValueError("Softmax-GS tape coverage diagnostic currently expects RGB/F=3 decoded splats.")
    device = decoded.xyz.device
    return project_for_fast_mac_batch(
        decoded.xyz.float(),
        decoded.scales.float(),
        decoded.quats.float(),
        decoded.opacities.float(),
        decoded.rgbs.float(),
        _camera_scalar_vector(cameras, "fx", device),
        _camera_scalar_vector(cameras, "fy", device),
        _camera_scalar_vector(cameras, "cx", device),
        _camera_scalar_vector(cameras, "cy", device),
        cameras=cameras,
        projection_mode=_resolve_camera_projection_mode(cameras, cfg["render"]["camera_projection"]),
        camera_to_world=_camera_to_world_batch(cameras, device),
        near_plane=float(cfg["render"]["near_plane"]),
        depth_mode=cfg["render"]["fast_mac"].get("depth_mode", "rank_depth"),
    )


def _quantiles(values: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = values.detach().float().cpu().flatten()
    if flat.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_p50": float(torch.quantile(flat, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(flat, 0.90).item()),
        f"{prefix}_p95": float(torch.quantile(flat, 0.95).item()),
        f"{prefix}_p99": float(torch.quantile(flat, 0.99).item()),
        f"{prefix}_max": float(flat.max().item()),
    }


def summarize_tape(
    selected_ids: torch.Tensor,
    selected_weights: torch.Tensor,
    residual_weight: torch.Tensor,
    final_alpha: torch.Tensor,
    *,
    alpha_eps: float,
) -> dict[str, float | int]:
    active = final_alpha > float(alpha_eps)
    active_count = int(active.sum().item())
    pixel_count = int(final_alpha.numel())
    selected_count = (selected_ids >= 0).sum(dim=-1).detach().float().cpu()
    selected_mass = selected_weights.sum(dim=-1)
    safe_alpha = final_alpha.clamp_min(float(alpha_eps))
    residual_ratio = residual_weight / safe_alpha
    selected_mass_ratio = selected_mass / safe_alpha

    summary: dict[str, float | int] = {
        "pixel_count": pixel_count,
        "active_pixel_count": active_count,
        "active_pixel_fraction": float(active_count / max(pixel_count, 1)),
        "selected_count_mean_all": float(selected_count.mean().item()) if selected_count.numel() else 0.0,
    }
    if active_count == 0:
        summary.update(_quantiles(residual_weight.new_empty((0,)), "active_residual"))
        summary.update(_quantiles(residual_ratio.new_empty((0,)), "active_residual_over_alpha"))
        summary.update(_quantiles(selected_mass_ratio.new_empty((0,)), "active_selected_mass_over_alpha"))
        return summary

    active_selected_count = selected_count[active.detach().cpu()]
    summary["selected_count_mean_active"] = float(active_selected_count.mean().item())
    summary["selected_count_p95_active"] = float(torch.quantile(active_selected_count, 0.95).item())
    summary.update(_quantiles(final_alpha[active], "active_final_alpha"))
    summary.update(_quantiles(residual_weight[active], "active_residual"))
    summary.update(_quantiles(residual_ratio[active], "active_residual_over_alpha"))
    summary.update(_quantiles(selected_mass_ratio[active], "active_selected_mass_over_alpha"))
    return summary


def _run_tape_sweep(
    cfg: dict[str, Any],
    trainer: Any,
    decoded: Any,
    views: list[str],
    clip_indices: torch.Tensor,
    k_values: list[int],
    *,
    alpha_eps: float,
) -> list[dict[str, Any]]:
    _ensure_fast_mac_v5_softmax_gs_on_path()
    from torch_gsplat_bridge_v5_softmax_gs import rasterize_softmax_gs_bounded_tape

    fast_cfg = FastMacRendererConfig.from_mapping(
        cfg["render"]["fast_mac"],
        fallback_tile_size=cfg["render"]["tile_size"],
        fallback_alpha_threshold=cfg["render"]["alpha_threshold"],
    )
    if fast_cfg.depth_mode != "center_camera_z":
        raise ValueError("Softmax-GS tape coverage diagnostics require fast_mac.depth_mode='center_camera_z'.")

    rows: list[dict[str, Any]] = []
    for view in views:
        view_label, cameras = _view_cameras(trainer, view, clip_indices)
        means2d, conics, colors, opacities, depths = _project_for_view(cfg, decoded, cameras)
        base_raster_cfg = _make_v5_softmax_gs_config(fast_cfg, int(cfg["render"]["render_size"]), int(cfg["render"]["render_size"]))
        for k in k_values:
            selected_ids, selected_weights, residual_weight, final_alpha = rasterize_softmax_gs_bounded_tape(
                means2d,
                conics,
                colors,
                opacities,
                depths,
                base_raster_cfg,
                k_limit=int(k),
            )
            row = {
                "view": view_label,
                "k": int(k),
                **summarize_tape(
                    selected_ids,
                    selected_weights,
                    residual_weight,
                    final_alpha,
                    alpha_eps=alpha_eps,
                ),
            }
            rows.append(row)
    return rows


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Softmax-GS Tape Coverage Diagnostic",
        "",
        f"Config: `{payload['config_path']}`",
        f"Train steps before diagnostic: `{payload['train_steps']}`",
        f"K values: `{payload['k_values']}`",
        f"Views: `{payload['views']}`",
        "",
        "| View | K | Active px | Residual/alpha mean | Residual/alpha p95 | Residual/alpha p99 | Residual max | Selected mass/alpha mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {view} | {k} | {active_pixel_count} | {active_residual_over_alpha_mean:.6f} | "
            "{active_residual_over_alpha_p95:.6f} | {active_residual_over_alpha_p99:.6f} | "
            "{active_residual_max:.6f} | {active_selected_mass_over_alpha_mean:.6f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _configure_for_diagnostic(load_config_file(args.config), train_steps=args.train_steps)
    trainer = instantiate_trainer_for_config(cfg, args.config)
    for _step in range(int(args.train_steps)):
        trainer.step(keep_preview=False)

    sequence_data = trainer.sequence_data
    clip_indices, clip_frames, clip_times = trainer.initial_clip_for_sequence(sequence_data)
    with trainer.model_eval_mode(), torch.no_grad():
        decoded = trainer._decode_clip(sequence_data, clip_frames, clip_times)

    rows = _run_tape_sweep(
        cfg,
        trainer,
        decoded,
        args.views,
        clip_indices,
        args.k_values,
        alpha_eps=float(args.alpha_eps),
    )
    payload = {
        "config_path": str(args.config),
        "train_steps": int(args.train_steps),
        "k_values": [int(k) for k in args.k_values],
        "views": list(args.views),
        "alpha_eps": float(args.alpha_eps),
        "resolved_render_config": serialize_config_value(cfg["render"]),
        "rows": rows,
    }
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        (output_dir / "summary.md").write_text(_markdown_report(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Softmax-GS bounded-tape residual coverage for a config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--k-values", type=_parse_int_csv, default=_parse_int_csv("1,2,4,8,16"))
    parser.add_argument("--views", type=_parse_views, default=_parse_views("train0,train1,heldout0"))
    parser.add_argument("--alpha-eps", type=float, default=1.0e-6)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
