from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from config_utils import load_config_file
from checkpoint_utils import load_checkpoint_mapping
from pipeline.diagnostics import reconstruction_eval_metrics
from powerfoam_metal_config import resolve_config
try:
    from .report_artifacts import relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import relative_to_project as rel, write_report_json
from train_devices import resolve_torch_device

from diagnose_powerfoam_heldout_error import load_model_for_checkpoint, render_split


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, dim=-1, eps=1.0e-6)


def heldout_subset(
    frame_indices: torch.Tensor,
    rays: torch.Tensor,
    targets: torch.Tensor,
    requested_frames: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not requested_frames:
        return frame_indices, rays, targets
    wanted = torch.tensor(requested_frames, device=frame_indices.device, dtype=frame_indices.dtype)
    mask = (frame_indices[:, None] == wanted[None, :]).any(dim=1)
    if not bool(mask.any().detach().cpu()):
        raise ValueError(f"No heldout samples matched requested frames {requested_frames}.")
    return frame_indices[mask], rays[mask], targets[mask]


def camera_axes(rays: torch.Tensor) -> dict[str, torch.Tensor]:
    if rays.ndim != 4 or rays.shape[-1] != 6:
        raise ValueError(f"Expected rays [B,H,W,6], got {tuple(rays.shape)}.")
    height = int(rays.shape[1])
    width = int(rays.shape[2])
    cy = height // 2
    cx = width // 2
    dirs = normalize(rays[..., 3:])
    forward = normalize(dirs[:, cy, cx])
    x0 = max(cx - 1, 0)
    x1 = min(cx + 1, width - 1)
    y0 = max(cy - 1, 0)
    y1 = min(cy + 1, height - 1)
    right = normalize(dirs[:, cy, x1] - dirs[:, cy, x0])
    down = normalize(dirs[:, y1, cx] - dirs[:, y0, cx])
    return {"right": right, "down": down, "forward": forward}


def rodrigues(axis: torch.Tensor, angle_radians: float) -> torch.Tensor:
    axis = normalize(axis)
    batch = int(axis.shape[0])
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    k = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(batch, 3, 3)
    angle = torch.as_tensor(float(angle_radians), device=axis.device, dtype=axis.dtype)
    return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * torch.bmm(k, k)


def rotate_rays(rays: torch.Tensor, axis: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    rot = rodrigues(axis, math.radians(float(angle_degrees)))
    out = rays.clone()
    out[..., 3:] = torch.einsum("bij,bhwj->bhwi", rot, rays[..., 3:])
    return out


def translate_rays(rays: torch.Tensor, axis: torch.Tensor, distance: float) -> torch.Tensor:
    out = rays.clone()
    out[..., :3] = out[..., :3] + axis[:, None, None, :] * float(distance)
    return out


def candidate_specs(rotation_degrees: list[float], translations: list[float]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"kind": "baseline", "name": "baseline"}]
    for axis_name in ("right", "down", "forward"):
        rotation_name = {"right": "pitch", "down": "yaw", "forward": "roll"}[axis_name]
        for degrees in rotation_degrees:
            for sign in (-1.0, 1.0):
                value = sign * float(degrees)
                specs.append(
                    {
                        "kind": "rotation",
                        "name": f"{rotation_name}_{value:+.3f}deg",
                        "axis": axis_name,
                        "degrees": value,
                    }
                )
    for axis_name in ("right", "down", "forward"):
        for distance in translations:
            for sign in (-1.0, 1.0):
                value = sign * float(distance)
                specs.append(
                    {
                        "kind": "translation",
                        "name": f"translate_{axis_name}_{value:+.4f}",
                        "axis": axis_name,
                        "distance": value,
                    }
                )
    return specs


def apply_candidate(rays: torch.Tensor, axes: dict[str, torch.Tensor], spec: dict[str, Any]) -> torch.Tensor:
    if spec["kind"] == "baseline":
        return rays
    axis = axes[str(spec["axis"])]
    if spec["kind"] == "rotation":
        return rotate_rays(rays, axis, float(spec["degrees"]))
    if spec["kind"] == "translation":
        return translate_rays(rays, axis, float(spec["distance"]))
    raise ValueError(f"Unknown candidate kind {spec['kind']!r}.")


def candidate_metrics(
    *,
    model,
    cfg: dict[str, Any],
    frame_indices: torch.Tensor,
    rays: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    spec: dict[str, Any],
    axes: dict[str, torch.Tensor],
) -> dict[str, Any]:
    perturbed_rays = apply_candidate(rays, axes, spec)
    rendered, alpha = render_split(model, cfg, frame_indices, perturbed_rays, batch_size=batch_size)
    metrics = reconstruction_eval_metrics(rendered, targets.detach().cpu(), cfg, prefix="heldout")
    metrics["heldout_alpha_mean"] = scalar(alpha.mean())
    metrics["heldout_alpha_gt_0_90"] = scalar((alpha > 0.9).to(dtype=torch.float32).mean())
    return {**spec, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen PowerFoam heldout camera perturbation probe.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, nargs="*", default=[0, 4, 8, 12])
    parser.add_argument("--rotation-degrees", type=float, nargs="*", default=[0.5, 1.0])
    parser.add_argument("--translations", type=float, nargs="*", default=[0.025, 0.05])
    args = parser.parse_args()

    cfg_preview = resolve_config(load_config_file(args.config))
    checkpoint = args.checkpoint or (cfg_preview["logging"]["output_dir"] / "checkpoint_best.pt")
    output = args.output or (cfg_preview["logging"]["output_dir"] / "heldout_camera_perturbations.json")
    device = resolve_torch_device(str(args.device), auto_cuda=False)
    cfg, training_data, model = load_model_for_checkpoint(args.config, checkpoint, device)
    checkpoint_payload = load_checkpoint_mapping(checkpoint, map_location="cpu")
    if training_data["heldout_targets"] is None or training_data["heldout_rays"] is None:
        raise ValueError("Camera perturbation probe requires a heldout split with rays.")

    frame_indices, rays, targets = heldout_subset(
        training_data["heldout_frame_indices"],
        training_data["heldout_rays"],
        training_data["heldout_targets"],
        [int(frame) for frame in args.frames],
    )
    axes = camera_axes(rays)
    rows = []
    for spec in candidate_specs([float(x) for x in args.rotation_degrees], [float(x) for x in args.translations]):
        rows.append(
            candidate_metrics(
                model=model,
                cfg=cfg,
                frame_indices=frame_indices,
                rays=rays,
                targets=targets,
                batch_size=int(args.batch_size),
                spec=spec,
                axes=axes,
            )
        )
    baseline = next(row for row in rows if row["kind"] == "baseline")
    best_psnr = max(rows, key=lambda row: float(row["heldout_psnr"]))
    best_ssim = max(rows, key=lambda row: float(row["heldout_ssim"]))
    report = {
        "schema_version": "powerfoam_camera_perturbation_probe_v1",
        "config": rel(args.config),
        "checkpoint": rel(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "output_dir": rel(cfg["logging"]["output_dir"]),
        "heldout_views": training_data["heldout_views"],
        "frames": [int(frame) for frame in frame_indices.detach().cpu().tolist()],
        "candidate_count": len(rows),
        "baseline": baseline,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "deltas": {
            "best_psnr_minus_baseline": float(best_psnr["heldout_psnr"]) - float(baseline["heldout_psnr"]),
            "best_ssim_minus_baseline": float(best_ssim["heldout_ssim"]) - float(baseline["heldout_ssim"]),
        },
        "candidates": rows,
    }
    write_report_json(output, report)
    print(
        json.dumps(
            {
                "output": rel(output),
                "baseline": baseline,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "deltas": report["deltas"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
