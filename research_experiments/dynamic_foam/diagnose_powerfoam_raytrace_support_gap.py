from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from config_utils import load_config_file
from multicam_video_data import camera_from_K_w2c
from renderers.projection import project_points_camera
from train_powerfoam_metal import POWERFOAM_SOFTPLUS_BETA, resolve_config
from verify_powerfoam_clean_init_coverage import multicam_matrices


ROOT = Path(__file__).resolve().parents[2]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_checkpoint_points(state: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.cat(
        [
            torch.tanh(state["raw_xy"]) * float(cfg["model"]["xy_extent"]),
            float(cfg["model"]["z_min"])
            + torch.sigmoid(state["raw_z"]) * (float(cfg["model"]["z_max"]) - float(cfg["model"]["z_min"])),
        ],
        dim=-1,
    ).to(dtype=torch.float32)
    radii = F.softplus(state["raw_radii"], beta=POWERFOAM_SOFTPLUS_BETA) + float(cfg["model"]["radius_min"])
    return points, radii.to(dtype=torch.float32)


def projection_row(
    points: torch.Tensor,
    *,
    K: torch.Tensor,
    w2c: torch.Tensor,
    lens_model: str,
    distortion: torch.Tensor | None,
    render_size: int,
) -> dict[str, Any]:
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype)], dim=-1)
    points_camera = (points_h @ w2c.T)[:, :3]
    camera = camera_from_K_w2c(
        K,
        torch.eye(4, dtype=points.dtype),
        lens_model=lens_model,
        distortion=distortion,
    )
    pixels, depths, _jacobian, front = project_points_camera(points_camera, camera, near_plane=1.0e-5)
    inside = (
        front
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(render_size))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(render_size))
    )
    if bool(inside.any()):
        pixel_coords = torch.stack(
            [
                pixels[inside, 1].floor().long().clamp(0, render_size - 1),
                pixels[inside, 0].floor().long().clamp(0, render_size - 1),
            ],
            dim=-1,
        )
        pixel_count = int(torch.unique(pixel_coords, dim=0).shape[0])
        depth_values = depths[inside]
        depth_min = scalar(depth_values.min())
        depth_median = scalar(torch.quantile(depth_values, 0.5))
        depth_max = scalar(depth_values.max())
    else:
        pixel_count = 0
        depth_min = depth_median = depth_max = None
    return {
        "visible_center_count": int(inside.sum().item()),
        "front_center_count": int(front.sum().item()),
        "center_pixel_count": pixel_count,
        "center_pixel_coverage": float(pixel_count) / float(render_size * render_size),
        "depth_min": depth_min,
        "depth_median": depth_median,
        "depth_max": depth_max,
    }


def per_view_projection(
    points: torch.Tensor,
    *,
    view_names: list[str],
    K: torch.Tensor,
    w2c: torch.Tensor,
    lens_models: list[str] | None,
    distortions: torch.Tensor | None,
    render_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for view_index, view_name in enumerate(view_names):
        frame_rows = [
            projection_row(
                points[frame],
                K=K[view_index],
                w2c=w2c[view_index, frame],
                lens_model="pinhole" if lens_models is None else lens_models[view_index],
                distortion=None if distortions is None else distortions[view_index],
                render_size=render_size,
            )
            for frame in range(int(points.shape[0]))
        ]
        rows.append(
            {
                "view_index": view_index,
                "view_name": view_name,
                "visible_center_min": min(row["visible_center_count"] for row in frame_rows),
                "visible_center_mean": sum(row["visible_center_count"] for row in frame_rows) / float(len(frame_rows)),
                "visible_center_max": max(row["visible_center_count"] for row in frame_rows),
                "center_pixel_min": min(row["center_pixel_count"] for row in frame_rows),
                "center_pixel_mean": sum(row["center_pixel_count"] for row in frame_rows) / float(len(frame_rows)),
                "center_pixel_coverage_mean": sum(row["center_pixel_coverage"] for row in frame_rows)
                / float(len(frame_rows)),
                "per_frame": frame_rows,
            }
        )
    return rows


def alpha_by_view(error_report: dict[str, Any] | None, split: str) -> dict[str, dict[str, Any]]:
    if error_report is None or error_report.get(split) is None:
        return {}
    return {row["view_name"]: row for row in error_report[split].get("per_view", [])}


def attach_alpha_and_flags(
    projection_rows: list[dict[str, Any]],
    alpha_rows: dict[str, dict[str, Any]],
    *,
    min_projected_centers: int,
    alpha_threshold: float,
) -> list[dict[str, Any]]:
    out = []
    for row in projection_rows:
        alpha = alpha_rows.get(row["view_name"], {})
        alpha_mean = alpha.get("alpha_mean")
        enriched = {
            **row,
            "raytrace_alpha_mean": alpha_mean,
            "raytrace_alpha_fraction_lt_0_05": alpha.get("alpha_fraction_lt_0_05"),
            "raytrace_l1": alpha.get("l1"),
            "raytrace_psnr": alpha.get("psnr"),
            "raytrace_support_gap": (
                row["visible_center_mean"] >= int(min_projected_centers)
                and alpha_mean is not None
                and float(alpha_mean) <= float(alpha_threshold)
            ),
        }
        out.append(enriched)
    return out


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(args.config))
    checkpoint = args.checkpoint or (cfg["logging"]["output_dir"] / "checkpoint_best.pt")
    error_report_path = args.error_report or (cfg["logging"]["output_dir"] / "heldout_error_diagnostics.json")
    error_report = load_json(error_report_path) if error_report_path.exists() else None
    state = torch.load(checkpoint, map_location="cpu")["model"]
    points, radii = decode_checkpoint_points(state, cfg)
    train_K, train_w2c, heldout_K, heldout_w2c, camera_meta = multicam_matrices(cfg)
    train_distortions = (
        None if camera_meta["train_distortions"] is None else torch.tensor(camera_meta["train_distortions"])
    )
    heldout_distortions = (
        None if camera_meta["heldout_distortions"] is None else torch.tensor(camera_meta["heldout_distortions"])
    )
    train_projection = per_view_projection(
        points,
        view_names=camera_meta["train_cameras"],
        K=train_K,
        w2c=train_w2c,
        lens_models=camera_meta["train_lens_models"],
        distortions=train_distortions,
        render_size=int(cfg["render"]["render_size"]),
    )
    heldout_projection = per_view_projection(
        points,
        view_names=camera_meta["heldout_cameras"],
        K=heldout_K,
        w2c=heldout_w2c,
        lens_models=camera_meta["heldout_lens_models"],
        distortions=heldout_distortions,
        render_size=int(cfg["render"]["render_size"]),
    )
    train = attach_alpha_and_flags(
        train_projection,
        alpha_by_view(error_report, "train"),
        min_projected_centers=int(args.min_projected_centers),
        alpha_threshold=float(args.alpha_threshold),
    )
    heldout = attach_alpha_and_flags(
        heldout_projection,
        alpha_by_view(error_report, "heldout"),
        min_projected_centers=int(args.min_projected_centers),
        alpha_threshold=float(args.alpha_threshold),
    )
    return {
        "config": rel(args.config),
        "checkpoint": rel(checkpoint),
        "error_report": None if error_report is None else rel(error_report_path),
        "render_use_raytrace": bool(cfg["render"]["use_raytrace"]),
        "feature_mode": str(cfg["model"]["feature_mode"]),
        "cells": int(cfg["model"]["cells"]),
        "radii": {
            "min": scalar(radii.min()),
            "mean": scalar(radii.mean()),
            "max": scalar(radii.max()),
        },
        "train": train,
        "heldout": heldout,
        "support_gap_views": [row["view_name"] for row in train + heldout if row["raytrace_support_gap"]],
        "interpretation": (
            "A support-gap view has many decoded cell centers projected into the camera but near-zero saved "
            "raytrace alpha. That implicates raytrace traversal/start-cell/connectivity before camera loading."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--error-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-projected-centers", type=int, default=64)
    parser.add_argument("--alpha-threshold", type=float, default=1.0e-6)
    args = parser.parse_args()
    report = build_report(args)
    cfg = resolve_config(load_config_file(args.config))
    output = args.output or (cfg["logging"]["output_dir"] / "raytrace_support_gap_diagnostics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": rel(output), "support_gap_views": report["support_gap_views"]}, indent=2))


if __name__ == "__main__":
    main()
