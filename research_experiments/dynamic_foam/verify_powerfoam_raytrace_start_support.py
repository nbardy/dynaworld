from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config_utils import load_config_file
from checkpoint_utils import load_checkpoint_mapping, model_state_dict_from_checkpoint
from multicam_video_data import camera_from_K_w2c
from powerfoam_direct import POWERFOAM_SOFTPLUS_BETA
from powerfoam_geometry import powerfoam_rays_from_camera
from powerfoam_metal_config import resolve_config
from powerfoam_raster_config import make_powerfoam_metal_raster_config as make_raster_config
try:
    from .report_artifacts import relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import relative_to_project as rel, write_report_json
from torch_powerfoam_metal.rasterize import _default_start_ids, _sampled_ray_support_counts
from verify_powerfoam_clean_init_coverage import multicam_matrices


def scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def decode_checkpoint_points(state: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.cat(
        [
            torch.tanh(state["raw_xy"]) * float(cfg["model"]["xy_extent"]),
            float(cfg["model"]["z_min"])
            + torch.sigmoid(state["raw_z"]) * (float(cfg["model"]["z_max"]) - float(cfg["model"]["z_min"])),
        ],
        dim=-1,
    ).to(dtype=torch.float32)
    radii = torch.nn.functional.softplus(state["raw_radii"], beta=POWERFOAM_SOFTPLUS_BETA)
    radii = radii + float(cfg["model"]["radius_min"])
    return points, radii.to(dtype=torch.float32)


def sample_view_rows(view_names: list[str], frame_count: int, sample_count: int) -> list[dict[str, Any]]:
    expected = int(frame_count) * len(view_names)
    if sample_count != expected:
        raise ValueError(
            f"Expected {expected} samples for {len(view_names)} views x {frame_count} frames; got {sample_count}."
        )
    rows = []
    for view_index, view_name in enumerate(view_names):
        for frame_offset in range(int(frame_count)):
            rows.append(
                {
                    "sample_index": view_index * int(frame_count) + frame_offset,
                    "view_index": view_index,
                    "view_name": view_name,
                    "frame_offset": frame_offset,
                }
            )
    return rows


def start_row(
    points: torch.Tensor,
    radii: torch.Tensor,
    rays: torch.Tensor,
    raster_config: Any,
    labels: dict[str, Any],
) -> dict[str, Any]:
    rays_b = rays.unsqueeze(0) if rays.ndim == 3 else rays
    origins = rays_b[:, 0, 0, :3]
    origin_power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    origin_id = torch.argmin(origin_power.detach(), dim=1)
    counts = _sampled_ray_support_counts(points, radii, rays_b, raster_config)
    support_id = counts.argmax(dim=1)
    start_ids = _default_start_ids(points, radii, rays_b, raster_config).to(torch.long)
    origin_support = counts.gather(1, origin_id.view(-1, 1)).squeeze(1)
    support_count = counts.gather(1, support_id.view(-1, 1)).squeeze(1)
    start_flat = start_ids.view(rays_b.shape[0], -1)
    start_count = counts.gather(1, start_flat).mean(dim=1)
    switch_fraction = (start_flat != origin_id.view(-1, 1)).to(torch.float32).mean(dim=1)
    unique_start_count = torch.unique(start_flat[0]).numel()
    return {
        **labels,
        "origin_id": int(origin_id[0].detach().cpu()),
        "support_id": int(support_id[0].detach().cpu()),
        "default_start_id": int(start_flat[0, 0].detach().cpu()) if unique_start_count == 1 else None,
        "default_start_unique_count": int(unique_start_count),
        "default_switched_from_origin": bool((switch_fraction[0] > 0.0).detach().cpu()),
        "default_switched_from_origin_fraction": scalar(switch_fraction[0]),
        "origin_support_count": scalar(origin_support[0]),
        "best_support_count": scalar(support_count[0]),
        "default_start_support_count": scalar(start_count[0]),
        "total_sample_hit_count": scalar(counts.sum()),
        "sampled_cell_count": int(counts.shape[1]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_view = []
    view_names = sorted({str(row["view_name"]) for row in rows})
    for view_name in view_names:
        view_rows = [row for row in rows if row["view_name"] == view_name]
        by_view.append(
            {
                "view_name": view_name,
                "sample_count": len(view_rows),
                "switch_fraction": sum(bool(row["default_switched_from_origin"]) for row in view_rows)
                / float(len(view_rows)),
                "pixel_switch_fraction_mean": sum(
                    float(row["default_switched_from_origin_fraction"]) for row in view_rows
                )
                / float(len(view_rows)),
                "origin_support_mean": sum(float(row["origin_support_count"]) for row in view_rows)
                / float(len(view_rows)),
                "default_start_support_mean": sum(float(row["default_start_support_count"]) for row in view_rows)
                / float(len(view_rows)),
                "total_sample_hit_mean": sum(float(row["total_sample_hit_count"]) for row in view_rows)
                / float(len(view_rows)),
            }
        )
    return {
        "view_count": len(by_view),
        "sample_count": len(rows),
        "switched_sample_count": sum(bool(row["default_switched_from_origin"]) for row in rows),
        "per_view": by_view,
    }


def train_lens_model(camera_meta: dict[str, Any], view_index: int) -> str:
    if camera_meta["train_lens_models"] is None:
        return "pinhole"
    return str(camera_meta["train_lens_models"][view_index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--views", nargs="*", default=["camera_0021", "camera_0013", "camera_0040"])
    args = parser.parse_args()

    cfg = resolve_config(load_config_file(args.config))
    checkpoint = args.checkpoint or (cfg["logging"]["output_dir"] / "checkpoint_best.pt")
    output = args.output or (cfg["logging"]["output_dir"] / "raytrace_start_support_diagnostics.json")
    checkpoint_payload = load_checkpoint_mapping(checkpoint, map_location="cpu")
    state = model_state_dict_from_checkpoint(checkpoint_payload)
    points, radii = decode_checkpoint_points(state, cfg)
    raster_config = make_raster_config(cfg["render"])
    train_K, train_w2c, _heldout_K, _heldout_w2c, camera_meta = multicam_matrices(cfg)
    train_distortions = (
        None if camera_meta["train_distortions"] is None else torch.tensor(camera_meta["train_distortions"])
    )

    frame_count = int(camera_meta["frame_count"])
    train_rows = sample_view_rows(
        list(camera_meta["train_cameras"]),
        frame_count=frame_count,
        sample_count=int(len(camera_meta["train_cameras"]) * frame_count),
    )
    selected = set(str(view) for view in args.views)
    rows = []
    for labels in train_rows:
        if labels["view_name"] not in selected:
            continue
        view_index = int(labels["view_index"])
        frame_index = int(labels["frame_offset"])
        camera = camera_from_K_w2c(
            train_K[view_index],
            train_w2c[view_index, frame_index],
            lens_model=train_lens_model(camera_meta, view_index),
            distortion=None if train_distortions is None else train_distortions[view_index],
        )
        rays = powerfoam_rays_from_camera(
            camera,
            height=int(cfg["render"]["render_size"]),
            width=int(cfg["render"]["render_size"]),
            device=torch.device("cpu"),
        )
        rows.append(
            start_row(
                points[frame_index],
                radii[frame_index],
                rays,
                raster_config,
                {**labels, "frame_index": frame_index},
            )
        )
    report = {
        "config": rel(args.config),
        "checkpoint": rel(checkpoint),
        "output": rel(output),
        "views": sorted(selected),
        "render_size": int(cfg["render"]["render_size"]),
        "summary": summarize(rows),
        "rows": rows,
        "interpretation": (
            "This checks default raytrace start-cell support on real train rays without rendering pixels. "
            "A switched row means the patched default start no longer uses the origin-nearest cell."
        ),
    }
    write_report_json(output, report)
    print(json.dumps({"output": rel(output), "summary": report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
