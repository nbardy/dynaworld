from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from config_utils import load_config_file
from checkpoint_utils import load_checkpoint_mapping, model_state_dict_from_checkpoint
from multicam_video_data import camera_from_K_w2c
from powerfoam_adjacency import build_csr_adjacency
from powerfoam_direct import POWERFOAM_SOFTPLUS_BETA
from powerfoam_geometry import powerfoam_rays_from_camera
from powerfoam_metal_config import resolve_config
from powerfoam_raster_config import make_powerfoam_metal_raster_config as make_raster_config
from train_devices import sync_torch_device

try:
    from .report_artifacts import relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import relative_to_project as rel, write_report_json
from torch_powerfoam_metal import (
    quaternion_frames,
    rasterize_power_foam_oriented_height_sv_texel_surface,
    raytrace_power_foam_oriented_height_sv_texel_surface,
)
from verify_powerfoam_clean_init_coverage import multicam_matrices


def scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def decode_checkpoint_state(state: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    points = torch.cat(
        [
            torch.tanh(state["raw_xy"]) * float(cfg["model"]["xy_extent"]),
            float(cfg["model"]["z_min"])
            + torch.sigmoid(state["raw_z"]) * (float(cfg["model"]["z_max"]) - float(cfg["model"]["z_min"])),
        ],
        dim=-1,
    ).to(dtype=torch.float32)
    radii = F.softplus(state["raw_radii"], beta=POWERFOAM_SOFTPLUS_BETA) + float(cfg["model"]["radius_min"])
    return {
        "points": points,
        "radii": radii.to(dtype=torch.float32),
        "densities": F.softplus(state["raw_densities"], beta=POWERFOAM_SOFTPLUS_BETA).to(dtype=torch.float32),
        "texel_sites": float(cfg["model"]["texel_site_scale"]) * torch.tanh(state["raw_texel_sites"]).to(dtype=torch.float32),
        "texel_sv_axis": state["raw_texel_sv_axis"].to(dtype=torch.float32),
        "texel_sv_rgb": state["raw_texel_sv_rgb"].to(dtype=torch.float32),
        "texel_heights_raw": state["raw_texel_heights"].to(dtype=torch.float32),
        "quaternions": state["raw_quaternions"].to(dtype=torch.float32),
    }


def frame_params(decoded: dict[str, torch.Tensor], cfg: dict[str, Any], frame_index: int, device: torch.device) -> dict[str, torch.Tensor]:
    frame = int(frame_index)
    point = decoded["points"][frame].to(device=device)
    radius = decoded["radii"][frame].to(device=device)
    normals, tangents, bitangents = quaternion_frames(
        decoded["quaternions"][frame].to(device=device),
        eps=float(cfg["render"]["eps"]),
    )
    return {
        "points": point,
        "radii": radius,
        "densities": decoded["densities"][frame].to(device=device),
        "texel_sites": decoded["texel_sites"][frame].to(device=device),
        "texel_heights": (
            radius[:, None]
            * float(cfg["model"]["texel_height_scale"])
            * torch.tanh(decoded["texel_heights_raw"][frame].to(device=device))
        ),
        "texel_sv_axis": decoded["texel_sv_axis"][frame].to(device=device),
        "texel_sv_rgb": decoded["texel_sv_rgb"][frame].to(device=device),
        "normals": normals,
        "tangents": tangents,
        "bitangents": bitangents,
    }


def sampled_indices(size: int, sample_size: int) -> torch.Tensor:
    samples = min(max(int(sample_size), 1), int(size))
    return torch.linspace(0, size - 1, samples, dtype=torch.float32).round().to(torch.long)


def sampled_rays(
    K: torch.Tensor,
    w2c: torch.Tensor,
    *,
    lens_model: str,
    distortion: torch.Tensor | None,
    render_size: int,
    sample_size: int,
    device: torch.device,
) -> torch.Tensor:
    camera = camera_from_K_w2c(K, w2c, lens_model=lens_model, distortion=distortion)
    rays = powerfoam_rays_from_camera(
        camera,
        height=int(render_size),
        width=int(render_size),
        device=device,
    )
    ids = sampled_indices(int(render_size), int(sample_size)).to(device=device)
    return rays.index_select(1, ids).index_select(2, ids).contiguous()


def origin_start_ids(points: torch.Tensor, radii: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    origins = rays[:, 0, 0, :3]
    power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    return torch.argmin(power.detach(), dim=1).to(device=points.device, dtype=torch.int32).contiguous()


def near_plane_start_ids(
    points: torch.Tensor,
    radii: torch.Tensor,
    rays: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    flat = rays.reshape(-1, 6)
    origins = flat[:, :3]
    dirs = F.normalize(flat[:, 3:], dim=-1, eps=float(cfg["render"]["eps"]))
    query = origins + dirs * float(cfg["render"]["near_plane"])
    power = (points.unsqueeze(0) - query.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    return torch.argmin(power.detach(), dim=1).to(device=points.device, dtype=torch.int32).contiguous()


def first_sphere_hit_start_ids(
    points: torch.Tensor,
    radii: torch.Tensor,
    rays: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    flat = rays.reshape(-1, 6)
    origins = flat[:, :3]
    dirs = F.normalize(flat[:, 3:], dim=-1, eps=float(cfg["render"]["eps"]))
    rel = origins[:, None, :] - points.unsqueeze(0)
    qa = dirs.square().sum(dim=-1, keepdim=True).clamp_min(float(cfg["render"]["eps"]))
    qb = 2.0 * (rel * dirs[:, None, :]).sum(dim=-1)
    qc = rel.square().sum(dim=-1) - radii.square().unsqueeze(0)
    disc = qb.square() - 4.0 * qa * qc
    hit = disc >= 0.0
    root = disc.clamp_min(0.0).sqrt()
    t_near = (-qb - root) / (2.0 * qa)
    t_far = (-qb + root) / (2.0 * qa)
    near_plane = float(cfg["render"]["near_plane"])
    hit = hit & (t_far > near_plane)
    t_first = torch.where(hit, t_near.clamp_min(near_plane), torch.full_like(t_near, float("inf")))
    first_ids = torch.argmin(t_first, dim=1)
    no_hit = ~torch.isfinite(t_first.gather(1, first_ids[:, None]).squeeze(1))
    if bool(no_hit.any().item()):
        fallback = near_plane_start_ids(points, radii, rays, cfg)
        first_ids = torch.where(no_hit, fallback.to(first_ids.device, dtype=first_ids.dtype), first_ids)
    return first_ids.to(device=points.device, dtype=torch.int32).contiguous()


def start_id_variants(
    points: torch.Tensor,
    radii: torch.Tensor,
    rays: torch.Tensor,
    cfg: dict[str, Any],
    names: list[str],
) -> dict[str, torch.Tensor | None]:
    variants: dict[str, torch.Tensor | None] = {}
    for name in names:
        if name == "default_per_ray":
            variants[name] = None
        elif name == "origin":
            variants[name] = origin_start_ids(points, radii, rays)
        elif name == "near_plane":
            variants[name] = near_plane_start_ids(points, radii, rays, cfg)
        elif name == "first_sphere_hit":
            variants[name] = first_sphere_hit_start_ids(points, radii, rays, cfg)
        else:
            raise ValueError(f"Unknown start mode {name!r}.")
    return variants


def all_pairs_adjacency(cell_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    offsets = [0]
    for i in range(int(cell_count)):
        rows.extend(j for j in range(int(cell_count)) if j != i)
        offsets.append(len(rows))
    return (
        torch.tensor(rows, device=device, dtype=torch.int32),
        torch.tensor(offsets, device=device, dtype=torch.int32),
    )


def verifier_adjacency(params: dict[str, torch.Tensor], cfg: dict[str, Any], mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "all_pairs":
        return all_pairs_adjacency(int(params["points"].shape[0]), params["points"].device)
    return build_csr_adjacency(
        params["points"],
        params["radii"],
        neighbor_count=int(cfg["model"]["neighbor_count"]),
        mode=mode,
    )


def raytrace_variant(
    params: dict[str, torch.Tensor],
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    rays: torch.Tensor,
    raster_config: Any,
    start_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ray_out, ray_alpha, steps = raytrace_power_foam_oriented_height_sv_texel_surface(
        params["points"],
        params["radii"],
        params["densities"],
        params["texel_sites"],
        params["texel_heights"],
        params["texel_sv_axis"],
        params["texel_sv_rgb"],
        params["normals"],
        adjacency,
        offsets,
        rays,
        raster_config,
        start_ids,
        tangents=params["tangents"],
        bitangents=params["bitangents"],
        return_steps=True,
    )
    return ray_out, ray_alpha, steps


def start_summary(
    name: str,
    start_ids: torch.Tensor | None,
    stream_out: torch.Tensor,
    stream_alpha: torch.Tensor,
    ray_out: torch.Tensor,
    ray_alpha: torch.Tensor,
    steps: torch.Tensor,
) -> dict[str, Any]:
    alpha_err = (stream_alpha - ray_alpha).abs()
    out_err = (stream_out - ray_out).abs()
    summary = {
        "start_mode": name,
        "start_shape": "implicit_default" if start_ids is None else list(start_ids.shape),
        "alpha_max": scalar(ray_alpha.max()),
        "alpha_mean": scalar(ray_alpha.mean()),
        "steps_mean": scalar(steps.float().mean()),
        "steps_max": int(steps.max().detach().cpu()),
        "vs_stream_alpha_max_error": scalar(alpha_err.max()),
        "vs_stream_alpha_mean_error": scalar(alpha_err.mean()),
        "vs_stream_feature_max_error": scalar(out_err.max()),
        "recovers_nonzero_alpha": bool((ray_alpha.max() > 0.05).detach().cpu()),
    }
    if start_ids is not None and start_ids.numel() > 0:
        flat = start_ids.detach().cpu().flatten()
        summary["start_first_id"] = int(flat[0])
        summary["start_unique_count"] = int(torch.unique(flat).numel())
    return summary


def render_row(
    params: dict[str, torch.Tensor],
    rays: torch.Tensor,
    cfg: dict[str, Any],
    adjacency_mode: str,
    start_modes: list[str],
    labels: dict[str, Any],
) -> dict[str, Any]:
    adjacency, offsets = verifier_adjacency(params, cfg, adjacency_mode)
    raster_config = make_raster_config(cfg["render"])
    stream_out, stream_alpha = rasterize_power_foam_oriented_height_sv_texel_surface(
        params["points"],
        params["radii"],
        params["densities"],
        params["texel_sites"],
        params["texel_heights"],
        params["texel_sv_axis"],
        params["texel_sv_rgb"],
        params["normals"],
        adjacency,
        offsets,
        rays,
        raster_config,
        tangents=params["tangents"],
        bitangents=params["bitangents"],
    )
    variants = start_id_variants(params["points"], params["radii"], rays, cfg, start_modes)
    start_rows = []
    for name, start_ids in variants.items():
        ray_out, ray_alpha, steps = raytrace_variant(params, adjacency, offsets, rays, raster_config, start_ids)
        start_rows.append(start_summary(name, start_ids, stream_out, stream_alpha, ray_out, ray_alpha, steps))
    sync_torch_device(rays.device)
    by_name = {row["start_mode"]: row for row in start_rows}
    origin = by_name.get("origin")
    default = by_name.get("default_per_ray")
    return {
        **labels,
        "adjacency_mode": adjacency_mode,
        "adjacency_edges": int(adjacency.numel()),
        "stream_alpha_max": scalar(stream_alpha.max()),
        "stream_alpha_mean": scalar(stream_alpha.mean()),
        "start_rows": start_rows,
        "old_origin_start_id": None if origin is None else origin.get("start_first_id"),
        "old_origin_alpha_max": 0.0 if origin is None else float(origin["alpha_max"]),
        "old_origin_alpha_mean": 0.0 if origin is None else float(origin["alpha_mean"]),
        "old_origin_steps_max": 0 if origin is None else int(origin["steps_max"]),
        "patched_alpha_max": 0.0 if default is None else float(default["alpha_max"]),
        "patched_alpha_mean": 0.0 if default is None else float(default["alpha_mean"]),
        "patched_steps_max": 0 if default is None else int(default["steps_max"]),
        "patched_vs_stream_alpha_max_error": 0.0 if default is None else float(default["vs_stream_alpha_max_error"]),
        "patched_vs_stream_alpha_mean_error": 0.0 if default is None else float(default["vs_stream_alpha_mean_error"]),
        "patched_vs_stream_feature_max_error": 0.0 if default is None else float(default["vs_stream_feature_max_error"]),
        "patched_recovers_nonzero_alpha": False if default is None else bool(default["recovers_nonzero_alpha"]),
        "old_origin_zero_or_weaker": False if origin is None or default is None else float(origin["alpha_max"]) < float(default["alpha_max"]),
    }


def train_lens_model(camera_meta: dict[str, Any], view_index: int) -> str:
    if camera_meta["train_lens_models"] is None:
        return "pinhole"
    return str(camera_meta["train_lens_models"][view_index])


def build_rows(args: argparse.Namespace) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(args.config))
    checkpoint = args.checkpoint or (cfg["logging"]["output_dir"] / "checkpoint_best.pt")
    checkpoint_payload = load_checkpoint_mapping(checkpoint, map_location="cpu")
    state = model_state_dict_from_checkpoint(checkpoint_payload)
    decoded = decode_checkpoint_state(state, cfg)
    train_K, train_w2c, _heldout_K, _heldout_w2c, camera_meta = multicam_matrices(cfg)
    train_distortions = (
        None if camera_meta["train_distortions"] is None else torch.tensor(camera_meta["train_distortions"])
    )
    device = torch.device(str(args.device))
    view_to_index = {name: index for index, name in enumerate(camera_meta["train_cameras"])}
    adjacency_modes = list(args.adjacency_modes or [args.adjacency_mode])
    rows = []
    for view_name in args.views:
        if view_name not in view_to_index:
            raise ValueError(f"{view_name!r} is not in train views {camera_meta['train_cameras']}.")
        view_index = int(view_to_index[view_name])
        for frame_index in args.frames:
            params = frame_params(decoded, cfg, int(frame_index), device)
            rays = sampled_rays(
                train_K[view_index],
                train_w2c[view_index, int(frame_index)],
                lens_model=train_lens_model(camera_meta, view_index),
                distortion=None if train_distortions is None else train_distortions[view_index],
                render_size=int(cfg["render"]["render_size"]),
                sample_size=int(args.sample_size),
                device=device,
            )
            for adjacency_mode in adjacency_modes:
                rows.append(
                    render_row(
                        params,
                        rays,
                        cfg,
                        str(adjacency_mode),
                        list(args.start_modes),
                        {
                            "view_name": view_name,
                            "view_index": view_index,
                            "frame_index": int(frame_index),
                            "sample_size": int(args.sample_size),
                        },
                    )
                )
    return {
        "config": rel(args.config),
        "checkpoint": rel(checkpoint),
        "render_size": int(cfg["render"]["render_size"]),
        "sample_size": int(args.sample_size),
        "adjacency_modes": adjacency_modes,
        "start_modes": list(args.start_modes),
        "rows": rows,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    start_rows = [start for row in rows for start in row.get("start_rows", [])]
    return {
        "row_count": len(rows),
        "patched_nonzero_rows": sum(bool(row["patched_recovers_nonzero_alpha"]) for row in rows),
        "old_origin_weaker_rows": sum(bool(row["old_origin_zero_or_weaker"]) for row in rows),
        "max_patched_vs_stream_alpha_error": max(float(row["patched_vs_stream_alpha_max_error"]) for row in rows),
        "max_patched_vs_stream_feature_error": max(float(row["patched_vs_stream_feature_max_error"]) for row in rows),
        "min_patched_alpha_max": min(float(row["patched_alpha_max"]) for row in rows),
        "max_old_origin_alpha_max": max(float(row["old_origin_alpha_max"]) for row in rows),
        "best_start_by_mean_alpha_error": min(
            start_rows,
            key=lambda row: float(row["vs_stream_alpha_mean_error"]),
        )
        if start_rows
        else None,
        "best_start_by_max_alpha_error": min(
            start_rows,
            key=lambda row: float(row["vs_stream_alpha_max_error"]),
        )
        if start_rows
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--views", nargs="*", default=["camera_0021", "camera_0013"])
    parser.add_argument("--frames", nargs="*", type=int, default=[0, 4, 8, 12])
    parser.add_argument("--sample-size", type=int, default=9)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--adjacency-mode", default=None)
    parser.add_argument("--adjacency-modes", nargs="*", default=None)
    parser.add_argument(
        "--start-modes",
        nargs="*",
        default=["origin", "default_per_ray", "near_plane", "first_sphere_hit"],
    )
    args = parser.parse_args()
    if args.adjacency_mode is None:
        cfg_preview = resolve_config(load_config_file(args.config))
        args.adjacency_mode = str(cfg_preview["model"]["adjacency_mode"])

    report = build_rows(args)
    report["summary"] = summarize(report["rows"])
    report["interpretation"] = (
        "This renders a tiny sampled ray grid for known zero-alpha real train views. "
        "It compares the patched default raytrace start, the old forced origin start, and the streaming renderer."
    )
    cfg_preview = resolve_config(load_config_file(args.config))
    output = args.output or (cfg_preview["logging"]["output_dir"] / "raytrace_real_view_alpha_diagnostics.json")
    report["output"] = rel(output)
    write_report_json(output, report)
    print(json.dumps({"output": rel(output), "summary": report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
