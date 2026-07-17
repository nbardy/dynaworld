from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from camera import CameraSpec
from checkpoint_utils import load_torch_checkpoint, model_state_dict_from_checkpoint
from config_utils import load_config_file
from dynamic_powerfoam_camera import build_camera_decoder
from dynamic_powerfoam_metal_config import TOKEN_RBF_FEATURE_MODE
from model_factories import build_model_from_config
from powerfoam_raster_config import make_dynamic_powerfoam_metal_raster_config
from train_logging import set_default_wandb_mode
from trainer_registry import instantiate_trainer_for_config, resolve_config_for_arch


@dataclass(frozen=True)
class FrustumLines:
    segments: torch.Tensor
    centers: torch.Tensor
    directions: torch.Tensor


def _resolve_config_by_arch(config: dict[str, Any]) -> dict[str, Any]:
    return resolve_config_for_arch(config)


def _normalized_model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return _resolve_config_by_arch(config)["model"]


def _infer_frame_count(config: dict[str, Any], requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    if "model" in config and config["model"].get("train_frame_count") is not None:
        return int(config["model"]["train_frame_count"])
    if "data" in config and config["data"].get("max_frames") is not None:
        return int(config["data"]["max_frames"])
    return 8


def _decode_raw_xyz(raw_xyz: torch.Tensor, model_cfg: dict[str, Any]) -> torch.Tensor:
    scene_extent = float(model_cfg["scene_extent"])
    xy_extent = float(model_cfg.get("xy_extent") if model_cfg.get("xy_extent") is not None else scene_extent)
    z_min = float(model_cfg.get("z_min") if model_cfg.get("z_min") is not None else -scene_extent)
    z_max = float(model_cfg.get("z_max") if model_cfg.get("z_max") is not None else scene_extent)
    return torch.cat(
        [
            torch.tanh(raw_xyz[..., :2]) * xy_extent,
            torch.sigmoid(raw_xyz[..., 2:]) * (z_max - z_min) + z_min,
        ],
        dim=-1,
    )


def _decode_points_from_state_dict(state: dict[str, torch.Tensor], model_cfg: dict[str, Any]) -> torch.Tensor | None:
    if "raw_xy" in state and "raw_z" in state:
        xy_extent = float(model_cfg.get("xy_extent") or model_cfg.get("scene_extent", 1.0))
        z_min = float(model_cfg.get("z_min", -xy_extent))
        z_max = float(model_cfg.get("z_max", xy_extent))
        xy = torch.tanh(state["raw_xy"].detach().cpu().float()) * xy_extent
        z = torch.sigmoid(state["raw_z"].detach().cpu().float()) * (z_max - z_min) + z_min
        return torch.cat([xy, z], dim=-1)
    if "raw_xyz" in state:
        return _decode_raw_xyz(state["raw_xyz"].detach().cpu().float(), model_cfg)
    if "base_raw_xyz" in state:
        return _decode_raw_xyz(state["base_raw_xyz"].detach().cpu().float(), model_cfg)
    return None


def _decoded_scene_from_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    *,
    frame_count: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, tuple[CameraSpec, ...]]:
    resolved = _resolve_config_by_arch(config)
    if str(resolved.get("arch", "")) == "multicam_relative_pose_implicit_camera":
        return _decoded_multicam_relative_pose_scene(resolved, checkpoint_path, frame_count=frame_count, device=device)
    if str(resolved.get("arch", "")) == "dynamic_powerfoam_metal":
        return _decoded_dynamic_powerfoam_scene(resolved, checkpoint_path, frame_count=frame_count, device=device)
    model = build_model_from_config(resolved).to(device)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    decode_times = torch.linspace(0.0, 1.0, int(frame_count), device=device).reshape(1, -1)
    with torch.no_grad():
        output = model(video=None, decode_times=decode_times)
    points = output.xyz.detach().cpu().float()
    cameras = output.cameras or ()
    return points, cameras


def _load_relpose_head_compat(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    load_result = model.load_state_dict(state, strict=False)
    allowed_missing = {"pair_delta_norm.weight", "pair_delta_norm.bias", "pair_delta_output.weight"}
    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected relative-pose checkpoint keys: {sorted(unexpected)}")
    if missing - allowed_missing:
        raise RuntimeError(f"Missing relative-pose checkpoint keys: {sorted(missing)}")


def _decoded_multicam_relative_pose_scene(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    *,
    frame_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[CameraSpec, ...]]:
    # Trainer construction initializes W&B in the shared base class. Keep this
    # diagnostic side-effect-free unless the caller explicitly overrides it.
    set_default_wandb_mode("disabled", silent=True)

    from camera_swap_sampling import CameraSwapPair, build_heldout_camera_swap_pairs
    from relative_pose import cameras_with_se3_transform

    trainer = instantiate_trainer_for_config(copy.deepcopy(cfg))
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=trainer.device)
    state = model_state_dict_from_checkpoint(checkpoint)
    trainer.model.load_state_dict(state, strict=True)
    if trainer.colorize is not None and isinstance(checkpoint, dict) and "colorizer" in checkpoint:
        trainer.colorize.load_state_dict(checkpoint["colorizer"])
    if isinstance(checkpoint, dict) and "camera_rig" in checkpoint:
        trainer.camera_rig.load_state_dict(checkpoint["camera_rig"])
    if isinstance(checkpoint, dict) and "relpose_head" in checkpoint:
        _load_relpose_head_compat(trainer.relpose_head, checkpoint["relpose_head"])
    trainer.model.eval()
    trainer.relpose_head.eval()

    requested = min(int(frame_count), int(trainer.sequence_data.frame_count))
    clip_indices = torch.arange(requested, device=trainer.device)
    with torch.no_grad():
        _sequence_data, _clip_frames, _clip_times, decoded = trainer._decode_source_view(0, clip_indices)
        memory_cache: dict[tuple[str, int], torch.Tensor] = {}
        cameras: list[CameraSpec] = []
        source_name = trainer.multicam_bundle.train_camera_names[0]
        for view, target_name in enumerate(trainer.multicam_bundle.train_camera_names):
            pair = CameraSwapPair(
                source_set="train",
                source_view=0,
                query_set="train",
                query_view=view,
                target_set="train",
                target_view=view,
                source_name=source_name,
                query_name=target_name,
                target_name=target_name,
            )
            prediction = trainer.full_relative_pose_for_pair(
                pair,
                clip_indices=clip_indices,
                memory_cache=memory_cache,
            )
            cameras.append(cameras_with_se3_transform(prediction.target_template_cameras, prediction.camera_to_world)[0])
        if trainer.multicam_bundle.heldout_frames is not None:
            heldout_pairs = build_heldout_camera_swap_pairs(
                trainer.multicam_bundle.train_view_count,
                trainer.multicam_bundle.heldout_view_count,
                train_camera_names=trainer.multicam_bundle.train_camera_names,
                heldout_camera_names=trainer.multicam_bundle.heldout_camera_names or None,
            )
            for pair in heldout_pairs:
                if int(pair.source_view) != 0:
                    continue
                prediction = trainer.full_relative_pose_for_pair(
                    pair,
                    clip_indices=clip_indices,
                    memory_cache=memory_cache,
                )
                cameras.append(cameras_with_se3_transform(prediction.target_template_cameras, prediction.camera_to_world)[0])
    return decoded.xyz.detach().cpu().float(), tuple(cameras)


def _decoded_dynamic_powerfoam_scene(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    *,
    frame_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[CameraSpec, ...]]:
    from dynamic_powerfoam_metal_trainer import DynamicMetalPowerFoamVideo, TokenDynamicPowerFoamFeatures

    camera_decoder = build_camera_decoder(cfg, frame_count=int(frame_count))
    model_kwargs = {
        "frame_count": int(frame_count),
        "cell_count": int(cfg["model"]["cells"]),
        "render_size": int(cfg["render"]["render_size"]),
        "fov_degrees": float(cfg["render"]["fov_degrees"]),
        "neighbor_count": int(cfg["model"]["neighbor_count"]),
        "adjacency_mode": str(cfg["model"]["adjacency_mode"]),
        "time_basis_count": int(cfg["model"]["time_basis_count"]),
        "time_basis_sigma_scale": float(cfg["model"]["time_basis_sigma_scale"]),
        "temporal_init_mode": str(cfg["model"]["temporal_init_mode"]),
        "dynamic_centers": bool(cfg["model"]["dynamic_centers"]),
        "dynamic_radii": bool(cfg["model"]["dynamic_radii"]),
        "dynamic_densities": bool(cfg["model"]["dynamic_densities"]),
        "dynamic_features": bool(cfg["model"]["dynamic_features"]),
        "dynamic_normals": bool(cfg["model"]["dynamic_normals"]),
        "dynamic_texel_sites": bool(cfg["model"]["dynamic_texel_sites"]),
        "xy_extent": float(cfg["model"]["xy_extent"]),
        "z_min": float(cfg["model"]["z_min"]),
        "z_max": float(cfg["model"]["z_max"]),
        "radius_init": float(cfg["model"]["radius_init"]),
        "radius_min": float(cfg["model"]["radius_min"]),
        "radius_scale": float(cfg["model"]["radius_scale"]),
        "density_init": float(cfg["model"]["density_init"]),
        "normal_init_jitter": float(cfg["model"]["normal_init_jitter"]),
        "num_texel_sites": int(cfg["model"]["num_texel_sites"]),
        "texel_site_scale": float(cfg["model"]["texel_site_scale"]),
        "color_init_mode": str(cfg["model"]["color_init_mode"]),
        "video_init_mode": str(cfg["model"]["video_init_mode"]),
        "seed": int(cfg["train"]["seed"]),
        "init_frames": None,
        "image_init_depth": None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"]),
        "image_init_jitter": float(cfg["model"]["image_init_jitter"]),
        "raster_config": make_dynamic_powerfoam_metal_raster_config(cfg["render"]),
        "camera_decoder": camera_decoder,
    }
    if str(cfg["model"]["dynamic_mode"]) == TOKEN_RBF_FEATURE_MODE:
        model = TokenDynamicPowerFoamFeatures(
            **model_kwargs,
            feature_dim=int(cfg["model"]["feature_dim"]),
            feature_init_noise=float(cfg["model"]["feature_init_noise"]),
            feature_rgb_init=str(cfg["model"]["feature_rgb_init"]),
            token_dim=int(cfg["model"]["token_dim"]),
            token_hidden_dim=int(cfg["model"]["token_hidden_dim"]),
            token_hidden_layers=int(cfg["model"]["token_hidden_layers"]),
            token_init_std=float(cfg["model"]["token_init_std"]),
            token_output_init_std=float(cfg["model"]["token_output_init_std"]),
            token_point_residual_scale=float(cfg["model"]["token_point_residual_scale"]),
            token_z_residual_scale=float(cfg["model"]["token_z_residual_scale"]),
            token_radius_residual_scale=float(cfg["model"]["token_radius_residual_scale"]),
            token_density_residual_scale=float(cfg["model"]["token_density_residual_scale"]),
            token_feature_residual_scale=float(cfg["model"]["token_feature_residual_scale"]),
            token_normal_residual_scale=float(cfg["model"]["token_normal_residual_scale"]),
            token_texel_site_residual_scale=float(cfg["model"]["token_texel_site_residual_scale"]),
            token_temporal_residual_scale=float(cfg["model"]["token_temporal_residual_scale"]),
            static_dynamic_split=bool(cfg["model"]["static_dynamic_split"]),
            dynamic_cells=None if cfg["model"]["dynamic_cells"] is None else int(cfg["model"]["dynamic_cells"]),
            dynamic_cell_fraction=float(cfg["model"]["dynamic_cell_fraction"]),
        )
    else:
        model = DynamicMetalPowerFoamVideo(
            **model_kwargs,
            dynamic_mode=str(cfg["model"]["dynamic_mode"]),
        )
    model = model.to(device)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    incompat = model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=False)
    unexpected = list(incompat.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys for dynamic_powerfoam_metal model: {unexpected[:8]}")
    model.eval()
    with torch.no_grad():
        points = model.decoded_parameters()[0].detach().cpu().float()
        cameras = () if camera_decoder is None else camera_decoder.cameras()
    return points, cameras


def load_scene(
    config_path: Path,
    checkpoint_path: Path,
    *,
    frame_count: int | None,
    frame_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[CameraSpec, ...], dict[str, Any]]:
    config = load_config_file(config_path)
    frame_count = _infer_frame_count(config, frame_count)
    checkpoint_cpu = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state = model_state_dict_from_checkpoint(checkpoint_cpu)
    model_cfg = _normalized_model_cfg(config)
    points = _decode_points_from_state_dict(state, model_cfg)
    cameras: tuple[CameraSpec, ...] = ()
    try:
        decoded_points, cameras = _decoded_scene_from_model(
            config,
            checkpoint_path,
            frame_count=frame_count,
            device=device,
        )
        if decoded_points is not None:
            points = decoded_points
    except Exception as exc:
        if points is None:
            raise RuntimeError(
                "Could not decode points by state_dict keys, and model forward decoding failed. "
                "This utility currently supports no-input implicit/free-bank checkpoints and PowerFoam-style raw_xy/raw_z checkpoints."
            ) from exc
        print(f"warning: model camera decode failed; drawing points only ({exc})")
    if points is None:
        raise ValueError("Could not find learned point parameters in the checkpoint.")
    if points.ndim == 3:
        frame = max(0, min(int(frame_index), int(points.shape[0]) - 1))
        points = points[frame]
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected decoded points with shape [N, 3], got {tuple(points.shape)}.")
    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "point_count": int(points.shape[0]),
        "camera_count": len(cameras),
    }
    if isinstance(checkpoint_cpu, dict) and "step" in checkpoint_cpu:
        metadata["checkpoint_step"] = int(checkpoint_cpu["step"])
    return points, cameras, metadata


def sample_point_cloud(points: torch.Tensor, fraction: float, *, seed: int, min_points: int = 1) -> torch.Tensor:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {tuple(points.shape)}.")
    finite = torch.isfinite(points).all(dim=-1)
    points = points[finite].detach().cpu().float()
    if points.numel() == 0:
        raise ValueError("No finite points to plot.")
    fraction = min(max(float(fraction), 0.0), 1.0)
    sample_count = max(int(min_points), int(round(points.shape[0] * fraction)))
    sample_count = min(sample_count, int(points.shape[0]))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return points.index_select(0, torch.randperm(points.shape[0], generator=generator)[:sample_count])


def camera_frustum_lines(cameras: tuple[CameraSpec, ...], *, length: float, width: float) -> FrustumLines:
    segments = []
    centers = []
    directions = []
    for camera in cameras:
        c2w = camera.camera_to_world.detach().cpu().float()
        center = c2w[:3, 3]
        right = c2w[:3, 0]
        up = c2w[:3, 1]
        forward = c2w[:3, 2]
        far_center = center + forward * float(length)
        corners = torch.stack(
            [
                far_center - right * width - up * width,
                far_center + right * width - up * width,
                far_center + right * width + up * width,
                far_center - right * width + up * width,
            ],
            dim=0,
        )
        for corner in corners:
            segments.append(torch.stack([center, corner], dim=0))
        for start, end in zip(corners, torch.roll(corners, shifts=-1, dims=0), strict=True):
            segments.append(torch.stack([start, end], dim=0))
        centers.append(center)
        directions.append(forward)
    if not segments:
        empty_segments = torch.empty((0, 2, 3), dtype=torch.float32)
        empty_vectors = torch.empty((0, 3), dtype=torch.float32)
        return FrustumLines(empty_segments, empty_vectors, empty_vectors)
    return FrustumLines(torch.stack(segments, dim=0), torch.stack(centers, dim=0), torch.stack(directions, dim=0))


def _plot_top_down(
    points: torch.Tensor,
    cameras: tuple[CameraSpec, ...],
    output_path: Path,
    *,
    fraction: float,
    seed: int,
    title: str,
    metadata: dict[str, Any],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    sampled = sample_point_cloud(points, fraction, seed=seed)
    span = (sampled.max(dim=0).values - sampled.min(dim=0).values).max().item()
    frustum_length = max(span * 0.18, 0.15)
    frustum_width = frustum_length * 0.35
    frustums = camera_frustum_lines(cameras, length=frustum_length, width=frustum_width)

    projected = [sampled[:, [0, 2]]]
    if frustums.segments.numel() > 0:
        projected.append(frustums.segments.reshape(-1, 3)[:, [0, 2]])
        projected.append(frustums.centers[:, [0, 2]])
    bounds = torch.cat(projected, dim=0)
    mins = bounds.min(dim=0).values
    maxs = bounds.max(dim=0).values
    center = (mins + maxs) * 0.5
    extent = float((maxs - mins).max().clamp_min(1.0e-6)) * 1.12
    image_size = 1400
    margin = 80
    drawable = image_size - 2 * margin

    def xy_to_pixel(values: torch.Tensor) -> tuple[int, int]:
        xy = (values - center) / extent + 0.5
        return int(margin + float(xy[0]) * drawable), int(margin + (1.0 - float(xy[1])) * drawable)

    def point_color(y_value: float, y_min: float, y_max: float) -> tuple[int, int, int]:
        t = 0.0 if y_max <= y_min else (float(y_value) - y_min) / (y_max - y_min)
        t = min(max(t, 0.0), 1.0)
        return int(58 + 68 * t), int(88 + 142 * t), int(160 - 96 * t)

    image = Image.new("RGB", (image_size, image_size), (252, 252, 249))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for index in range(6):
        x = margin + index * drawable / 5.0
        y = margin + index * drawable / 5.0
        draw.line([(x, margin), (x, image_size - margin)], fill=(225, 225, 220), width=1)
        draw.line([(margin, y), (image_size - margin, y)], fill=(225, 225, 220), width=1)
    draw.rectangle((margin, margin, image_size - margin, image_size - margin), outline=(170, 170, 165), width=1)

    y_min = float(sampled[:, 1].min())
    y_max = float(sampled[:, 1].max())
    for point in sampled:
        x, y = xy_to_pixel(point[[0, 2]])
        color = point_color(float(point[1]), y_min, y_max)
        draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=color)

    for segment in frustums.segments:
        draw.line([xy_to_pixel(segment[0, [0, 2]]), xy_to_pixel(segment[1, [0, 2]])], fill=(210, 42, 42), width=3)
    if frustums.centers.numel() > 0:
        for center_3d, direction_3d in zip(frustums.centers, frustums.directions, strict=True):
            center_px = xy_to_pixel(center_3d[[0, 2]])
            tip_px = xy_to_pixel((center_3d + direction_3d * frustum_length)[[0, 2]])
            draw.ellipse((center_px[0] - 5, center_px[1] - 5, center_px[0] + 5, center_px[1] + 5), fill=(210, 42, 42))
            draw.line([center_px, tip_px], fill=(210, 42, 42), width=4)

    draw.text((margin, 28), title, fill=(20, 20, 20), font=font)
    draw.text((margin, image_size - 52), "world x / world z top-down projection", fill=(80, 80, 76), font=font)
    draw.text(
        (margin, image_size - 32),
        json.dumps(
            {
                "points_plotted": int(sampled.shape[0]),
                "point_count": metadata["point_count"],
                "camera_count": metadata["camera_count"],
            },
            sort_keys=True,
        ),
        fill=(80, 80, 76),
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a top-down learned-point/camera-frustum diagnostic PNG from a checkpoint."
    )
    parser.add_argument("config", type=Path, help="JSONC train config used by the checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint containing a model state_dict.")
    parser.add_argument("--output", type=Path, default=None, help="PNG output path.")
    parser.add_argument("--device", default="cpu", help="Device for optional model decode.")
    parser.add_argument("--frame-count", type=int, default=None, help="Number of camera times to decode when available.")
    parser.add_argument("--frame-index", type=int, default=0, help="Point frame to plot for per-frame point banks.")
    parser.add_argument("--point-fraction", type=float, default=0.05, help="Fraction of learned points to plot.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic point subsampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (args.checkpoint.parent / "camera_scene_topdown.png")
    points, cameras, metadata = load_scene(
        args.config,
        args.checkpoint,
        frame_count=args.frame_count,
        frame_index=args.frame_index,
        device=torch.device(str(args.device)),
    )
    _plot_top_down(
        points,
        cameras,
        output,
        fraction=args.point_fraction,
        seed=args.seed,
        title="Top-down learned points and camera frustums",
        metadata=metadata,
    )
    metadata["output"] = str(output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
