from __future__ import annotations

from typing import Any

import torch

from multicam_video_data import cameras_from_K_w2c, heldout_cameras_from_K_w2c, load_multicam_video_bundle
from paper_training_protocol import normalize_image_size
from powerfoam_geometry import powerfoam_rays_from_camera_grid
from powerfoam_training import flatten_multiview_powerfoam_samples
from sequence_data import load_video_sequence


def load_powerfoam_training_data(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    render_size = int(cfg["render"]["render_size"])
    image_size = normalize_image_size(cfg["render"]["image_size"])
    frame_source = str(cfg["data"]["frame_source"])
    if frame_source == "multicam_val":
        bundle = load_multicam_video_bundle(
            data_cfg=cfg["data"],
            camera_cfg=cfg["camera"],
            target_size=(image_size.height, image_size.width),
            device=device,
        )
        train_cameras = cameras_from_K_w2c(
            bundle.train_K,
            bundle.train_w2c,
            lens_models=bundle.train_lens_models,
            distortions=bundle.train_distortions,
        )
        train_rays = powerfoam_rays_from_camera_grid(
            train_cameras,
            height=image_size.height,
            width=image_size.width,
            device=device,
        )
        targets, sample_frame_indices, sample_rays = flatten_multiview_powerfoam_samples(
            bundle.train_frames.to(device=device, dtype=torch.float32),
            train_rays,
        )

        heldout_targets = None
        heldout_frame_indices = None
        heldout_rays = None
        if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
            heldout_camera_grid = heldout_cameras_from_K_w2c(
                bundle.heldout_K,
                bundle.heldout_w2c,
                lens_models=bundle.heldout_lens_models,
                distortions=bundle.heldout_distortions,
            )
            heldout_ray_grid = powerfoam_rays_from_camera_grid(
                heldout_camera_grid,
                height=image_size.height,
                width=image_size.width,
                device=device,
            )
            heldout_targets, heldout_frame_indices, heldout_rays = flatten_multiview_powerfoam_samples(
                bundle.heldout_frames.to(device=device, dtype=torch.float32),
                heldout_ray_grid,
            )

        return {
            "targets": targets,
            "sample_frame_indices": sample_frame_indices,
            "sample_rays": sample_rays,
            "heldout_targets": heldout_targets,
            "heldout_frame_indices": heldout_frame_indices,
            "heldout_rays": heldout_rays,
            "init_frames": bundle.condition_sequence.frames.detach().cpu(),
            "frame_count": bundle.frame_count,
            "train_view_count": bundle.train_view_count,
            "video_fps": float(bundle.condition_sequence.video_fps),
            "source_label": str(bundle.metadata.get("sample_id")) if bundle.metadata else "multicam_val",
            "train_views": bundle.train_camera_names,
            "heldout_views": bundle.heldout_camera_names or [],
            "pose_source": bundle.pose_source,
            "world_to_model": None
            if bundle.anchor_c2w is None
            else torch.linalg.inv(bundle.anchor_c2w.detach().to(device="cpu", dtype=torch.float32)),
            "point_cloud_visibility_train_K": bundle.train_K.detach().to(device="cpu", dtype=torch.float32),
            "point_cloud_visibility_train_w2c": bundle.train_w2c.detach().to(device="cpu", dtype=torch.float32),
            "point_cloud_visibility_train_lens_models": bundle.train_lens_models,
            "point_cloud_visibility_train_distortions": None
            if bundle.train_distortions is None
            else bundle.train_distortions.detach().to(device="cpu", dtype=torch.float32),
        }

    if cfg["data"]["video_path"] is None:
        raise ValueError("data.video_path is required unless data.frame_source is 'multicam_val'.")
    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=render_size,
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=frame_source,
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    return {
        "targets": targets,
        "sample_frame_indices": torch.arange(targets.size(0), device=device, dtype=torch.long),
        "sample_rays": None,
        "heldout_targets": None,
        "heldout_frame_indices": None,
        "heldout_rays": None,
        "init_frames": targets.detach().cpu(),
        "frame_count": int(targets.size(0)),
        "train_view_count": 1,
        "video_fps": float(sequence.video_fps),
        "source_label": str(cfg["data"]["video_path"]),
        "train_views": [],
        "heldout_views": [],
        "pose_source": None,
        "world_to_model": None,
        "point_cloud_visibility_train_K": None,
        "point_cloud_visibility_train_w2c": None,
        "point_cloud_visibility_train_lens_models": None,
        "point_cloud_visibility_train_distortions": None,
    }
