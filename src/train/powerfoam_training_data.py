from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from multicam_video_data import cameras_from_K_w2c, heldout_cameras_from_K_w2c, load_multicam_video_bundle
from paper_training_protocol import normalize_image_size
from camera import CameraSpec
from powerfoam_geometry import powerfoam_rays_from_camera, powerfoam_rays_from_camera_grid
from powerfoam_training import flatten_multiview_powerfoam_samples
from sequence_data import load_video_sequence


@dataclass(frozen=True)
class PowerFoamRayProvider:
    cameras: tuple[tuple[CameraSpec, ...], ...]
    height: int
    width: int
    device: torch.device

    @property
    def view_count(self) -> int:
        return len(self.cameras)

    @property
    def frame_count(self) -> int:
        return len(self.cameras[0]) if self.cameras else 0

    @property
    def sample_count(self) -> int:
        return self.view_count * self.frame_count

    def select(self, sample_indices: torch.Tensor) -> torch.Tensor:
        flat_indices = sample_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        if not flat_indices:
            raise ValueError("PowerFoam ray selection requires at least one sample")
        rays = []
        for index in flat_indices:
            if index < 0 or index >= self.sample_count:
                raise IndexError(f"PowerFoam ray sample index {index} is outside [0, {self.sample_count})")
            view, frame = divmod(int(index), self.frame_count)
            rays.append(
                powerfoam_rays_from_camera(
                    self.cameras[view][frame],
                    height=self.height,
                    width=self.width,
                    device=self.device,
                )
            )
        return torch.cat(rays, dim=0)


def load_powerfoam_training_data(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    render_size = int(cfg["render"]["render_size"])
    image_size = normalize_image_size(cfg["render"]["image_size"])
    frame_source = str(cfg["data"]["frame_source"])
    if frame_source == "multicam_val":
        stream_rays = bool(cfg.get("paper_protocol", {}).get("enabled", False))
        bundle = load_multicam_video_bundle(
            data_cfg=cfg["data"],
            camera_cfg=cfg["camera"],
            target_size=(image_size.height, image_size.width),
            device=device,
            frame_device=torch.device("cpu") if stream_rays else device,
        )
        train_cameras = cameras_from_K_w2c(
            bundle.train_K,
            bundle.train_w2c,
            lens_models=bundle.train_lens_models,
            distortions=bundle.train_distortions,
        )
        if stream_rays:
            train_ray_provider = PowerFoamRayProvider(
                cameras=train_cameras,
                height=image_size.height,
                width=image_size.width,
                device=device,
            )
            targets = bundle.train_frames.reshape(
                bundle.train_view_count * bundle.frame_count, *bundle.train_frames.shape[2:]
            ).contiguous()
            sample_frame_indices = torch.arange(bundle.frame_count, device=device, dtype=torch.long).repeat(
                bundle.train_view_count
            )
            sample_rays = None
        else:
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
            train_ray_provider = None

        heldout_targets = None
        heldout_frame_indices = None
        heldout_rays = None
        heldout_ray_provider = None
        if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
            heldout_camera_grid = heldout_cameras_from_K_w2c(
                bundle.heldout_K,
                bundle.heldout_w2c,
                lens_models=bundle.heldout_lens_models,
                distortions=bundle.heldout_distortions,
            )
            if stream_rays:
                heldout_ray_provider = PowerFoamRayProvider(
                    cameras=heldout_camera_grid,
                    height=image_size.height,
                    width=image_size.width,
                    device=device,
                )
                heldout_targets = bundle.heldout_frames.reshape(
                    bundle.heldout_view_count * bundle.frame_count, *bundle.heldout_frames.shape[2:]
                ).contiguous()
                heldout_frame_indices = torch.arange(
                    bundle.frame_count, device=device, dtype=torch.long
                ).repeat(bundle.heldout_view_count)
                heldout_rays = None
            else:
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
            "sample_ray_provider": train_ray_provider,
            "heldout_targets": heldout_targets,
            "heldout_frame_indices": heldout_frame_indices,
            "heldout_rays": heldout_rays,
            "heldout_ray_provider": heldout_ray_provider,
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
        "sample_ray_provider": None,
        "heldout_targets": None,
        "heldout_frame_indices": None,
        "heldout_rays": None,
        "heldout_ray_provider": None,
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
