import torch
import torch.nn as nn
from camera import build_plucker_ray_grid, build_plucker_ray_grid_batch, make_default_camera
from runtime_types import GaussianSequence

from .blocks import TokenGSBackbone


class DynamicTokenGS(TokenGSBackbone):
    def __init__(
        self,
        num_tokens=128,
        feat_dim=128,
        gaussians_per_token=4,
        xy_extent=1.5,
        z_min=0.5,
        z_max=2.5,
        scale_init=0.05,
        scale_init_log_jitter=0.0,
        opacity_init=None,
        token_init_std=1.0,
        head_hidden_dim=64,
        head_hidden_layers=1,
        head_output_init_std=None,
        position_init_extent_coverage=0.0,
        rotation_init="random",
    ):
        super().__init__(
            num_tokens=num_tokens,
            feat_dim=feat_dim,
            gaussians_per_token=gaussians_per_token,
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
            scale_init=scale_init,
            scale_init_log_jitter=scale_init_log_jitter,
            opacity_init=opacity_init,
            token_init_std=token_init_std,
            head_hidden_dim=head_hidden_dim,
            head_hidden_layers=head_hidden_layers,
            head_output_init_std=head_output_init_std,
            position_init_extent_coverage=position_init_extent_coverage,
            rotation_init=rotation_init,
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def _build_plucker_batch(self, camera, batch_size, image_size, device):
        if camera is None:
            default_camera = make_default_camera(image_size=image_size, device=device)
            return build_plucker_ray_grid(
                default_camera, image_size=image_size, device=device, channels_first=True
            ).expand(batch_size, -1, -1, -1)
        if isinstance(camera, (list, tuple)):
            if len(camera) != batch_size:
                raise ValueError(f"Expected {batch_size} cameras, got {len(camera)}")
            return build_plucker_ray_grid_batch(camera, image_size=image_size, device=device, channels_first=True)
        return build_plucker_ray_grid(camera, image_size=image_size, device=device, channels_first=True).expand(
            batch_size, -1, -1, -1
        )

    def forward(self, img, camera=None, frame_times=None):
        batch_size = img.shape[0]
        _, _, _, width = img.shape
        plucker_grid = self._build_plucker_batch(
            camera=camera,
            batch_size=batch_size,
            image_size=width,
            device=img.device,
        )

        if frame_times is None:
            frame_times = torch.zeros((batch_size, 1), device=img.device, dtype=img.dtype)
        else:
            frame_times = frame_times.to(device=img.device, dtype=img.dtype).reshape(batch_size, 1)
        time_offsets = self.time_proj(frame_times).unsqueeze(1)
        xyz, scales, quats, opacities, rgbs = self.predict_gaussians(
            img,
            plucker_grid,
            token_offsets=time_offsets,
        )
        return GaussianSequence(xyz=xyz, scales=scales, quats=quats, opacities=opacities, rgbs=rgbs)
