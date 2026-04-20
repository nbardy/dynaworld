from __future__ import annotations

import torch
import torch.nn.functional as F
from camera import CameraSpec
from renderers.common import build_pixel_grid
from renderers.dense import render_pytorch_3dgs
from renderers.tiled import render_pytorch_3dgs_tiled
from runtime_types import GaussianFrame, ResolvedRendererMode


def resize_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    if images.shape[-2:] == (image_size, image_size):
        return images
    leading_shape = images.shape[:-3]
    flattened = images.reshape(-1, *images.shape[-3:])
    resized = F.interpolate(flattened, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.reshape(*leading_shape, *resized.shape[-3:])


def pick_renderer_mode(
    renderer: str,
    gaussian_count: int,
    height: int,
    width: int,
    auto_dense_limit: int,
) -> ResolvedRendererMode:
    if renderer == "auto":
        return "dense" if gaussian_count * height * width <= auto_dense_limit else "tiled"
    if renderer == "dense" or renderer == "tiled":
        return renderer
    raise ValueError(f"Unknown renderer mode: {renderer}")


def build_or_reuse_grid(
    height: int,
    width: int,
    device: torch.device | str,
    grid: torch.Tensor | None = None,
) -> torch.Tensor:
    resolved_device = torch.device(device)
    if grid is not None and tuple(grid.shape) == (height, width, 2) and grid.device == resolved_device:
        return grid
    return build_pixel_grid(height, width, resolved_device)


def render_gaussian_frame(
    frame: GaussianFrame,
    camera: CameraSpec,
    height: int,
    width: int,
    mode: str,
    dense_grid: torch.Tensor | None = None,
    tile_size: int = 8,
    bound_scale: float = 3.0,
    alpha_threshold: float = 1.0 / 255.0,
) -> torch.Tensor:
    if mode == "dense":
        return render_pytorch_3dgs(
            frame.xyz.float(),
            frame.scales.float(),
            frame.quats.float(),
            frame.opacities.float(),
            frame.rgbs.float(),
            height,
            width,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            grid=build_or_reuse_grid(height, width, frame.xyz.device, dense_grid),
            camera_to_world=camera.camera_to_world.float(),
        )
    if mode == "tiled":
        return render_pytorch_3dgs_tiled(
            frame.xyz.float(),
            frame.scales.float(),
            frame.quats.float(),
            frame.opacities.float(),
            frame.rgbs.float(),
            height,
            width,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            tile_size=tile_size,
            bound_scale=bound_scale,
            alpha_threshold=alpha_threshold,
            camera_to_world=camera.camera_to_world.float(),
        )
    raise ValueError(f"Unknown renderer mode: {mode}")


__all__ = [
    "ResolvedRendererMode",
    "build_or_reuse_grid",
    "pick_renderer_mode",
    "render_gaussian_frame",
    "resize_images",
]
