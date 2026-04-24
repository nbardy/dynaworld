from __future__ import annotations

import torch
import torch.nn.functional as F
from camera import CameraSpec, make_camera_like
from renderers.common import build_pixel_grid
from renderers.dense import render_pytorch_3dgs, render_pytorch_3dgs_batch
from renderers.fast_mac import FastMacRendererConfig, render_fast_mac_3dgs, render_fast_mac_3dgs_batch
from renderers.taichi import TaichiRendererConfig, render_taichi_3dgs, render_taichi_3dgs_batch
from renderers.tiled import render_pytorch_3dgs_tiled
from runtime_types import GaussianFrame, GaussianSequence, ResolvedRendererMode

CAMERA_PROJECTION_MODES = {"auto", "legacy_pinhole", "camera_model"}


def resize_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    if images.shape[-2:] == (image_size, image_size):
        return images
    leading_shape = images.shape[:-3]
    flattened = images.reshape(-1, *images.shape[-3:])
    resized = F.interpolate(flattened, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.reshape(*leading_shape, *resized.shape[-3:])


def camera_for_viewport(
    camera: CameraSpec,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> CameraSpec:
    """Return the same camera pose with intrinsics expressed in target viewport pixels."""
    if source_height < 1 or source_width < 1:
        raise ValueError(f"source viewport must be positive, got {source_width}x{source_height}.")
    if target_height < 1 or target_width < 1:
        raise ValueError(f"target viewport must be positive, got {target_width}x{target_height}.")

    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    return make_camera_like(
        camera,
        fx=camera.fx * scale_x,
        fy=camera.fy * scale_y,
        cx=camera.cx * scale_x,
        cy=camera.cy * scale_y,
        camera_to_world=camera.camera_to_world,
    )


def pick_renderer_mode(
    renderer: str,
    gaussian_count: int,
    height: int,
    width: int,
    auto_dense_limit: int,
) -> ResolvedRendererMode:
    if renderer == "auto":
        return "dense" if gaussian_count * height * width <= auto_dense_limit else "tiled"
    if renderer == "dense" or renderer == "tiled" or renderer == "taichi" or renderer == "fast_mac":
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


def _normalize_camera_projection_mode(camera_projection: str | None) -> str:
    mode = "auto" if camera_projection is None else str(camera_projection).lower()
    if mode == "legacy":
        mode = "legacy_pinhole"
    if mode not in CAMERA_PROJECTION_MODES:
        raise ValueError(
            f"Unknown render.camera_projection={camera_projection!r}. "
            "Expected auto, legacy_pinhole, or camera_model."
        )
    return mode


def _resolve_camera_projection_mode(
    cameras: CameraSpec | list[CameraSpec] | tuple[CameraSpec, ...],
    camera_projection: str | None,
) -> str:
    mode = _normalize_camera_projection_mode(camera_projection)
    camera_list = list(cameras) if isinstance(cameras, (list, tuple)) else [cameras]
    has_non_pinhole = any(camera.lens_model != "pinhole" for camera in camera_list)
    if mode == "legacy_pinhole" and has_non_pinhole:
        lens_models = sorted({camera.lens_model for camera in camera_list})
        raise ValueError(
            "render.camera_projection='legacy_pinhole' cannot render non-pinhole CameraSpec values "
            f"(lens models: {lens_models}). Use render.camera_projection='camera_model' or 'auto'."
        )
    if mode == "auto":
        return "camera_model" if has_non_pinhole else "legacy_pinhole"
    return mode


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
    near_plane: float = 1.0e-4,
    taichi_options: dict | None = None,
    fast_mac_options: dict | None = None,
    camera_projection: str | None = "auto",
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    projection_mode = _resolve_camera_projection_mode(camera, camera_projection)
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
            near_plane=near_plane,
            return_aux=return_aux,
            camera=camera,
            projection_mode=projection_mode,
        )
    if return_aux:
        raise ValueError("return_aux is only supported by the dense renderer.")
    if mode == "taichi":
        return render_taichi_3dgs(
            frame.xyz,
            frame.scales,
            frame.quats,
            frame.opacities,
            frame.rgbs,
            height,
            width,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera_to_world=camera.camera_to_world,
            near_plane=near_plane,
            camera=camera,
            projection_mode=projection_mode,
            config=TaichiRendererConfig.from_mapping(
                taichi_options,
                fallback_tile_size=tile_size,
                fallback_alpha_threshold=alpha_threshold,
            ),
        )
    if mode == "fast_mac":
        return render_fast_mac_3dgs(
            frame.xyz,
            frame.scales,
            frame.quats,
            frame.opacities,
            frame.rgbs,
            height,
            width,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera_to_world=camera.camera_to_world,
            near_plane=near_plane,
            camera=camera,
            projection_mode=projection_mode,
            config=FastMacRendererConfig.from_mapping(
                fast_mac_options,
                fallback_tile_size=tile_size,
                fallback_alpha_threshold=alpha_threshold,
            ),
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
            near_plane=near_plane,
            camera=camera,
            projection_mode=projection_mode,
        )
    raise ValueError(f"Unknown renderer mode: {mode}")


def _camera_scalar_vector(cameras: list[CameraSpec] | tuple[CameraSpec, ...], field_name: str, device) -> torch.Tensor:
    values = [getattr(camera, field_name) for camera in cameras]
    if any(torch.is_tensor(value) for value in values):
        return torch.stack(
            [
                value.to(device=device, dtype=torch.float32)
                if torch.is_tensor(value)
                else torch.tensor(value, device=device, dtype=torch.float32)
                for value in values
            ],
            dim=0,
        ).reshape(-1)
    return torch.tensor(values, device=device, dtype=torch.float32)


def render_gaussian_frames(
    sequence: GaussianSequence,
    cameras: list[CameraSpec] | tuple[CameraSpec, ...],
    height: int,
    width: int,
    mode: str,
    dense_grid: torch.Tensor | None = None,
    tile_size: int = 8,
    bound_scale: float = 3.0,
    alpha_threshold: float = 1.0 / 255.0,
    near_plane: float = 1.0e-4,
    taichi_options: dict | None = None,
    fast_mac_options: dict | None = None,
    camera_projection: str | None = "auto",
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if sequence.frame_count != len(cameras):
        raise ValueError(f"Expected {sequence.frame_count} cameras, got {len(cameras)}.")

    projection_mode = _resolve_camera_projection_mode(cameras, camera_projection)
    if mode == "dense":
        device = sequence.xyz.device
        return render_pytorch_3dgs_batch(
            sequence.xyz.float(),
            sequence.scales.float(),
            sequence.quats.float(),
            sequence.opacities.float(),
            sequence.rgbs.float(),
            height,
            width,
            _camera_scalar_vector(cameras, "fx", device),
            _camera_scalar_vector(cameras, "fy", device),
            _camera_scalar_vector(cameras, "cx", device),
            _camera_scalar_vector(cameras, "cy", device),
            grid=build_or_reuse_grid(height, width, device, dense_grid),
            camera_to_world=torch.stack(
                [camera.camera_to_world.to(device=device, dtype=torch.float32) for camera in cameras],
                dim=0,
            ),
            near_plane=near_plane,
            return_aux=return_aux,
            cameras=cameras,
            projection_mode=projection_mode,
        )

    if return_aux:
        raise ValueError("return_aux is only supported by the dense renderer.")
    if mode == "taichi":
        device = sequence.xyz.device
        return render_taichi_3dgs_batch(
            sequence.xyz,
            sequence.scales,
            sequence.quats,
            sequence.opacities,
            sequence.rgbs,
            height,
            width,
            _camera_scalar_vector(cameras, "fx", device),
            _camera_scalar_vector(cameras, "fy", device),
            _camera_scalar_vector(cameras, "cx", device),
            _camera_scalar_vector(cameras, "cy", device),
            camera_to_world=torch.stack(
                [camera.camera_to_world.to(device=device, dtype=torch.float32) for camera in cameras],
                dim=0,
            ),
            near_plane=near_plane,
            cameras=cameras,
            projection_mode=projection_mode,
            config=TaichiRendererConfig.from_mapping(
                taichi_options,
                fallback_tile_size=tile_size,
                fallback_alpha_threshold=alpha_threshold,
            ),
        )
    if mode == "fast_mac":
        device = sequence.xyz.device
        return render_fast_mac_3dgs_batch(
            sequence.xyz,
            sequence.scales,
            sequence.quats,
            sequence.opacities,
            sequence.rgbs,
            height,
            width,
            _camera_scalar_vector(cameras, "fx", device),
            _camera_scalar_vector(cameras, "fy", device),
            _camera_scalar_vector(cameras, "cx", device),
            _camera_scalar_vector(cameras, "cy", device),
            camera_to_world=torch.stack(
                [camera.camera_to_world.to(device=device, dtype=torch.float32) for camera in cameras],
                dim=0,
            ),
            near_plane=near_plane,
            cameras=cameras,
            projection_mode=projection_mode,
            config=FastMacRendererConfig.from_mapping(
                fast_mac_options,
                fallback_tile_size=tile_size,
                fallback_alpha_threshold=alpha_threshold,
            ),
        )
    return torch.stack(
        [
            render_gaussian_frame(
                sequence.frame(index),
                camera=camera,
                height=height,
                width=width,
                mode=mode,
                dense_grid=dense_grid,
                tile_size=tile_size,
                bound_scale=bound_scale,
                alpha_threshold=alpha_threshold,
                near_plane=near_plane,
                taichi_options=taichi_options,
                fast_mac_options=fast_mac_options,
                camera_projection=camera_projection,
            )
            for index, camera in enumerate(cameras)
        ],
        dim=0,
    )


__all__ = [
    "ResolvedRendererMode",
    "build_or_reuse_grid",
    "camera_for_viewport",
    "pick_renderer_mode",
    "render_gaussian_frame",
    "render_gaussian_frames",
    "resize_images",
]
