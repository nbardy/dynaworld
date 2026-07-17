from __future__ import annotations

import torch

from camera import CameraSpec
from rendering import render_gaussian_frames_rasterized
from runtime_types import GaussianSequence, RasterizedClip


def _cfg() -> dict[str, object]:
    return {
        "render": {
            "render_size": 4,
            "tile_size": 2,
            "bound_scale": 3.0,
            "alpha_threshold": 1.0 / 255.0,
            "near_plane": 0.01,
            "camera_projection": "legacy_pinhole",
            "fast_mac": None,
        }
    }


def _sequence() -> GaussianSequence:
    return GaussianSequence(
        xyz=torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32),
        scales=torch.tensor([[[0.1, 0.1, 0.1]]], dtype=torch.float32),
        quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        opacities=torch.tensor([[[0.5]]], dtype=torch.float32),
        rgbs=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
    )


def _camera() -> CameraSpec:
    return CameraSpec(
        fx=4.0,
        fy=4.0,
        cx=2.0,
        cy=2.0,
        camera_to_world=torch.eye(4, dtype=torch.float32),
    )


def test_render_gaussian_frames_rasterized_returns_typed_dense_payload() -> None:
    rasterized = render_gaussian_frames_rasterized(_cfg(), _sequence(), (_camera(),), mode="dense")

    assert isinstance(rasterized, RasterizedClip)
    assert rasterized.features.shape == (1, 3, 4, 4)
    assert rasterized.alpha is None
