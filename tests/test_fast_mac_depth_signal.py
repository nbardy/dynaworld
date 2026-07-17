from __future__ import annotations

import torch
import pytest

from renderers.fast_mac import (
    FastMacRendererConfig,
    _make_v5_softmax_gs_config,
    _rasterize_rgb_projected,
    describe_fast_mac_depth_signal,
    project_for_fast_mac,
    project_for_fast_mac_batch,
)


def _projection_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    scales = torch.full((3, 3), 0.1, dtype=torch.float32)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float32)
    opacities = torch.full((3, 1), 0.5, dtype=torch.float32)
    rgbs = torch.eye(3, dtype=torch.float32)
    return means, scales, quats, opacities, rgbs


def _project(depth_mode: str):
    return project_for_fast_mac(
        *_projection_inputs(),
        8.0,
        8.0,
        4.0,
        4.0,
        depth_mode=depth_mode,
    )


def test_fast_mac_depth_mode_defaults_to_rank_depth() -> None:
    config = FastMacRendererConfig.from_mapping(
        {},
        fallback_tile_size=16,
        fallback_alpha_threshold=1.0 / 255.0,
    )

    assert describe_fast_mac_depth_signal(config) == {
        "kind": "rank_depth",
        "softmax_gs_ready": False,
    }


def test_project_for_fast_mac_can_return_center_camera_z_depths() -> None:
    _means2d, _conics, colors, _opacities, depths = _project("center_camera_z")

    torch.testing.assert_close(depths, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    torch.testing.assert_close(colors, torch.eye(3, dtype=torch.float32)[torch.tensor([1, 2, 0])])


def test_project_for_fast_mac_default_rank_depths_are_unchanged() -> None:
    _means2d, _conics, _colors, _opacities, depths = _project("rank_depth")

    torch.testing.assert_close(depths, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32))


def test_project_for_fast_mac_batch_can_return_center_camera_z_depths() -> None:
    means, scales, quats, opacities, rgbs = _projection_inputs()
    means_batch = torch.stack([means, means + torch.tensor([0.0, 0.0, 2.0])], dim=0)

    _means2d, _conics, _colors, _opacities, depths = project_for_fast_mac_batch(
        means_batch,
        scales.expand(2, -1, -1),
        quats.expand(2, -1, -1),
        opacities.expand(2, -1, -1),
        rgbs.expand(2, -1, -1),
        8.0,
        8.0,
        4.0,
        4.0,
        depth_mode="center_camera_z",
    )

    expected = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=torch.float32)
    torch.testing.assert_close(depths, expected)


def test_fast_mac_accepts_softmax_gs_noop_rgb_variant() -> None:
    config = FastMacRendererConfig.from_mapping(
        {
            "rgb_variant": "v5_softmax_gs",
            "softmax_gs_beta": 4.0,
            "softmax_gs_gamma": 3.0,
        },
        fallback_tile_size=16,
        fallback_alpha_threshold=1.0 / 255.0,
    )

    raster_config = _make_v5_softmax_gs_config(config, height=8, width=8)

    assert config.rgb_variant == "v5_softmax_gs"
    assert raster_config.softmax_gs_enabled is False
    assert raster_config.softmax_gs_beta == 4.0
    assert raster_config.softmax_gs_gamma == 3.0


def test_softmax_gs_enabled_requires_center_camera_depth() -> None:
    config = FastMacRendererConfig.from_mapping(
        {
            "rgb_variant": "v5_softmax_gs",
            "softmax_gs_enabled": True,
            "depth_mode": "rank_depth",
        },
        fallback_tile_size=16,
        fallback_alpha_threshold=1.0 / 255.0,
    )
    means2d = torch.zeros((1, 2), dtype=torch.float32)
    conics = torch.ones((1, 3), dtype=torch.float32)
    colors = torch.zeros((1, 3), dtype=torch.float32)
    opacities = torch.ones((1,), dtype=torch.float32)
    depths = torch.zeros((1,), dtype=torch.float32)

    with pytest.raises(ValueError, match="center_camera_z"):
        _rasterize_rgb_projected(means2d, conics, colors, opacities, depths, config, height=8, width=8)
