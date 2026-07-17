from __future__ import annotations

import pytest
import torch

from star_uvt_feature_targets import adapt_rgb_to_grid, upsample_grid_rgb
from star_uvt_feature_rgb_probe_config import resolve_config as resolve_rgb_probe_config
from star_uvt_rendered_feature_probe_config import resolve_config as resolve_rendered_probe_config
from star_uvt_rendered_feature_probe_objective import (
    _stratified_grid_pixel_ids_for_chunk,
    compose_sparse_rgb,
    gather_target_rgb_values,
    sparse_rgb_loss_and_grads,
)


def test_rgb_grid_downsample_and_upsample_shapes() -> None:
    frames = torch.linspace(0.0, 1.0, steps=4 * 3 * 6 * 8).reshape(4, 3, 6, 8)

    grid = adapt_rgb_to_grid(frames, target_shape=(2, 3, 4), mode="trilinear")
    restored = upsample_grid_rgb(grid, target_shape=(4, 6, 8), mode="trilinear")

    assert list(grid.shape) == [2, 3, 3, 4]
    assert list(restored.shape) == list(frames.shape)
    assert float(grid.min()) >= 0.0
    assert float(grid.max()) <= 1.0


def test_rgb_grid_nearest_uses_valid_samples() -> None:
    frames = torch.zeros((4, 3, 6, 8))
    frames[2] = 1.0

    grid = adapt_rgb_to_grid(frames, target_shape=(2, 3, 4), mode="nearest")

    assert set(grid.unique().tolist()) <= {0.0, 1.0}


def test_probe_config_requires_target_grid_materialization() -> None:
    cfg = {
        "data": {},
        "features": {},
        "feature_target": {"enabled": True, "materialization": "cached_chunks"},
        "feature_uvt": {"feature_dim": 32},
        "probe": {
            "steps": 1,
            "lr": 0.01,
            "device": "cpu",
            "seed": 1,
            "target_rgb_adapter": "trilinear",
            "require_loss_decrease": True,
        },
        "colorize": {
            "hidden_dim": None,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": None,
            "checkpoint": None,
            "contact_sheet": None,
            "contact_sheet_frames": 1,
            "contact_sheet_mode": "linspace",
            "side_by_side_video": None,
            "side_by_side_fps": None,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "x",
            "wandb_run_name": "x",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }

    with pytest.raises(ValueError, match="target_grid"):
        resolve_rgb_probe_config(cfg)


def _rendered_probe_cfg() -> dict:
    return {
        "data": {
            "video_path": "dummy.mp4",
            "start_seconds": 0.0,
            "fps": 8.0,
            "duration_seconds": 1.0,
            "image_crop_mode": "center",
            "target_size": 8,
            "max_frames": 4,
        },
        "probe": {
            "steps": 1,
            "lr": 0.01,
            "device": "mps",
            "seed": 1,
            "frame_chunk_size": 2,
            "resume_checkpoint": "checkpoint.pt",
            "pixel_source": "stratified_grid",
            "sample_grid_shape": [4, 2, 2],
            "sample_grid_adapter": "trilinear",
            "require_loss_decrease": True,
        },
        "feature_uvt": {
            "tube_count": 16,
            "feature_dim": 4,
            "tile_t": 4,
            "tile_capacity": 64,
            "alpha_threshold": 0.0,
            "max_alpha": 1.0,
        },
        "colorize": {
            "hidden_dim": None,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": None,
            "checkpoint": None,
            "contact_sheet": None,
            "contact_sheet_frames": 1,
            "contact_sheet_mode": "linspace",
            "side_by_side_video": None,
            "side_by_side_fps": None,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "x",
            "wandb_run_name": "x",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }


def test_rendered_feature_probe_config_defaults_and_requires_resume_checkpoint() -> None:
    cfg = _rendered_probe_cfg()
    resolved = resolve_rendered_probe_config(cfg)
    assert resolved["probe"]["train_star_model"] is False
    assert resolved["probe"]["train_colorizer"] is True
    assert resolved["probe"]["colorizer_init_checkpoint"] is None

    missing_resume = _rendered_probe_cfg()
    missing_resume["probe"]["resume_checkpoint"] = None
    with pytest.raises(ValueError, match="resume_checkpoint"):
        resolve_rendered_probe_config(missing_resume)


def test_rendered_feature_probe_config_bounds_stratified_grid_shape() -> None:
    cfg = _rendered_probe_cfg()
    cfg["probe"]["sample_grid_shape"] = [5, 2, 2]

    with pytest.raises(ValueError, match="sample_grid_shape"):
        resolve_rendered_probe_config(cfg)


def test_rendered_feature_probe_gathers_target_values() -> None:
    target = torch.arange(2 * 3 * 2 * 3, dtype=torch.float32).reshape(2, 3, 2, 3)
    gathered = gather_target_rgb_values(target, torch.tensor([0, 5, 6, 11], dtype=torch.int32))

    expected = target.permute(0, 2, 3, 1).reshape(-1, 3)[[0, 5, 6, 11]]
    assert torch.equal(gathered, expected)


def test_rendered_feature_probe_stratified_grid_pixel_ids_are_chunk_local() -> None:
    ids = _stratified_grid_pixel_ids_for_chunk(
        chunk_frames=2,
        height=8,
        width=8,
        render_frames=4,
        frame_start=2,
        sample_grid_shape=(4, 2, 2),
        device=torch.device("cpu"),
    )

    assert ids.tolist() == [18, 22, 50, 54, 82, 86, 114, 118]


def test_rendered_feature_probe_stratified_grid_empty_chunk_keeps_int32() -> None:
    ids = _stratified_grid_pixel_ids_for_chunk(
        chunk_frames=1,
        height=8,
        width=8,
        render_frames=4,
        frame_start=0,
        sample_grid_shape=(2, 2, 2),
        device=torch.device("cpu"),
    )

    assert ids.dtype == torch.int32
    assert ids.numel() == 0


def test_rendered_feature_probe_sparse_compose_uses_alpha() -> None:
    colorizer = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
    with torch.no_grad():
        colorizer.weight.zero_()
        colorizer.weight[0, 0, 0, 0] = 1.0
        colorizer.weight[1, 1, 0, 0] = 1.0
    feature_values = torch.tensor([[0.25, 0.5], [1.0, 0.0]], dtype=torch.float32)
    alpha_values = torch.tensor([0.5, 0.25], dtype=torch.float32)

    composed = compose_sparse_rgb(feature_values, alpha_values, colorizer)  # type: ignore[arg-type]

    assert composed.shape == (2, 3)
    assert torch.allclose(composed[0], torch.tensor([0.125, 0.25, 0.0]))
    assert torch.allclose(composed[1], torch.tensor([0.25, 0.0, 0.0]))


def test_rendered_feature_probe_sparse_rgb_loss_returns_local_grads() -> None:
    colorizer = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
    with torch.no_grad():
        colorizer.weight.zero_()
        colorizer.weight[0, 0, 0, 0] = 1.0
        colorizer.weight[1, 1, 0, 0] = 1.0
    feature_values = torch.tensor([[0.25, 0.5], [1.0, 0.0]], dtype=torch.float32)
    alpha_values = torch.tensor([0.5, 0.25], dtype=torch.float32)
    target_values = torch.zeros((2, 3), dtype=torch.float32)

    loss, grad_feature, grad_alpha = sparse_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        colorizer,  # type: ignore[arg-type]
        total_loss_elems=6,
    )

    assert float(loss) > 0.0
    assert grad_feature.shape == feature_values.shape
    assert grad_alpha.shape == alpha_values.shape
    assert float(grad_feature.abs().sum()) > 0.0
    assert float(grad_alpha.abs().sum()) > 0.0
