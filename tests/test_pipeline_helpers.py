from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pipeline.render import gaussian_sequence_slice, stack_complete_frame_list
from pipeline.validation_media import (
    compose_multicam_feature_gt_render_grid,
    compose_gt_pred_alpha_pca_grid,
    render_diagnostics_payload,
)
from runtime_types import GaussianSequence, RasterizedClip, RenderedClip, SequenceData
from sequence_data import prepare_clip


def test_compose_gt_pred_alpha_pca_grid_returns_only_diagnostic_composites() -> None:
    gt = torch.zeros(2, 3, 4, 5)
    pred = torch.ones(2, 3, 4, 5)

    assert compose_gt_pred_alpha_pca_grid(gt=gt, pred=pred) is None

    alpha_video = torch.full_like(gt, 0.25)
    pca_video = torch.full_like(gt, 0.75)
    grid = compose_gt_pred_alpha_pca_grid(
        gt=gt,
        pred=pred,
        alpha_video=alpha_video,
        pca_video=pca_video,
    )

    assert grid is not None
    assert grid.shape == (2, 3, 4, 20)
    assert torch.equal(grid[..., :5], gt)
    assert torch.equal(grid[..., 5:10], pred)
    assert torch.equal(grid[..., 10:15], alpha_video)
    assert torch.equal(grid[..., 15:20], pca_video)


def test_compose_multicam_feature_gt_render_grid_uses_camera_rows_and_media_columns() -> None:
    feature0 = torch.full((2, 3, 4, 5), 0.10)
    feature1 = torch.full((2, 3, 4, 5), 0.20)
    gt0 = torch.full((2, 3, 4, 5), 0.30)
    gt1 = torch.full((2, 3, 4, 5), 0.40)
    render0 = torch.full((2, 3, 4, 5), 0.50)
    render1 = torch.full((2, 3, 4, 5), 0.60)

    grid = compose_multicam_feature_gt_render_grid(
        feature_videos=[feature0, feature1],
        gt_videos=[gt0, gt1],
        render_videos=[render0, render1],
    )

    assert grid is not None
    assert grid.shape == (2, 3, 8, 15)
    assert torch.equal(grid[..., 0:4, 0:5], feature0)
    assert torch.equal(grid[..., 0:4, 5:10], gt0)
    assert torch.equal(grid[..., 0:4, 10:15], render0)
    assert torch.equal(grid[..., 4:8, 0:5], feature1)
    assert torch.equal(grid[..., 4:8, 5:10], gt1)
    assert torch.equal(grid[..., 4:8, 10:15], render1)


def test_render_diagnostics_payload_requires_features_when_pca_enabled() -> None:
    cfg = {"logging": {"feature_pca_log": True}}
    frames = torch.zeros(2, 3, 4, 4)

    with pytest.raises(ValueError, match="feature_pca_log=True requires"):
        render_diagnostics_payload(
            cfg,
            prefix="Eval",
            target=frames,
            pred=frames,
            alpha=None,
            features=None,
            fps=4.0,
        )


def test_prepare_clip_returns_batched_frames_and_normalized_times() -> None:
    frames = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
    sequence = SequenceData(
        frames=frames,
        frame_times=torch.tensor([0.0, 0.12, 0.25, 0.75, 1.0]).reshape(5, 1),
        frame_source="unit",
        video_fps=4.0,
        source_path=Path("sample.mp4"),
        all_frame_count=5,
    )
    indices = torch.tensor([1, 3])

    clip_frames, clip_times = prepare_clip(sequence, indices)

    assert torch.equal(clip_frames, frames[indices].unsqueeze(0))
    assert torch.allclose(clip_times, torch.tensor([[0.12, 0.75]]))


def test_stack_complete_frame_list_rejects_missing_frames() -> None:
    frames = [torch.zeros(3, 2, 2), None, torch.ones(3, 2, 2)]

    with pytest.raises(RuntimeError, match="missing indices: 1"):
        stack_complete_frame_list(frames, name="unit render")


def test_gaussian_sequence_slice_preserves_auxiliary_and_slices_cameras() -> None:
    sequence = GaussianSequence(
        xyz=torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3),
        scales=torch.ones(4, 2, 3),
        quats=torch.ones(4, 2, 4),
        opacities=torch.ones(4, 2, 1),
        rgbs=torch.ones(4, 2, 3),
        cameras=("c0", "c1", "c2", "c3"),
        auxiliary={"tag": "kept"},
    )

    sliced = gaussian_sequence_slice(sequence, 1, 3)

    assert sliced.xyz.shape == (2, 2, 3)
    assert sliced.cameras == ("c1", "c2")
    assert sliced.auxiliary == {"tag": "kept"}


def test_render_clip_payload_types_live_in_runtime_types() -> None:
    features = torch.ones(2, 4, 3, 3)
    alpha = torch.full((2, 3, 3), 0.5)
    rasterized = RasterizedClip(features=features, alpha=alpha)
    rendered = RenderedClip(
        rgb_sequence=torch.zeros(2, 3, 3, 3),
        camera_state=None,
        temporal_metrics={"Eval/DecodedXYZAdjacentL2": 0.0},
        feature_sequence=features,
        alpha_sequence=alpha,
    )

    assert rasterized.features is features
    assert rendered.feature_sequence is features
    assert rendered.alpha_sequence is alpha
