from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "train"))

from objective import (  # noqa: E402
    BackgroundSample,
    BackgroundSpec,
    ObjectiveSpec,
    RGBReconObjective,
    RasterizedView,
    ReconstructionLossSpec,
    TargetView,
)


def _target(frames: torch.Tensor) -> TargetView:
    frame_count = frames.shape[0]
    return TargetView(
        view_id="view0",
        role="train",
        camera_role="target",
        camera_owner="target",
        frames=frames,
        frame_indices=torch.arange(frame_count),
        frame_times=torch.linspace(0.0, 1.0, frame_count).unsqueeze(-1),
        video_fps=4.0,
    )


class ConstantColorizer:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, features: torch.Tensor, view_dirs: torch.Tensor | None = None) -> torch.Tensor:
        return features.new_full((features.shape[0], 3, features.shape[-2], features.shape[-1]), self.value)


def test_f3_legacy_path_passes_rgb_through_without_alpha_composition() -> None:
    features = torch.rand(2, 3, 4, 4)
    rasterized = RasterizedView(view=_target(features.clone()), features=features, alpha=None)
    objective = RGBReconObjective(ObjectiveSpec(version="test"))

    rendered = objective.compose_rasterized(rasterized, phase="train")

    assert rendered.background.mode == "random_rgb"
    assert torch.allclose(rendered.rgb, features)


def test_feature_splatting_requires_colorizer_before_rgb_loss() -> None:
    frames = torch.zeros(1, 3, 2, 2)
    rasterized = RasterizedView(view=_target(frames), features=torch.zeros(1, 8, 2, 2), alpha=None)
    objective = RGBReconObjective(ObjectiveSpec(version="test"))

    with pytest.raises(ValueError, match="requires a colorizer"):
        objective.compose_rasterized(rasterized, phase="eval")


def test_alpha_composition_uses_colorized_rgb_and_background() -> None:
    frames = torch.zeros(1, 3, 2, 2)
    rasterized = RasterizedView(
        view=_target(frames),
        features=torch.zeros(1, 8, 2, 2),
        alpha=torch.full((1, 2, 2), 0.5),
    )
    background = BackgroundSample(
        rgb=torch.ones(1, 3, 1, 1),
        mode="white",
        phase="eval",
        scope="step",
    )
    objective = RGBReconObjective(
        ObjectiveSpec(version="test", reconstruction=ReconstructionLossSpec(kind="l1")),
        colorizer=ConstantColorizer(0.25),
    )

    rendered, loss = objective.loss_for_rasterized(rasterized, phase="eval", background=background)

    assert torch.allclose(rendered.rgb, torch.full((1, 3, 2, 2), 0.625))
    assert torch.allclose(loss.per_image, torch.tensor([0.625]))


def test_background_policy_random_train_white_eval() -> None:
    spec = ObjectiveSpec(
        version="test",
        background=BackgroundSpec(train_mode="random_rgb", eval_mode="white", sample_scope="step"),
    )
    objective = RGBReconObjective(spec)
    features = torch.zeros(3, 3, 5, 5)
    rasterized = RasterizedView(view=_target(features), features=features, alpha=None)
    generator = torch.Generator().manual_seed(123)

    train_bg = objective.background_for_view(rasterized, phase="train", generator=generator)
    eval_bg = objective.background_for_view(rasterized, phase="eval")

    assert train_bg.mode == "random_rgb"
    assert train_bg.rgb is not None
    assert train_bg.rgb.shape == (1, 3, 1, 1)
    assert not torch.allclose(train_bg.rgb, torch.ones_like(train_bg.rgb))
    assert eval_bg.mode == "white"
    assert eval_bg.rgb is not None
    assert torch.allclose(eval_bg.rgb, torch.ones(1, 3, 1, 1))
