from __future__ import annotations

import unittest

import torch

from objective.background import BackgroundPolicy
from objective.objective import compose_rgb
from objective.types import BackgroundSpec, ColorizedView, RasterizedView, TargetView


def _target_view(frame_count: int = 1, height: int = 4, width: int = 5) -> TargetView:
    return TargetView(
        view_id="unit",
        role="train",
        camera_role="target",
        camera_owner="target",
        frames=torch.zeros(frame_count, 3, height, width),
        frame_indices=torch.arange(frame_count),
        frame_times=torch.linspace(0.0, 1.0, steps=frame_count),
        video_fps=4.0,
    )


class ObjectiveBackgroundCompositionTests(unittest.TestCase):
    def test_random_train_background_and_fixed_eval_background_use_like_tensor_properties(self) -> None:
        like = torch.zeros(2, 3, 4, 5, dtype=torch.float64)
        policy = BackgroundPolicy(BackgroundSpec(train_mode="random_rgb", eval_mode="white"))
        generator = torch.Generator().manual_seed(123)

        train_bg = policy.sample(phase="train", like=like, frame_count=like.shape[0], generator=generator)
        eval_bg = policy.sample(phase="eval", like=like, frame_count=like.shape[0], generator=generator)

        self.assertEqual(train_bg.rgb.shape, (1, 3, 1, 1))
        self.assertEqual(train_bg.rgb.dtype, like.dtype)
        self.assertEqual(train_bg.rgb.device, like.device)
        self.assertEqual(train_bg.mode, "random_rgb")
        self.assertTrue(bool(torch.all((train_bg.rgb >= 0.0) & (train_bg.rgb <= 1.0))))

        self.assertEqual(eval_bg.mode, "white")
        self.assertTrue(torch.equal(eval_bg.rgb, torch.ones(1, 3, 1, 1, dtype=like.dtype)))

    def test_alpha_composition_matches_closed_form_and_supports_gradients(self) -> None:
        splat_rgb = torch.tensor(
            [
                [
                    [[0.0, 0.2], [0.4, 0.6]],
                    [[0.1, 0.3], [0.5, 0.7]],
                    [[0.2, 0.4], [0.6, 0.8]],
                ]
            ],
            requires_grad=True,
        )
        alpha = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]], requires_grad=True)
        background = BackgroundPolicy(BackgroundSpec(train_mode="white")).sample(
            phase="train",
            like=splat_rgb,
            frame_count=splat_rgb.shape[0],
        )
        rasterized = RasterizedView(
            view=_target_view(frame_count=1, height=2, width=2),
            features=splat_rgb,
            alpha=alpha,
        )
        colorized = ColorizedView(splat_rgb=splat_rgb)

        composed = compose_rgb(rasterized=rasterized, colorized=colorized, background=background)
        expected = alpha.unsqueeze(1) * splat_rgb + (1.0 - alpha.unsqueeze(1)) * background.rgb

        self.assertTrue(torch.allclose(composed, expected))
        composed.sum().backward()
        self.assertTrue(torch.allclose(splat_rgb.grad, alpha.unsqueeze(1).expand_as(splat_rgb)))
        self.assertIsNotNone(alpha.grad)

    def test_alpha_none_returns_rgb_without_requiring_background(self) -> None:
        splat_rgb = torch.rand(2, 3, 4, 5)
        rasterized = RasterizedView(view=_target_view(frame_count=2), features=splat_rgb, alpha=None)
        background = BackgroundPolicy(BackgroundSpec(train_mode="white")).sample(
            phase="train",
            like=splat_rgb,
            frame_count=splat_rgb.shape[0],
        )

        composed = compose_rgb(rasterized=rasterized, colorized=None, background=background)

        self.assertIs(composed, splat_rgb)

    def test_black_background_composition_preserves_alpha_signal(self) -> None:
        features = torch.rand(2, 32, 4, 5)
        splat_rgb = torch.rand(2, 3, 4, 5)
        alpha = torch.rand(2, 4, 5)
        raster = RasterizedView(view=_target_view(frame_count=2), features=features, alpha=alpha)
        colorized = ColorizedView(splat_rgb=splat_rgb)
        background = BackgroundPolicy(BackgroundSpec(train_mode="black")).sample(
            phase="train",
            like=features,
            frame_count=features.shape[0],
        )

        rendered_rgb = compose_rgb(rasterized=raster, colorized=colorized, background=background)

        self.assertEqual(rendered_rgb.shape, splat_rgb.shape)
        self.assertTrue(torch.allclose(rendered_rgb, alpha.unsqueeze(1) * splat_rgb))

    def test_alpha_composition_rejects_shape_drift(self) -> None:
        cases: tuple[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, str], ...] = (
            (
                torch.rand(1, 32, 3, 3),
                torch.rand(1, 4, 3, 3),
                torch.rand(1, 3, 3),
                "ColorizedView.splat_rgb",
            ),
            (
                torch.rand(1, 3, 3, 3),
                None,
                torch.rand(1, 1, 3, 3),
                "alpha must have shape",
            ),
        )
        for features, colorized_rgb, alpha, message in cases:
            rasterized = RasterizedView(
                view=_target_view(frame_count=1, height=3, width=3),
                features=features,
                alpha=alpha,
            )
            colorized = None if colorized_rgb is None else ColorizedView(splat_rgb=colorized_rgb)
            background = BackgroundPolicy(BackgroundSpec(train_mode="white")).sample(
                phase="train",
                like=features,
                frame_count=features.shape[0],
            )
            with self.assertRaisesRegex(ValueError, message):
                compose_rgb(rasterized=rasterized, colorized=colorized, background=background)


if __name__ == "__main__":
    unittest.main()
