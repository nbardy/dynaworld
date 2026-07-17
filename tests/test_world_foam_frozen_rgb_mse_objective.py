from __future__ import annotations

import unittest

import torch

from objective.world_foam_frozen_rgb_mse import (
    FRAMEGROUP16_TAPE_KEYS,
    PROMOTED_FRAMEGROUP16_TAPE_MODE,
    WorldFoamFrozenRGBMSEObjective,
    WorldFoamTargetLayout,
    target_rgb_to_track_major,
    validate_world_foam_frozen_rgb_mse_scope,
)


def _dummy_tape() -> dict[str, torch.Tensor]:
    return {key: torch.zeros(1) for key in FRAMEGROUP16_TAPE_KEYS}


def _fake_fused_loss_fn(**kwargs) -> torch.Tensor:
    target = kwargs["target_rgb_f32"]
    site_rgba = kwargs["site_rgba_f32"]
    track_count = int(kwargs["track_count"])
    frame_count = int(kwargs["frame_count"])
    if tuple(target.shape) != (track_count, frame_count, 3):
        raise AssertionError(f"unexpected target shape {tuple(target.shape)}")
    pred = site_rgba[0, :3].reshape(1, 1, 3).expand_as(target)
    return (pred - target).square().mean()


class WorldFoamFrozenRGBMSEObjectiveTests(unittest.TestCase):
    def test_target_rgb_to_track_major_converts_image_layout(self) -> None:
        layout = WorldFoamTargetLayout(view_count=2, frame_count=3, height=2, width=2)
        image = torch.arange(2 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(2 * 3, 3, 2, 2)

        track_major = target_rgb_to_track_major(image, layout)
        expected = image.reshape(2, 3, 3, 2, 2).permute(0, 3, 4, 1, 2).reshape(8, 3, 3)

        self.assertEqual(tuple(track_major.shape), (8, 3, 3))
        self.assertTrue(torch.equal(track_major, expected))

    def test_target_rgb_to_track_major_accepts_view_major_and_track_major(self) -> None:
        layout = WorldFoamTargetLayout(view_count=1, frame_count=2, height=2, width=3)
        view_major = torch.arange(1 * 2 * 3 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3, 2, 3)
        track_major = target_rgb_to_track_major(view_major, layout)

        self.assertTrue(torch.equal(target_rgb_to_track_major(track_major, layout), track_major))
        track_layout = WorldFoamTargetLayout.from_track_major(track_count=6, frame_count=2)
        self.assertEqual(track_layout.track_count, layout.track_count)
        self.assertEqual(track_layout.frame_count, layout.frame_count)

    def test_objective_loss_flows_gradient_only_through_site_rgba(self) -> None:
        layout = WorldFoamTargetLayout.from_track_major(track_count=4, frame_count=2)
        objective = WorldFoamFrozenRGBMSEObjective(
            tape=_dummy_tape(),
            config=object(),
            boundary_count=4,
            layout=layout,
            fused_loss_fn=_fake_fused_loss_fn,
        )
        site_rgba = torch.tensor([[0.2, 0.3, 0.4, 0.8]], requires_grad=True)
        target_rgb = torch.zeros(layout.view_count * layout.frame_count, 3, layout.height, layout.width)

        loss = objective.loss(site_rgba=site_rgba, target_rgb=target_rgb)
        loss.backward()

        self.assertGreater(float(site_rgba.grad[:, :3].abs().sum()), 0.0)
        self.assertEqual(float(site_rgba.grad[:, 3].abs().sum()), 0.0)
        self.assertEqual(objective.scope.tape_mode, PROMOTED_FRAMEGROUP16_TAPE_MODE)
        self.assertFalse(objective.scope.full_trainer_claim)
        self.assertFalse(objective.scope.full_geometry_gradient_claim)
        self.assertFalse(objective.scope.renderer_backend_claim)

    def test_scope_validator_rejects_unsupported_trainer_features(self) -> None:
        validate_world_foam_frozen_rgb_mse_scope(loss_kind="mse", feature_dim=3)

        cases = (
            {"loss_kind": "l1", "feature_dim": 3},
            {"loss_kind": "mse", "feature_dim": 8},
            {"loss_kind": "mse", "feature_dim": 3, "vjepa_feature_weight": 0.1},
            {"loss_kind": "mse", "feature_dim": 3, "uses_colorizer": True},
            {"loss_kind": "mse", "feature_dim": 3, "uses_background_composition": True},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    validate_world_foam_frozen_rgb_mse_scope(**kwargs)

    def test_missing_tape_keys_are_rejected(self) -> None:
        tape = _dummy_tape()
        del tape["change_record_i16"]

        with self.assertRaisesRegex(ValueError, "missing keys"):
            WorldFoamFrozenRGBMSEObjective(
                tape=tape,
                config=object(),
                boundary_count=4,
                layout=WorldFoamTargetLayout(view_count=1, frame_count=1, height=1, width=1),
                fused_loss_fn=_fake_fused_loss_fn,
            )


if __name__ == "__main__":
    unittest.main()
