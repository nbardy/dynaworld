from __future__ import annotations

import torch

from camera import build_look_at_camera_to_world
from powerfoam_implicit_camera import PowerFoamImplicitCameraDecoder


def test_powerfoam_implicit_camera_zero_init_looks_at_origin() -> None:
    decoder = PowerFoamImplicitCameraDecoder(
        frame_count=3,
        image_size=8,
        fov_degrees=60.0,
        base_radius=2.0,
        token_dim=8,
        hidden_dim=16,
        time_basis_count=2,
    )

    camera_to_world = decoder.camera_to_world_matrices()
    expected = build_look_at_camera_to_world(torch.tensor([0.0, 0.0, -2.0]))

    assert camera_to_world.shape == (3, 4, 4)
    assert torch.allclose(camera_to_world, expected.expand_as(camera_to_world), atol=1.0e-6)

    cameras = decoder.cameras()
    assert len(cameras) == 3
    assert torch.allclose(cameras[0].camera_to_world[:3, 2], torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)

    origins, directions = decoder.rays(height=4, width=4)
    assert origins.shape == (3, 4, 4, 3)
    assert directions.shape == (3, 4, 4, 3)
    assert torch.allclose(origins[0], torch.tensor([0.0, 0.0, -2.0]).expand_as(origins[0]), atol=1.0e-6)
    assert torch.allclose(directions.norm(dim=-1), torch.ones(3, 4, 4), atol=1.0e-6)
    assert decoder.regularization_loss().item() == 0.0


def test_powerfoam_implicit_camera_nonzero_offset_changes_pose_and_rays() -> None:
    decoder = PowerFoamImplicitCameraDecoder(
        frame_count=2,
        image_size=8,
        fov_degrees=60.0,
        base_radius=2.0,
        token_dim=8,
        hidden_dim=16,
        time_basis_count=2,
        max_translation=0.5,
        max_rotation_degrees=20.0,
    )
    before_c2w = decoder.camera_to_world_matrices().detach().clone()
    before_origins, before_directions = decoder.rays(height=4, width=4)

    with torch.no_grad():
        decoder.offset_head[-1].bias[0] = 0.5
        decoder.offset_head[-1].bias[3] = 0.5

    after_c2w = decoder.camera_to_world_matrices()
    after_origins, after_directions = decoder.rays(height=4, width=4)
    terms = decoder.regularization_terms()
    metrics = decoder.metrics()

    assert not torch.allclose(after_c2w, before_c2w)
    assert not torch.allclose(after_origins, before_origins)
    assert not torch.allclose(after_directions, before_directions)
    assert terms["camera_rotation_l2"].item() > 0.0
    assert terms["camera_translation_l2"].item() > 0.0
    assert metrics["Camera/RotationDeltaMeanDegrees"] > 0.0
    assert metrics["Camera/TranslationDeltaMean"] > 0.0


def test_powerfoam_implicit_camera_orbit_base_path_spans_half_turn() -> None:
    decoder = PowerFoamImplicitCameraDecoder(
        frame_count=5,
        image_size=8,
        fov_degrees=60.0,
        base_radius=2.0,
        token_dim=8,
        hidden_dim=16,
        time_basis_count=2,
        base_path_mode="orbit_yaw",
        orbit_yaw_start_degrees=0.0,
        orbit_yaw_end_degrees=180.0,
    )

    camera_to_world = decoder.camera_to_world_matrices()
    assert torch.allclose(camera_to_world[0, :3, 3], torch.tensor([0.0, 0.0, -2.0]), atol=1.0e-5)
    assert torch.allclose(camera_to_world[-1, :3, 3], torch.tensor([0.0, 0.0, 2.0]), atol=1.0e-5)
    assert torch.allclose(camera_to_world[0, :3, 2], torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-5)
    assert torch.allclose(camera_to_world[-1, :3, 2], torch.tensor([0.0, 0.0, -1.0]), atol=1.0e-5)

    subset = decoder.camera_to_world_matrices(torch.tensor([0, 4]))
    assert subset.shape == (2, 4, 4)
    assert torch.allclose(subset[1], camera_to_world[-1], atol=1.0e-6)
