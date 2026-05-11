from __future__ import annotations

import torch

from camera import CameraSpec, build_look_at_camera_to_world
from visualize_camera_scene_diagnostic import camera_frustum_lines, sample_point_cloud


def test_sample_point_cloud_uses_requested_fraction_deterministically() -> None:
    points = torch.arange(300, dtype=torch.float32).reshape(100, 3)

    first = sample_point_cloud(points, 0.05, seed=7)
    second = sample_point_cloud(points, 0.05, seed=7)

    assert first.shape == (5, 3)
    assert torch.equal(first, second)


def test_camera_frustum_lines_follow_camera_forward_axis() -> None:
    c2w = build_look_at_camera_to_world(torch.tensor([0.0, 0.0, -2.0]))
    camera = CameraSpec(
        fx=8.0,
        fy=8.0,
        cx=4.0,
        cy=4.0,
        camera_to_world=c2w,
    )

    frustums = camera_frustum_lines((camera,), length=1.0, width=0.25)

    assert frustums.segments.shape == (8, 2, 3)
    assert torch.allclose(frustums.centers[0], torch.tensor([0.0, 0.0, -2.0]), atol=1.0e-6)
    assert torch.allclose(frustums.directions[0], torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6)
    assert torch.allclose(frustums.segments[0, 1, 2], torch.tensor(-1.0), atol=1.0e-6)
