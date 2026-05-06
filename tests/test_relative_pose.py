from __future__ import annotations

import torch

from camera import CameraSpec
from relative_pose import (
    RelativePoseCrossAttentionHead,
    cameras_with_se3_transform,
    compose_cameras_with_se3_residual,
    se3_cycle_loss,
    se3_residual_identity_loss,
    se3_transform_l2_loss,
)


def test_relative_pose_head_starts_at_zero_residual() -> None:
    head = RelativePoseCrossAttentionHead(dim=8, num_heads=2, layers=1, hidden_dim=8)
    source = torch.randn(1, 4, 8)
    target = torch.randn(1, 4, 8)

    rotation, translation = head(source, target)

    assert rotation.shape == (1, 3)
    assert translation.shape == (1, 3)
    assert torch.allclose(rotation, torch.zeros_like(rotation))
    assert torch.allclose(translation, torch.zeros_like(translation))
    assert torch.equal(se3_residual_identity_loss(rotation, translation), rotation.new_zeros(()))


def test_identity_residual_preserves_camera_pose() -> None:
    camera = CameraSpec(
        fx=torch.tensor(10.0),
        fy=torch.tensor(10.0),
        cx=torch.tensor(4.0),
        cy=torch.tensor(4.0),
        camera_to_world=torch.eye(4),
    )

    cameras = compose_cameras_with_se3_residual(
        (camera,),
        torch.zeros(1, 3),
        torch.zeros(1, 3),
    )

    assert len(cameras) == 1
    assert torch.allclose(cameras[0].camera_to_world, camera.camera_to_world)


def test_inverse_relative_transforms_have_zero_cycle_loss() -> None:
    source_to_target = torch.eye(4)
    source_to_target[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    target_to_source = torch.linalg.inv(source_to_target)

    loss = se3_cycle_loss(source_to_target.unsqueeze(0), target_to_source.unsqueeze(0))

    assert torch.allclose(loss, torch.zeros_like(loss))


def test_full_pose_camera_helper_replaces_pose_and_preserves_intrinsics() -> None:
    camera = CameraSpec(
        fx=torch.tensor(10.0),
        fy=torch.tensor(11.0),
        cx=torch.tensor(4.0),
        cy=torch.tensor(5.0),
        camera_to_world=torch.eye(4),
        lens_model="radial_tangential",
        distortion=torch.tensor([0.1, 0.01, 0.0, 0.0, 0.0]),
    )
    predicted = torch.eye(4)
    predicted[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

    cameras = cameras_with_se3_transform((camera,), predicted)

    assert len(cameras) == 1
    assert torch.allclose(cameras[0].camera_to_world, predicted)
    assert cameras[0].fx is camera.fx
    assert cameras[0].fy is camera.fy
    assert cameras[0].cx is camera.cx
    assert cameras[0].cy is camera.cy
    assert cameras[0].lens_model == camera.lens_model
    assert cameras[0].distortion is camera.distortion


def test_full_pose_transform_loss_is_zero_for_matching_transforms_and_positive_for_offset() -> None:
    target = torch.eye(4).unsqueeze(0)
    predicted = target.clone()

    zero_loss = se3_transform_l2_loss(predicted, target)

    predicted[:, 0, 3] = 1.0
    positive_loss = se3_transform_l2_loss(predicted, target)

    assert torch.allclose(zero_loss, torch.zeros_like(zero_loss))
    assert positive_loss > 0
