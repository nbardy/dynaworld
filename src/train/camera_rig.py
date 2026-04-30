from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from camera import CameraSpec


def _skew_symmetric(vectors: torch.Tensor) -> torch.Tensor:
    batch_size = int(vectors.shape[0])
    skew = vectors.new_zeros((batch_size, 3, 3))
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axes = axis_angle / angles.clamp_min(1.0e-8)
    skew = _skew_symmetric(axes)
    eye = (
        torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(axis_angle.shape[0], -1, -1)
    )
    sin_term = torch.sin(angles).unsqueeze(-1)
    cos_term = (1.0 - torch.cos(angles)).unsqueeze(-1)
    rotation = eye + sin_term * skew + cos_term * (skew @ skew)
    small_angle = angles.squeeze(-1) < 1.0e-6
    if torch.any(small_angle):
        small_skew = _skew_symmetric(axis_angle[small_angle])
        rotation[small_angle] = eye[small_angle] + small_skew
    return rotation


def se3_delta_transform(
    rotation_raw: torch.Tensor,
    translation_raw: torch.Tensor,
    *,
    max_rotation_degrees: float,
    max_translation: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = torch.tanh(rotation_raw) * math.radians(float(max_rotation_degrees))
    translation = torch.tanh(translation_raw) * float(max_translation)
    transform = torch.eye(4, dtype=rotation.dtype, device=rotation.device).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = axis_angle_to_matrix(rotation)
    transform[:, :3, 3] = translation
    return transform, rotation, translation


def _camera_scalar(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), dtype=dtype, device=device)


class LearnableCameraRig(nn.Module):
    """Small learnable alignment layer around a fixed camera rig.

    Base camera-to-world transforms are buffers. A shared global SE3 can align
    the entire rig to the learned world frame, and a small per-view SE3 can
    correct individual camera references without making each camera fully free.
    """

    def __init__(
        self,
        train_cameras: Sequence[Sequence[CameraSpec]],
        *,
        heldout_cameras: Sequence[Sequence[CameraSpec]] | None = None,
        learn_global_se3: bool = True,
        learn_per_camera_se3: bool = True,
        anchor_policy: str = "soft",
        max_rotation_degrees: float = 15.0,
        max_translation_ratio: float = 0.2,
        radius: float = 3.0,
    ) -> None:
        super().__init__()
        if len(train_cameras) < 1:
            raise ValueError("LearnableCameraRig requires at least one train camera view.")
        frame_count = len(train_cameras[0])
        if frame_count < 1:
            raise ValueError("LearnableCameraRig requires at least one frame per train view.")
        if any(len(view) != frame_count for view in train_cameras):
            raise ValueError("All train camera views must have the same frame count.")
        normalized_anchor_policy = str(anchor_policy).lower()
        if normalized_anchor_policy not in {"soft", "fixed_first", "free"}:
            raise ValueError("camera.rig_anchor_policy must be one of: soft, fixed_first, free")

        first_camera = train_cameras[0][0]
        device = first_camera.camera_to_world.device
        dtype = first_camera.camera_to_world.dtype
        self.view_count = len(train_cameras)
        self.frame_count = frame_count
        self.learn_global_se3 = bool(learn_global_se3)
        self.learn_per_camera_se3 = bool(learn_per_camera_se3)
        self.anchor_policy = normalized_anchor_policy
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.max_translation = float(radius) * float(max_translation_ratio)

        train_c2w = []
        train_intrinsics = []
        for view in train_cameras:
            train_c2w.append(torch.stack([camera.camera_to_world.to(device=device, dtype=dtype) for camera in view], dim=0))
            train_intrinsics.append(
                torch.stack(
                    [
                        torch.stack(
                            [
                                _camera_scalar(camera.fx, device=device, dtype=dtype),
                                _camera_scalar(camera.fy, device=device, dtype=dtype),
                                _camera_scalar(camera.cx, device=device, dtype=dtype),
                                _camera_scalar(camera.cy, device=device, dtype=dtype),
                            ]
                        )
                        for camera in view
                    ],
                    dim=0,
                )
            )
        self.register_buffer("train_c2w", torch.stack(train_c2w, dim=0), persistent=False)
        self.register_buffer("train_intrinsics", torch.stack(train_intrinsics, dim=0), persistent=False)

        if heldout_cameras is not None:
            if len(heldout_cameras) < 1:
                raise ValueError("heldout_cameras must be None or contain at least one held-out view.")
            if any(len(view) != frame_count for view in heldout_cameras):
                raise ValueError("All held-out camera views must have the same frame count as the train views.")
            self.register_buffer(
                "heldout_c2w",
                torch.stack(
                    [
                        torch.stack(
                            [camera.camera_to_world.to(device=device, dtype=dtype) for camera in view],
                            dim=0,
                        )
                        for view in heldout_cameras
                    ],
                    dim=0,
                ),
                persistent=False,
            )
            self.register_buffer(
                "heldout_intrinsics",
                torch.stack(
                    [
                        torch.stack(
                            [
                                torch.stack(
                                    [
                                        _camera_scalar(camera.fx, device=device, dtype=dtype),
                                        _camera_scalar(camera.fy, device=device, dtype=dtype),
                                        _camera_scalar(camera.cx, device=device, dtype=dtype),
                                        _camera_scalar(camera.cy, device=device, dtype=dtype),
                                    ]
                                )
                                for camera in view
                            ]
                        )
                        for view in heldout_cameras
                    ],
                    dim=0,
                ),
                persistent=False,
            )
        else:
            self.heldout_c2w = None
            self.heldout_intrinsics = None

        self.global_rotation_raw = nn.Parameter(torch.zeros(1, 3, dtype=dtype, device=device))
        self.global_translation_raw = nn.Parameter(torch.zeros(1, 3, dtype=dtype, device=device))
        self.view_rotation_raw = nn.Parameter(torch.zeros(self.view_count, 3, dtype=dtype, device=device))
        self.view_translation_raw = nn.Parameter(torch.zeros(self.view_count, 3, dtype=dtype, device=device))
        self.global_rotation_raw.requires_grad_(self.learn_global_se3)
        self.global_translation_raw.requires_grad_(self.learn_global_se3)
        self.view_rotation_raw.requires_grad_(self.learn_per_camera_se3)
        self.view_translation_raw.requires_grad_(self.learn_per_camera_se3)

    @property
    def heldout_view_count(self) -> int:
        if self.heldout_c2w is None:
            return 0
        return int(self.heldout_c2w.shape[0])

    def _global_transform(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return se3_delta_transform(
            self.global_rotation_raw,
            self.global_translation_raw,
            max_rotation_degrees=self.max_rotation_degrees,
            max_translation=self.max_translation,
        )

    def _view_transform(self, view: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.anchor_policy == "fixed_first" and int(view) == 0:
            raw_rot = torch.zeros_like(self.view_rotation_raw[0:1])
            raw_trans = torch.zeros_like(self.view_translation_raw[0:1])
        else:
            raw_rot = self.view_rotation_raw[int(view) : int(view) + 1]
            raw_trans = self.view_translation_raw[int(view) : int(view) + 1]
        return se3_delta_transform(
            raw_rot,
            raw_trans,
            max_rotation_degrees=self.max_rotation_degrees,
            max_translation=self.max_translation,
        )

    def _make_camera(self, c2w: torch.Tensor, intrinsics: torch.Tensor) -> CameraSpec:
        return CameraSpec(
            fx=intrinsics[0],
            fy=intrinsics[1],
            cx=intrinsics[2],
            cy=intrinsics[3],
            camera_to_world=c2w,
            lens_model="pinhole",
        )

    def cameras_for_view(self, view: int, frame_indices: torch.Tensor) -> tuple[CameraSpec, ...]:
        global_transform, _global_rot, _global_trans = self._global_transform()
        view_transform, _view_rot, _view_trans = self._view_transform(int(view))
        cameras = []
        for frame in frame_indices.detach().cpu().tolist():
            base_c2w = self.train_c2w[int(view), int(frame)]
            c2w = global_transform[0] @ base_c2w @ view_transform[0]
            cameras.append(self._make_camera(c2w, self.train_intrinsics[int(view), int(frame)]))
        return tuple(cameras)

    def heldout_cameras_for(self, view: int, frame_indices: torch.Tensor) -> tuple[CameraSpec, ...]:
        if self.heldout_c2w is None or self.heldout_intrinsics is None:
            raise ValueError("This camera rig does not have held-out cameras.")
        global_transform, _global_rot, _global_trans = self._global_transform()
        cameras = []
        for frame in frame_indices.detach().cpu().tolist():
            c2w = global_transform[0] @ self.heldout_c2w[int(view), int(frame)]
            cameras.append(self._make_camera(c2w, self.heldout_intrinsics[int(view), int(frame)]))
        return tuple(cameras)

    def regularization_loss(self) -> torch.Tensor:
        _global_transform, global_rot, global_trans = self._global_transform()
        _view_transform, view_rot0, view_trans0 = self._view_transform(0)
        del _global_transform, _view_transform, view_rot0, view_trans0
        loss = global_rot.square().mean() + global_trans.square().mean()
        if self.anchor_policy == "fixed_first":
            view_rot = self.view_rotation_raw[1:]
            view_trans = self.view_translation_raw[1:]
        else:
            view_rot = self.view_rotation_raw
            view_trans = self.view_translation_raw
        if view_rot.numel() > 0:
            _view_delta, bounded_rot, bounded_trans = se3_delta_transform(
                view_rot,
                view_trans,
                max_rotation_degrees=self.max_rotation_degrees,
                max_translation=self.max_translation,
            )
            loss = loss + bounded_rot.square().mean() + bounded_trans.square().mean()
        return loss

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        _global_transform, global_rot, global_trans = self._global_transform()
        _view_transform, _view_rot0, _view_trans0 = self._view_transform(0)
        del _global_transform, _view_transform, _view_rot0, _view_trans0
        _view_delta, view_rot, view_trans = se3_delta_transform(
            self.view_rotation_raw,
            self.view_translation_raw,
            max_rotation_degrees=self.max_rotation_degrees,
            max_translation=self.max_translation,
        )
        return {
            "Rig/GlobalRotationDegrees": float(torch.rad2deg(torch.linalg.norm(global_rot, dim=-1)).mean().item()),
            "Rig/GlobalTranslation": float(torch.linalg.norm(global_trans, dim=-1).mean().item()),
            "Rig/ViewRotationDegrees": float(torch.rad2deg(torch.linalg.norm(view_rot, dim=-1)).mean().item()),
            "Rig/ViewTranslation": float(torch.linalg.norm(view_trans, dim=-1).mean().item()),
        }


__all__ = ["LearnableCameraRig", "axis_angle_to_matrix", "se3_delta_transform"]
