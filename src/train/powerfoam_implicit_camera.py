from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from camera import (
    CameraSpec,
    LensModel,
    build_camera_rays,
    build_look_at_camera_to_world,
)
from camera_rig import axis_angle_to_matrix
from runtime_types import CameraState


@dataclass(frozen=True)
class PowerFoamImplicitCameraBatch:
    cameras: tuple[CameraSpec, ...]
    camera_to_world: torch.Tensor
    camera_state: CameraState


def _zero_init_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def _make_gaussian_time_basis(frame_count: int, basis_count: int, sigma_scale: float) -> torch.Tensor:
    if frame_count < 1:
        raise ValueError("frame_count must be positive.")
    if basis_count < 1:
        raise ValueError("time_basis_count must be positive.")
    if sigma_scale <= 0.0:
        raise ValueError("time_basis_sigma_scale must be positive.")
    times = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32)
    centers = torch.linspace(0.0, 1.0, basis_count, dtype=torch.float32)
    spacing = 1.0 / float(max(basis_count - 1, 1))
    sigma = max(spacing * float(sigma_scale), 1.0e-4)
    basis = torch.exp(-0.5 * ((times[:, None] - centers[None, :]) / sigma).square())
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _as_vector3(value: Sequence[float] | torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {tensor.numel()}.")
    return tensor


def _pose_delta_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4, device=rotation.device, dtype=rotation.dtype).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = axis_angle_to_matrix(rotation)
    transform[:, :3, 3] = translation
    return transform


def _transform_from_rotation_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4, device=rotation.device, dtype=rotation.dtype).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    return transform


def _matrix_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    angle = torch.acos(cos_angle)
    vee = torch.stack(
        [
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ],
        dim=-1,
    )
    scale = angle / (2.0 * torch.sin(angle).clamp_min(1.0e-6))
    axis_angle = vee * scale.unsqueeze(-1)
    small = angle < 1.0e-4
    if torch.any(small):
        axis_angle[small] = 0.5 * vee[small]
    return axis_angle


def _make_orbit_camera_to_world_path(
    *,
    frame_count: int,
    radius: float,
    look_at: torch.Tensor,
    up: torch.Tensor,
    yaw_start_degrees: float,
    yaw_end_degrees: float,
    pitch_degrees: float,
) -> torch.Tensor:
    yaw = torch.linspace(math.radians(float(yaw_start_degrees)), math.radians(float(yaw_end_degrees)), int(frame_count))
    pitch = math.radians(float(pitch_degrees))
    cos_pitch = math.cos(pitch)
    offsets = torch.stack(
        [
            torch.sin(yaw) * float(radius) * cos_pitch,
            torch.full_like(yaw, math.sin(pitch) * float(radius)),
            -torch.cos(yaw) * float(radius) * cos_pitch,
        ],
        dim=-1,
    )
    return torch.stack(
        [build_look_at_camera_to_world(look_at + offset, target=look_at, up=up) for offset in offsets],
        dim=0,
    )


class PowerFoamImplicitCameraDecoder(nn.Module):
    """Lean object-centric implicit camera decoder for dynamic PowerFoam.

    The fixed base camera starts on the negative z axis and looks at the
    origin. A learnable start token predicts a global residual shared by all
    frames, while a Gaussian time-basis token path predicts per-frame SE(3)
    offsets. Intrinsics and lens parameters are fixed by constructor args.
    """

    def __init__(
        self,
        *,
        frame_count: int,
        image_size: int,
        fov_degrees: float = 55.0,
        base_radius: float = 3.0,
        token_dim: int = 32,
        hidden_dim: int = 64,
        time_basis_count: int = 8,
        time_basis_sigma_scale: float = 0.75,
        token_init_std: float = 0.02,
        max_rotation_degrees: float = 10.0,
        max_translation: float = 0.25,
        base_position: Sequence[float] | torch.Tensor | None = None,
        look_at: Sequence[float] | torch.Tensor = (0.0, 0.0, 0.0),
        up: Sequence[float] | torch.Tensor = (0.0, 1.0, 0.0),
        base_path_mode: str = "static",
        path_parameterization: str = "pose_delta",
        orbit_yaw_start_degrees: float = 0.0,
        orbit_yaw_end_degrees: float = 0.0,
        orbit_pitch_degrees: float = 0.0,
        drone_integration_horizon: float = 1.0,
        drone_damping: float = 0.98,
        drone_max_linear_velocity_ratio: float = 0.35,
        drone_max_linear_acceleration_ratio: float = 0.7,
        drone_max_angular_velocity_degrees: float = 45.0,
        drone_max_angular_acceleration_degrees: float = 90.0,
        drone_gimbal_max_rotation_degrees: float = 5.0,
        drone_body_frame_translation: bool = True,
        lens_model: LensModel = "pinhole",
        distortion: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if image_size < 1:
            raise ValueError("image_size must be positive.")
        if token_dim < 1:
            raise ValueError("token_dim must be positive.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if base_radius <= 0.0:
            raise ValueError("base_radius must be positive.")
        if max_rotation_degrees < 0.0:
            raise ValueError("max_rotation_degrees must be non-negative.")
        if max_translation < 0.0:
            raise ValueError("max_translation must be non-negative.")
        if drone_integration_horizon <= 0.0:
            raise ValueError("drone_integration_horizon must be positive.")
        if not (0.0 <= drone_damping <= 1.0):
            raise ValueError("drone_damping must be in [0, 1].")
        if drone_max_linear_velocity_ratio < 0.0:
            raise ValueError("drone_max_linear_velocity_ratio must be non-negative.")
        if drone_max_linear_acceleration_ratio < 0.0:
            raise ValueError("drone_max_linear_acceleration_ratio must be non-negative.")
        if drone_max_angular_velocity_degrees < 0.0:
            raise ValueError("drone_max_angular_velocity_degrees must be non-negative.")
        if drone_max_angular_acceleration_degrees < 0.0:
            raise ValueError("drone_max_angular_acceleration_degrees must be non-negative.")
        if drone_gimbal_max_rotation_degrees < 0.0:
            raise ValueError("drone_gimbal_max_rotation_degrees must be non-negative.")

        self.frame_count = int(frame_count)
        self.image_size = int(image_size)
        self.fov_degrees = float(fov_degrees)
        self.base_radius = float(base_radius)
        self.max_rotation_radians = math.radians(float(max_rotation_degrees))
        self.max_translation = float(max_translation)
        self.lens_model = lens_model
        self.base_path_mode = str(base_path_mode)
        self.path_parameterization = str(path_parameterization)
        self.drone_integration_horizon = float(drone_integration_horizon)
        self.drone_damping = float(drone_damping)
        self.drone_max_linear_velocity = float(base_radius) * float(drone_max_linear_velocity_ratio)
        self.drone_max_linear_acceleration = float(base_radius) * float(drone_max_linear_acceleration_ratio)
        self.drone_max_angular_velocity_radians = math.radians(float(drone_max_angular_velocity_degrees))
        self.drone_max_angular_acceleration_radians = math.radians(float(drone_max_angular_acceleration_degrees))
        self.drone_gimbal_max_rotation_radians = math.radians(float(drone_gimbal_max_rotation_degrees))
        self.drone_body_frame_translation = bool(drone_body_frame_translation)
        if self.path_parameterization not in {"pose_delta", "integrated_drone"}:
            raise ValueError("path_parameterization must be 'pose_delta' or 'integrated_drone'.")

        position = (
            _as_vector3(base_position, name="base_position")
            if base_position is not None
            else torch.tensor([0.0, 0.0, -float(base_radius)], dtype=torch.float32)
        )
        target = _as_vector3(look_at, name="look_at")
        base_up = _as_vector3(up, name="up")
        if self.base_path_mode == "static":
            base_camera_to_world = build_look_at_camera_to_world(position, target=target, up=base_up)
        elif self.base_path_mode == "orbit_yaw":
            base_camera_to_world = _make_orbit_camera_to_world_path(
                frame_count=int(frame_count),
                radius=float(base_radius),
                look_at=target,
                up=base_up,
                yaw_start_degrees=float(orbit_yaw_start_degrees),
                yaw_end_degrees=float(orbit_yaw_end_degrees),
                pitch_degrees=float(orbit_pitch_degrees),
            )
        else:
            raise ValueError("base_path_mode must be 'static' or 'orbit_yaw'.")
        self.register_buffer(
            "base_camera_to_world",
            base_camera_to_world,
            persistent=False,
        )
        self.register_buffer(
            "time_basis",
            _make_gaussian_time_basis(int(frame_count), int(time_basis_count), float(time_basis_sigma_scale)),
            persistent=False,
        )
        self.register_buffer(
            "base_distortion",
            torch.as_tensor((), dtype=torch.float32)
            if distortion is None
            else torch.as_tensor(distortion, dtype=torch.float32).flatten(),
            persistent=False,
        )

        image_extent = torch.tensor(float(image_size), dtype=torch.float32)
        fov = torch.tensor(math.radians(float(fov_degrees)), dtype=torch.float32)
        focal = 0.5 * image_extent / torch.tan(0.5 * fov)
        self.register_buffer("base_intrinsics", torch.stack([focal, focal, image_extent * 0.5, image_extent * 0.5]))

        self.start_camera_token = nn.Parameter(torch.zeros(int(token_dim)))
        self.time_basis_tokens = nn.Parameter(torch.randn(int(time_basis_count), int(token_dim)) * float(token_init_std))
        self.global_head = _zero_init_mlp(int(token_dim), int(hidden_dim), 6)
        self.offset_head = _zero_init_mlp(int(token_dim), int(hidden_dim), 6)
        self.velocity_head = _zero_init_mlp(int(token_dim), int(hidden_dim), 6)
        self.acceleration_head = _zero_init_mlp(int(token_dim), int(hidden_dim), 6)
        self.gimbal_head = _zero_init_mlp(int(token_dim), int(hidden_dim), 3)

    def _bounded_pose(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = torch.tanh(raw[..., :3]) * self.max_rotation_radians
        translation = torch.tanh(raw[..., 3:6]) * self.max_translation
        return rotation, translation

    def _temporal_tokens(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        basis = self.time_basis
        if frame_indices is not None:
            basis = basis[frame_indices.to(device=basis.device, dtype=torch.long)]
        temporal_token = basis.to(dtype=self.time_basis_tokens.dtype) @ self.time_basis_tokens
        return temporal_token + self.start_camera_token.view(1, -1)

    def _pose_delta_components(
        self,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path_raw = self.offset_head(self._temporal_tokens(frame_indices))
        global_raw = self.global_head(self.start_camera_token.view(1, -1))
        global_rotation, global_translation = self._bounded_pose(global_raw)
        path_rotation, path_translation = self._bounded_pose(path_raw)
        rotation = global_rotation + path_rotation
        translation = global_translation + path_translation
        return global_raw, rotation, translation, _pose_delta_matrix(rotation, translation)

    def _bounded_drone_dynamics(self) -> dict[str, torch.Tensor]:
        global_raw = self.global_head(self.start_camera_token.view(1, -1))
        start_rotation, start_translation = self._bounded_pose(global_raw)
        velocity_raw = self.velocity_head(self.start_camera_token.view(1, -1))
        temporal_tokens = self._temporal_tokens()
        acceleration_raw = self.acceleration_head(temporal_tokens)
        gimbal_raw = self.gimbal_head(temporal_tokens)
        return {
            "global_raw": global_raw,
            "start_rotation": start_rotation,
            "start_translation": start_translation,
            "velocity_raw": velocity_raw,
            "angular_velocity": torch.tanh(velocity_raw[..., :3]) * self.drone_max_angular_velocity_radians,
            "linear_velocity": torch.tanh(velocity_raw[..., 3:6]) * self.drone_max_linear_velocity,
            "acceleration_raw": acceleration_raw,
            "angular_acceleration": torch.tanh(acceleration_raw[..., :3]) * self.drone_max_angular_acceleration_radians,
            "linear_acceleration": torch.tanh(acceleration_raw[..., 3:6]) * self.drone_max_linear_acceleration,
            "gimbal_rotation": torch.tanh(gimbal_raw) * self.drone_gimbal_max_rotation_radians,
        }

    def _drone_delta_components(
        self,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dynamics = self._bounded_drone_dynamics()
        full_delta = self._integrate_drone_delta_matrices(dynamics)
        if frame_indices is not None:
            full_delta = full_delta[frame_indices.to(device=full_delta.device, dtype=torch.long)]
        rotation = _matrix_to_axis_angle(full_delta[:, :3, :3])
        translation = full_delta[:, :3, 3]
        return dynamics["global_raw"], rotation, translation, full_delta

    def _integrate_drone_delta_matrices(self, dynamics: dict[str, torch.Tensor]) -> torch.Tensor:
        step_count = max(self.frame_count - 1, 1)
        dt = float(self.drone_integration_horizon) / float(step_count)
        damping = float(self.drone_damping)

        rotation = axis_angle_to_matrix(dynamics["start_rotation"])[0]
        translation = dynamics["start_translation"].squeeze(0)
        angular_velocity = dynamics["angular_velocity"].squeeze(0)
        linear_velocity = dynamics["linear_velocity"].squeeze(0)
        rotations = []
        translations = []
        for frame in range(self.frame_count):
            gimbal = axis_angle_to_matrix(dynamics["gimbal_rotation"][frame : frame + 1])[0]
            rotations.append(rotation @ gimbal)
            translations.append(translation)
            if frame == self.frame_count - 1:
                continue
            angular_velocity = damping * angular_velocity + dt * dynamics["angular_acceleration"][frame]
            rotation = rotation @ axis_angle_to_matrix((dt * angular_velocity).view(1, 3))[0]
            linear_velocity = damping * linear_velocity + dt * dynamics["linear_acceleration"][frame]
            step_velocity = rotation @ linear_velocity if self.drone_body_frame_translation else linear_velocity
            translation = translation + dt * step_velocity
        return _transform_from_rotation_matrix(torch.stack(rotations, dim=0), torch.stack(translations, dim=0))

    def pose_deltas(self, frame_indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_raw, rotation, translation, _delta = self._delta_components(frame_indices)
        return global_raw, rotation, translation

    def _delta_components(
        self,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.path_parameterization == "integrated_drone":
            return self._drone_delta_components(frame_indices)
        return self._pose_delta_components(frame_indices)

    def camera_to_world_matrices(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        _global_raw, _rotation, _translation, delta = self._delta_components(frame_indices)
        return self.base_camera_to_world_matrices(frame_indices, device=delta.device, dtype=delta.dtype) @ delta

    def base_camera_to_world_matrices(
        self,
        frame_indices: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        base = self.base_camera_to_world
        if base.dim() == 2:
            count = self.frame_count if frame_indices is None else int(frame_indices.numel())
            base = base.unsqueeze(0).expand(count, -1, -1)
        elif frame_indices is not None:
            base = base[frame_indices.to(device=base.device, dtype=torch.long)]
        if device is not None or dtype is not None:
            base = base.to(device=device or base.device, dtype=dtype or base.dtype)
        return base

    def _make_camera(self, camera_to_world: torch.Tensor) -> CameraSpec:
        intrinsics = self.base_intrinsics.to(device=camera_to_world.device, dtype=camera_to_world.dtype)
        distortion = None
        if self.base_distortion.numel() > 0:
            distortion = self.base_distortion.to(device=camera_to_world.device, dtype=camera_to_world.dtype)
        return CameraSpec(
            fx=intrinsics[0],
            fy=intrinsics[1],
            cx=intrinsics[2],
            cy=intrinsics[3],
            camera_to_world=camera_to_world,
            lens_model=self.lens_model,
            distortion=distortion,
        )

    def camera_state(self, frame_indices: torch.Tensor | None = None) -> CameraState:
        global_raw, rotation, translation = self.pose_deltas(frame_indices)
        path_residuals = torch.cat([rotation, translation], dim=-1)
        return CameraState(
            fov_degrees=rotation.new_tensor(self.fov_degrees),
            radius=rotation.new_tensor(self.base_radius),
            global_residuals=global_raw.squeeze(0),
            rotation_delta=rotation,
            translation_delta=translation,
            path_residuals=path_residuals,
        )

    def cameras(self, frame_indices: torch.Tensor | None = None) -> tuple[CameraSpec, ...]:
        return tuple(self._make_camera(c2w) for c2w in self.camera_to_world_matrices(frame_indices))

    def forward(self, frame_indices: torch.Tensor | None = None) -> PowerFoamImplicitCameraBatch:
        c2w = self.camera_to_world_matrices(frame_indices)
        return PowerFoamImplicitCameraBatch(
            cameras=tuple(self._make_camera(matrix) for matrix in c2w),
            camera_to_world=c2w,
            camera_state=self.camera_state(frame_indices),
        )

    def rays(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
        frame_indices: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        pixel_center: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ray_bundles = [
            build_camera_rays(
                camera,
                height=int(height or self.image_size),
                width=int(width or self.image_size),
                device=camera.camera_to_world.device,
                dtype=dtype,
                pixel_center=float(pixel_center),
            )
            for camera in self.cameras(frame_indices)
        ]
        origins = torch.stack([bundle[0] for bundle in ray_bundles], dim=0)
        directions = torch.stack([bundle[1] for bundle in ray_bundles], dim=0)
        return origins, directions

    def regularization_terms(self, frame_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        state = self.camera_state(frame_indices)
        rotation_l2 = state.rotation_delta.square().mean()
        translation_l2 = state.translation_delta.square().mean()
        if state.rotation_delta.shape[0] > 1:
            motion = state.motion_features()
            temporal_l2 = (motion[1:] - motion[:-1]).square().mean()
        else:
            temporal_l2 = rotation_l2.new_tensor(0.0)
        terms = {
            "camera_rotation_l2": rotation_l2,
            "camera_translation_l2": translation_l2,
            "camera_temporal_l2": temporal_l2,
            "camera_global_l2": state.global_residuals.square().mean(),
        }
        if self.path_parameterization == "integrated_drone":
            dynamics = self._bounded_drone_dynamics()
            terms.update(
                {
                    "camera_velocity_l2": dynamics["angular_velocity"].square().mean()
                    + dynamics["linear_velocity"].square().mean(),
                    "camera_acceleration_l2": dynamics["angular_acceleration"].square().mean()
                    + dynamics["linear_acceleration"].square().mean(),
                    "camera_gimbal_l2": dynamics["gimbal_rotation"].square().mean(),
                }
            )
        return terms

    def regularization_loss(
        self,
        *,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        temporal_weight: float = 0.0,
        global_weight: float = 1.0,
        frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        terms = self.regularization_terms(frame_indices)
        return (
            float(rotation_weight) * terms["camera_rotation_l2"]
            + float(translation_weight) * terms["camera_translation_l2"]
            + float(temporal_weight) * terms["camera_temporal_l2"]
            + float(global_weight) * terms["camera_global_l2"]
        )

    @torch.no_grad()
    def metrics(self, frame_indices: torch.Tensor | None = None) -> dict[str, float]:
        state = self.camera_state(frame_indices)
        return {
            "Camera/FovDegrees": float(state.fov_degrees.item()),
            "Camera/Radius": float(state.radius.item()),
            "Camera/RotationDeltaMeanDegrees": float(
                torch.rad2deg(torch.linalg.norm(state.rotation_delta, dim=-1)).mean().item()
            ),
            "Camera/TranslationDeltaMean": float(torch.linalg.norm(state.translation_delta, dim=-1).mean().item()),
            "Camera/GlobalResidualL2": float(state.global_residuals.square().mean().item()),
        }


__all__ = ["PowerFoamImplicitCameraBatch", "PowerFoamImplicitCameraDecoder"]
