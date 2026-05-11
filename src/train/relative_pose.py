from __future__ import annotations

import math

import torch
import torch.nn as nn

from camera import make_camera_like
from camera_rig import axis_angle_to_matrix
from gs_models.dynamic_video_token_gs_implicit_camera import QueryCrossAttentionBlock


class RelativePoseCrossAttentionHead(nn.Module):
    """Predict a bounded source-to-target SE(3) residual from paired memories."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        layers: int = 1,
        query_count: int = 1,
        mlp_ratio: float = 2.0,
        hidden_dim: int | None = None,
        query_init_std: float = 0.02,
        output_init_std: float = 0.0,
        pair_delta_init_std: float = 0.0,
        max_rotation_degrees: float = 5.0,
        max_translation: float = 0.15,
    ) -> None:
        super().__init__()
        if query_count < 1:
            raise ValueError(f"query_count must be >= 1, got {query_count}.")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if output_init_std < 0:
            raise ValueError(f"output_init_std must be >= 0, got {output_init_std}.")
        if pair_delta_init_std < 0:
            raise ValueError(f"pair_delta_init_std must be >= 0, got {pair_delta_init_std}.")
        self.dim = int(dim)
        self.query_count = int(query_count)
        self.max_rotation_radians = math.radians(float(max_rotation_degrees))
        self.max_translation = float(max_translation)
        self.source_role = nn.Parameter(torch.zeros(self.dim))
        self.target_role = nn.Parameter(torch.zeros(self.dim))
        self.query_tokens = nn.Parameter(torch.randn(self.query_count, self.dim) * float(query_init_std))
        self.blocks = nn.ModuleList(
            [
                QueryCrossAttentionBlock(dim=self.dim, num_heads=int(num_heads), mlp_ratio=float(mlp_ratio))
                for _ in range(int(layers))
            ]
        )
        hidden = int(hidden_dim or self.dim)
        self.output = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        if float(pair_delta_init_std) > 0.0:
            self.pair_delta_norm = nn.LayerNorm(self.dim)
            self.pair_delta_output = nn.Linear(self.dim, 6, bias=False)
            nn.init.normal_(self.pair_delta_output.weight, mean=0.0, std=float(pair_delta_init_std))
        else:
            self.pair_delta_norm = None
            self.pair_delta_output = None
        final = self.output[-1]
        if float(output_init_std) > 0.0:
            nn.init.normal_(final.weight, mean=0.0, std=float(output_init_std))
        else:
            nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, source_memory: torch.Tensor, target_memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if source_memory.ndim != 3 or target_memory.ndim != 3:
            raise ValueError(
                f"Expected source/target memory as [B,N,C], got {tuple(source_memory.shape)} "
                f"and {tuple(target_memory.shape)}."
            )
        if source_memory.shape[0] != target_memory.shape[0]:
            raise ValueError("source_memory and target_memory must have the same batch size.")
        if source_memory.shape[-1] != self.dim or target_memory.shape[-1] != self.dim:
            raise ValueError(
                f"Expected memory dim {self.dim}, got {source_memory.shape[-1]} and {target_memory.shape[-1]}."
            )

        dtype = self.query_tokens.dtype
        source = source_memory.to(dtype=dtype) + self.source_role.view(1, 1, -1)
        target = target_memory.to(dtype=dtype) + self.target_role.view(1, 1, -1)
        memory = torch.cat([source, target], dim=1)
        queries = self.query_tokens.to(device=memory.device, dtype=dtype).unsqueeze(0).expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        raw = self.output(queries[:, 0, :])
        if self.pair_delta_norm is not None and self.pair_delta_output is not None:
            pair_delta = self.pair_delta_norm(target.mean(dim=1) - source.mean(dim=1))
            raw = raw + self.pair_delta_output(pair_delta)
        rotation = torch.tanh(raw[:, :3]) * self.max_rotation_radians
        translation = torch.tanh(raw[:, 3:]) * self.max_translation
        return rotation, translation


def compose_cameras_with_se3_residual(cameras, rotation: torch.Tensor, translation: torch.Tensor):
    if rotation.shape != (1, 3) or translation.shape != (1, 3):
        raise ValueError(
            f"Expected one residual rotation/translation with shape [1,3], got {tuple(rotation.shape)} "
            f"and {tuple(translation.shape)}."
        )
    residual = torch.eye(4, device=rotation.device, dtype=rotation.dtype)
    residual[:3, :3] = axis_angle_to_matrix(rotation)[0]
    residual[:3, 3] = translation[0]
    return tuple(
        make_camera_like(
            camera,
            camera_to_world=camera.camera_to_world.to(device=rotation.device, dtype=rotation.dtype) @ residual,
        )
        for camera in cameras
    )


def cameras_with_se3_transform(cameras, camera_to_world: torch.Tensor):
    if camera_to_world.ndim == 2:
        if camera_to_world.shape != (4, 4):
            raise ValueError(f"Expected camera_to_world shape [4,4], got {tuple(camera_to_world.shape)}.")
        transforms = camera_to_world.unsqueeze(0).expand(len(cameras), -1, -1)
    elif camera_to_world.ndim == 3:
        if camera_to_world.shape[-2:] != (4, 4):
            raise ValueError(f"Expected camera_to_world shape [B,4,4], got {tuple(camera_to_world.shape)}.")
        if camera_to_world.shape[0] == 1:
            transforms = camera_to_world.expand(len(cameras), -1, -1)
        elif camera_to_world.shape[0] == len(cameras):
            transforms = camera_to_world
        else:
            raise ValueError(
                f"Transform batch {camera_to_world.shape[0]} must be 1 or match camera count {len(cameras)}."
            )
    else:
        raise ValueError(f"Expected camera_to_world as [4,4] or [B,4,4], got {tuple(camera_to_world.shape)}.")
    return tuple(make_camera_like(camera, camera_to_world=transform) for camera, transform in zip(cameras, transforms))


def se3_residual_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    if rotation.ndim != 2 or translation.ndim != 2 or rotation.shape != translation.shape or rotation.shape[-1] != 3:
        raise ValueError(
            f"Expected rotation/translation as matching [B,3] tensors, got {tuple(rotation.shape)} "
            f"and {tuple(translation.shape)}."
        )
    residual = torch.eye(4, device=rotation.device, dtype=rotation.dtype).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    residual[:, :3, :3] = axis_angle_to_matrix(rotation)
    residual[:, :3, 3] = translation
    return residual


def compose_transform_with_se3_residual(
    camera_to_world: torch.Tensor,
    rotation: torch.Tensor | None,
    translation: torch.Tensor | None,
) -> torch.Tensor:
    composed = camera_to_world
    if rotation is None or translation is None:
        return composed
    residual = se3_residual_matrix(rotation, translation)
    if composed.ndim == 2:
        return composed.to(device=rotation.device, dtype=rotation.dtype) @ residual[0]
    if composed.ndim == 3:
        if residual.shape[0] != 1 and residual.shape[0] != composed.shape[0]:
            raise ValueError(
                f"Residual batch {residual.shape[0]} must be 1 or match transform batch {composed.shape[0]}."
            )
        if residual.shape[0] == 1:
            residual = residual.expand(composed.shape[0], -1, -1)
        return composed.to(device=rotation.device, dtype=rotation.dtype) @ residual
    raise ValueError(f"Expected camera_to_world as [4,4] or [B,4,4], got {tuple(composed.shape)}.")


def se3_cycle_loss(source_to_target_c2w: torch.Tensor, target_to_source_c2w: torch.Tensor) -> torch.Tensor:
    if source_to_target_c2w.shape != target_to_source_c2w.shape or source_to_target_c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected matching [...,4,4] transforms, got {tuple(source_to_target_c2w.shape)} "
            f"and {tuple(target_to_source_c2w.shape)}."
        )
    eye = torch.eye(
        4,
        device=source_to_target_c2w.device,
        dtype=source_to_target_c2w.dtype,
    )
    cycle = source_to_target_c2w @ target_to_source_c2w
    return (cycle - eye).square().mean()


def se3_transform_l2_loss(predicted_c2w: torch.Tensor, target_c2w: torch.Tensor) -> torch.Tensor:
    if predicted_c2w.shape != target_c2w.shape or predicted_c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected matching [...,4,4] transforms, got {tuple(predicted_c2w.shape)} "
            f"and {tuple(target_c2w.shape)}."
        )
    return (predicted_c2w - target_c2w.to(device=predicted_c2w.device, dtype=predicted_c2w.dtype)).square().mean()


def se3_residual_identity_loss(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return rotation.square().mean() + translation.square().mean()


__all__ = [
    "RelativePoseCrossAttentionHead",
    "cameras_with_se3_transform",
    "compose_cameras_with_se3_residual",
    "compose_transform_with_se3_residual",
    "se3_cycle_loss",
    "se3_residual_matrix",
    "se3_residual_identity_loss",
    "se3_transform_l2_loss",
]
