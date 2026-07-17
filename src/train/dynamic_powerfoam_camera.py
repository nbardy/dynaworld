from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from json_io import load_json
from pipeline.diagnostics import camera_state_summary_metrics
from powerfoam_implicit_camera import PowerFoamImplicitCameraDecoder
from train_artifacts import write_json


def build_camera_decoder(cfg: dict[str, Any], *, frame_count: int) -> PowerFoamImplicitCameraDecoder | None:
    camera_cfg = cfg["camera"]
    if not bool(camera_cfg["enabled"]):
        return None
    base_fov = cfg["render"]["fov_degrees"] if camera_cfg["base_fov_degrees"] is None else camera_cfg["base_fov_degrees"]
    basis_count = cfg["model"]["time_basis_count"] if camera_cfg["time_basis_count"] is None else camera_cfg["time_basis_count"]
    sigma_scale = (
        cfg["model"]["time_basis_sigma_scale"]
        if camera_cfg["time_basis_sigma_scale"] is None
        else camera_cfg["time_basis_sigma_scale"]
    )
    max_translation = (
        float(camera_cfg["base_radius"]) * float(camera_cfg["max_translation_ratio"])
        if camera_cfg["max_translation"] is None
        else float(camera_cfg["max_translation"])
    )
    return PowerFoamImplicitCameraDecoder(
        frame_count=int(frame_count),
        image_size=int(cfg["render"]["render_size"]),
        fov_degrees=float(base_fov),
        base_radius=float(camera_cfg["base_radius"]),
        token_dim=int(camera_cfg["token_dim"]),
        hidden_dim=int(camera_cfg["hidden_dim"]),
        time_basis_count=int(basis_count),
        time_basis_sigma_scale=float(sigma_scale),
        token_init_std=float(camera_cfg["token_init_std"]),
        max_rotation_degrees=float(camera_cfg["max_rotation_degrees"]),
        max_translation=max_translation,
        base_position=camera_cfg["base_position"],
        look_at=camera_cfg["look_at"],
        up=camera_cfg["up"],
        base_path_mode=str(camera_cfg["base_path_mode"]),
        path_parameterization=str(camera_cfg["path_parameterization"]),
        orbit_yaw_start_degrees=float(camera_cfg["orbit_yaw_start_degrees"]),
        orbit_yaw_end_degrees=float(camera_cfg["orbit_yaw_end_degrees"]),
        orbit_pitch_degrees=float(camera_cfg["orbit_pitch_degrees"]),
        drone_integration_horizon=float(camera_cfg["drone_integration_horizon"]),
        drone_damping=float(camera_cfg["drone_damping"]),
        drone_max_linear_velocity_ratio=float(camera_cfg["drone_max_linear_velocity_ratio"]),
        drone_max_linear_acceleration_ratio=float(camera_cfg["drone_max_linear_acceleration_ratio"]),
        drone_max_angular_velocity_degrees=float(camera_cfg["drone_max_angular_velocity_degrees"]),
        drone_max_angular_acceleration_degrees=float(camera_cfg["drone_max_angular_acceleration_degrees"]),
        drone_gimbal_max_rotation_degrees=float(camera_cfg["drone_gimbal_max_rotation_degrees"]),
        drone_body_frame_translation=bool(camera_cfg["drone_body_frame_translation"]),
        initial_zoom_steps=int(camera_cfg["initial_zoom_steps"]),
        initial_zoom_translation=float(camera_cfg["initial_zoom_translation"]),
        lens_model=str(camera_cfg["lens_model"]),  # type: ignore[arg-type]
        distortion=camera_cfg["distortion"],
    )


def camera_param_group(
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    train_cfg: dict[str, Any],
) -> dict[str, object] | None:
    if camera_decoder is None:
        return None
    return {
        "params": list(camera_decoder.parameters()),
        "lr": float(train_cfg["camera_lr_multiplier"]) * float(train_cfg["lr"]),
        "name": "implicit_camera",
    }


def camera_regularization(
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    if camera_decoder is None:
        return None, {}
    terms = camera_decoder.regularization_terms()
    motion = terms["camera_rotation_l2"] + terms["camera_translation_l2"]
    loss = (
        float(loss_cfg["camera_motion_weight"]) * motion
        + float(loss_cfg["camera_temporal_weight"]) * terms["camera_temporal_l2"]
        + float(loss_cfg["camera_global_weight"]) * terms["camera_global_l2"]
    )
    for key, weight_key in (
        ("camera_velocity_l2", "camera_velocity_weight"),
        ("camera_acceleration_l2", "camera_acceleration_weight"),
        ("camera_gimbal_l2", "camera_gimbal_weight"),
    ):
        if key in terms:
            loss = loss + float(loss_cfg[weight_key]) * terms[key]
    return loss, terms


def compact_camera_metrics(camera_decoder: PowerFoamImplicitCameraDecoder | None) -> dict[str, float]:
    if camera_decoder is None:
        return {}
    state = camera_decoder.camera_state()
    c2w = camera_decoder.camera_to_world_matrices()
    base = camera_decoder.base_camera_to_world_matrices(device=c2w.device, dtype=c2w.dtype)
    origin_delta = torch.linalg.vector_norm(c2w[:, :3, 3] - base[:, :3, 3], dim=-1)
    forward_delta = torch.linalg.vector_norm(c2w[:, :3, 2] - base[:, :3, 2], dim=-1)
    state_metrics = camera_state_summary_metrics(state)
    metrics = {
        "state_camera_fov_degrees": state_metrics["fov_degrees"],
        "state_camera_radius": state_metrics["radius"],
        "state_camera_rotation_delta_mean_degrees": state_metrics["rotation_delta_mean_degrees"],
        "state_camera_translation_delta_mean": state_metrics["translation_delta_mean"],
        "state_camera_origin_delta_mean": float(origin_delta.mean().detach().cpu()),
        "state_camera_forward_delta_mean": float(forward_delta.mean().detach().cpu()),
        "state_camera_global_residual_l2": float(state.global_residuals.square().mean().detach().cpu()),
        "state_camera_active_frames": float(camera_decoder.active_frame_count or camera_decoder.frame_count),
    }
    terms = camera_decoder.regularization_terms()
    for key in ("camera_velocity_l2", "camera_acceleration_l2", "camera_gimbal_l2"):
        if key in terms:
            metrics[f"state_{key}"] = float(terms[key].detach().cpu())
    return metrics


def _camera_pose_geodesic(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = predicted[:, :3, :3].transpose(1, 2) @ target[:, :3, :3]
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.where(
        cos_angle > 1.0 - 1.0e-6,
        torch.zeros_like(cos_angle),
        torch.acos(cos_angle.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)),
    )


def _camera_velocity_alignment(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.shape[0] < 2:
        return predicted.new_zeros(())
    pred_translation_delta = predicted[1:, :3, 3] - predicted[:-1, :3, 3]
    target_translation_delta = target[1:, :3, 3] - target[:-1, :3, 3]
    pred_relative = predicted[:-1, :3, :3].transpose(1, 2) @ predicted[1:, :3, :3]
    target_relative = target[:-1, :3, :3].transpose(1, 2) @ target[1:, :3, :3]
    trace = (pred_relative.transpose(1, 2) @ target_relative).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.where(
        cos_angle > 1.0 - 1.0e-6,
        torch.zeros_like(cos_angle),
        torch.acos(cos_angle.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)),
    )
    return F.mse_loss(pred_translation_delta, target_translation_delta) + angle.square().mean()


def _extract_camera_to_world_records(payload: Any) -> torch.Tensor:
    if isinstance(payload, dict):
        for key in ("camera_to_world", "camera_to_world_matrices", "cameras"):
            if key in payload:
                return _extract_camera_to_world_records(payload[key])
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return torch.tensor([record["camera_to_world"] for record in payload], dtype=torch.float32)
    tensor = torch.as_tensor(payload, dtype=torch.float32)
    if tensor.ndim == 2 and tuple(tensor.shape) == (4, 4):
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tuple(tensor.shape[-2:]) != (4, 4):
        raise ValueError(f"Expected camera-to-world payload shape [T,4,4], got {tuple(tensor.shape)}")
    return tensor


def load_teacher_camera_to_world(camera_cfg: dict[str, Any], frame_count: int) -> torch.Tensor | None:
    path = camera_cfg["init_teacher_path"]
    if path is None:
        return None
    teacher = _extract_camera_to_world_records(load_json(path))
    if teacher.shape[0] < int(frame_count):
        raise ValueError(f"camera.init_teacher_path has {teacher.shape[0]} frames but training needs {frame_count}")
    teacher = teacher[: int(frame_count)]
    if bool(camera_cfg["init_teacher_normalize_to_first"]):
        teacher = torch.linalg.inv(teacher[0]) @ teacher
    return teacher


def camera_teacher_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    camera_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    rotation = _camera_pose_geodesic(predicted, target).square().mean()
    translation = F.mse_loss(predicted[:, :3, 3], target[:, :3, 3])
    velocity = _camera_velocity_alignment(predicted, target)
    loss = (
        float(camera_cfg["init_teacher_rotation_weight"]) * rotation
        + float(camera_cfg["init_teacher_translation_weight"]) * translation
        + float(camera_cfg["init_teacher_velocity_weight"]) * velocity
    )
    return loss, {
        "teacher_camera_loss": float(loss.detach().cpu()),
        "teacher_camera_rotation_loss": float(rotation.detach().cpu()),
        "teacher_camera_translation_loss": float(translation.detach().cpu()),
        "teacher_camera_velocity_loss": float(velocity.detach().cpu()),
    }


def prefit_camera_decoder_from_teacher(
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    cfg: dict[str, Any],
    *,
    frame_count: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float]:
    if camera_decoder is None or int(cfg["camera"]["init_teacher_steps"]) == 0:
        return {}
    camera_decoder.to(device)
    teacher = load_teacher_camera_to_world(cfg["camera"], int(frame_count))
    if teacher is None:
        return {}
    teacher = teacher.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(camera_decoder.parameters(), lr=float(cfg["camera"]["init_teacher_lr"]))
    last_metrics: dict[str, float] = {}
    for _step in range(1, int(cfg["camera"]["init_teacher_steps"]) + 1):
        predicted = camera_decoder.camera_to_world_matrices()
        loss, last_metrics = camera_teacher_alignment_loss(predicted, teacher, cfg["camera"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    last_metrics = {"teacher_camera_init_steps": float(cfg["camera"]["init_teacher_steps"]), **last_metrics}
    write_json(output_dir / "camera_teacher_init_metrics.json", last_metrics)
    return last_metrics


def decoded_powerfoam_rays(
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    fixed_rays: torch.Tensor,
    frame_indices: torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if camera_decoder is None:
        return fixed_rays.to(device=device, dtype=dtype)
    origins, directions = camera_decoder.rays(
        height=height,
        width=width,
        frame_indices=frame_indices,
        dtype=dtype,
    )
    return torch.cat([origins, directions], dim=-1).to(device=device, dtype=dtype).contiguous()
