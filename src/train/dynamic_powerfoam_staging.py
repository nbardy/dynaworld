from __future__ import annotations

from typing import Any


def camera_curriculum_active_frames(cfg: dict[str, Any], step: int, frame_count: int) -> int:
    if not bool(cfg["train"].get("camera_curriculum_enabled", False)):
        return int(frame_count)
    active_frames = 1
    for start_step, schedule_frames in cfg["train"].get("camera_curriculum_schedule", []):
        if int(step) < int(start_step):
            break
        active_frames = int(schedule_frames)
    return max(1, min(int(active_frames), int(frame_count)))


def apply_training_stage(model: Any, cfg: dict[str, Any], step: int) -> dict[str, float]:
    static_only_steps = int(cfg["train"]["static_only_steps"])
    no_repaint_steps = int(cfg["train"]["no_repaint_steps"])
    static_only = static_only_steps > 0 and int(step) <= static_only_steps
    no_repaint = no_repaint_steps > 0 and int(step) <= no_repaint_steps
    active_frame_count = camera_curriculum_active_frames(cfg, step, int(getattr(model, "frame_count", 1)))
    controls = {
        "stage_temporal_geometry_scale": 0.0 if static_only else 1.0,
        "stage_temporal_feature_scale": 0.0 if no_repaint else 1.0,
        "stage_camera_active_frames": float(active_frame_count),
    }
    if hasattr(model, "set_training_controls"):
        model.set_training_controls(
            temporal_geometry_scale=controls["stage_temporal_geometry_scale"],
            temporal_feature_scale=controls["stage_temporal_feature_scale"],
        )
    camera_decoder = getattr(model, "camera_decoder", None)
    if camera_decoder is not None and hasattr(camera_decoder, "set_active_frame_count"):
        camera_decoder.set_active_frame_count(active_frame_count)
    return controls
