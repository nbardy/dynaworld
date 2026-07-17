# PowerFoam Camera Metric Helper Reuse

## Context

Continued the trainer modularization goal after the val-log, scalar-payload,
and camera-state diagnostics cleanup. The next live duplication check showed
that Token-GS had only a one-line `camera_metrics(...)` wrapper left, while
PowerFoam implicit-camera compact metrics still duplicated the same
fov/radius/rotation/translation scalar math already centralized in
`pipeline.diagnostics.camera_state_summary_metrics(...)`.

## Changes

- `src/train/dynamic_powerfoam_camera.py`
  - Imports `camera_state_summary_metrics(...)`.
  - Uses it for:
    - `state_camera_fov_degrees`
    - `state_camera_radius`
    - `state_camera_rotation_delta_mean_degrees`
    - `state_camera_translation_delta_mean`
  - Keeps PowerFoam-only metrics local:
    - origin delta
    - forward-axis delta
    - global residual L2
    - active frame count
    - velocity/acceleration/gimbal regularizer terms
- `src/train/train_video_token_implicit_dynamic.py`
  - Removed the one-line `Trainer.camera_metrics(...)` wrapper.
  - `progress_message(...)` now calls `camera_state_summary_metrics(...)`
    directly.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/dynamic_powerfoam_camera.py src/train/train_video_token_implicit_dynamic.py src/train/pipeline/diagnostics.py`
- `rtk rg -n "def camera_metrics\\(|self\\.camera_metrics\\(|state_camera_(fov_degrees|radius|rotation_delta_mean_degrees|translation_delta_mean)" src/train tests`
  - Confirmed no `camera_metrics(...)` wrapper/call remains.
  - The remaining `state_camera_*` hits are the PowerFoam key mapping and
    downstream summary consumers.

## Notes

This is a helper-reuse cleanup, not a training-quality claim. It should not go
into `agent_notes/key_learnings.md`: the key lesson already exists, and this
file is at 199 lines. The useful invariant is simply that `CameraState` scalar
math should live in `pipeline.diagnostics`, while trainer-specific metric names
and extra branch-only diagnostics stay near their owning trainer.
