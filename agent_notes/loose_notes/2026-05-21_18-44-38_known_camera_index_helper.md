# Known-Camera Index Helper

## Context

The modularization pass is keeping shared mechanics behind small helpers while
leaving trainer math explicit. In the known-camera branch, camera tuple
extraction had drifted into multiple local shapes:

- `sample_clip(...)` validated `clip.cameras`.
- `initial_step_result(...)` manually indexed `sequence_data.cameras`.
- `_eval_decode_clip(...)` manually indexed `sequence_data.cameras` again.

The repeated code was small, but it was the same known-camera contract:
given a `SequenceData` and frame indices, return the camera tuple or fail with
the known-camera missing-camera error.

## Change

- Added `KnownCameraTrainer.known_cameras_for_indices(...)`.
- Updated known-camera initial eval to use it.
- Updated known-camera full-sequence eval decode to use it.
- Updated known-camera train step to use the existing `sample_clip(...)`
  boundary, which already validates sampled camera batches.

No rendering, loss, backward, or camera math changed.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile` passed for:
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_relative_pose_implicit_dynamic.py`
  - `src/train/train_mixed_same_heldout_implicit_dynamic.py`
- `git diff --check` passed on `src/train/train_video_token_implicit_dynamic.py`.
- A lightweight runtime smoke verified:
  - selected camera tuple ordering for indices `[0, 2, 4]`,
  - the expected `ValueError` when `SequenceData.cameras is None`.

This is known-camera wiring evidence only. It does not prove convergence,
renderer math, or visual quality.

## Next

Keep extracting only repeated mechanics visible in the current tree. Avoid
turning branch-local camera semantics into a vague generic camera abstraction.
