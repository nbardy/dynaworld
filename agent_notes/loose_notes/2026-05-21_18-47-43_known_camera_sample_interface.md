# Known-Camera Sample Interface Cleanup

## Context

The active cleanup goal is to establish simple shared trainer interfaces. One
quiet interface drift remained in the known-camera branch:

- Base `Trainer.sample_clip(...)` returned `(sequence_data, clip_frames,
  clip_times)`.
- `KnownCameraTrainer.sample_clip(...)` overrode that name but returned
  `(sequence_data, clip_frames, clip_times, clip_cameras)`.

The four-value return was valid for known-camera training, but keeping it under
the same method name made the base interface ambiguous.

## Change

- Renamed the known-camera four-value helper to `sample_known_clip(...)`.
- Updated `KnownCameraTrainer.step(...)` to call `sample_known_clip(...)`.
- Left base `Trainer.sample_clip(...)` as the only `sample_clip(...)` method in
  the trainer hierarchy.

No sampling behavior, camera selection, rendering, loss, or backward logic
changed.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile` passed for:
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_relative_pose_implicit_dynamic.py`
  - `src/train/train_mixed_same_heldout_implicit_dynamic.py`
- `git diff --check` passed on `src/train/train_video_token_implicit_dynamic.py`.
- A lightweight runtime smoke verified:
  - `sample_known_clip(...)` returns the known-camera four-value batch with
    expected frames, times, and cameras,
  - `KnownCameraTrainer.sample_clip is Trainer.sample_clip`.

This is trainer-interface evidence only. It does not prove convergence,
renderer math, or visual quality.

## Next

Continue eliminating misleading trainer interfaces before adding broader
abstractions. Keep branch-specific data semantics explicit, especially
same-view versus heldout/novel-view paths.
