# Frame Sampling Config

## What changed

- Added `src/train/temporal_sampling.py` as the shared temporal-frame sampler.
- Supported modes:
  - `random`: sorted unique random subset of `model.train_frame_count` frames.
  - `contiguous`: legacy random contiguous window behavior.
  - `temporal-dilation`: centered power-of-two stencil with default offsets `[-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]`.
- Temporal dilation wraps with modulo indexing when offsets fall outside sequence bounds.
- Accepted `contigous` as a typo alias, but normalized resolved configs to `contiguous`.
- Wired base video-token trainer and multicam precomputed trainer through the shared sampler, covering inherited relative-pose / camera-swap training paths.
- Left current behavior unchanged by default: missing `train.frame_sampling` resolves to `{"mode": "contiguous"}`.
- Added explicit contiguous sampling config to the two latest 1920 multires configs so the frame policy is visible alongside the resolution schedule.

## Important contract

- `temporal-dilation` returns one frame per configured offset.
- The default stencil has 11 offsets, so configs using it should set `model.train_frame_count` to `11`, or provide a custom `train.frame_sampling.offsets` list whose length matches the existing frame count.
- The resolver raises early if `temporal-dilation` offset count and `model.train_frame_count` disagree; this avoids silently feeding a different temporal length into models or feature caches.

## Validation

- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_temporal_sampling.py tests/test_multicam_relative_pose_trainer.py tests/test_config_factory_helpers.py -q`
  - `32 passed`
- `PYTHONPATH=src/train uv run python -m py_compile src/train/temporal_sampling.py src/train/train_video_token_implicit_dynamic.py src/train/train_multicam_precomputed_feature_implicit_dynamic.py src/train/train_multicam_relative_pose_implicit_dynamic.py tests/test_temporal_sampling.py`
  - passed
