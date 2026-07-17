# Tensor Cache Atomic Save Boundary

## Context

Continued the trainer modularization cleanup by removing the remaining
hand-rolled tensor-cache save pattern in `src/train`. The checkpoint and STAR
paths already used `checkpoint_utils.atomic_torch_save(...)`; the direct
video-window frame cache and V-JEPA feature cache still had local
`torch.save(tmp_path); tmp_path.replace(path)` blocks.

## Changes

- `sequence_data._save_frame_cache(...)` now calls
  `checkpoint_utils.atomic_torch_save(...)`.
- `video_feature_cache.VideoFeatureCache._save_cached(...)` now calls
  `checkpoint_utils.atomic_torch_save(...)`.
- Cache keys, fingerprints, payload schemas, and load validation remain local
  to their owning modules.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

Commands run:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/sequence_data.py \
  src/train/video_feature_cache.py \
  src/train/checkpoint_utils.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_sequence_data_single_frame.py::test_load_video_window_sequence_reuses_frame_cache \
  tests/test_video_feature_cache.py::test_video_feature_cache_hit_miss_and_key_busting \
  tests/test_powerfoam_direct.py::test_atomic_torch_save_preserves_existing_checkpoint_on_failure \
  -q

rg -n "torch\\.save\\(|tmp_path = path\\.with_suffix|atomic_torch_save" \
  src/train/sequence_data.py src/train/video_feature_cache.py src/train/checkpoint_utils.py
```

Results:

- `py_compile` passed for the touched modules.
- Focused cache/atomic tests: `3 passed`.
- The only remaining `torch.save(...)` among those files is inside
  `checkpoint_utils.atomic_torch_save(...)`.

## Remaining

This only centralizes cache persistence. It does not alter dataset semantics,
feature extraction, cache keys, or trainer behavior. Further cleanup should
continue from live duplication scans, not historical delete lists.
