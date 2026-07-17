# Render Dispatch Boundary Cleanup

## Goal

Continue the trainer modularization work without inventing a large trainer
framework. This slice targeted the model-aware renderer selection helpers that
were still living in `train_video_token_implicit_dynamic.py` and being imported
from the relative-pose trainer.

## What Changed

- Added `src/train/render_dispatch.py`.
- Moved decoded-token counting, token-layout detail-level accounting,
  token-summary text, effective Gaussian counting, and
  `pick_renderer_mode_from_config(...)` into that module.
- Updated `train_video_token_implicit_dynamic.py` to import the helpers from
  `render_dispatch.py`.
- Updated `train_multicam_relative_pose_implicit_dynamic.py` to import
  `pick_renderer_mode_from_config(...)` and `token_layout_detail_levels(...)`
  from `render_dispatch.py` instead of importing from the token-GS trainer.
- Added `tests/test_render_dispatch.py` to cover legacy token counts,
  token-layout counts, invalid active detail levels, token summaries, and the
  dense/tiled renderer auto-selection path.

This is deliberately a narrow boundary: it does not change renderer policy, low
level Metal kernels, alpha-aware render payloads, or trainer scheduling.

## Validation

- `py_compile` passed for:
  - `src/train/render_dispatch.py`
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_multicam_relative_pose_implicit_dynamic.py`
  - `tests/test_render_dispatch.py`
- Focused tests passed:
  - `tests/test_render_dispatch.py`
  - `tests/test_multicam_relative_pose_trainer.py`
  - result: `17 passed in 3.18s`
- Import audit passed: no remaining `from train_video_token_implicit_dynamic import (...)`
  dependency for renderer-mode helpers, no `resolve_renderer_mode`, and no
  `model_token_layout_detail_levels` references under `src/train` or `tests`.
- Runtime smoke through `src/train/train.py` passed with
  `src/train_configs/local_mac_single_video_pretrain_100_local_encoder_64f_tiny_smoke.jsonc`.
  The run used MPS, loaded the lazy 100-sequence sampler, selected the dense
  renderer for the tiny effective Gaussian count, completed the 1-step smoke,
  and wrote offline W&B output under
  `wandb/offline-run-20260521_221651-ls53zqtb`.

## Remaining Gaps

- P4 render-dispatch convergence is not finished. The active clip render path
  uses `runtime_types.RasterizedClip`, but lower-level renderer wrappers are
  still not fully normalized around one alpha-aware return contract.
- There is still no broader `RenderConfig` dataclass. That may be useful later,
  but this slice intentionally avoided a larger abstraction.
- The mixed same-view plus heldout-view bridge has only smoke evidence. It
  still needs a W&B-enabled benchmark trace before it becomes a real baseline
  claim.
- Cleanup smokes prove that the refactored code paths execute. They do not
  prove convergence quality or solve the STAR UVT visual-quality problem.
