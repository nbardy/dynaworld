# Review Request: Desloppify Cleanup After Code-Growth Review

Date: 2026-04-30

Status: review before commit. This pass corrected the worst code-growth mistakes
from the first desloppify run, but it still is not the originally requested
50-75% shrink.

## Current Measurement

Tracked diff only:

```text
staged total:           +822 / -968 = -146 lines
staged prod py/js:      +509 / -820 = -311 lines
staged tests:           +39  / -29  = +10 lines
tracked config/lock:    +122 / -21  = +101 lines
```

Comparable tracked surfaces against `HEAD`:

```text
src/train Python:
  base:            58 files, 17865 lines
  current tracked: 52 files, 17444 lines
  delta:           -6 files,  -421 lines

src/train + src/train_scripts Python/shell:
  base:            66 files, 18272 lines
  current tracked: 60 files, 17851 lines
  delta:           -6 files,  -421 lines

src/train_configs JSONC:
  base:            82 files, 9449 lines
  current tracked: 82 files, 9449 lines
  delta:            0 files,    0 lines
```

Untracked non-generated additions remain:

```text
optional new test files                       +350
loose notes                                   +135
desloppify skill guide                        +318
```

`.desloppify/` and `scorecard.png` are now ignored because they are tool output,
not source.

## Worth Keeping

- Module init flattening:
  - `src/train/objective/__init__.py`
  - `src/train/gs_models/__init__.py`
  - `src/train/renderers/__init__.py`
- Deleted stale model/shared modules:
  - `src/train/gs_models/token_gs.py`
  - `src/train/gs_models/dynamic_token_gs.py`
  - `src/train/gs_models/dynamic_token_gs_implicit_camera.py`
  - `src/train/gs_models/dynamic_token_gs_separated_implicit_camera.py`
  - `src/train/tokenGS_shared.py`
  - `src/train/dynamicTokenGS_shared.py`
- Removed thin Trainer delegate methods and route callers to `pipeline.*`.
- Moved manifest loading into `sequence_data.py`.
- Moved model construction into `model_factories.py`.
- Moved `StepResult` into `runtime_types.py`; multicam no longer imports it
  from the executable single-cam trainer.
- Replaced tuple render output with `RasterizedClip`.
- Added `objective/choices.py::checked_choice()` as a typed helper so literal
  choice validation does not require repeated casts or hand-written if ladders.
- Kept `download_utils.py` as a small dataset-script helper that removes
  repeated download/open-json boilerplate.

## Reverted Or Trimmed From The First Attempt

- Reverted the half-done `config_utils.py` sys.path shim migration.
- Reverted `camera.py` annotation-only expansion.
- Reverted `image_utils.py` defensive URL/error wrapping.
- Removed tautological image/alpha-expansion tests.
- Removed multicam chunking feature growth from this cleanup pass.
- Excluded `research_experiments/gauge_fields` from the active desloppify scope
  because user guidance was to leave research/gauge work alone here.
- Trimmed the `video_feature_cache.py` descriptor registry back to a direct
  five-way builder plus a single cache-key list.
- Deleted unused planned `StepLosses` / `TrainStepResult` dataclasses.
- Deleted 8 unused `model.variant` alias keys from `model_factories.py` and
  the trainer/probe validation sets. Every remaining factory key is referenced
  by at least one checked-in config.

## Test Audit

Five new test files were audited:

- `tests/conftest.py`: staged. Centralizes test import paths and removes repeated
  sys.path mutation from existing tests.
- `tests/test_config_and_dataset_io.py`: left untracked for follow-up. Checks config/JSONL/download helper
  error context. This protects CLI/data-pipeline failure messages rather than
  mirroring an implementation.
- `tests/test_multicam_video_data.py`: left untracked for follow-up. Protects ViVo per-camera timestamp
  offsets and the train/heldout/condition bundle contract. Keep.
- `tests/test_pipeline_helpers.py`: left untracked for follow-up. Protects extracted render/diagnostic helper
  contracts, including missing-frame failures and diagnostic composite columns.
  The prior pure alpha-expansion tautology was removed.
- `tests/test_video_feature_cache.py`: left untracked for follow-up. Protects cache hit/miss/key-bust behavior
  and feature-channel inference. Keep.

No remaining untracked test is obviously equivalent to the deleted
`test_image_utils.py` tautologies, but they are still optional if the first
cleanup commit should be production-shrink-only.

## Remaining Risks

1. The codebase is smaller in tracked production lines, but not by the requested
   50-75%. This is a cleanup commit, not the full architectural demolition.
2. The four optional new behavior tests are intentionally left untracked so
   this cleanup commit remains a shrink. They are not tautologies, but should
   be committed separately if kept.
3. `train_video_token_implicit_dynamic.py` remains the main monolith.
   This pass reduced executable-script coupling around it but did not replace
   the inheritance chain.
4. `train_precomputed_feature_implicit_dynamic.py` still inherits from the
   single-cam trainer. That is an explicit remaining design debt.
5. `export_dynaworld_browser_bundle.py` and probe scripts still import
   `resolve_config` from the single-cam trainer. Moving config normalization out
   of that script is a logical next cleanup.

## Verification

```bash
PYTHONPATH=src:src/train:src/dataset_pipeline \
  uv run --with pytest python -m pytest tests -q
```

Result:

```text
50 passed in 1.32s
```

Compile/import smoke:

```bash
PYTHONPATH=src:src/train:src/dataset_pipeline .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train.py \
  src/train/export_dynaworld_browser_bundle.py \
  src/train/probe_init_diagnostics.py \
  src/train/probe_colorize_init.py \
  src/train/probe_colorize_matrix.py
```

Trainer dispatch smoke:

```text
local_mac_unconditioned_tokens_fast.jsonc
  -> train_video_token_implicit_dynamic.run_training
local_mac_overfit_precomputed_vjepa2_1_torchhub_vitb_384.jsonc
  -> train_precomputed_feature_implicit_dynamic.run_training
local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc
  -> train_multicam_precomputed_feature_implicit_dynamic.run_training
```

Desloppify after rescan:

```text
overall 75.8
objective 73.3
strict 75.6
verified 73.0
```

The score did not improve. The tool registered resolved work, but the remaining
score bottlenecks are stale subjective dimensions and broad test-health metrics,
not the concrete line-count cleanup.

## Review Ask

Review this as a pragmatic cleanup commit. The optional behavior tests were
split from the staged shrinkage commit; the staged diff keeps the production
helper additions required by the refactor while remaining net negative overall.
