# Review Request: Trainer Consolidation Cleanup

Date: 2026-04-30

Status: request review before commit. This is now a real consolidation pass, but it still does **not** meet the original 50-75% shrink goal.

## Summary

The first refactor fixed the objective semantics but grew the repo. The follow-up cleanup deleted the legacy trainer surface and stale one-config launch wrappers so the measured train surface is now slightly smaller than the clean checkpoint `9b68192`.

Measured against `9b68192`:

```text
src/train Python:
  base:    54 files, 17366 lines
  current: 53 files, 17360 lines
  delta:   -1 file,  -6 lines

src/train + src/train_scripts Python/shell:
  base:    73 files, 18042 lines
  current: 62 files, 17905 lines
  delta:  -11 files, -137 lines

src/train_configs JSONC:
  base:    96 files, 10303 lines
  current: 82 files,  9449 lines
  delta:  -14 files, -854 lines
```

This is no longer a code-growth disaster, but it is also not the intended 50-75% deletion. Review should decide whether this is an acceptable first cleanup commit or whether the objective/factory abstractions still need to be compressed before commit.

## Major Deletions

- Deleted legacy trainers:
  - `src/train/dynamicTokenGS.py`
  - `src/train/dynamicTokenGS_tiled.py`
  - `src/train/tokenGS.py`
  - `src/train/tokenGS_tiled.py`
  - `src/train/train_camera_implicit_dynamic.py`
  - `src/train/train_camera_implict_dynamic.py`
  - `src/train/train_image_encoder_implicit_camera_baseline.py`
  - `src/train/train_ltx_feature_implicit_dynamic.py`
- Deleted stale one-config train scripts:
  - prebaked-camera wrappers
  - image-implicit wrapper
  - LTX/Wan/V-JEPA precomputed wrappers
  - old smoke wrapper
  - old multicam static/dynamic wrapper
- Deleted stale legacy configs for the removed arch values:
  - `tokengs_prebaked_camera*`
  - `tokengs_single_image*`
  - `tokengs_image_implicit_camera*`

## Major Additions

- `src/train/train.py`
  - single config-dispatch entrypoint based on top-level `arch`
  - replaces the deleted one-config scripts for active trainer families
- `src/train/objective/`
  - shared `TargetView -> RasterizedView -> RenderedView` RGB reconstruction path
  - alpha-aware composition and random train background live in one objective
  - used by both single-cam and multicam trainers
- `src/train/model_factories.py`
  - centralized model/colorizer constructor kwargs boundary
  - rejects unknown model/camera/colorize config keys

## Behavior Fixed

- F=32 feature splatting no longer compares raw features to RGB in multicam.
- Multicam now uses the same objective semantics as single-cam:
  - feature colorizer
  - alpha-aware RGB composition
  - random train background
  - white eval background
- Multicam heldout logging now includes alpha/PCA/composite diagnostics.
- The import dependency on `dynamicTokenGS.py` was removed; `pick_device` now lives beside the fast-attention helpers in `fast_attn.py`.

## Review Risks

1. Deleting the old prebaked-camera, TokenGS single-image, and image-implicit config lanes is intentionally breaking. They are now available only from git history.
2. `src/train/objective/` may still be larger than necessary for the current trainer surface.
3. `src/train/model_factories.py` is smaller than the first attempt, but still a large constructor map.
4. `src/train/train_video_token_implicit_dynamic.py` remains the 1900-line core monolith; this pass consolidated around it rather than splitting it deeply.
5. The result shrinks files and lines slightly, but it is not close to the requested 50-75% reduction.

## Verification Run

Compile:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train.py \
  src/train/fast_attn.py \
  src/train/model_factories.py \
  src/train/objective/*.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train/probe_init_diagnostics.py \
  src/train/export_dynaworld_browser_bundle.py \
  src/train/init_diagnostics.py
```

Tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_objective_background_and_composition.py \
  tests/test_rgb_recon_objective.py \
  tests/test_config_factory_helpers.py \
  tests/test_fast_mac_feature_background.py -q
```

Result:

```text
18 passed in 0.45s
```

Runtime smokes through the new dispatcher:

```text
F=3 single-cam:          wandb/offline-run-20260430_124127-7pzvjtjk
F=32 single-cam:         wandb/offline-run-20260430_124137-xhowbvfd
F=32 multicam ultimate:  wandb/offline-run-20260430_124212-x2sqanwc
```

`git diff --check` passed.

## Review Ask

Review this as a cleanup commit with a hard eye on whether the deleted legacy surface justifies the new objective/factory modules. The key question is not "does it pass tests"; it is whether this is the right first consolidation commit or whether we should compress `objective/` and `model_factories.py` further before committing.
