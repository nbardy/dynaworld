# PowerFoam Direct And Dynamic Gauge Owner Splits

## Context

After the larger owner-module batch, the only remaining routed `train_*.py`
files with warm-path trainer loops were:

- `src/train/train_powerfoam_direct.py`
- `src/train/train_dynamic_gauge_foam.py`

This slice moved those loops to non-CLI owner modules and left the historical
files as thin CLI wrappers.

## Changes

### PowerFoam Direct

- New owner: `src/train/powerfoam_direct_trainer.py`
- Thin wrapper: `src/train/train_powerfoam_direct.py`
- Registry arch: `powerfoam_direct -> powerfoam_direct_trainer.run_training`
- Initial wrapper re-exports before the later wrapper-surface trim:
  `DIRECT_POWERFOAM_DATA_KEYS`, `build_wandb_artifact_payload`,
  `load_direct_powerfoam_training_data`, `log_artifacts`, `run_training`, and
  wrapper `main`
- `powerfoam_eval_render.render_powerfoam_samples(...)` now accepts either
  tuple/list outputs or structured render results with `.rendered` and `.alpha`.
  The Direct trainer smoke exposed this because `DirectPowerFoamVideo` returns
  `PowerFoamRenderResult`, not a tuple.

Validation:

- `py_compile` passed for the Direct owner, wrapper, registry, and focused
  tests.
- Wrapper identity smoke passed.
- Focused tests passed:
  `tests/test_powerfoam_direct.py::test_powerfoam_eval_render_accepts_structured_result`,
  `tests/test_powerfoam_direct.py::test_powerfoam_direct_config_dispatches_to_trainer`,
  and `tests/test_trainer_registry.py` reported `16 passed`.
- One-step runtime smoke through `src/train/train.py` passed with:
  config `/tmp/dynaworld_powerfoam_direct_owner_split_smoke.jsonc`, CPU, 4
  frames, 32 cells, 64px render, W&B disabled, `/tmp` output, initial eval,
  one train step, and final checkpoint write.

### Dynamic Gauge Foam

- New owner: `src/train/dynamic_gauge_foam_trainer.py`
- Thin wrapper: `src/train/train_dynamic_gauge_foam.py`
- Registry arch: `dynamic_gauge_foam -> dynamic_gauge_foam_trainer.run_training`
- Initial wrapper re-exports before the later wrapper-surface trim:
  `DynamicGaugeFoamVideo`, `build_knn_edges`, `optimizer_param_groups`,
  `log_artifacts`, `run_training`, and wrapper `main`

Validation:

- `py_compile` passed for the Gauge owner, wrapper, registry, and focused
  tests.
- Wrapper identity smoke passed.
- Focused tests passed:
  `tests/test_powerfoam_direct.py::test_powerfoam_direct_config_dispatches_to_trainer`,
  `tests/test_trainer_registry.py`, and `tests/test_dynamic_gauge_foam.py`
  reported `17 passed`.
- One-step runtime smoke through `src/train/train.py` passed with:
  config `/tmp/dynaworld_dynamic_gauge_foam_owner_split_smoke.jsonc`, MPS, 4
  frames, 64 primitives, 64px render, W&B disabled, `/tmp` output, initial eval,
  one train step, and final checkpoint write.

## Follow-Up Cleanup

After the owner split, the remaining non-test wrapper imports were removed:

- `research_experiments/dynamic_foam/diagnose_powerfoam_sections.py` and
  `research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py` now
  import `MetalPowerFoamVideo` from `powerfoam_metal_trainer.py`.
- `src/train/visualize_camera_scene_diagnostic.py` now imports Dynamic
  PowerFoam model classes from `dynamic_powerfoam_metal_trainer.py`.

Validation:

- `py_compile` passed for all three scripts.
- `--help` passed for all three scripts.
- A follow-up pass moved test-side PowerFoam/Dynamic PowerFoam model imports to
  owner modules, so `rg` no longer finds active wrapper imports in `src`,
  `research_experiments`, or `tests`.

Late subagent audit cleanup:

- `research_experiments/star_uvt_feature_tubes/report_artifacts.py` now launches
  STAR UVT feature trainer subprocesses through `src/train/train.py` instead of
  the historical wrapper path.
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py`
  now inspects `star_uvt_feature_overfit_trainer.py` for implementation checks
  and `trainer_registry.py` for the STAR route instead of treating
  `train_star_uvt_feature_overfit.py` as the owner file.
- `src/train/train_star_uvt_feature_overfit.py` stopped re-exporting private
  owner helpers; it now exposes only `run_training` plus CLI `main`.
- `src/train/wandb_media.py` now owns the generic W&B image constructor used
  by `pipeline.validation_media`, removing the last direct W&B import from that
  validation-media helper layer.
- `src/train/wandb_media.py` also owns existing-file image/video media wrapping
  for row-output logging; `train_logging.py` now keeps W&B run lifecycle,
  scalar flattening, cadence, and payload submission but no longer constructs
  saved-path `wandb.Image` / `wandb.Video` objects itself.
- A follow-up boundary pass made that dependency private in `train_logging.py`;
  `add_existing_wandb_media(...)` is now public only from `wandb_media.py`.
- The explicit trainer anti-pattern scan found a tiny local `train_cfg` alias in
  `dynamic_powerfoam_staging.py`; it was removed so the helper reads the
  canonical `cfg["train"]` container at the use site.
- The same pass removed the small STAR feature-overfit `alpha_background_cfg`
  pass-through alias; the trainer reads the normalized `cfg["alpha_background"]`
  keys directly at the use site.
- `video_io.save_rgb_alpha_eval_media(...)` now owns the repeated RGB+alpha
  eval file-artifact pattern for Direct PowerFoam, shared PowerFoam eval
  artifacts, Dynamic PowerFoam Metal, and Dynamic Gauge Foam: preview PNG,
  optional heldout preview PNG, render MP4, and side-by-side MP4.
- `wandb_media.build_rgb_alpha_eval_media_payload(...)` now owns the matching
  W&B preview-plus-optional-video payload for those same RGB+alpha eval paths;
  trainers keep metric maps and branch-specific extras local.
- `train_optim.optimizer_backward_step(...)` now owns the repeated PowerFoam
  and Gauge zero-grad/backward/optional-grad-clip/optimizer-step sequence.
  The trainer owners still construct their own optimizers and parameter groups;
  the helper only removes the duplicated mechanical step block.
- `train_logging.wandb_run_lifecycle(...)` now wraps shared W&B init and
  finally-finish for Direct PowerFoam, PowerFoam Metal, Dynamic PowerFoam
  Metal, and Dynamic Gauge Foam. This makes cleanup exception-safe without
  moving trainer loops or payload schemas into a shared framework.
- `train_logging.log_wandb_run_payload_lazy(...)` now owns the disabled-W&B
  guard for expensive explicit-run payload factories. Token-GS, multicam
  relative-pose, Direct PowerFoam, shared PowerFoam eval artifacts, Dynamic
  PowerFoam Metal, and Dynamic Gauge eval/media paths now avoid building
  images/videos when no W&B run exists while keeping metric/media schemas local.
- PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic
  Gauge train-loop scalar W&B logs now use trainer-local key-map tuples with
  `train_logging.mapped_metric_payload(...)` plus the null-safe
  `log_wandb_run_payload(...)`. This removes the remaining repeated
  `if wandb_run is not None` branches in those scalar loops without moving
  user-facing metric names into a shared schema.
- The four PowerFoam/Gauge historical `train_*.py` wrappers stopped
  re-exporting owner internals. They now import/export only `run_training` plus
  CLI `main`; code that needs model classes, config helpers, or artifact helpers
  should import the owner/helper modules directly.
- The remaining Token-GS/precomputed/multicam/mixed historical `train_*.py`
  wrappers now follow the same rule: `main` plus `run_training` only. Classes,
  defaults, schedule helpers, and result dataclasses live on the owner modules
  and are reached through `trainer_registry` or direct owner imports.
- The PowerFoam/Gauge owner modules also now keep narrow `__all__` exports:
  Direct and Gauge trainer owners advertise only `run_training(...)`, and
  PowerFoam Metal advertises only its structural model/run surface. Helper
  modules remain the public home for config defaults, raster builders, geometry,
  data loading, objectives, and artifact utilities.
- A follow-up runtime pass generated temporary one-step smoke configs under
  `/tmp/dynaworld_registry_smokes` and ran them through `src/train/train.py`.
  Passing routes: `tokengs_video_implicit_camera` F=3, multicam RGB-pyramid,
  mixed same/heldout RGB-pyramid, multicam relative-pose RGB-pyramid, Direct
  PowerFoam, PowerFoam Metal, Dynamic PowerFoam RBF, Dynamic PowerFoam
  token/F32, and Dynamic Gauge. The first Token-GS temp config failed because
  it requested `split='eval'` from a manifest with no eval rows; switching to
  the checked-in tiny-30 config with a real `test` split fixed the smoke. A
  Direct PowerFoam offline-W&B variant also passed with local media/checkpoint
  artifacts in `/tmp/dynaworld_registry_smokes/powerfoam_direct_wandb_offline/outputs`
  and W&B offline run `wandb/offline-run-20260522_152129-r22iyau1`.

## State After This Slice

The routed warm-path trainer owner/wrapper sweep is complete for the current
`src/train/train.py` registry. Remaining `train_*.py` files are thin wrappers
or shared helper modules (`train_cli.py`, `train_logging.py`, `train_devices.py`,
`train_artifacts.py`, `train_optim.py`), not large trainer-loop owners.

No `key_learnings.md` entry was added. The only surprising issue was the
structured Direct PowerFoam render result mismatch, and it is now covered by a
focused behavior test in `tests/test_powerfoam_direct.py`.
