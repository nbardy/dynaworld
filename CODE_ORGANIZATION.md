# DynaWorld Code Organization Roadmap

This roadmap exists so cleanup work improves the codebase instead of moving
complexity around. Prefer small shared helpers and explicit contracts over a
large trainer framework.

## Target Shape

The repo should have these boundaries:

| Boundary | Owner files | Contract |
| --- | --- | --- |
| Configs | `src/train_configs/*.jsonc` | Checked-in experiment settings; no env-var fanout. |
| Entrypoints | `src/train/train.py`, thin scripts in `src/train_scripts/` | Pick a config and dispatch to one trainer. |
| Data | `src/train/sequence_data.py`, `src/train/multicam_video_data.py` | Return typed sequence/bundle data and clip slices; do not hide same-view vs heldout semantics. |
| JSON file I/O | `src/train/json_io.py` | Load JSON/JSONL files once; callers own split, shape, and domain validation. |
| Mixed data scheduling | New small module/trainer bridge | Sample same-view and heldout-view batches; log losses separately. |
| Runtime payloads | `src/train/runtime_types.py` | Named dataclasses, not positional tuple contracts. Includes clip render payloads. |
| Models | `src/train/gs_models/` | Decode tokens/cameras/gaussians; no logging or W&B. |
| Rendering | `src/train/renderers/`, renderer wrappers | Alpha-aware return shape; no trainer-specific media logic. |
| Loss/composition | Shared helper module | One source for `alpha * colorize + bg`, metric payloads, and feature RGB composition. |
| Logging/media | `src/train/train_logging.py`, `src/train/wandb_media.py`, and `src/train/pipeline/validation_media.py` | W&B setup/cadence/scalars, low-level media constructors, and trainer validation media payloads. |
| Research forks | `research_experiments/`, `star_uvt/`, `third_party/*/variants/` | Isolated experiments with result JSONs and notes. |
| Paper ablation protocol | `src/train/paper_training_types.py`, `src/train/paper_training_protocol.py` | Shared typed dataset, space-time sampling, stage, and cost contracts; representation-specific trainers retain their kernels and model state. |

The paper runner follows the same small-helper rule: it is an orchestrator over
three explicit adapters, not a replacement trainer hierarchy. World Tubes
keeps STAR UVT projection/Metal autograd, WorldFoam keeps PowerFoam Metal and
state-preserving resampling, and dynamic 3DGS keeps fast-mac. Only data order,
stage schedule, target/raster cost accounting, and report validation are
shared.

The World Tubes boundary must also preserve the distinction between the STAR
UVT representation/backend and the gauged camera-path compiler layered on it.
Do not simplify away the camera-ray pullback, depth-fiber marginalization plus
conditional-depth metadata, projective gauge domains, visibility-order event
strata, or certified local fallback. Those mechanisms are the large-motion
overlap repair and a primary paper contribution, not legacy wrappers around
STAR UVT.

## Dedupe Priorities

### P0 - Shared RGB Composition

Problem: `alpha * colorize(features) + (1 - alpha) * bg` is duplicated across
single-cam paths and has historically been missing from multicam paths.

Desired helper:

```text
compose_rendered_rgb(features, alpha, colorize, cameras, *, bg)
```

Rules:

- Caller chooses train/eval background.
- Helper handles F=3 direct RGB and F32 feature colorization.
- Multicam and single-cam validation must use the same helper.

Progress:

- Current tree: `objective.compose_rgb` and `RGBReconObjective.render_view`
  own RGB composition. The single-cam token-GS trainer and multicam
  precomputed-feature trainer call `rgb_objective.render_view(...)`, so
  random/fixed background and alpha composition go through one objective
  boundary instead of trainer-local formulas.
- 2026-05-21: `RGBReconObjective.require_alpha_for_feature_background(...)`
  now owns the F32 safety guard that used to be duplicated in the single-cam,
  multicam train-view, heldout-view, and camera-swap loops. Feature-splat
  training with an RGB background still requires alpha-aware raster output, but
  the invariant now lives at the objective boundary.
- 2026-05-21: the objective layer now exposes tensor-level
  `compose_rgb_background_tensor(...)`, `compose_feature_background_tensor(...)`,
  and `colorize_and_compose_feature_rgb(...)`. STAR UVT feature rendering, the
  background-cheat diagnostic, and the dense feature-tube prototype compatibility
  shim call those helpers, so RGB-after-colorizer and feature-before-colorizer
  background math no longer has a separate STAR-local formula in active code.

### P1 - Validation Media Builder

Problem: GT/render/alpha/feature-PCA videos are assembled in several trainer
variants, and new diagnostics land in only one path.

Desired helper:

```text
build_validation_media_bundle(gt, rendered, alpha=None, features=None, cameras=None)
```

Rules:

- Media names should be stable across single-cam and multicam trainers.
- Multicam rows should make train views and heldout views explicit.
- Feature PCA and alpha panels should be optional, not forked trainer logic.

Progress:

- Current tree: `pipeline.validation_media` owns the single-cam and multicam
  validation-video payload helpers, alpha mask conversion, feature-PCA grids,
  and diagnostic media composition. Token-GS and multicam trainers call those
  helpers while keeping trainer-specific evaluation loops local.

### P2 - Eval Metrics And Log Cadence

Move duplicate L1/MSE/PSNR/SSIM/DSSIM payload code into one metrics helper.
Move repeated `step % every == 0 or last_step` checks into `train_logging.py`.

Only centralize W&B setup when the logging config contract is identical across
multiple trainers. Keep trainer-specific payload names and artifact choices
local.

Progress:

- 2026-05-21: `train_logging.py` now owns `should_log_step`,
  `should_log_scalar`, `should_log_image`, and `should_log_video`. The
  token-GS trainer plus the PowerFoam, Dynamic PowerFoam, and Dynamic Gauge Foam
  trainers call the shared cadence helpers instead of open-coding modulo plus
  last-step checks.
- 2026-05-22: base Token-GS cadence wrapper methods now call the config-aware
  `should_log_scalar/image/video(...)` helpers directly. The trainer still keeps
  named wrapper methods for branch overrides, but it no longer rebuilds the
  shared `should_log_step(...)` argument bundle locally.
- 2026-05-21: `train_logging.py` now owns `init_wandb_run` for the shared
  `logging.wandb_*` config contract. PowerFoam, Dynamic PowerFoam, Dynamic
  Gauge Foam, and STAR UVT probe/overfit scripts call it while keeping media
  payload construction local.
- 2026-05-21: the main token-GS trainer now uses
  `train_logging.init_wandb_run(...)` too. Config normalization preserves the
  legacy default by setting missing `logging.wandb_enabled=true`, and the run
  loop skips W&B log/finish calls cleanly when a config explicitly disables
  W&B.
- 2026-05-21: `train_logging.scalar_payload(...)` now owns the base
  `StepResult` scalar W&B payload: loss scalars, train/eval sequence counts,
  input/render size, camera metrics, bank-rate terms, and aux loss terms.
  `pipeline.validation_media` is now media-only, while trainer subclasses still
  extend scalars through their `scalar_payload(...)` methods.
- 2026-05-22: `pipeline.diagnostics.camera_state_summary_metrics(...)` and
  `camera_state_payload(...)` now own camera fov/radius/rotation/translation
  scalar math. Training scalar payloads, progress messages, and full-sequence
  eval payloads share the same camera-state summaries with caller-selected key
  prefixes.
- 2026-05-22: PowerFoam implicit-camera compact metrics now reuse
  `camera_state_summary_metrics(...)` for the shared fov/radius/rotation/
  translation math while keeping PowerFoam-only origin/forward/global/active-
  frame keys local. The Token-GS trainer no longer keeps a one-line
  `camera_metrics(...)` wrapper around the same helper.
- 2026-05-22: `powerfoam_diagnostics.powerfoam_parameter_delta_metrics(...)`
  now owns the repeated PowerFoam state-delta scalar payload for center,
  radius, density, feature, normal, and texel-site drift. PowerFoam Metal and
  the two Dynamic PowerFoam Metal model classes keep temporal, camera, token,
  quaternion, and texel-SV extras local, but no longer rebuild the common
  delta dictionary independently.
- 2026-05-22: `train_logging.log_wandb_payload(...)` now owns the generic
  W&B payload submit call. The base Token-GS trainer and the multicam
  relative-pose trainer still assemble their own scalar/media payloads, but no
  longer import W&B directly only to call `wandb.log(payload, step=step)`.
- 2026-05-22: `train_logging.log_wandb_run_payload(...)` now owns the explicit
  run-object W&B submit call for trainers that keep a `wandb_run` handle.
  PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, Dynamic Gauge
  Foam, and PowerFoam eval artifact logging still assemble payloads locally,
  but no longer call `wandb_run.log(...)` directly.
- 2026-05-22: `train_logging.log_wandb_run_payload_lazy(...)` now owns the
  matching disabled-run guard for expensive payload factories. Token-GS,
  multicam relative-pose, Direct PowerFoam, shared PowerFoam eval, Dynamic
  PowerFoam Metal, and Dynamic Gauge eval/media paths now skip image/video
  payload construction when W&B is disabled, while scalar-only train-loop logs
  stay direct and local.
- 2026-05-22: `train_logging.wandb_run_lifecycle(...)` now owns the shared
  init/yield/finally-finish wrapper for PowerFoam Direct, PowerFoam Metal,
  Dynamic PowerFoam Metal, and Dynamic Gauge Foam. It keeps W&B cleanup safe on
  exceptions without moving trainer loops, payload schemas, or cadence policy.
- 2026-05-22: base Token-GS and multicam relative-pose `val_log(...)` now share
  the `Trainer.log_gate_flags(...)` cadence bundle and both submit through
  `log_wandb_run_payload(...)`. The relative-pose override keeps its render-size
  contexts, but disabled W&B now exits before building payloads, matching the
  base trainer.
- 2026-05-22: base `Trainer.scalar_payload(...)` now honors
  `result.render_size` when a multires branch attaches it. The relative-pose
  scalar extension no longer restamps `RenderSize` / `Render/BaseSize`; it only
  adds relative-pose and multires-specific metrics.
- 2026-05-22: Gauge Fields material-surfel training now uses
  `train_logging.log_wandb_payload(...)` and `finish_wandb_run(...)` for W&B
  submit/finish boundaries. Its custom `log_to_wandb` config, W&B init, scalar
  payload, and final media payload stay local because that research trainer has
  a different logging schema.
- 2026-05-22: `train_logging.set_default_wandb_mode(...)` now owns the repeated
  `WANDB_MODE` / optional `WANDB_SILENT` `setdefault` pattern. Trainer-phase,
  train-step memory, camera-scene diagnostics, and the V-JEPA performance
  scripts default W&B side effects through the same helper while preserving
  caller overrides.
- 2026-05-22: `train_logging.mapped_metric_payload(...)` now owns the repeated
  "copy this metric key into this W&B key" pattern with explicit
  required-versus-optional behavior. Direct PowerFoam, shared PowerFoam eval
  artifacts, Dynamic PowerFoam Metal, and Dynamic Gauge Foam now keep metric
  key maps as local data tables instead of open-coded payload assignments and
  optional `if key in metrics` blocks.
- 2026-05-22: PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, and
  Dynamic Gauge train-loop scalar W&B logs now use local key maps plus
  `mapped_metric_payload(...)` and the null-safe `log_wandb_run_payload(...)`.
  The user-facing metric names remain trainer-local, but the dict-copy and
  disabled-run submit mechanics no longer fork across those loops.
- 2026-05-21: `wandb_media.py` now owns low-level W&B media constructors:
  `make_wandb_video(...)`, `make_preview_image(...)`,
  `make_wandb_image(...)`, `add_existing_wandb_media(...)`, and
  `build_validation_video_payload(...)`. PowerFoam trainers and
  `pipeline.validation_media` import those media helpers directly, leaving
  `train_logging.py` focused on W&B setup, cadence, scalars, and row-output
  logging. `pipeline.validation_media` no longer imports W&B directly.
- 2026-05-22: `wandb_media.build_rgb_alpha_validation_video_payload(...)` and
  `build_rgb_alpha_eval_media_payload(...)` now own the repeated W&B eval media
  payload for RGB reconstructions with alpha masks: preview image, optional
  render video, render/GT side-by-side video, GT video, and alpha video. Direct
  PowerFoam, shared PowerFoam eval artifacts, Dynamic PowerFoam Metal, and
  Dynamic Gauge Foam use it while keeping branch-specific scalar and depth-video
  payloads local.
- 2026-05-22: `video_io.rgb_alpha_preview(...)`,
  `save_rgb_alpha_preview(...)`, `save_render_side_by_side_videos(...)`, and
  `save_rgb_alpha_eval_media(...)` now own the matching file-artifact pattern
  for RGB+alpha eval paths: `preview_step_*.png`, optional
  `heldout_preview_step_*.png`, `render_step_*.mp4`, and
  `side_by_side_step_*.mp4`. The same PowerFoam/Gauge eval paths use those
  helpers without changing filenames or log cadence.
- 2026-05-22: `video_io.video_fps_from_config(...)` and
  `wandb_media.make_step_preview_image(...)` now own the repeated PowerFoam/
  Gauge media defaults: the top-level `video_fps` fallback to `4.0` and the
  `step {step}: GT | render` W&B preview caption. The eval paths keep their
  local cadence decisions but no longer repeat those small media-policy literals.
- 2026-05-22: `powerfoam_eval_render.powerfoam_eval_batch_size(...)` now owns
  the eval-render batch-size policy derived from `train.frames_per_step`, with
  the existing minimum of one frame. Direct PowerFoam, shared PowerFoam eval
  artifacts, and Dynamic PowerFoam Metal use it for artifact/eval renders while
  train-time frame sampling remains local.
- 2026-05-22: `powerfoam_training.powerfoam_train_batch_indices(...)` now owns
  the matching train-loop random index sampling for PowerFoam-family trainers.
  Direct PowerFoam, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic Gauge
  Foam use it while keeping their target/ray/stage-specific batch assembly
  local.
- 2026-05-21: `pipeline.diagnostics` now owns shared full-clip reconstruction
  metric helpers for snake_case result JSON keys. PowerFoam, Dynamic PowerFoam,
  and Dynamic Gauge Foam use those helpers for eval L1/MSE; PowerFoam Metal
  uses the shared L1/MSE/PSNR/SSIM helper.
- 2026-05-21: `train_logging.log_wandb_row_outputs(...)` now owns the STAR UVT
  row-metric flattening and output-media attachment pattern. RGB STAR, feature
  STAR, feature RGB probe, and rendered-feature RGB probe keep their metric
  prefixes/media key choices local, but no longer carry four copies of the
  recursive scalar-flattening plus existing-file W&B media code.
- 2026-05-22: `star_uvt_outputs.log_star_uvt_row_outputs(...)` now owns the
  STAR-specific W&B row-output convention on top of
  `train_logging.log_wandb_row_outputs(...)`: default contact-sheet and
  side-by-side media keys, with an override for the feature-overfit RGB-probe
  media pair. RGB STAR, feature STAR, target-grid RGB probe, and
  rendered-feature RGB probe no longer carry local `_log_wandb_outputs(...)`
  wrappers.
- 2026-05-22: `model_factories.build_colorizer(...)` is the Token-GS colorizer
  factory boundary for trainer and probe code. `probe_colorize_init.py` and
  `probe_colorize_matrix.py` now use it instead of reconstructing
  `FeatureToColor` locally, so view-conditioning, detach flags, unknown-key
  validation, and future colorize kwargs stay aligned with the trainer path.
- 2026-05-22: the Token-GS trainer stopped hoisting low-use normalized config
  aliases for `feature_dim`, recon backward strategy, temporal microbatch size,
  profile timing sync/cadence, and profile backward split. It now reads those
  values from `self.model_cfg` or `self.train_cfg` at the actual use sites,
  while keeping the heavily used section aliases.
- 2026-05-22: `star_uvt_colorizers.build_default_feature_colorizer(...)` owns
  the STAR feature-tube default colorizer settings. `star_uvt_feature_tube_model`
  and the feature-tube autograd overfit benchmark use that helper instead of
  repeating LN + kaiming gain-4 `FeatureToColor` construction.
- 2026-05-22: `powerfoam_colorizers.py` owns Dynamic PowerFoam feature-colorizer
  defaults, RGB identity initialization, and the token-feature-mode colorizer
  builder. `train_dynamic_powerfoam_metal.py` now calls the shared builder
  directly instead of keeping a local `build_colorizer(...)` pass-through
  wrapper.
- 2026-05-22: `dynamic_powerfoam_metal_config.py` owns Dynamic PowerFoam Metal
  config normalization: defaults, `token_rbf_features`, camera/render
  validation, colorize-default wiring, and `resolve_config(...)`. The trainer
  imports/re-exports that boundary for compatibility, matching the
  PowerFoam Metal config split. `dynamic_powerfoam_metal_trainer.py` now owns
  the full Dynamic PowerFoam implementation and `run_training(...)`, while the
  historical `train_dynamic_powerfoam_metal.py` file is a thin CLI wrapper that
  imports only `run_training(...)`. The registry routes
  `dynamic_powerfoam_metal` to the owner module. One-step `src/train/train.py`
  smokes passed for both the RBF branch and the token/F32 branch with 4 frames,
  64 cells, MPS, `/tmp` outputs, and disabled W&B.
- 2026-05-22: focused Dynamic PowerFoam tests now import defaults and
  `resolve_config(...)` from `dynamic_powerfoam_metal_config.py`, and raster
  config construction from `powerfoam_raster_config.py`. The full trainer is
  still imported for model classes because those classes have not been split
  out yet, but pure config/raster helper imports no longer route through the
  trainer file.
- 2026-05-22: `visualize_camera_scene_diagnostic.py` now follows the same
  boundary for Dynamic PowerFoam checkpoint visualization: it imports the
  token-feature mode constant from `dynamic_powerfoam_metal_config.py` and the
  raster config helper from `powerfoam_raster_config.py`, while importing only
  structural model classes from `dynamic_powerfoam_metal_trainer.py`.
- 2026-05-22: `dynamic_powerfoam_temporal.py` owns pure Dynamic PowerFoam
  temporal helpers: Gaussian time bases, temporal basis fitting, acceleration
  regularizers, bounded `atanh`, and temporal motion metrics. The trainer
  imports/re-exports those helpers for compatibility, but tests now exercise the
  motion-metric contract from the helper module directly.
- 2026-05-22: `dynamic_powerfoam_camera.py` owns Dynamic PowerFoam implicit
  camera construction, camera optimizer groups, regularization and compact
  metrics, teacher camera loading/alignment/prefit, and camera-decoded ray
  assembly. The dynamic trainer keeps compatibility imports, while tests and
  the camera-scene diagnostic import the light camera helper directly.
- 2026-05-22: `dynamic_powerfoam_initialization.py` owns Dynamic PowerFoam
  initialization geometry: camera/world transforms, orbit-camera normal
  initialization, orbit-video point/texel initialization, and token-feature
  texel initialization. The dynamic trainer imports these helpers while keeping
  the model classes and train/eval loop local.
- 2026-05-22: `dynamic_powerfoam_rendering.py` owns Dynamic PowerFoam
  premultiplied-feature rendering helpers: RGB background sampling,
  alpha-normalized feature-to-RGB composition, no-grad full-video eval renders,
  per-frame reconstruction metrics, and temporal alpha metrics. The trainer
  keeps W&B/media artifact policy local, but no longer owns the pure render and
  metric helpers.
- 2026-05-22: `dynamic_gauge_rendering.py` owns Dynamic Gauge Foam render
  kwargs and no-grad full-video eval rendering. The Gauge trainer still owns
  optimizer groups, losses, metrics payloads, checkpointing, and media policy,
  but no longer carries render-argument parsing or the per-frame eval loop
  inline.
- 2026-05-22: `dynamic_gauge_config.py` owns Dynamic Gauge Foam defaults and
  `resolve_config(...)`, matching the PowerFoam-family config-module pattern.
  The Gauge trainer imports that boundary instead of carrying default dicts and
  config validation inline. `dynamic_gauge_foam_trainer.py` now owns the full
  Dynamic Gauge trainer implementation and `run_training(...)`, while the
  historical `train_dynamic_gauge_foam.py` file is a thin CLI wrapper that
  imports only `run_training(...)`. The registry routes `dynamic_gauge_foam` to
  the owner module. A one-step
  `src/train/train.py` smoke passed on MPS with 4 frames, 64 primitives, `/tmp`
  output, disabled W&B, and final checkpoint write.
- 2026-05-22: `dynamic_gauge_objectives.py` owns Dynamic Gauge Foam training
  loss assembly: RGB L1/MSE, gauge connection loss, temporal acceleration loss,
  opacity/radius regularizers, atlas total variation, and weighted total loss.
  The Gauge trainer keeps sampling, optimizer stepping, scalar naming, artifact
  logging, and checkpointing local.
- 2026-05-22: `dynamic_powerfoam_staging.py` owns Dynamic PowerFoam stage
  controls: static-geometry warmup, no-repaint warmup, camera-curriculum active
  frame selection, and optional camera active-prefix mutation. The trainer
  imports/re-exports the helpers while keeping step-loop logging local.
- 2026-05-22: scale/pretrain shell launchers now use
  `trainer_registry.resolve_config_for_arch(...)` and `run_config_dict(...)`
  for embedded Python config checks/probes instead of importing concrete
  precomputed or multicam trainers as generic resolver/runner namespaces.
- 2026-05-22: the multicam scale/pretrain launcher now launches via the
  registry CLI `src/train/train.py` instead of calling
  `train_multicam_precomputed_feature_implicit_dynamic.py` directly. The
  checked-in config still resolves to the multicam trainer module through
  `trainer_registry`, but the shell launcher no longer bypasses the shared
  entrypoint.
- 2026-05-22: the older Token-GS/precomputed shell launchers for local
  ablations, scene-distinct baselines, single-video pretrain, and compare
  matrices now launch registered configs through `src/train/train.py` instead
  of naming `train_video_token_implicit_dynamic.py` or
  `train_precomputed_feature_implicit_dynamic.py` directly. The STAR
  fast-overfit launcher now also sends all registered STAR RGB/feature and
  dynamic-gsplat configs through the same registry CLI, including the compact
  visual and native full-cell feature-overfit modes.
- 2026-05-22: the Dynamic Foam external-blocker runner now builds its
  PowerFoam Metal train command with `src/train/train.py` as well, while
  preserving the `src/train:third_party/powerfoam-metal` path contract.
- 2026-05-22: the scale/pretrain launchers also route complete patched config
  writes through `train_artifacts.write_json(...)`: the multicam per-record
  temp config and the 1k single-video smoke config no longer open-code
  `json.dumps(...)` file writes. Manifest JSONL row copying remains text-local
  because it is not a complete JSON object artifact.
- 2026-05-22: `powerfoam_geometry.py` owns pure PowerFoam geometry helpers:
  pinhole rays, camera rays, camera-ray grids, stable tangent fallback, and
  orthonormal surface frames. PowerFoam Direct and Metal import/re-export those
  ray helpers for compatibility, while Dynamic PowerFoam imports the shared
  geometry helpers directly instead of reaching through the full Metal trainer.
- 2026-05-22: `powerfoam_adjacency.py` owns CSR adjacency construction and
  adjacency stats for PowerFoam-family trainers. Regular-triangulation adjacency
  lazy-loads the Metal extension only when requested. PowerFoam Metal re-exports
  the helpers for compatibility, while Dynamic PowerFoam and Dynamic Foam
  diagnostics import adjacency directly from the helper module.
- 2026-05-22: Dynamic Foam diagnostics now use
  `train_devices.resolve_torch_device(...)` for device selection instead of
  importing `resolve_device` from the full PowerFoam Metal trainer.
- 2026-05-22: the stale `train_powerfoam_metal.resolve_device(...)`
  compatibility wrapper was removed after `rg` found no live imports. PowerFoam
  Metal and Dynamic Foam callers now use `resolve_torch_device(...)` directly
  with their explicit auto policy.
- 2026-05-22: Dynamic Foam diagnostics now import `POWERFOAM_SOFTPLUS_BETA`
  from `powerfoam_direct` and `reconstruction_eval_metrics(...)` from
  `pipeline.diagnostics` instead of reaching through the PowerFoam Metal trainer
  for pure constants/helpers.
- 2026-05-21: `train_optim.adam_with_device_fused(...)` now owns the repeated
  Token-GS Adam fused-kernel policy (`fused=true` on CUDA/MPS only). The base
  token-GS trainer and the relative-pose-only scope use it. PowerFoam,
  Dynamic PowerFoam, Dynamic Gauge Foam, and STAR UVT optimizers stay local
  because their parameter groups, Adam/AdamW choice, LR multipliers, or probe
  semantics differ.
- 2026-05-22: `train_optim.optimizer_backward_step(...)` now owns the small
  zero-grad/backward/optional-grad-clip/optimizer-step lifecycle shared by
  PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic Gauge
  Foam. Optimizer construction and parameter groups stay local; only the
  mechanical step sequence is shared.
- 2026-05-22: the PowerFoam/Gauge owner modules now keep their public
  `__all__` surfaces narrow. `powerfoam_direct_trainer.py` and
  `dynamic_gauge_foam_trainer.py` export only `run_training(...)`;
  `powerfoam_metal_trainer.py` exports only the structural model/run surface.
  Config defaults, raster config builders, data loaders, geometry helpers, and
  artifact helpers are public from their owning helper modules instead.
- 2026-05-22: `powerfoam_metal_config.py` owns pure PowerFoam Metal config
  normalization: defaults, feature-mode sets, LR group specs, and
  `resolve_config(...)`. `powerfoam_metal_trainer.py` owns the full Metal
  trainer implementation and `run_training(...)`, while the historical
  `train_powerfoam_metal.py` file is now a thin CLI wrapper that imports only
  `run_training(...)`. Dynamic Foam diagnostics import config resolution from
  the config module instead of the full trainer, and the registry routes
  `powerfoam_metal` to the owner module. A one-step
  `src/train/train.py` smoke based on the 64px local Mac config passed on MPS
  with `/tmp` output, final checkpoint write, and eval L1 improving from
  0.07646 at step 0 to 0.07628 at step 1.
- 2026-05-22: `powerfoam_raster_config.py` owns `FoamRasterConfig` construction
  for PowerFoam Metal and Dynamic PowerFoam Metal. Both trainers preserve their
  `make_raster_config(...)` compatibility aliases, while Dynamic Foam
  diagnostics import the light helper instead of the full Metal trainer for
  raster config construction.
- 2026-05-22: `powerfoam_training.py` owns shared PowerFoam train-step
  primitives: multiview frame/ray flattening and exponential schedule
  interpolation. PowerFoam Direct and PowerFoam Metal import/re-export those
  primitives.
- 2026-05-22: `powerfoam_objectives.scheduled_loss_weights(...)` now also
  covers the Direct PowerFoam schedule. Direct keeps its `rgb_mse_sum_weight`
  loss term as an optional schedule payload key, and Direct defaults now carry
  the same explicit auxiliary start-step keys as Metal so the shared objective
  helper is not relying on implicit missing-key behavior.
- 2026-05-22: `powerfoam_point_cloud.py` owns PowerFoam point-cloud
  initialization: PLY/COLMAP parsing, color normalization, fit/clamp to the
  model box, train-view visibility filtering, duplicate backfill, and the
  `PointCloudInitialization` payload. PowerFoam Metal keeps compatibility
  re-exports, while diagnostics use the lighter point-cloud module directly.
- 2026-05-22: `powerfoam_objectives.py` owns trainer-independent PowerFoam
  objective helpers: Metal SSIM loss wrapping, Metal loss-weight scheduling,
  Direct PowerFoam loss assembly, contribution and normal-distance losses,
  depth-to-normal-map targets, normal-map loss, and alpha/background
  compositing. PowerFoam Metal keeps compatibility re-exports; Direct now calls
  the objective helper from the train loop instead of owning the formula inline.
- 2026-05-22: `tests/test_powerfoam_direct.py` now imports pure PowerFoam
  helpers from their owning modules (`powerfoam_geometry`, `powerfoam_training`,
  `powerfoam_metal_config`, `powerfoam_objectives`, `powerfoam_point_cloud`,
  `powerfoam_raster_config`, `powerfoam_adjacency`, and direct
  `torch_powerfoam_metal` raster fixture symbols) instead of using
  `train_powerfoam_metal.py` as a helper namespace. Its remaining
  `train_powerfoam_metal` imports are structural `MetalPowerFoamVideo` model
  backcompat checks; non-test Dynamic Foam diagnostics import the owner
  `powerfoam_metal_trainer.py` module directly.
- 2026-05-22: `powerfoam_eval_color.py` owns PowerFoam eval color-calibration
  helpers: channel-affine and RGB-matrix affine fit/apply, pixel
  flattening/bias-column utilities, frame-index summaries, and calibration
  provenance serialization. PowerFoam Metal keeps compatibility re-exports, and
  the color-affine diagnostic now uses the same affine helper implementation.
- 2026-05-22: `powerfoam_optim.py` owns pure PowerFoam Metal optimizer schedule
  helpers: cosine LR, LR-group initial/final/warmup metadata, and param-group
  LR updates. PowerFoam Metal keeps compatibility re-exports, while focused
  tests can exercise the LR contract without importing the full Metal trainer.
- 2026-05-22: `powerfoam_resampling.py` owns the official-style PowerFoam
  resample cadence predicate and geometric target-cell schedule. PowerFoam
  Metal keeps compatibility re-exports, but the train loop now consumes the
  light helper module.
- 2026-05-22: `powerfoam_checkpoints.py` owns PowerFoam checkpoint artifact
  helpers: best-metric selection, atomic checkpoint writes, and
  `best_metrics.json` updates. The helper supports both the Metal metric-rich
  payload and the Direct minimal final checkpoint payload, so checkpoint schema
  construction is no longer embedded in either train-loop body.
- 2026-05-22: Dynamic PowerFoam Metal and Dynamic Gauge Foam final checkpoints
  now call `checkpoint_utils.atomic_torch_save(...)` too. Their checkpoint
  payload schemas stay trainer-local, but the final `checkpoint_final.pt`
  persistence path shares the same temporary-file replace behavior as PowerFoam
  Direct, PowerFoam Metal, and STAR UVT saves.
- 2026-05-22: direct video-window frame caches in `sequence_data.py` and
  precomputed V-JEPA feature caches in `video_feature_cache.py` now use
  `checkpoint_utils.atomic_torch_save(...)` as well. Their cache payload
  schemas stay local, but tensor-cache writes no longer hand-roll
  `torch.save(tmp) -> replace(...)` without shared cleanup-on-failure behavior.
- 2026-05-22: `checkpoint_utils.py` now owns the matching read-side checkpoint
  primitives: raw torch checkpoint loading, mapping-payload validation, and
  extraction of either wrapped `payload["model"]` or raw state-dict payloads.
  STAR UVT checkpoint loading, STAR UVT colorizer-init loading, relative-pose
  checkpoint resume, and the camera-scene diagnostic share those helpers instead
  of carrying local mapping/state-dict shape checks; checkpoint schemas and
  model-specific loading remain local.
- 2026-05-22: `checkpoint_utils.load_torch_checkpoint(...)` also supports the
  `weights_only=True` torch-load policy used by frame caches, V-JEPA feature
  caches, and browser-bundle state-dict export. Those callers now route binary
  loads through the helper while keeping cache keys, stale-cache handling, and
  state-dict schema checks local; `src/train` has no direct `torch.load(...)`
  calls outside `checkpoint_utils.py`.
- 2026-05-22: `powerfoam_eval_render.py` owns the no-grad PowerFoam batch
  sample renderer used by evaluation and diagnostics. PowerFoam Metal keeps
  `render_samples` as a compatibility alias, while color-affine and heldout
  diagnostics import the renderer from the light helper module. The helper now
  accepts PowerFoam-style call outputs with extra tensors and keeps only the
  first `(rendered, alpha)` pair. PowerFoam Direct now calls it directly for
  eval and heldout artifact renders instead of carrying a local pass-through
  render wrapper.
- 2026-05-22: `powerfoam_training_data.py` owns the PowerFoam train/eval data
  dict contract for single-video and multicam validation inputs: targets,
  sample frame indices, optional sample rays, heldout tensors, view metadata,
  pose metadata, FPS, and point-cloud visibility metadata. PowerFoam Metal keeps
  a compatibility alias, while Dynamic Foam diagnostics and World Foam Gate 1
  feeder/reference scripts import the data loader directly. PowerFoam Direct
  now wraps the same loader and prunes back to its historical key set instead
  of keeping a duplicate multicam/single-video loader.
- 2026-05-22: `powerfoam_direct_config.py` owns PowerFoam Direct defaults and
  `resolve_config(...)`. `powerfoam_direct_trainer.py` now owns the Direct
  trainer implementation and `run_training(...)`; the historical
  `train_powerfoam_direct.py` file is a thin CLI wrapper that imports only
  `run_training(...)`. The registry routes `powerfoam_direct` to the owner
  module, matching the Metal/Dynamic PowerFoam owner-module pattern. A
  one-step `src/train/train.py` smoke passed on CPU with 4 frames, 32 cells,
  64px render, `/tmp` output, disabled W&B, and final checkpoint write.
- 2026-05-22: `powerfoam_direct.py` owns Direct PowerFoam render-option
  construction through `direct_powerfoam_render_options(...)`, next to the
  `PowerFoamRenderOptions` dataclass it builds. The Direct trainer now passes
  normalized render config into that light model/render module instead of
  keeping a local config-to-options helper.
- 2026-05-22: the focused PowerFoam tests now import Direct defaults from
  `powerfoam_direct_config.py` and the shared schedule from
  `powerfoam_objectives.py` instead of using `train_powerfoam_direct.py` as a
  helper namespace. The Direct trainer imports only `resolve_config(...)` from
  its config module.
- 2026-05-22: `powerfoam_eval_artifacts.py` owns PowerFoam eval artifact
  assembly: no-grad render calls, fixed-background compositing, optional eval
  color calibration, reconstruction metrics, aux/drift metrics, preview PNGs,
  optional MP4s, and W&B eval payloads. PowerFoam Metal keeps `log_artifacts`
  as a compatibility alias, but the train loop file no longer carries the media
  and metrics assembly block.
- 2026-05-21: `train_cli.py` now owns the repeated one-config script boundary:
  `run_config_arg(...)` loads `sys.argv[1]` with a per-script usage string, and
  `run_config_or_path(...)` preserves the public `main(config_or_path)` pattern
  for trainer modules used directly by tests/scripts. Active Token-GS,
  precomputed-feature, multicam, mixed same-heldout, PowerFoam, and STAR UVT
  train/probe modules now use this helper instead of local `sys` plus
  `load_config_file` boilerplate.
- 2026-05-22: `train_cli.py` also owns `run_path_arg(...)` for path-dispatch
  CLIs that must pass the config path through unchanged. `src/train/train.py`
  now uses that helper for its arity/usage handling while still dispatching by
  path through `trainer_registry.run_config(...)`.
- 2026-05-22: `train_cli.parse_csv_ints(...)` now owns the small
  comma-separated integer-list parser used by train-local CLI probes. The
  colorize init and matrix probes use it for `--seeds`, so new train probes do
  not need another local `args.foo.split(",")` branch for the same CLI shape.
- 2026-05-21: `train_logging.finish_wandb_run(...)` now owns the shared
  non-null W&B finish guard. Token-GS, PowerFoam-family, and STAR UVT
  train/probe modules call it instead of open-coding `wandb_run.finish()` or
  `wandb.finish()` branches.
- 2026-05-22: `finish_wandb_run(...)` also handles the global active
  `wandb.run` case for benchmark/probe scripts that do not retain an explicit
  run object. The reusable `src/benchmarks/*` timing/parity CLIs and the
  V-JEPA performance CLIs now use that shared finish boundary instead of
  importing W&B only to call `wandb.finish()`.
- 2026-05-22: `wandb_run_lifecycle(...)` wraps the same init/finish primitives
  for PowerFoam/Gauge trainer owners that otherwise held a run object for the
  whole training loop. This is intentionally only a safety wrapper; payload
  construction, cadence, checkpointing, and train-loop control remain local.
- 2026-05-21: `train_devices.py` now owns shared `auto` device resolution and
  device synchronization helpers. PowerFoam-family trainers keep their
  existing auto policy by passing `auto_cuda=False`; Dynamic Gauge keeps its
  CUDA fallback with `auto_cuda=True`; STAR UVT keeps requested-device
  availability checks with `validate_requested=True`; Token-GS profile timing
  now uses `sync_torch_device(...)` instead of local CUDA/MPS branches.
- 2026-05-21: colorize init probes, V-JEPA performance benchmarks, and the
  STAR alpha-background ablation orchestrator now reuse `train_devices` for
  auto-device resolution and/or MPS/CUDA timing synchronization. These scripts
  are still experiment/probe surfaces, but they no longer carry local copies of
  the same device primitive used by trainer timing code.
- 2026-05-22: `train_devices.clear_torch_device_cache(...)` now owns the
  repeated Python GC plus MPS/CUDA cache-clear primitive. `video_feature_cache`
  uses it after feature baking, and `src/benchmarks/benchmark_memory.py` keeps
  its public `clear_device_cache(...)` wrapper while delegating the device
  details to `train_devices`.
- 2026-05-22: `research_experiments/vjepa_performance/vjepa_benchmark_common.py`
  now owns V-JEPA performance benchmark repo-root/train-path bootstrap,
  positive-int/string CSV parsing, deterministic seed setup, device-synchronized
  timing fences, and timing-summary formatting. The five V-JEPA performance
  scripts are still experiment-specific, but no longer each carry local
  `sys.path` mutation, import-order comments, seed setup, or timing helpers.
  The common module keeps V-JEPA-specific argparse validation, while generic CSV
  tokenization now delegates to `train_cli.parse_csv_ints(...)` and
  `parse_csv_strings(...)`.
- 2026-05-22: the same V-JEPA benchmark common module now owns total splat-count
  config validation, effective splat-count reporting, optional video benchmark
  shape/step patching, and quiet trainer logging gates for benchmark/profiling
  runs. Throughput, phase-profile, quality-parity, and multicam V-JEPA
  benchmark scripts keep their case-specific config patches local, but no
  longer repeat the same divisibility check, render/clip/step assignments, or
  media-log suppression keys.
- 2026-05-22: PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, and
  Dynamic Gauge Foam now call `resolve_torch_device(...)` directly instead of
  keeping identical one-line `resolve_device(...)` wrappers. DUSt3R video export
  also delegates its auto-device selection to `train_devices` while preserving
  MPS-first, CUDA-second fallback.
- 2026-05-22: `external_paths.py` now owns project and third-party path
  bootstrap for train-local entrypoints. PowerFoam Metal, Dynamic PowerFoam
  Metal, DUSt3R video export, STAR UVT runtime, Taichi renderer bootstrap, and
  the two v12a objective helpers use it for root/path insertion. The v12a
  helpers also share the compiled-module origin guard, so wrong-variant
  protection stays intact without two local copies.
- 2026-05-22: `renderers.fast_mac` now uses the same third-party path helper
  and compiled-module origin guard. The wrapper keeps explicit variant names and
  dispatch local, but the repeated repo-root/`third_party/fast-mac-gsplat`
  expressions and duplicate bridge-origin guard are gone.
- 2026-05-22: `sync_torch_device(...)` now skips unavailable MPS and CUDA
  devices safely. The depth-aware DOF demo and fast-mac benchmark probes route
  their local timing fences through it, so benchmark/probe code no longer needs
  device-specific synchronize guards at each call site.
- 2026-05-22: Dynamic Foam 4K trainability generation and real-view raytrace
  alpha diagnostics now route MPS/CUDA timing fences through
  `sync_torch_device(...)` as well. The diagnostic math and artifact schemas
  stay local; only the device synchronization primitive is shared.
- 2026-05-22: `resolve_torch_device(...)` now has an explicit
  `auto_prefer_cuda` flag for legacy renderer benchmark CLIs whose historical
  `auto` policy chose CUDA before MPS. `splat_renderer_benchmark.py`,
  `splat_renderer_accuracy.py`, and `trainer_phase_benchmark.py` now use the
  shared device/sync primitives without changing that benchmark policy.
- 2026-05-22: `src/benchmarks/renderer_benchmark_cli.py` now owns shared
  renderer-benchmark CLI primitives: resolution parsing, comma-separated list
  parsing, config deep-merge, torch dtype lookup, project-relative output
  paths, safe filename parts, save-target selection, save-image CLI override
  application, row-target matching, and CHW tensor preview-image writes.
  `splat_renderer_benchmark.py` and
  `splat_renderer_accuracy.py` use those helpers instead of maintaining
  parallel parser/path/image-selection/image-write blocks. The same
  comma-separated string parser is also used by `mac_renderer_stack_compare.py`
  and `fast_mac_v13_iteration_matrix.py` for renderer/version selection. The
  temporal raster overlap profile now uses the same int/float CSV parser
  boundary while keeping its positive/nonempty argparse validation local.
  `depth_aware_dof_demo.py` uses the same safe-filename helper for comparison
  panel names while preserving its historical no-dot stem behavior.
- 2026-05-22: the STAR/dynamic alpha-background ablation now calls
  `finish_wandb_run(...)` instead of importing W&B only to finish the trainer
  run. The ablation still owns its orchestration, stdout capture, and row
  schemas.
- 2026-05-22: the STAR UVT feature-kernel benchmark cluster now uses shared
  device synchronization and report JSON writes. Direct feature, feature
  autograd overfit, sparse hidden sigmoid-MSE, and sparse hidden target-area
  benchmarks keep their kernel-specific timing/parity logic local, but no
  longer carry local MPS sync calls or direct output-file writes.
- 2026-05-21: `train_artifacts.py` now owns common trainer artifact primitives:
  `write_resolved_config(output_dir, cfg)` for `resolved_config.json` and
  `append_jsonl(path, payload)` for serialized metrics-history rows. PowerFoam
  Direct, Dynamic Gauge Foam, Dynamic PowerFoam Metal, and PowerFoam Metal use
  the shared resolved-config writer; the two PowerFoam Metal trainers use the
  shared JSONL appender.
- 2026-05-21: `train_artifacts.py` also owns reusable benchmark/result artifact
  writes through `write_json(...)`, `write_jsonl(...)`, `write_csv(...)`, and
  `write_text(...)`.
  V-JEPA performance benchmarks, including the fixed fast-mac variant matrix,
  STAR UVT diagnostics/reports, STAR UVT feature1 continuation reports, and the
  STAR alpha-background ablation orchestrator use those helpers instead of
  open-coding parent-directory creation plus sorted JSON, JSONL, or
  markdown/text result writes. STAR UVT row-output helpers now delegate their
  JSON file writes to the same artifact primitive while keeping their stdout
  rows local.
- 2026-05-22: `research_experiments/report_artifacts.py` now owns generic
  top-level research report path and artifact helpers: Dynaworld root
  resolution, `src/train` path bootstrap, JSON/JSONL/CSV reads, and
  JSON/CSV/text writes backed by `train_artifacts.py`. The renderer scaling
  report uses that boundary for its STAR/dynamic/feature table inputs plus CSV
  and markdown outputs while keeping renderer-specific table assembly local.
  The multicam train2/holdout1 split smoke now also uses the same helper for
  repo-root `chdir` and config path resolution instead of carrying a local
  `Path(__file__)`/`sys.path.insert(...)` preamble.
- 2026-05-22: Gauge Fields report/config/metrics JSON outputs now route through
  the same artifact primitive via `research_experiments/gauge_fields/common.py`.
  Gauge trainers, probe scripts, run matrix wall-clock reports, and JSON
  summaries keep the Gauge-local helper name, but no longer own a separate
  parent-mkdir plus sorted-JSON writer.
- 2026-05-22: Gauge Fields run-matrix CLIs now share
  `common.parse_gauge_matrix_args(...)` and `common.run_gauge_matrix(...)`.
  The DeepView 3-cam holdout and incidence-matrix scripts keep their run lists
  and default descriptions local, but the repeated `--output-root`/`--steps`/
  `--device`/`--no-wandb`/`--only` args, subprocess command construction,
  wall-clock JSON, and failure exit handling are no longer duplicated.
- 2026-05-22: Gauge Fields RGB MP4 output now shares
  `common.save_rgb_mp4(...)`. `smiley_smoke.py` and
  `cheat_probe_material_gauge.py` no longer carry duplicate OpenCV writer
  loops for single RGB videos; side-by-side MP4s still use
  `common.save_side_by_side_mp4(...)`.
- 2026-05-22: Gauge Fields media sidecar legends now share
  `common.write_columns_legend(...)`. Preview strips, smiley smoke output, and
  cheat-probe xmap/flow/probe strips no longer hand-write `*_columns.txt`
  files.
- 2026-05-22: the same Gauge Fields common module now owns both experiment-dir
  and `src/train` path setup. `cheat_probe_material_gauge.py` and
  `smiley_smoke.py` no longer duplicate the local `EXPERIMENT_DIR`/`sys.path`
  preamble, and the helper enforces experiment-dir priority so `from train`
  still resolves the Gauge-local `train.py`.
- 2026-05-22: Gauge Fields `common.resolve_device(...)` now delegates to
  `train_devices.resolve_torch_device(...)` with `auto_cuda=True` and
  `auto_prefer_cuda=True`, preserving Gauge's CUDA-first auto policy while
  removing a local CUDA/MPS/CPU branch copy.
- 2026-05-22: Gauge Fields `common.write_checkpoint(...)` now owns atomic
  checkpoint persistence through `checkpoint_utils.atomic_torch_save(...)`.
  Material-surfel Gauge and free-dynamic 3DGS Gauge keep their checkpoint payload
  schemas local, but neither trainer calls `torch.save(...)` directly for
  `checkpoint.pt`; the splat baseline also uses the shared Gauge repo-path
  resolver instead of keeping a local copy.
- 2026-05-22: Gauge Fields sweep config generation now shares
  `common.parse_csv_strings/ints/floats/bools(...)`, `clone_jsonable(...)`, and
  `write_generated_jsonc(...)`. `make_sweep_configs.py` keeps sweep-specific
  slug/tag/config patching local, but no longer owns local CSV parsers,
  JSON-roundtrip clone code, or generated-JSONC file writing.
- 2026-05-22: Gauge Fields matrix/report CLIs now use the same
  `parse_csv_strings(...)` helper for `--only` run selection and summary
  `--columns`. Matrix execution, run schemas, markdown layout, and metric
  sorting remain local to the owning script.
- 2026-05-22: `fast_attn.pick_device()` now delegates to
  `train_devices.resolve_torch_device("auto", auto_cuda=True,
  auto_prefer_cuda=True)`. Base Token-GS and browser export keep their legacy
  CUDA-first auto policy, but the CUDA/MPS/CPU branch lives in one device
  helper.
- 2026-05-22: the general `src/benchmarks` timing/parity scripts now route
  optional JSON outputs through `train_artifacts.write_json(...)` too:
  trainer-phase timing, fixed-render variant parity, camera-swap variant
  parity, fixed-render backward-mode parity, and train-step memory reports.
- 2026-05-22: `src/benchmarks/benchmark_bootstrap.py` now owns the shared
  Dynaworld/train path bootstrap for reusable benchmark CLIs. Trainer-phase,
  train-step memory, fixed-render variant parity, fixed-render backward-mode
  parity, and camera-swap parity scripts use it instead of local
  `Path(__file__)` root discovery, `sys.path` mutation, and import-order
  `# noqa` comments. The train-step memory script also now imports
  `sync_torch_device` directly from `train_devices` instead of a stale
  `trainer_phase_benchmark.sync_device` name.
- 2026-05-22: the same benchmark bootstrap exposes `PROJECT_ROOT`,
  `BENCHMARK_DIR`, and `ensure_sys_path(...)` for benchmark-specific extra
  paths. `depth_aware_dof_demo.py`, `splat_renderer_benchmark.py`,
  `splat_renderer_accuracy.py`, and `mac_renderer_stack_compare.py` use it for
  shared project/train/benchmark setup while leaving their vendored renderer
  path choices local.
- 2026-05-22: `benchmark_bootstrap.py` also exposes `VENV_PYTHON` for
  subprocess benchmark matrices. `fast_mac_project3d_benchmark.py` now uses the
  shared project/train path bootstrap, and
  `fast_mac_v13_iteration_matrix.py` uses the shared repo root and venv Python
  path while keeping variant-specific commands local. A help smoke also caught
  and fixed a stale missing `Path` import in `mac_renderer_stack_compare.py`.
- 2026-05-22: `fast_mac_project3d_benchmark.py` now also uses
  `renderer_benchmark_cli.parse_csv_strings(...)` for comma-separated case
  tokenization. The `name:size:gaussians:batch` case schema and validation stay
  local to the project3d benchmark.
- 2026-05-22: `raw_metal_mlx_bridge.py` now uses the same benchmark bootstrap
  for repo-root and path insertion. Its MLX import handling and raw-Metal
  settings stay local to the backend bridge.
- 2026-05-22: `world_foam_gate0_paired_benchmark.py` now uses
  `benchmark_bootstrap` for WorldFoam lane path setup and
  `train_artifacts.write_json(...)` for optional report output. Its paired
  benchmark math remains script-local.
- 2026-05-22: parent-safe CSV/JSONL benchmark outputs now route through
  `train_artifacts.write_csv(...)` / `write_jsonl(...)` where the payload is a
  complete in-memory table. `splat_renderer_benchmark.py` uses those helpers
  for optional result files, `mac_renderer_stack_compare.py` uses shared CSV
  output, and `depth_aware_dof_demo.py` uses shared JSON summary output.
  Row-at-a-time CSV/log streaming stays local because it is not the same
  artifact contract.
- 2026-05-22: `build_clip_dataset.py` now uses the same artifact helpers for
  per-clip summaries, manifest JSONL files, split manifest JSONL files, and
  `dataset.json`. Dataset construction still owns video probing, frame
  extraction, and manifest schema; only serialized artifact writes are shared.
- 2026-05-22: `train_artifacts.write_jsonl(..., compact=True)` now preserves
  compact newline-delimited manifest rows while sharing parent-safe artifact
  writes. `build_single_video_pretrain_manifest.py` uses that mode instead of
  carrying a local JSONL writer.
- 2026-05-22: browser-bundle export, DUSt3R video export, Dynamic PowerFoam
  Metal metric JSONs, and PowerFoam Metal eval/best-metric JSONs now use the
  same artifact writer. Tensor binaries, checkpoints, PNGs, MP4s, and NumPy
  arrays stay local because they have distinct serialization contracts.
- 2026-05-22: `research_experiments/dynamic_foam/report_artifacts.py` now owns
  Dynamic Foam JSON report reads/writes for simple diagnostic/report scripts.
  Heldout-error, topology-edge, support-gap, camera-perturbation, color-affine,
  CUDA-vs-Metal, motion-vs-repaint, video-motion ranking, raytrace-support, and
  PowerFoam-vs-splats reports share parent creation, sorted JSON, newline, and
  object-load validation while keeping PLY writers, Modal file mirroring,
  config dumps, and streaming logs local. The Dynamic Foam helper preserves the
  report-facing `write_report_json(...)` API but delegates the low-level JSON
  write to `train_artifacts.write_json(...)`.
- 2026-05-22: the same Dynamic Foam report helper now owns project-relative
  path display via `relative_to_project(...)` and exposes `PROJECT_ROOT` for
  report defaults. The routed Dynamic Foam report scripts no longer carry local
  `ROOT = Path(...parents[2])` plus `rel(...)` copies. The ALIKED Modal
  geometry orchestrator also uses `relative_to_project(...)` for output display
  while keeping its local/remote repo-root fallback and Modal staging local.
- 2026-05-22: the Dynamic Foam report helper also owns shared frame-index list
  parsing and optional range validation through `parse_frame_indices(...)` and
  `validate_frame_indices(...)`. Feature-triangulation, known-pose pycolmap,
  ALIKED geometry orchestration, and section diagnostics use that helper while
  keeping their CLI defaults, `all` support, Modal staging, and per-script
  command semantics local.
- 2026-05-22: `research_experiments/dynamic_foam/experiment_paths.py` now owns
  the Dynamic Foam repo-root and `src/train` bootstrap. The smoke dataset
  exporter, multiview plane-sweep builder, feature-triangulation builder,
  EX4DGS anchor prep, and known-pose pycolmap builder import that boundary
  through `report_artifacts` instead of carrying local `ROOT`/`TRAIN_SRC`/
  `sys.path.insert(...)` preambles. Their PLY/image/manifest semantics stay
  local.
- 2026-05-22: Dynamic Foam verifier/runner scripts now use the same path
  boundary too: 4K benchmark/trainability verifiers, clean-init coverage,
  section diagnostics, completion audit, paper acceptance, CUDA smoke result
  verification, external-blocker orchestration, and the CUDA smoke runner no
  longer duplicate repo-root/train-path/PowerFoam-Metal bootstrap code. The
  Modal ALIKED geometry launcher keeps its custom local/remote root detector
  because that script stages files into `/root/dynaworld`.
- 2026-05-22: strict Dynamic Foam report-object readers now route through
  `load_report_json(...)` in motion-vs-repaint, raytrace support-gap,
  dynamic-geometry verification, CUDA-smoke verification, 4K trainability,
  paper-acceptance, clean-init coverage, and completion-audit scripts. Direct
  CLI execution and package imports are both supported; tolerant loaders, JSONL
  readers, JSON list artifacts, copied remote artifacts, and embedded upstream
  runner internals stay local because they are different contracts.
- 2026-05-22: the remaining pass-through Dynamic Foam JSON-object wrappers
  were removed from PowerFoam-vs-splats comparison, raytrace support-gap
  diagnosis, and external-blocker orchestration. Those scripts now call
  `load_report_json(...)` directly, and the raytrace support-gap diagnostic
  also bootstraps `src/train`/Dynamic Foam paths before importing train-local
  modules so direct `--help` works.
- 2026-05-22: `load_report_jsonl(...)` now owns strict Dynamic Foam JSONL row
  loading for report-shaped object histories, with line-number errors and an
  explicit `missing_ok` option for optional histories. CUDA-vs-Metal comparison
  and the PowerFoam paper-acceptance verifier use it instead of local JSONL
  loops, while row-specific metric interpretation stays in the verifier.
- 2026-05-22: Dynamic Foam point-cloud and 4K trainability summary artifacts
  now write adjacent JSON through `write_report_json(...)`: multiview
  plane-sweep summaries, feature-triangulation summaries and failure
  diagnostics, known-pose pycolmap summaries, merged-PLY summaries, EX4DGS
  anchor summaries, generated 4K trainability artifacts, and PowerFoam parity
  fixtures. PLY payloads, dataset exports, manifests, and Modal runner internals
  keep their local serialization contracts. The known-pose pycolmap builder
  also lazily imports optional `pycolmap`, so `--help` works in lightweight
  environments.
- 2026-05-22: PowerFoam CUDA smoke `summary.json`, lane settings JSON, and
  Modal `modal_return.json` writes now use `write_report_json(...)` too; lane
  metrics reads use `load_report_json(...)`. Copied remote JSON files and the
  embedded upstream smoke entry remain local because they are inputs/outputs
  inside the cloned upstream PowerFoam checkout, not Dynaworld report artifacts.
- 2026-05-22: Dynamic Foam checkpoint/report diagnostics now share the train
  checkpoint read boundary too. Heldout-error, color-affine, raytrace-support,
  section/topology, camera-perturbation, real-view-alpha, start-support,
  official-parity, CUDA-smoke, runner, and external-blocker scripts route
  checkpoint mappings through `checkpoint_utils.load_checkpoint_mapping(...)`,
  unwrap model state through `model_state_dict_from_checkpoint(...)`, and use
  `load_report_json(...)` / `write_report_json(...)` for report-shaped objects.
  Embedded upstream settings, copied remote JSON, row-list artifacts, PLY
  metadata, and ffprobe output stay local because they are different contracts.
- 2026-05-22: ALIKED/Colmap Modal geometry report files now share
  `write_report_json(...)`: `plan.json`, local probe/full result JSONs,
  `onnx_check.json`, `colmap_cli_onnx_check.json`, and generated remote config
  JSONC. Remote JSONL manifests and copied returned artifacts stay local or
  byte-preserving because they are not generic report outputs.
- 2026-05-22: Dynamic Foam external-blocker generated training configs now use
  `write_report_json(..., sort_keys=False)` for the complete patched JSON
  config artifact. The dry-run stdout summary and Modal/remote input files
  remain local because they are different contracts.
- 2026-05-22: `research_experiments/star_uvt_feature_tubes/report_artifacts.py`
  now owns the local STAR report path bootstrap plus root-relative report JSON
  and text writes. It also owns shared report JSON loading plus markdown-table
  cell/pair formatting for the feature1 report family, so those scripts no
  longer repeat `sys.path` mutation, `ROOT / out_path` artifact boilerplate, or
  identical `_load`/`_fmt`/`_pair` helpers.
- 2026-05-22: `report_artifacts.py` now inserts the STAR UVT variant root on
  `sys.path` as part of the same bootstrap. `firstclass_scale_report.py` uses
  that boundary for path setup plus report JSON/markdown writes instead of
  rebuilding Dynaworld/train/variant roots and importing `train_artifacts`
  directly.
- 2026-05-22: `star_uvt_sparse_forward_profile.py` and
  `star_uvt_targetgrid_vjp_bridge_profile.py` now use the report-artifacts
  bootstrap plus `summary_stats(...)` directly, instead of importing the old
  private `_stats` helper from `star_uvt_feature1_wholegraph_profile.py`.
- 2026-05-22: `sparse_forward_batched_target_vjp_profile.py` and
  `sparse_forward_batched_step_benchmark.py` now use the report-artifacts
  bootstrap plus `distribution_stats(...)` directly, instead of rebuilding
  Dynaworld/train/STAR-UVT roots and mutating `sys.path` locally.
- 2026-05-22: `sparse_hidden_sigmoid_mse_kernel_benchmark.py` and
  `sparse_hidden_target_area_kernel_benchmark.py` now rely on the same
  report-artifacts bootstrap instead of carrying local
  `Path(__file__)` root discovery, STAR UVT variant path setup, and
  import-order `# noqa` comments.
- 2026-05-22: `alpha_only_visibility_profile.py`,
  `dense_alpha_failure_diagnostic.py`, `sparse_visual_loss_vjp_profile.py`, and
  `star_uvt_logit_handoff_rgb_vjp_profile.py` also rely on that shared
  bootstrap. Scripts with default config/output paths import `ROOT` from
  `report_artifacts`; timing/profile math stays local.
- 2026-05-22: `support_birth_split_sweep.py`,
  `targetgrid_render_mode_trainer_matrix.py`, `sparse_forward_scale_matrix.py`,
  and `sparse_forward_timing_repeat.py` dropped stale import-order
  `# noqa: E402` comments after their path setup moved into
  `report_artifacts`.
- 2026-05-22: `firstclass_backward_breakdown.py`,
  `star_uvt_feature1_wholegraph_profile.py`, `star_uvt_vjepa_bridge_audit.py`,
  and `run_alpha_background_ablation.py` now rely on the shared report
  bootstrap too. STAR report-shaped JSON/text outputs go through
  `write_report_json(...)` / `write_report_text(...)`; diagnostic math and run
  orchestration stay local.
- 2026-05-22: `background_cheat_diagnostic.py`,
  `compare_compact_visual_vjp_gate.py`, and
  `star_uvt_vjepa_vs_gaussian_comparison.py` now use the STAR report helper for
  report-shaped JSON/text/root handling. The background diagnostic imports the
  bootstrap before objective modules so direct script imports no longer depend
  on an external `PYTHONPATH`.
- 2026-05-22: the STAR report helper also owns logged subprocess execution and
  the standard STAR UVT feature-overfit trainer subprocess wrapper. Sparse
  forward timing repeat, sparse forward scale matrix, and target-grid render
  mode matrix reports share the `PYTHONPATH`, `TMPDIR`, timeout, stdout/stderr
  log capture, return-code status, and elapsed-time contract instead of each
  carrying a local trainer-launch block.
- 2026-05-22: the top-level `star_uvt_backward_kernel_matrix.py` now uses the
  same STAR report helper for dual-mode import bootstrap, optional report JSON
  loading, manifest JSON writing, summary CSV writing, markdown summary writing,
  and logged subprocess execution. The kernel-case definitions and row schemas
  stay local because they are specific to the v0/PRT benchmark matrix.
- 2026-05-22: support birth/split sweeps now use the same STAR logged
  subprocess helper too. The helper supports env defaults and overrides, so the
  sweep keeps its `WANDB_MODE=offline` default and `STAR_UVT_TILE_CAPACITY`
  override without local `subprocess.run(...)` blocks; its dense-support
  diagnostic uses the generic logged subprocess wrapper with a custom command.
- 2026-05-22: the target-grid analytic VJP trainer report, Gate 4 quality
  bracket report, and logit-handoff reducer report now use the same STAR report
  helper for JSON loading plus JSON/markdown output writes. Those report scripts
  still keep their domain-specific parsing and table formatting local, but no
  longer repeat parent-directory creation or sorted JSON/text artifact writes.
- 2026-05-22: target-cache budget, first-class scale summary, V-JEPA versus
  Gaussian comparison, and feature1 whole-graph reference timing now use
  `report_artifacts.load_report_json(...)` for report-shaped JSON object reads.
  Keep tolerant audit loaders local when they intentionally return `_load_error`
  rows instead of failing.
- 2026-05-22: the STAR report helper now owns optional report JSON loading and
  basic comma-separated string/int/float parsing. Direct-feature mode matrices,
  target-grid render-mode matrices, sparse-forward scale/repeat reports, and
  support birth/split sweeps no longer carry local `_read_json` or CSV-split
  helpers. The tile-slot accumulator budget now uses the same int CSV parser.
  The direct-feature mode matrix also uses the shared logged subprocess wrapper
  for its benchmark launches while keeping its CSV summary writer local.
- 2026-05-22: the same CSV parser boundary now covers direct feature-kernel,
  sparse-hidden sigmoid-MSE, sparse-hidden target-area, dense-alpha failure,
  background-cheat, and first-class backward-breakdown scripts. Feature-dim
  lists, raw-opacity bias lists, alpha sweeps, and backward mode lists share
  `split_csv_ints(...)`, `split_csv_floats(...)`, or `split_csv_strings(...)`;
  validation of allowed alpha ranges and all kernel/diagnostic math remains
  local to the owning script.
- 2026-05-22: the same report-matrix cluster now imports `ROOT`, `TRAIN_ROOT`,
  and `STAR_UVT_ROOT` from `report_artifacts` instead of rebuilding local root
  constants and mutating `sys.path` for `config_utils`. Target-grid render-mode,
  sparse-forward repeat/scale, and support birth/split scripts rely on the
  shared bootstrap while keeping their own config patching and report rows.
- 2026-05-22: `report_artifacts.write_report_csv(...)` now wraps the shared
  `train_artifacts.write_csv(...)` for root-relative STAR report CSVs while
  preserving first-seen column order. `direct_feature_mode_matrix.py` uses it
  for `summary.csv` instead of a local `csv.DictWriter` helper.
- 2026-05-22: `report_artifacts.read_report_csv(...)` now owns root-relative
  STAR report CSV reads. `logit_handoff_reduce_report.py` and
  `gate4_quality_bracket_report.py` use it instead of local `csv.DictReader`
  blocks while keeping their row filtering and report-specific type conversion
  local.
- 2026-05-22: `report_artifacts.mean_timing_without_first(...)` now owns the
  repeated `step_timings_ms[1:]` mean helper used by target-grid render-mode
  matrices, sparse-forward repeat/scale reports, and support birth/split
  sweeps. The feature1 LR reset/schedule reports use the same helper now.
  Reports still choose which timing keys to expose locally.
- 2026-05-22: `report_artifacts.summary_stats(...)` and
  `report_artifacts.distribution_stats(...)` now own the two repeated STAR
  profile timing-stat contracts: zero-empty sample/mean/min/max summaries and
  count/stdev distributions with `None` empty values. Keep both contracts
  explicit; do not collapse them into one vague stats helper.
- 2026-05-22: the STAR report helper now also bootstraps the Dynaworld root
  path, not only `src/train`, for directly launched report/prototype scripts
  that import `research_experiments.*`. The V-JEPA bridge audit,
  dense-alpha failure diagnostic, dense feature-tube prototype, V-JEPA versus
  Gaussian comparison, alpha-only visibility profile, visibility support bridge,
  visibility birth/split gate, feature1 whole-graph profile, and target-cache
  budget now use shared report JSON/text writers instead of open-coding
  parent-directory creation plus artifact writes. The dense prototype also uses
  shared device resolution/synchronization through `star_uvt_runtime`.
- 2026-05-22: STAR UVT report/prototype scripts now use the same dual-mode
  import boundary as Dynamic Foam: package imports use
  `.report_artifacts`, while direct script execution falls back to
  `report_artifacts`. A package-level global alias was rejected because pytest
  can import the Dynamic Foam helper as top-level `report_artifacts` first,
  causing STAR modules to see the wrong report API.
- 2026-05-22: the remaining STAR feature-tube profile/matrix report surfaces
  now use shared artifact helpers too. Target-grid render-mode matrices,
  sparse-forward scale/repeat/profile reports, direct-feature mode manifests,
  tile-slot budget summaries, batched sparse-forward VJP/step reports, support
  birth/split sweeps, compact visual VJP comparisons, sparse visual loss VJP
  profiles, and logit/target-grid VJP bridge profiles no longer call
  `.write_text(...)` directly for report artifacts. Log files and CSV streaming
  still use explicit file handles because those are different I/O contracts.
- 2026-05-22: `report_artifacts.load_optional_report_json_or_error(...)` now
  owns the tolerant STAR report-object read used when a report should preserve a
  `{"_load_error": ...}` row instead of failing. The V-JEPA bridge audit uses
  that helper, and the dense-alpha failure diagnostic routes checkpoint loads
  through `checkpoint_utils.load_checkpoint_mapping(...)`.
- 2026-05-21: `config_utils.require_config_keys(...)` now owns the repeated
  required-key error contract used by the STAR UVT train/probe scripts, and
  those scripts use the existing shared `config_utils.path_or_none(...)`
  instead of local `_path_or_none` helpers. The rendered-feature RGB probe no
  longer imports path handling from the feature overfit trainer just to convert
  optional output/checkpoint paths.
- 2026-05-22: STAR UVT profiling scripts now follow the same optional-path
  helper boundary. `star_uvt_logit_handoff_rgb_vjp_profile.py` and
  `star_uvt_feature1_wholegraph_profile.py` import
  `config_utils.path_or_none(...)` instead of carrying local `_path_or_none`
  helpers.
- 2026-05-21: `star_uvt_runtime.py` now owns the STAR UVT checkout path,
  optional Dynaworld-root path insertion, and loss-to-PSNR helper. Its
  device-resolution/synchronization functions now delegate to `train_devices`
  while preserving STAR's stricter requested-device validation. RGB STAR,
  feature STAR, feature RGB probe, and rendered-feature RGB probe use this
  shared runtime boundary. The probes still import feature/model helpers from
  the feature-overfit trainer, but no longer import generic runtime/device
  helpers from a trainer module.
- 2026-05-21: `star_uvt_common.py` now owns shared STAR UVT training/probe
  helpers that are not trainer policy: source video sequence loading,
  colorizer-init checkpoint loading, model/colorizer gradient-norm extraction,
  and target-grid chunk slicing. Feature STAR and both RGB probe scripts import
  those helpers from the common module instead of from
  `train_star_uvt_feature_overfit.py`. Later STAR UVT helper slices moved the
  remaining feature-target loading, render-RGB media chunks, and sparse
  target-grid mechanics into narrower modules below.
- 2026-05-21: `star_uvt_feature_targets.py` now owns cached feature-target
  materialization and adaptation for STAR UVT feature overfit/probe scripts:
  target tensor chunking, RGB-to-target-grid adapters, render-to-target-grid
  adapters, channel adaptation, normalization, streaming stats, and cache
  loading. The target-grid RGB probe imports those helpers directly instead of
  importing them through `train_star_uvt_feature_overfit.py`, so the remaining
  rendered-feature probe trainer imports are limited to STAR render/sparse-pixel
  mechanics.
- 2026-05-22: `star_uvt_feature_targets.py` now also exposes the public
  grid-RGB probe adapter/loss boundary: `FEATURE_TARGET_GRID_ADAPTERS`,
  `adapt_rgb_to_grid(...)`, `upsample_grid_rgb(...)`, and
  `mean_rgb_grid_loss(...)`. The target-grid RGB probe no longer aliases
  private adapter helpers or keeps a local `_mean_loss(...)`; the
  rendered-feature probe objective reuses the same adapter set for
  target-grid sparse-pixel sampling.
- 2026-05-22: `star_uvt_feature_rgb_probe_config.py` now owns target-grid RGB
  probe config validation: required sections/keys, grid adapter validation,
  positive step/LR checks, and target-grid materialization requirements.
  `star_uvt_feature_rgb_probe_trainer.py` imports `resolve_config(...)` from
  that module and owns target-grid RGB probe run orchestration. The historical
  `train_star_uvt_feature_rgb_probe.py` file is a thin CLI wrapper for
  `run_probe(...)`, and the registry points `star_uvt_feature_rgb_probe` at the owner
  module.
- 2026-05-22: `star_uvt_rendered_feature_probe_config.py` now owns rendered
  feature RGB probe config validation: required STAR data/colorize/output/
  logging sections, sparse pixel-source and grid-adapter validation, default
  trainable-scope flags, resume/checkpoint requirements, sample-grid bounds,
  frame-chunk checks, and feature-dim checks. The rendered-feature probe trainer
  imports `resolve_config(...)` from that module and keeps render/optimizer/
  checkpoint/logging orchestration local. The historical
  `train_star_uvt_rendered_feature_rgb_probe.py` file is now a thin
  CLI wrapper for `run_probe(...)`, and the registry points
  `star_uvt_rendered_feature_rgb_probe` at
  `star_uvt_rendered_feature_rgb_probe_trainer.run_probe`.
- 2026-05-22: `star_uvt_video_overfit_config.py` now owns RGB STAR video
  overfit config validation: required data/train/UVT/per-frame/output/logging
  sections and key sets. `star_uvt_video_trainer.py` imports
  `resolve_config(...)` from that module and owns the external
  `run_video_fit_comparison(...)` bridge, result assertions, W&B media, and row
  output orchestration. The historical `train_star_uvt_video_overfit.py` file
  is now a thin CLI wrapper for `run_training(...)`, and the registry points
  `star_uvt_video_overfit` at the owner module.
- 2026-05-21: `star_uvt_feature_rendering.py` now owns STAR UVT feature
  alpha-background RGB composition and chunked media rendering, while
  `star_uvt_sparse_grid.py` owns sparse target-grid forward/VJP plans and
  sparse pixel-id selection. The rendered-feature RGB probe no longer imports
  helpers from `train_star_uvt_feature_overfit.py`; it imports the shared media
  renderer and sparse-grid helper modules directly. This also fixed the stale
  rendered-probe media call that still passed an old `colorize_and_compose`
  argument and omitted alpha-background arguments.
- 2026-05-21: `star_uvt_colorizers.py` now owns STAR UVT `FeatureToColor`
  construction and module train/eval toggling. Feature STAR overfit, the
  target-grid RGB probe, the rendered-feature RGB probe, and the STAR UVT
  diagnostics/profilers that load feature-overfit configs use the same
  colorizer factory, including `hidden_dim=null` single-layer probes, instead
  of open-coding constructor kwargs in each script.
- 2026-05-21: `star_uvt_render_configs.py` now owns the normalized
  config-to-render-config boundary for STAR UVT feature tubes. The feature
  overfit trainer and feature-overfit diagnostics/profilers build
  `FeatureTubeRenderConfig` plus `UVTRenderConfig` through one helper instead
  of repeating the same `data`/`feature_uvt` field mapping.
- 2026-05-21: `star_uvt_models.py` now owns config-to-model construction for
  STAR UVT feature tubes. The feature overfit trainer, rendered-feature RGB
  probe, and feature-overfit diagnostics/profilers use the same
  `build_feature_tube_model(...)` helper, including the probe seed-section
  override, instead of importing the prototype model class at each call site.
- 2026-05-21: `star_uvt_feature_tube_model.py` now owns the reusable
  `FeatureTubeRenderConfig`, `FeatureScreenTimeTubeModel`, dense feature-tube
  CPU renderer, default colorizer, and legacy `colorize_and_compose(...)`
  compatibility helper. `src/train` no longer imports model/config objects from
  `research_experiments/.../dense_feature_tube_prototype.py`; that prototype
  re-exports the shared implementation for old benchmark scripts.
- 2026-05-21: `star_uvt_outputs.py` now owns STAR UVT prediction-media output
  and result-row JSON emission. RGB overfit, feature overfit, target-grid RGB
  probe, and rendered-feature RGB probe still choose their own row schemas and
  W&B media keys, but contact-sheet/video writing, FPS fallback, output path
  handling, and sorted pretty JSON row persistence live in one helper.
- 2026-05-22: the same `star_uvt_outputs.py` module now owns the STAR W&B
  row-output wrapper too. Trainers choose only `metric_prefix` and, for feature
  overfit, the extra RGB-probe media keys.
- 2026-05-21: `star_uvt_timing.py` now owns STAR UVT timing-summary helpers.
  Feature overfit, target-grid RGB probe, and rendered-feature RGB probe use
  one implementation for mean timing tables; feature overfit also uses the
  shared first/last/min/max trace-summary helper.
- 2026-05-21: `star_uvt_config_keys.py` now owns shared STAR UVT section-key
  contracts and validation helpers for common `data`, `colorize`, `output`,
  and `logging` sections. RGB STAR overfit, feature overfit config
  normalization, target-grid RGB probe, and rendered-feature RGB probe use
  those helpers instead of carrying local copies of the same key tuples.
- 2026-05-21: `star_uvt_checkpoints.py` now owns STAR UVT training checkpoint
  save/load, target-grid RGB probe checkpoint save/load, optimizer LR helpers,
  the feature-overfit RGB-probe checkpoint loader wrapper, and model-only
  resume metadata plus checkpoint saves for rendered feature probes. The
  feature overfit trainer, target-grid RGB probe, and rendered-feature RGB
  probe no longer carry separate checkpoint payload validation/save paths for
  the shared STAR model/colorizer checkpoint contracts.
- 2026-05-22: STAR UVT checkpoint saves now use
  `checkpoint_utils.atomic_torch_save(...)`. The training checkpoint helper,
  feature RGB probe checkpoint helper, and rendered-feature RGB probe checkpoint
  helper share the same parent-directory creation and temporary-file replace
  behavior instead of trainer-local `torch.save(...)` or `atomic_torch_save(...)`
  blocks.
- 2026-05-22: STAR UVT checkpoint loads now delegate mapping-payload validation
  to `checkpoint_utils.load_checkpoint_mapping(...)`; the STAR helper and
  common colorizer-init helper keep the required-key checks plus
  colorizer/model/optimizer semantics local.
- 2026-05-21: `star_uvt_sparse_visual_sampling.py` now owns sparse-visual
  pixel-source enums, VJP-mode enums, stratified/patch pixel selection, patch
  phase cycling, local-frame selection, and loss sample-count math. The feature
  overfit trainer, sparse-visual VJP profiler, and sparse-visual tests import
  those sampling helpers directly instead of treating the trainer as the helper
  module.
- 2026-05-21: `star_uvt_sparse_visual_losses.py` now owns sparse-visual RGB
  composition, autograd/manual colorizer VJP helpers, target-area loss helpers,
  alpha/black-hole losses, and native target-area VJP mode mapping. The feature
  overfit trainer and sparse-visual VJP profiler import that loss contract
  directly; tests no longer import sparse-visual loss math through the trainer.
- 2026-05-22: `star_uvt_rendered_feature_probe_objective.py` now owns the
  rendered-feature RGB probe's sparse-pixel objective helpers: target-grid
  pixel ids, stratified-grid pixel ids, target RGB gathers, sparse RGB
  composition, and local feature/alpha VJPs. It delegates to the shared STAR
  sparse-grid, sparse-visual sampling, and sparse-visual loss contracts, so the
  rendered-feature probe trainer no longer carries its own duplicate
  stratified sampling or sparse colorizer-loss formulas.
- 2026-05-21: `star_uvt_visibility_support.py` now owns STAR UVT visibility
  proxy target sampling/losses and support-birth-split tube placement. The
  feature overfit trainer keeps config validation and lifecycle orchestration,
  while tests import the geometry/loss helpers from the dedicated module rather
  than from the trainer.
- 2026-05-21: `star_uvt_schedules.py` now owns STAR UVT feature-target weight
  schedules, optimizer LR schedules, JSON schedule serialization, and the
  feature-target enabled/RGB-weight predicates. The feature overfit trainer and
  STAR profiling scripts import the schedule contract directly instead of
  re-exporting it through `train_star_uvt_feature_overfit.py`.
- 2026-05-21: `star_uvt_feature_losses.py` now owns STAR UVT feature-target
  loss/VJP mechanics: dense target-grid VJPs, sparse target-grid forward/VJPs,
  RGB-probe grid gradients, trainable colorizer grid gradients, sparse image
  VJP packing, and the VJP result records. The feature overfit trainer still
  chooses which path runs, but tests and profilers no longer need to import
  those loss helpers through the trainer.
- 2026-05-21: `star_uvt_feature_config.py` now owns STAR UVT feature-overfit
  config normalization and validation, including feature-target, sparse-visual,
  dense-alpha, visibility-proxy, and support-birth-split gates. Diagnostics,
  profilers, and tests import `resolve_config` from that module instead of
  importing `train_star_uvt_feature_overfit.py` just to normalize configs.
- 2026-05-22: `star_uvt_feature_overfit_trainer.py` now owns STAR UVT feature
  overfit run orchestration and `run_training(...)`. The historical
  `train_star_uvt_feature_overfit.py` file is a thin CLI wrapper for
  `run_training(...)`, and the registry routes `star_uvt_feature_overfit` to the
  owner module. STAR report subprocess helpers now launch through
  `src/train/train.py`, and the V-JEPA bridge audit inspects the owner module
  plus registry route instead of reading the wrapper as if it owned the
  implementation. A tiny 8-frame, 64px, 512-tube, 1-step `src/train/train.py`
  smoke passed on MPS with direct-atomic rendering, disabled W&B, `/tmp`
  output, gradient flow through STAR parameters and the colorizer, and zero
  tile overflow.
- 2026-05-21: `star_uvt_tile_stats.py` now owns STAR UVT tile-load summary
  reporting, and `star_uvt_render_modes.py` owns the shared feature-render-mode
  contract: mode order, validation set, backward-mode dispatch, effective-mode
  reporting, fallback reporting, and the feature-gradcache cap. Feature overfit
  plus backward-breakdown, logit-handoff, whole-graph, target-grid VJP, and
  target-grid render-mode profiling scripts import those helper contracts
  directly instead of treating the feature-overfit trainer as a
  statistics/constants/mode-mapping namespace. The whole-graph profiler also
  imports the RGB-probe checkpoint loader from `star_uvt_checkpoints.py`,
  leaving `resolve_config` as its only feature-overfit trainer import.

### P3 - Mixed Data Scheduler

Build the smallest bridge that samples:

```text
same_view_manifest -> same_view batch -> same_view_recon
multicam_manifest -> heldout bundle -> heldout_view_recon
```

Rules:

- Keep `same_view_recon` and `heldout_view_recon` separate in logs and configs.
- Do not invent a third manifest format until the scheduler proves it needs one.
- Start with smoke configs before large runs.

Status: smoke trainer bridge started; benchmark-ready mixed training is still
future work. Do not mark the mixed same-view plus heldout-view lane complete
until a longer checked-in smoke/benchmark config logs both loss names and result
artifacts cleanly.

Progress:

- 2026-05-21: `sequence_data.prepare_clip` now owns legacy trainer clip tensor
  preparation on top of `make_clip(...) -> ClipBatch`. Active token-GS,
  multicam, relative-pose, and export callers import it from the data module.
  The old `pipeline.render.prepare_clip` compatibility re-export was removed
  after `rg` found no real code imports.
- 2026-05-22: temporary 1-step registry smokes under
  `/tmp/dynaworld_registry_smokes` passed through `src/train/train.py` for
  Token-GS F=3, multicam RGB-pyramid, mixed same/heldout, multicam relative
  pose, Direct PowerFoam, PowerFoam Metal, Dynamic PowerFoam RBF, Dynamic
  PowerFoam token/F32, and Dynamic Gauge. A Direct PowerFoam offline-W&B smoke
  also passed and produced local eval media/checkpoint artifacts plus
  `wandb/offline-run-20260522_152129-r22iyau1`.
- 2026-05-21: `clip_sampling.sample_clip_batch` now owns the common
  frame-sampling plus `ClipBatch` construction step. Token-GS, known-camera,
  multicam, and camera-swap sample paths call it and adapt back to legacy tuple
  returns at their trainer boundaries.
- 2026-05-21: `sequence_data.ManifestSequenceSampler` now owns same-view
  manifest entry loading, eager/lazy sequence sampling, cycle/random order, and
  optional one-worker prefetch. The base token-GS trainer and the mixed
  same-heldout trainer both use it instead of carrying trainer-local manifest
  cursors.
- 2026-05-22: `json_io.load_json(...)` now owns low-level JSON file reads for
  train-local data/camera paths. `sequence_data.py`, `multicam_video_data.py`,
  and `dynamic_powerfoam_camera.py` use it for sequence summaries, camera JSON
  sequences, DeepView/CamXTime/AIST/ViVo camera metadata, and PowerFoam teacher
  camera initialization. Each caller still owns domain shape validation.
- 2026-05-22: `json_io.load_jsonl_objects(...)` now owns strict JSONL object-row
  decoding with blank-line skipping and file:line errors. Same-view manifest
  loading and multicam validation manifest loading use it, while their split
  defaults and empty-result errors stay in `sequence_data.py` and
  `multicam_val_data.py`.
- 2026-05-21: `mixed_data_scheduler.py` now defines typed `SameViewBatch`,
  `NovelViewBatch`, and `MixedStepBatch` records with explicit
  `same_view_recon` and `heldout_view_recon` names. It also owns shared
  multicam view sampling and `both`/`alternate` loss-kind scheduling. The active
  multicam trainer calls the shared view sampler, and the first mixed trainer
  consumes the same batch boundary.
- 2026-05-21: `mixed_data_scheduler.sample_mixed_step_batch(...)` now owns the
  full same-view/novel-view schedule branch for the mixed trainer. It accepts a
  lazy same-view sequence provider, so alternate novel-view steps do not load a
  same-view sequence just to skip it.
- 2026-05-21/22: `mixed_same_heldout_trainer.py` now provides the first trainer
  consumer of `MixedStepBatch`, dispatched by
  `arch=mixed_same_heldout_precomputed_feature_implicit_camera`. A checked-in
  10-step smoke config lives at
  `src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`;
  it exercised alternating same-view and heldout-view optimizer steps and wrote
  separate `Loss/same_view_recon` and `Loss/heldout_view_recon` aux keys. This
  is a bridge smoke, not a benchmark row or convergence claim. The historical
  `train_mixed_same_heldout_implicit_dynamic.py` file is now only a thin CLI
  wrapper that imports `run_training(...)`.
- 2026-05-22: current-state runtime evidence for that bridge passed at
  `wandb/offline-run-20260522_004727-ka4lm8g5`. The checked-in smoke went
  through the shared registry/CLI path, W&B offline logging, feature-cache
  load, mixed scheduler, objective/loss plumbing, and multicam validation media.
  The offline record includes both mixed loss names, train/heldout eval PSNR
  keys, final preview image, and TrainView0/TrainView1/Heldout0 rendered+GT
  videos. Treat this as interface evidence only.
- 2026-05-21: `MixedBackwardResult` and `MixedStepAccumulator` now own mixed
  same-view/heldout step aggregation inside the mixed trainer. The raw and
  weighted loss names remain explicit, but bank-rate terms, aux terms, preview
  selection, and total-loss payload assembly no longer live in two parallel
  same-view versus heldout blocks.
- 2026-05-21: `MulticamPrecomputedFeatureImplicitTrainer._recon_loss_for_views`
  now owns the shared multicam train-view and heldout-view render/loss loop.
  `multicam_recon_loss(...)` and `heldout_recon_loss(...)` only choose the
  target frame bank and render function, so preview capture, background
  sampling, alpha/background guarding, and reconstruction-loss accumulation no
  longer drift between the two paths.
- 2026-05-21: `MulticamPrecomputedFeatureImplicitTrainer._rendered_view_recon_loss`
  now owns rendered-view alpha/background guarding, reconstruction-loss
  profiling, and optional preview capture for multicam train-view, heldout-view,
  and camera-swap renders. Camera-swap keeps its relpose-specific logic local,
  but no longer carries its own copy of the rendered-view loss mechanics.
- 2026-05-21: `MulticamPrecomputedFeatureImplicitTrainer._step_result(...)`
  now owns `StepResult` assembly for multicam initial eval, normal multicam
  train steps, and camera-swap train/eval steps. The branches still own their
  math, but source path, frame count, zero camera regularizers, detach policy,
  and bank-rate term detaching now have one multicam boundary.
- 2026-05-21: `MulticamPrecomputedFeatureImplicitTrainer.multicam_validation_payload_from_renders(...)`
  now owns train/heldout target resizing, `multicam_validation_video_payload`
  dispatch, `gt_video_logged` updates, camera-rig metrics, fps wiring, and
  best-heldout bookkeeping for both base multicam and relative-pose validation.
  The branches still own only how they render the views.
- 2026-05-22: `pipeline.diagnostics.decoded_temporal_payload_from_sequence(...)`
  now owns the full-sequence decoded Gaussian temporal metric assembly. Base
  multicam external-view, base multicam oracle-relative, and full relative-pose
  validation render paths call the helper instead of rebuilding the same
  `xyz`/`scales`/`opacities`/`rgbs` detach-to-CPU buffer dict inline.
- 2026-05-21: `Trainer.temporary_render_size(...)` now owns the generic
  render-size context used by relative-pose multires train and validation
  logging. `MulticamRelativePoseImplicitTrainer` still overrides
  `_activate_render_size(...)` for token-detail-aware renderer dispatch, but it
  no longer carries a separate dense-grid cache helper or context manager.
- 2026-05-21: `runtime_types.build_step_result(...)` now owns the shared
  `StepResult` payload contract for base token-GS, known-camera, multicam, and
  mixed same-view/heldout trainers. Trainers still own sampling, math,
  backward, and optimizer behavior; the helper only centralizes source
  metadata, zero camera-loss defaults, detach policy, bank-rate terms, and aux
  loss payloads.
- 2026-05-21: `pipeline.validation_media.training_preview_payload(...)` now
  owns per-step preview image plus optional feature-PCA image payload assembly
  for the base and relative-pose `val_log` paths. The trainers still choose the
  log cadence and render-size context; the shared helper keeps the
  `feature_pca_log` missing-preview failure mode in one place.
- 2026-05-21: `Trainer.run_training_loop(...)` and `print_training_header(...)`
  now own the shared base/known-camera loop. `KnownCameraTrainer` only provides
  its start banner, camera summary, completion message, and no-browser-export
  policy, so initialization logging, step iteration, profile-print hooks,
  prefetch cleanup, and W&B finish no longer live in a copied run method.
- 2026-05-21: `Trainer.training_preamble_messages(...)` and
  `after_training_complete(...)` now cover the remaining lifecycle-only
  overrides. `PrecomputedFeatureImplicitTrainer` reports feature-cache metadata
  through the preamble hook, and `MulticamRelativePoseImplicitTrainer` saves its
  optional checkpoint through the post-success hook, so neither trainer owns a
  custom `run(...)` just to wrap the shared loop.
- 2026-05-21: `Trainer.model_eval_mode(...)` now owns eval/train restoration
  around initial diagnostics for base token-GS, known-camera, and multicam
  trainers. Those branches still own clip selection, decode, and loss math; the
  shared helper only prevents eval-mode restore drift when initial logging
  changes.
- 2026-05-21: `Trainer.train_step_context(...)` and `optimizer_step(...)` now
  own the shared zero-grad, step-total profiling, optimizer-step profiling, and
  timing-finalization envelope for base token-GS, known-camera, multicam, and
  mixed same-view/heldout steps. The branches still own sampling, decode,
  backward, and result payloads; known-camera no longer bypasses the same
  profiling shell used by the other token-GS trainers.
- 2026-05-21: `Trainer.initial_clip_indices(...)` and
  `initial_clip_for_sequence(...)` now own the repeated first-window diagnostic
  clip setup. Base token-GS, known-camera, and multicam initial diagnostics use
  the same train-frame-count index range and `prepare_clip(...)` wrapper while
  keeping branch-specific camera/decode/loss handling local.
- 2026-05-21: `KnownCameraTrainer.known_cameras_for_indices(...)` now owns the
  known-camera tuple extraction and missing-camera error used by initial eval
  and full-sequence eval. The train step also reuses the existing
  `sample_clip(...)` boundary, so known-camera camera extraction no longer lives
  in three different local shapes.
- 2026-05-21: `KnownCameraTrainer.sample_clip(...)` no longer overrides the
  base `Trainer.sample_clip(...)` with a different tuple arity. The
  camera-aware training batch helper is now `sample_known_clip(...)`, while
  `sample_clip(...)` keeps the base three-value interface across trainer
  classes.
- 2026-05-21: `Trainer.initial_recon_step_result(...)` now owns the shared
  initial eval render/reconstruction/V-JEPA/payload step used by implicit-camera
  and known-camera trainers. The branches still own how they prepare cameras and
  decode the clip; the common helper owns the first preview render and
  `StepResult` assembly.
- 2026-05-21: `KnownCameraTrainer` no longer overrides
  `render_full_sequence(...)`; the inherited base method already dispatches to
  the known-camera `_eval_decode_clip(...)`. Full-sequence validation render
  wiring now has one single-cam implementation.
- 2026-05-21: `MulticamRelativePoseImplicitTrainer` now reuses the inherited
  `_rendered_view_recon_loss(...)` helper for full relative-pose camera-swap
  renders. Relative-pose still owns residual/cycle/bank-rate math, but
  alpha/background guarding, recon-loss profiling, and preview capture no
  longer fork from the multicam implementation.

### P4 - Render Dispatch Convergence

Renderer wrappers should converge on the typed `RasterizedClip` payload.
Older wrappers that only return RGB should either wrap the typed result or be
marked legacy.

Progress:

- Current tree: active token-GS rendering goes through
  `pipeline.render.render_clip_sequence(...) -> runtime_types.RasterizedClip`,
  which carries features plus optional alpha. Full validation rendering returns
  `runtime_types.RenderedClip`. The old procedural render wrappers referenced
  by earlier cleanup notes are not present under `src/train/` in this tree.
- 2026-05-21: `render_dispatch.py` now owns model-aware renderer selection:
  decoded-token counting, token-layout detail-level accounting, token-summary
  text, effective Gaussian count, and `pick_renderer_mode_from_config(...)`.
  Token-GS and relative-pose trainers import this helper directly, removing the
  trainer-to-trainer dependency for renderer-mode selection.
- 2026-05-21: `rendering.render_gaussian_frames_rasterized(...)` now provides
  the typed alpha-aware batch render wrapper returning `RasterizedClip`. It
  centralizes the fast-mac batch call shared by tensor-only and alpha-aware
  callers, and `pipeline.render.render_clip_sequence(...)` now returns that
  payload directly.
- 2026-05-21: removed the obsolete
  `render_gaussian_frames_alpha_aware(...)` tuple wrapper after `rg` found no
  real code imports. `render_gaussian_frames_rasterized(...) -> RasterizedClip`
  is now the only public alpha-aware batch-render API.
- 2026-05-21: removed `RasterizedClip`/`RenderedClip` from
  `pipeline.render.__all__` after `rg` found no imports through that module.
  The payload dataclasses are owned and imported from `runtime_types`.

### P5 - Entrypoint Cleanup

Delete or fold only after checking active configs:

- 2026-05-21: `src/train/trainer_registry.py` now owns the
  `arch -> TrainerEntry` map, config arch validation, and shared
  `run_config(...)` dispatch. `src/train/train.py` remains the thin CLI and
  backcompat re-export surface, while active dynamic-load tests exercise the
  underlying `trainer_registry.py` module directly.
- 2026-05-21: `trainer_registry.py` now also records
  `EXTERNAL_TRAINER_BY_ARCH` for checked-in configs that are intentionally
  launched by research CLIs instead of `src/train/train.py`: gauge-field
  material surfels and the static/free-dynamic 3DGS gauge baselines. The
  registry error now points to the external launcher instead of reporting a
  generic unsupported arch.
- 2026-05-21/22: added `star_uvt_feature_rgb_probe` and
  `star_uvt_rendered_feature_rgb_probe` to the train.py registry as `run_probe`
  routes, matching the checked-in RGB-probe configs and their owner modules.
- 2026-05-21: `tests/test_trainer_registry.py` now asserts every checked-in
  `src/train_configs/*.json*` arch is either train.py-routed or explicitly
  external.
- 2026-05-21: `trainer_registry.resolve_config_for_arch(...)` now owns the
  arch-aware config-resolution boundary, and
  `trainer_registry.trainer_class_for_config(...)` owns the legacy Token-GS
  class-factory lookup for benchmark/probe callers. Colorize/init diagnostics,
  browser export, camera-scene visualization, V-JEPA benchmark scripts, and
  config-factory tests no longer import `train_video_token_implicit_dynamic.py`
  as a generic helper namespace.
- 2026-05-22: `src/dataset_scripts/visualize_multicam_rig.py` now resolves
  multicam configs through `trainer_registry.resolve_config_for_arch(...)`
  instead of importing `MulticamPrecomputedFeatureImplicitTrainer` only to call
  its config resolver. The script keeps camera-rig visualization logic local.
- 2026-05-22: `src/dataset_scripts/script_paths.py` now owns dataset-script
  repo-root resolution, train-path insertion, repo-relative path resolution,
  and repo-relative display strings. The single-video pretrain manifest builder
  and multicam rig visualizer both use it, so direct dataset CLIs no longer
  repeat local `Path(__file__).parents[2]` plus `sys.path.insert(...)` blocks.
- 2026-05-22: those dataset CLIs also route complete JSON/text artifact writes
  through `train_artifacts.write_json(...)` and `write_text(...)`.
  `build_single_video_pretrain_manifest.py` also routes compact manifest JSONL
  through `train_artifacts.write_jsonl(..., compact=True)`, so its old local
  writer is gone without changing row formatting.
- 2026-05-22: the single-video pretrain launchers now route their embedded
  manifest audits, load checks, cache-status counts, prebake totals, and
  full-cache guards through `json_io.load_jsonl_objects(...)` when the file is a
  complete JSONL object manifest. This preserves their stdout schemas while
  sharing strict object-row decoding and file:line errors with train-local
  manifest loaders.
- 2026-05-22: `trainer_registry.TrainerEntry` now optionally records a
  `trainer_class` name, and `instantiate_trainer_for_config(...)` builds
  class-based trainers through the same arch registry. Precomputed, multicam
  precomputed, mixed same-heldout, and multicam relative-pose configs no longer
  need benchmark/probe code to import their concrete trainer classes directly.
- 2026-05-22: `precomputed_feature_trainer.py` now owns
  `PrecomputedFeatureImplicitTrainer`, feature-cache defaults, and the
  precomputed-feature `run_training(...)` implementation. The old
  `train_precomputed_feature_implicit_dynamic.py` file is now a thin CLI
  wrapper that imports only `run_training(...)`, while the registry points
  precomputed-feature arches at the non-CLI module and multicam imports the
  base class from that module.
  The trainer-as-helper chain no longer starts at this CLI-named file.
- 2026-05-22: `multicam_precomputed_trainer.py` now owns
  `MulticamPrecomputedFeatureImplicitTrainer`, multicam defaults, and multicam
  `run_training(...)`. The old
  `train_multicam_precomputed_feature_implicit_dynamic.py` is now a thin CLI
  wrapper that imports only `run_training(...)`. Mixed same-heldout,
  relative-pose, temporal tests, and the registry import the multicam base from
  the non-CLI module, so the live trainer-as-helper chain no longer routes
  through the multicam launcher.
  The remaining large trainer owners are non-CLI modules.
- 2026-05-22: `token_gs_trainer.py` now owns the base Token-GS implementation:
  `Trainer`, `KnownCameraTrainer`, Token-GS config defaults, render-size
  schedule normalization, `trainer_class_for_config(...)`, and
  `run_training(...)`. The old `train_video_token_implicit_dynamic.py` is now a
  thin CLI wrapper that imports only `run_training(...)`. Registry entries for
  `tokengs`, implicit-camera Token-GS, and known-camera Token-GS point at the
  non-CLI owner, and `precomputed_feature_trainer.py` subclasses the base
  trainer from that module. This removes the last major live trainer-as-helper
  import through a CLI-named Token-GS launcher.
- 2026-05-22: `mixed_same_heldout_trainer.py` now owns
  `MixedSameHeldoutPrecomputedFeatureTrainer`, `MixedBackwardResult`,
  `MixedStepAccumulator`, mixed-train defaults, and `run_training(...)`. The old
  `train_mixed_same_heldout_implicit_dynamic.py` file is a thin CLI wrapper
  that imports only `run_training(...)`. The registry now points
  `mixed_same_heldout_precomputed_feature_implicit_camera` at the non-CLI owner,
  and tests import the owner module directly.
- 2026-05-22: `multicam_relative_pose_trainer.py` now owns
  `MulticamRelativePoseImplicitTrainer`, relative-pose defaults, multires
  normalization helpers, first-frame feature sequence construction, and
  `run_training(...)`. The old
  `train_multicam_relative_pose_implicit_dynamic.py` file is now a thin
  CLI wrapper that imports only `run_training(...)`. The registry points
  `multicam_relative_pose_implicit_camera` at the non-CLI owner, and focused
  tests import the owner module directly.
- 2026-05-22: `src/benchmarks/trainer_phase_benchmark.py` now imports
  `instantiate_trainer_for_config(...)` from `trainer_registry`, and the
  300-video pretrain launcher resolves decoded token counts through
  `render_dispatch`. `camera_swap_variant_parity.py` and
  `benchmark_multicam_vjepa.py` also instantiate multicam trainers through the
  registry. Remaining imports from concrete Token-GS/precomputed trainer
  modules are structural tests, not generic helper lookups.
- 2026-05-22: `export_dynaworld_browser_bundle.py` now uses the already
  resolved feature config directly, `visualize_camera_scene_diagnostic.py`
  instantiates the multicam relative-pose trainer through
  `trainer_registry.instantiate_trainer_for_config(...)`, and fixed-render
  parity CLIs use `trainer_uses_multicam_phase(...)` instead of class-name
  string checks.
- 2026-05-22: `trainer_capabilities.py` now owns the multicam trainer
  capability predicate. Trainer phase, fixed-render, camera-swap,
  alpha-background, and V-JEPA performance entrypoints instantiate trainers
  through `trainer_registry.instantiate_trainer_for_config(...)`; none of those
  live benchmark/probe scripts import a trainer class factory just to build an
  instance.
- 2026-05-22: `src/benchmarks/benchmark_memory.py` now owns device memory
  snapshots, cache-clear call policy, sampled peak tracking, and
  `run_with_memory_sampling(...)`. `trainer_phase_benchmark.py` and
  `train_step_memory_benchmark.py` share that utility instead of one benchmark
  importing generic runtime helpers from the other. Low-level GC/MPS/CUDA cache
  clearing lives in `train_devices.clear_torch_device_cache(...)`.
- 2026-05-22: `src/benchmarks/benchmark_compare.py` now owns benchmark seed
  setup, tensor diff stats, gradient diff stats, and max-diff reduction.
  `src/benchmarks/fixed_render_cases.py` owns heldout fixed-render case
  assembly. Fixed-render variant parity, fixed-render backward-mode parity, and
  camera-swap parity no longer import private helpers from each other.
- 2026-05-22: `src/benchmarks/benchmark_gradients.py` now owns
  `GaussianSequence` leaf-gradient snapshots and named module parameter
  gradient snapshots. Fixed-render variant parity, fixed-render backward-mode
  parity, and camera-swap parity share the same gradient-capture contract
  instead of carrying local clone/grad collectors.
- 2026-05-22: `src/benchmarks/fixed_render_graph.py` now owns fixed-render
  graph primitives: `PhaseTimer`, `RasterGraph`, `FixedRenderCase`, fast-mac
  project/raster split, train-target chunking, fixed-render sequence
  detach/clone helpers, background slicing, and `prepare_fixed_render_case`.
  `trainer_phase_benchmark.py` is now a consumer of those primitives instead of
  the owner that parity CLIs import through.
- 2026-05-22: `profile_fast_mac_render_phases.py` no longer reaches into
  stale v5-only config helpers. Its projected-raster probe now calls the shared
  fast-mac projected RGB/feature rasterizer helpers, so it follows the active
  `rgb_variant`/`feature_variant` dispatch instead of hard-coding one bridge.
- 2026-05-21: `trainer_registry.run_config_dict(...)` now owns in-memory
  config dispatch through the same arch registry. STAR feature-tube experiment
  scripts that patch configs in memory no longer import
  `train_star_uvt_feature_overfit.run_training` directly just to launch the
  configured trainer.
- 2026-05-21/22: `src/train/train_cli.py` now owns repeated one-config CLI
  loading, public `main(config_or_path)` dispatch for train/probe modules, and
  raw path-argument dispatch for `src/train/train.py`. It also owns the small
  `parse_csv_ints(...)` helper for train-probe CLI lists such as colorize
  `--seeds`. Keep `src/train/train.py` custom only in the sense that it still
  routes by config path through `trainer_registry.run_config(...)`; do not force
  it through the config-dict helper.
- 2026-05-21: `train_logging.finish_wandb_run(...)` now owns the shared
  `run is not None` finish guard across Token-GS, PowerFoam-family, and STAR UVT
  modules. This is intentionally a tiny lifecycle helper, not a broader W&B
  context-manager refactor.
- `ltx_feature_implicit_camera` and `wan_vace_feature_implicit_camera` already
  dispatch through `src/train/train.py` to
  `train_precomputed_feature_implicit_dynamic`.
- The older typo/image/dynamicTokenGS shim files referenced by earlier notes
  are absent from the current `src/train/` tree. Keep this section focused on
  live entrypoints only.

## Non-Goals

- Do not build a giant abstract base trainer.
- Do not unify PowerFoam/WorldFoam trainers into the token-GS trainer surface.
- Do not hide different data semantics behind one vague "unified loader."
- Do not scatter config defaults across use sites.
- Do not add environment-variable knobs for every experiment setting.

## Experiment Organization Rules

For a reusable training lane:

1. Config lives in `src/train_configs/`.
2. Script launcher lives in `src/train_scripts/` and only chooses a config.
3. Logs go to `outputs/run_logs/` or a lane-specific `results/` folder.
4. Benchmark rows go to `BASELINES.md`.
5. Active-lane status goes to `EXPERIMENTS.md`.
6. Raw chronology goes to `agent_notes/loose_notes/`.
7. Surprising lessons go to `agent_notes/key_learnings.md`.

For shader/research forks:

1. Keep fork-specific code under `research_experiments/`, `star_uvt/`, or
   `third_party/*/variants/`.
2. Save result JSONs next to the experiment.
3. Promote only after parity, timing, and quality gates are all named.
4. Keep failed variants documented long enough to prevent repeat work.

## How To Add A New Trainer Feature

1. Check `TODO/trainer_landscape_unification.md`.
2. Add the feature as a helper if two trainers need it.
3. Wire the helper into single-cam first, then multicam.
4. Run the relevant smoke gate from `AGENTS.md`.
5. Add a loose note if the change affects experiment interpretation.

The standard of success is fewer behavior forks, not just fewer lines.
