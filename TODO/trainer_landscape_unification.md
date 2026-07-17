# Trainer Landscape Unification

## TL;DR

- The original audit found real trainer drift around feature/RGB composition,
  validation media, render payloads, CLI boilerplate, registry dispatch,
  device/sync policy, W&B lifecycle, artifact writes, and mixed same/heldout
  sampling. Most of those have now been landed as small shared helpers.
- The active background/composition path is no longer trainer-local:
  `objective.compose_rgb`, `RGBReconObjective.render_view(...)`, and
  `objective.colorize_and_compose_feature_rgb(...)` own the alpha/background
  contract for Token-GS, multicam feature training, and STAR UVT feature
  surfaces.
- Old delete candidates from the first audit are already absent or folded in
  the current `src/train/` tree. Do not chase historical filenames; use the
  live-file checks below before deleting anything.
- The current cleanup posture is conservative: keep experiment-specific loops,
  optimizer contracts, result schemas, checkpoints, images, and videos local;
  keep unifying repeated boundaries such as config CLI, registry dispatch,
  device/sync, artifact I/O, logging cadence, and benchmark/report plumbing.
- Remaining work is not another abstract base class. The useful next slices are
  to run the STAR/dynamic alpha-background ablation, continue live-file helper
  routing where duplication is still active, and use real W&B/media/benchmark
  evidence before calling any trainer path solved.

## 2026-05-21 Progress

- `src/train/train_logging.py` now owns shared log-cadence gates:
  `should_log_step`, `should_log_scalar`, `should_log_image`, and
  `should_log_video`.
- PowerFoam, Dynamic PowerFoam, Dynamic Gauge Foam, and the main token-GS
  trainer now call the shared cadence helpers instead of open-coding modulo
  plus last-step checks.
- `src/train/train_logging.py` now owns `init_wandb_run(cfg)` for trainers that
  use the same `logging.wandb_*` config contract. PowerFoam, Dynamic PowerFoam,
  Dynamic Gauge Foam, and STAR UVT probe/overfit scripts use it.
- The base token-GS trainer now uses `init_wandb_run(cfg)` as well. Missing
  `logging.wandb_enabled` still normalizes to `true` for old configs, while
  explicit `false` skips W&B logging and finishes without an implicit W&B run.
- `src/train/train_logging.py` now owns the base `scalar_payload(...)` helper
  for `StepResult` scalar W&B logs: losses, sequence counts, camera metrics,
  bank-rate terms, and aux loss terms. `pipeline.validation_media` is now
  media-only; trainers still own branch-specific scalar extensions.
- `src/train/train_logging.py` now owns `log_wandb_payload(...)` for the
  generic W&B payload submit call. Token-GS and multicam relative-pose trainers
  keep their payload assembly local, but no longer import W&B directly only to
  call `wandb.log(payload, step=step)`.
- `src/train/train_logging.py` now owns `log_wandb_run_payload(...)` for
  explicit run-object W&B submit calls. PowerFoam-family and Dynamic
  Gauge/Dynamic PowerFoam paths keep their payload assembly and logging cadence
  local, but the current `src/train` tree no longer has direct
  `wandb_run.log(...)` call sites.
- `src/train/train_logging.py` now owns `log_wandb_run_payload_lazy(...)` for
  expensive explicit-run payloads. Token-GS, multicam relative-pose,
  PowerFoam-family, and Dynamic Gauge eval/media paths preserve disabled-W&B
  laziness without repeating `if wandb_run is not None` guards around image and
  video payload construction.
- `src/train/train_logging.py` now owns `wandb_run_lifecycle(...)` for
  PowerFoam/Gauge trainer owners that use the shared `logging.wandb_*` config
  contract. It guarantees `finish_wandb_run(...)` in a `finally` block without
  moving the train loop, checkpoint policy, or media payload schemas.
- Gauge Fields material-surfel training now shares the W&B submit/finish
  helpers too: `wandb_log_training_logs(...)` and the final media log call route
  through `log_wandb_payload(...)`, and the finally block calls
  `finish_wandb_run()`. The Gauge-specific `log_to_wandb` config and W&B init
  stay local because its logging schema differs from `logging.wandb_enabled`.
- `src/train/train_logging.py` now owns `set_default_wandb_mode(...)` for
  benchmark/probe entrypoints that should default W&B off without overriding a
  caller-provided environment. Trainer-phase, train-step memory, camera-scene
  diagnostics, and V-JEPA performance scripts share the same helper.
- `src/train/wandb_media.py` now owns low-level W&B media helpers:
  `make_wandb_video(...)`, `make_preview_image(...)`, and
  `make_wandb_image(...)`, plus `add_existing_wandb_media(...)` and
  `build_validation_video_payload(...)`. PowerFoam-family trainers and
  `pipeline.validation_media` import those directly, so `train_logging.py` no
  longer mixes log cadence/scalars with image/video constructors and
  `pipeline.validation_media` no longer imports W&B directly.
- `src/train/pipeline/diagnostics.py` now owns shared reconstruction metric
  helpers for trainer result JSONs. PowerFoam, Dynamic PowerFoam, Dynamic Gauge
  Foam, and PowerFoam Metal eval paths call them without changing existing
  metric keys.
- The current tree already has the RGB composition interface that this doc
  originally proposed: `objective.compose_rgb` plus
  `RGBReconObjective.render_view`. Single-cam token-GS and multicam feature
  trainers call that objective path instead of owning separate
  `alpha * colorize + background` formulas.
- `RGBReconObjective.require_alpha_for_feature_background(...)` now owns the
  F32 random/fixed-background safety guard. The base token-GS path, multicam
  train/heldout paths, and camera-swap paths call the same guard instead of
  carrying duplicate `feature_dim != 3 and alpha is None` checks.
- STAR UVT feature rendering, the background-cheat diagnostic, and the dense
  feature-tube prototype compatibility shim now use
  `objective.colorize_and_compose_feature_rgb(...)`, backed by shared tensor
  helpers for RGB-after-colorizer and feature-before-colorizer background
  composition. This keeps the alpha-background ablation strategies configurable
  while removing the separate STAR-local composition formula from active code.
- The current tree already has the validation-media helper layer that this doc
  originally proposed: `pipeline.validation_media` builds single-cam and
  multicam video payloads, alpha/feature-PCA diagnostics, and composite grids.
- W&B media payload assembly remains trainer-local. That is intentional: media
  naming and artifact choices still differ by trainer family.
- The routed owner/wrapper sweep is now complete for the current warm-path
  trainers: `src/train/train.py` routes to non-CLI owner modules, while the
  historical `train_*.py` files are thin CLI wrappers. Remaining
  cleanup should be live-helper duplication, experiment evidence, or
  model-class extraction only when it removes a real dependency. Do not
  introduce a base trainer class.
- `runtime_types.RasterizedClip` and `runtime_types.RenderedClip` now own the
  clip-level render payload contracts. `pipeline.render` owns render
  orchestration functions and imports those payload dataclasses instead of
  defining runtime types locally. `pipeline.render.__all__` no longer re-exports
  those dataclasses; callers import them from `runtime_types`.
- `sequence_data.prepare_clip` now owns frame/time clip tensor preparation on
  top of `make_clip(...) -> ClipBatch`. Active trainer/export callers import it
  from the data module; the old `pipeline.render.prepare_clip` compatibility
  re-export was removed after `rg` found no real code imports.
- `json_io.load_json(...)` now owns low-level JSON file reads for train-local
  data/camera paths. The sequence loader, multicam loader, and Dynamic
  PowerFoam camera-teacher path use it while keeping shape and domain
  validation local.
- `json_io.load_jsonl_objects(...)` now owns strict JSONL object-row decoding
  for train-local manifests. Same-view and multicam validation loaders share
  blank-line skipping plus file:line errors, while each loader keeps its own
  split default and no-records error.
- `clip_sampling.sample_clip_batch` now owns the repeated
  `select_frame_indices(...) -> make_clip(...)` pattern. Token-GS,
  known-camera, multicam, and camera-swap sample paths call it while preserving
  their legacy return tuple shapes.
- `render_dispatch.py` now owns decoded-token counting, token-layout detail
  levels, token-summary text, and `pick_renderer_mode_from_config(...)`.
  Token-GS and relative-pose trainers use this shared boundary, so the
  relative-pose trainer no longer imports renderer-mode helpers from the
  token-GS trainer.
- `rendering.render_gaussian_frames_rasterized(...)` now exposes the typed
  `RasterizedClip(features, alpha)` batch-render contract at the renderer
  wrapper layer. `pipeline.render.render_clip_sequence(...)` calls that typed
  wrapper directly. The old `render_gaussian_frames_alpha_aware(...)` tuple API
  was removed after `rg` found no real code imports.
- `trainer_registry.py` now separates train.py-routed arches from explicit
  external research launcher arches. `star_uvt_feature_rgb_probe` and
  `star_uvt_rendered_feature_rgb_probe` are routed to their owner modules'
  `run_probe` entrypoints, while gauge-field material surfels and
  static/free-dynamic 3DGS gauge baselines are recorded as external launchers
  under `research_experiments/gauge_fields/`.
- `tests/test_trainer_registry.py` now guards the config/registry boundary: all
  checked-in `src/train_configs/*.json*` arches must be either train.py-routed
  or explicitly external.
- `trainer_registry.resolve_config_for_arch(...)` now shares the arch-aware
  config-resolution boundary for diagnostics and export/profiling scripts.
  Token-GS colorize/init probes, browser export, camera-scene visualization,
  V-JEPA benchmark scripts, and config-factory tests no longer import
  `train_video_token_implicit_dynamic.py` just to reach `resolve_config` or the
  Token-GS trainer class factory.
- `trainer_registry.TrainerEntry` now optionally records a `trainer_class`
  name, and `instantiate_trainer_for_config(...)` builds class-based trainers
  through the registry. Precomputed, multicam precomputed, mixed same-heldout,
  and multicam relative-pose configs now share the same instantiation boundary
  as Token-GS factory configs.
- `src/benchmarks/trainer_phase_benchmark.py`,
  `src/benchmarks/camera_swap_variant_parity.py`, and
  `research_experiments/vjepa_performance/benchmark_multicam_vjepa.py` no
  longer import concrete precomputed or multicam trainer classes directly.
  Direct imports left by the current scan are structural temporal-sampling
  tests that call class methods on object shells.
- `export_dynaworld_browser_bundle.py` uses the already resolved feature config
  directly, `visualize_camera_scene_diagnostic.py` instantiates the multicam
  relative-pose trainer through `trainer_registry`, and fixed-render parity
  CLIs use `trainer_uses_multicam_phase(...)` instead of fragile class-name
  string checks.
- `trainer_capabilities.py` now owns the multicam trainer capability predicate,
  so benchmark scripts no longer import that API from
  `trainer_phase_benchmark.py`. Trainer phase, fixed-render, camera-swap,
  alpha-background, and V-JEPA performance entrypoints instantiate through
  `trainer_registry.instantiate_trainer_for_config(...)`.
- `src/benchmarks/benchmark_memory.py` now owns benchmark memory sampling and
  cache-clearing helpers. `trainer_phase_benchmark.py` and
  `train_step_memory_benchmark.py` use it directly instead of sharing generic
  runtime helpers through one benchmark module.
- `src/benchmarks/benchmark_compare.py` now owns shared benchmark seed setup,
  tensor diff stats, gradient diff stats, and max-diff reduction.
  `src/benchmarks/fixed_render_cases.py` owns heldout fixed-render case setup.
  Fixed-render and camera-swap parity CLIs no longer import private helpers
  from one another.
- `src/benchmarks/benchmark_gradients.py` now owns `GaussianSequence`
  leaf-gradient snapshots and named module parameter gradient snapshots.
  Fixed-render and camera-swap parity CLIs share that one contract instead of
  duplicating clone/grad collector helpers.
- `src/benchmarks/fixed_render_graph.py` now owns fixed-render graph
  primitives: `PhaseTimer`, `RasterGraph`, `FixedRenderCase`, fast-mac
  project/raster split, target chunking, fixed-render sequence detach/clone
  helpers, background slicing, and `prepare_fixed_render_case`.
  `trainer_phase_benchmark.py` now consumes that module instead of serving as
  the shared helper namespace for parity CLIs.
- `research_experiments/vjepa_performance/vjepa_benchmark_common.py` now owns
  V-JEPA performance benchmark bootstrap, positive CSV parsing, seed setup,
  device-synchronized timing, and timing summaries. The V-JEPA performance
  scripts no longer carry one-off `sys.path` mutations or local timing/seed
  helper copies. The generic CSV tokenization is shared with `train_cli.py`;
  V-JEPA keeps only the positive/nonempty argparse semantics.
- `model_factories.build_colorizer(...)` now owns Token-GS colorizer
  construction for the trainer plus the colorize-init and colorize-matrix
  probes. The probes no longer hand-build `FeatureToColor`, so
  view-conditioning, detach-view flags, unknown-key validation, and future
  colorize kwargs do not fork from the trainer path.
- The Token-GS trainer no longer stores low-use copies of normalized config
  values such as `feature_dim`, recon backward strategy, temporal microbatch
  size, profile timing sync/cadence, and profile backward split. Those values
  are read from `self.model_cfg` or `self.train_cfg` where used; broader
  section aliases remain because they are high-use and keep call sites shorter.
- `star_uvt_colorizers.build_default_feature_colorizer(...)` now owns the
  default STAR feature-tube colorizer settings. The dense feature-tube model and
  the autograd overfit benchmark no longer repeat the LN + kaiming gain-4
  `FeatureToColor` constructor locally.
- `powerfoam_colorizers.py` now owns Dynamic PowerFoam feature-colorizer
  defaults, RGB identity initialization, and the token-feature-mode colorizer
  builder. The dynamic PowerFoam trainer now calls that shared builder directly
  instead of keeping a local `build_colorizer(...)` pass-through wrapper.
- `dynamic_powerfoam_metal_config.py` now owns Dynamic PowerFoam Metal defaults,
  the `token_rbf_features` mode name, colorize defaults wiring, camera/render
  config validation, and `resolve_config(...)`. The trainer imports/re-exports
  these names for existing tests/scripts but no longer carries the large config
  normalization block inline. `dynamic_powerfoam_metal_trainer.py` now owns the
  full Dynamic PowerFoam implementation and `run_training(...)`; the historical
  `train_dynamic_powerfoam_metal.py` file is a thin CLI wrapper that imports
  only `run_training(...)`. The registry routes `dynamic_powerfoam_metal` to the
  owner module. One-step `src/train/train.py` smokes passed for both the RBF
  branch and the token/F32 branch with 4 frames, 64 cells, MPS, `/tmp` outputs,
  and disabled W&B.
- Focused Dynamic PowerFoam tests now import defaults and
  `resolve_config(...)` from `dynamic_powerfoam_metal_config.py`, and raster
  config construction from `powerfoam_raster_config.py`. The full trainer is
  still imported for model classes, but pure config/raster helper imports no
  longer route through the trainer file.
- `visualize_camera_scene_diagnostic.py` now follows the same Dynamic PowerFoam
  boundary: mode constants come from `dynamic_powerfoam_metal_config.py`,
  raster config construction comes from `powerfoam_raster_config.py`, and the
  owner-trainer import is limited to structural model classes for checkpoint
  decode.
- `dynamic_powerfoam_temporal.py` now owns pure Dynamic PowerFoam temporal
  helpers: Gaussian time bases, temporal basis fitting, acceleration
  regularizers, bounded `atanh`, and temporal motion metrics. The trainer keeps
  compatibility imports, while the focused dynamic test imports the motion
  metric helper directly.
- `dynamic_powerfoam_camera.py` now owns Dynamic PowerFoam implicit camera
  construction, camera optimizer groups, regularization and compact metrics,
  teacher camera loading/alignment/prefit, and camera-decoded ray assembly. The
  trainer keeps compatibility imports, while focused tests and the camera-scene
  diagnostic use the light helper directly.
- `dynamic_powerfoam_initialization.py` now owns Dynamic PowerFoam
  initialization geometry: frame-to-camera and camera-to-world transforms,
  orbit-camera normal initialization, orbit-video point/texel initialization,
  and token-feature texel initialization. The trainer imports those helpers,
  while its two model variants keep only their parameterization and
  training-facing behavior.
- `dynamic_powerfoam_rendering.py` now owns Dynamic PowerFoam premultiplied
  feature rendering helpers: RGB background sampling, alpha-normalized
  feature-to-RGB composition, no-grad full-video eval rendering, per-frame
  reconstruction metrics, and temporal alpha metrics. The trainer keeps
  W&B/media artifact policy local because payload names and artifact files are
  train-lane specific.
- `dynamic_gauge_rendering.py` now owns Dynamic Gauge Foam render kwargs and
  no-grad full-video eval rendering. The Gauge trainer keeps optimizer groups,
  losses, metrics payloads, checkpointing, and media policy local.
- `dynamic_gauge_config.py` now owns Dynamic Gauge Foam defaults and
  `resolve_config(...)`, matching the PowerFoam-family config-module pattern.
  The Gauge trainer no longer carries default dicts and config validation
  inline. `dynamic_gauge_foam_trainer.py` now owns the full Dynamic Gauge
  trainer implementation and `run_training(...)`; the historical
  `train_dynamic_gauge_foam.py` file is a thin CLI wrapper that imports only
  `run_training(...)`. The registry routes `dynamic_gauge_foam` to the owner
  module. A one-step `src/train/train.py` smoke passed on MPS with
  4 frames, 64 primitives, `/tmp` output, disabled W&B, and final checkpoint
  write.
- `dynamic_gauge_objectives.py` now owns Dynamic Gauge Foam training loss
  assembly: RGB L1/MSE, gauge connection loss, temporal acceleration loss,
  opacity/radius regularizers, atlas total variation, and weighted total loss.
  The Gauge trainer keeps sampling, optimizer stepping, scalar naming,
  artifact logging, and checkpointing local.
- `dynamic_powerfoam_staging.py` now owns Dynamic PowerFoam stage controls:
  static-geometry warmup, no-repaint warmup, camera-curriculum active-frame
  selection, and optional camera active-prefix mutation. The trainer keeps
  stage logging local, while focused tests import the staging helpers directly.
- Scale/pretrain shell launchers now route embedded Python config checks,
  probes, and runs through `trainer_registry.resolve_config_for_arch(...)` and
  `run_config_dict(...)` instead of importing concrete precomputed or multicam
  trainers as generic helper namespaces.
- The multicam scale/pretrain launcher now uses the registry CLI
  `src/train/train.py` for sample/sweep runs instead of launching
  `train_multicam_precomputed_feature_implicit_dynamic.py` directly. The arch
  still resolves to `MulticamPrecomputedFeatureImplicitTrainer`; the shell
  entrypoint now follows the same dispatch path as the other routed configs.
- Older Token-GS/precomputed shell launchers now use the same registry CLI for
  configs that already resolve to registered arches: static/dynamic V-JEPA
  ablations, local 30-clip baselines, 256px scene-distinct baselines,
  single-video pretrain 100/all-YouTube 64f routes, and the V-JEPA/local
  single-overfit comparison matrix. A fake-runner registry smoke covered all
  31 config paths without launching training. The STAR fast-overfit launcher
  now also routes its registered STAR RGB/feature and dynamic-gsplat configs
  through `src/train/train.py`, including compact visual and native full-cell
  feature-overfit modes.
- The Dynamic Foam external-blocker runner now emits its PowerFoam Metal train
  command through `src/train/train.py`, preserving the existing
  `src/train:third_party/powerfoam-metal` environment. This removes one more
  direct trainer-script launch without changing the PowerFoam path setup.
- Scale/pretrain launchers now also use `train_artifacts.write_json(...)` for
  complete patched config artifacts. The multicam per-record temp config and
  1k single-video smoke config share the parent-safe config writer; manifest
  JSONL row copying stays text-local because it is not the same artifact
  contract.
- `powerfoam_geometry.py` now owns pure PowerFoam ray and surface-frame helpers:
  pinhole rays, camera rays, camera-ray grids, stable tangent fallback, and
  orthonormal surface frames. PowerFoam Direct and Metal import/re-export the
  shared ray helpers, and Dynamic PowerFoam no longer imports those geometry
  utilities from the full Metal trainer.
- `powerfoam_adjacency.py` now owns CSR adjacency construction and adjacency
  stats for PowerFoam-family trainers. Regular-triangulation adjacency lazy-loads
  the Metal extension only when requested. PowerFoam Metal re-exports the
  helpers for compatibility, while Dynamic PowerFoam and Dynamic Foam
  diagnostics import adjacency directly from the helper module.
- Dynamic Foam diagnostics now use `train_devices.resolve_torch_device(...)`
  for device selection instead of importing `resolve_device` from the full
  PowerFoam Metal trainer. The later cleanup removed the stale
  `train_powerfoam_metal.resolve_device(...)` compatibility wrapper after `rg`
  found no live imports; PowerFoam Metal and diagnostics now call the shared
  device helper directly with their explicit auto policy.
- Dynamic Foam diagnostics now import `POWERFOAM_SOFTPLUS_BETA` from
  `powerfoam_direct` and `reconstruction_eval_metrics(...)` from
  `pipeline.diagnostics` instead of reaching through the PowerFoam Metal
  trainer for those pure constants/helpers.
- `src/benchmarks/trainer_phase_benchmark.py` now reaches the Token-GS class
  factory through `trainer_registry`, and
  `train_single_video_pretrain_300_64f.sh` imports decoded-token accounting
  from `render_dispatch`. A targeted scan now only finds structural
  subclass/test imports from the main Token-GS trainer.
- The single-video pretrain shell launchers now use
  `json_io.load_jsonl_objects(...)` for embedded manifest audits, load checks,
  cache-status counts, prebake totals, and full-cache guards when operating on
  complete JSONL object manifests. This keeps strict object-row parsing aligned
  with `sequence_data`/`multicam_val_data` without changing launcher output
  schemas.
- `precomputed_feature_trainer.py` now owns the precomputed-feature base trainer
  class, feature-cache defaults, and `run_training(...)`. The historical
  `train_precomputed_feature_implicit_dynamic.py` module remains as a thin CLI
  wrapper that imports only `run_training(...)`; registry entries for
  precomputed-feature arches route to the non-CLI module, and multicam imports
  the base class from that module instead of using the CLI file as a helper
  namespace.
- `multicam_precomputed_trainer.py` now owns the multicam base trainer class,
  multicam defaults, and `run_training(...)`. The historical
  `train_multicam_precomputed_feature_implicit_dynamic.py` module remains as a
  thin CLI wrapper that imports only `run_training(...)`; mixed same-heldout,
  relative-pose, temporal tests, and registry entries now import the multicam
  base from the non-CLI module. The remaining large trainer owners are non-CLI
  modules.
- `token_gs_trainer.py` now owns the base Token-GS trainer implementation:
  `Trainer`, `KnownCameraTrainer`, config defaults, render-size schedule
  normalization, the legacy class factory, and `run_training(...)`. The
  historical `train_video_token_implicit_dynamic.py` module remains as a thin
  CLI wrapper that imports only `run_training(...)`; registry entries for
  `tokengs`, implicit-camera Token-GS, and known-camera Token-GS route to the
  non-CLI module, and precomputed-feature subclasses import the base trainer
  from there.
  This clears the last major live trainer-as-helper import through a CLI-named
  Token-GS file.
- `trainer_registry.run_config_dict(...)` now shares in-memory config dispatch
  for scripts that patch configs programmatically. STAR feature-tube scale and
  alpha-background ablation scripts use the registry instead of importing
  `train_star_uvt_feature_overfit.run_training` directly.
- `train_optim.adam_with_device_fused(...)` now owns the repeated Token-GS
  Adam fused-kernel policy. Base token-GS and the relative-pose-only scope use
  it; PowerFoam, Dynamic PowerFoam, Dynamic Gauge Foam, and STAR UVT optimizers
  remain local because their optimizer contracts differ.
- `powerfoam_metal_config.py` now owns pure PowerFoam Metal defaults, feature
  mode sets, LR group specs, and `resolve_config(...)`.
  `powerfoam_metal_trainer.py` now owns the full Metal trainer implementation
  and `run_training(...)`; the historical `train_powerfoam_metal.py` file is a
  thin CLI wrapper that imports only `run_training(...)`. Dynamic Foam
  diagnostics and point-cloud builders import config resolution from the config
  module instead of using the trainer as a helper, and the registry routes
  `powerfoam_metal` to the owner module. A one-step
  `src/train/train.py` smoke based on the 64px local Mac config passed on MPS
  with `/tmp` output, final checkpoint write, and eval L1 improving from
  0.07646 at step 0 to 0.07628 at step 1.
- `powerfoam_raster_config.py` now owns PowerFoam Metal and Dynamic PowerFoam
  Metal `FoamRasterConfig` construction. Both trainers keep their old
  `make_raster_config(...)` aliases for tests/scripts, while Dynamic Foam
  diagnostics import raster config construction from the light helper instead
  of reaching through the full Metal trainer.
- `powerfoam_training.py` now owns shared PowerFoam train-step primitives:
  `flatten_multiview_powerfoam_samples(...)` and
  `exp_scheduled_weight(...)`. PowerFoam Direct and PowerFoam Metal import and
  re-export those names.
- `powerfoam_objectives.scheduled_loss_weights(...)` now covers Direct
  PowerFoam as well as Metal-style PowerFoam schedules. Direct keeps
  `rgb_mse_sum_weight` as an optional schedule key and adds explicit auxiliary
  start-step defaults so the shared schedule does not depend on missing-key
  behavior.
- `powerfoam_point_cloud.py` now owns PowerFoam point-cloud initialization:
  PLY/COLMAP point loading, color normalization, model-box fit/clamp,
  train-view visibility filtering, duplicate backfill, and the
  `PointCloudInitialization` payload. PowerFoam Metal imports/re-exports the
  public helpers for compatibility, while Dynamic Foam point-cloud diagnostics
  import them from the light module.
- `powerfoam_objectives.py` now owns trainer-independent PowerFoam objective
  helpers: Metal SSIM loss wrapping, Metal loss-weight scheduling, Direct
  PowerFoam loss assembly, contribution and normal-distance losses,
  depth-to-normal-map targets, normal-map loss, and alpha/background
  compositing. PowerFoam Metal imports/re-exports the helpers for
  compatibility, while Direct and diagnostics call the light objective module
  instead of embedding formula helpers in trainer files.
- `tests/test_powerfoam_direct.py` now imports pure PowerFoam helpers from
  their owning modules instead of using `train_powerfoam_metal.py` as a helper
  namespace. The raw Metal raster fixture check now imports
  `FoamRasterConfig` and
  `rasterize_power_foam_quaternion_height_sv_texel_surface` from
  `torch_powerfoam_metal` through an explicit third-party bootstrap helper.
  Remaining imports from the Metal trainer are structural
  `MetalPowerFoamVideo` checks, and the focused gate after this cleanup was
  `44 passed, 1 skipped`.
- `powerfoam_eval_color.py` now owns PowerFoam eval color-calibration helpers:
  channel-affine and RGB-matrix affine fit/apply, pixel flattening/bias-column
  utilities, train/heldout frame-index summaries, and calibration provenance
  serialization. PowerFoam Metal imports/re-exports the public fit/apply/serialize
  functions for compatibility, while the color-affine diagnostic reuses the same
  affine helpers instead of carrying a second implementation.
- `powerfoam_optim.py` now owns PowerFoam Metal LR scheduling helpers:
  cosine LR, group initial/final/warmup metadata, and optimizer param-group LR
  updates. Compatibility aliases remain on `train_powerfoam_metal.py`, but the
  behavior can be tested without importing the full Metal trainer.
- `powerfoam_resampling.py` now owns PowerFoam Metal resample scheduling:
  the resample-step predicate and geometric target-cell growth. Compatibility
  aliases remain on `train_powerfoam_metal.py`, while the train loop consumes
  the light helper module.
- `powerfoam_checkpoints.py` now owns PowerFoam checkpoint artifact helpers:
  best-metric selection, atomic checkpoint writes, and `best_metrics.json`
  updates. It supports both the Metal metric-rich checkpoint payload and the
  Direct minimal final checkpoint payload, so neither train loop embeds its
  own checkpoint schema.
- Dynamic PowerFoam Metal and Dynamic Gauge Foam final checkpoints now use
  `checkpoint_utils.atomic_torch_save(...)` as well. Their payload schemas stay
  local to the trainer family, but final `checkpoint_final.pt` persistence no
  longer has a separate direct `torch.save(...)` path.
- `checkpoint_utils.py` now owns shared checkpoint read-side primitives too:
  raw `torch.load(...)`, mapping-payload validation, and wrapped-or-raw model
  state-dict extraction. STAR UVT checkpoint loading, STAR UVT colorizer-init
  loading, relative-pose checkpoint resume, and
  `visualize_camera_scene_diagnostic.py` use those helpers, while checkpoint
  schemas and model-specific load semantics remain local.
- The same read helper now carries the explicit `weights_only=True` policy for
  frame-cache reads, V-JEPA feature-cache reads, and browser-bundle state-dict
  export. Cache and export schemas stay local, but direct `torch.load(...)`
  calls are gone from `src/train` outside `checkpoint_utils.py`.
- Gauge Fields material-surfel and free-dynamic 3DGS trainers now use
  `common.write_checkpoint(...)` for `checkpoint.pt`, backed by
  `checkpoint_utils.atomic_torch_save(...)`. Their checkpoint payload contents
  remain trainer-local, but persistence and path resolution share the Gauge
  common boundary instead of local `torch.save(...)` and duplicate repo-root
  resolution.
- Direct video-window frame caches and precomputed feature caches now use
  `checkpoint_utils.atomic_torch_save(...)` too. `sequence_data.py` and
  `video_feature_cache.py` still own their cache keys and payload schemas, but
  they no longer carry local `torch.save(tmp) -> replace(...)` implementations.
- `powerfoam_eval_render.py` now owns the no-grad PowerFoam batch sample
  renderer used by eval and diagnostics. `train_powerfoam_metal.py` keeps the
  old `render_samples` alias for backcompat, while Dynamic Foam diagnostics
  import owner/helper modules directly. The helper now accepts PowerFoam-style call outputs with
  extra tensors and keeps only the first `(rendered, alpha)` pair, so
  PowerFoam Direct calls it directly instead of carrying a second batch-render
  loop or local pass-through wrapper.
- `powerfoam_training_data.py` now owns the PowerFoam training-data dict
  contract for single-video and multicam validation inputs: targets, sample
  frame indices, optional sample rays, heldout tensors, view metadata, pose
  metadata, FPS, and point-cloud visibility metadata. The Metal trainer keeps a
  compatibility alias, while Dynamic Foam diagnostics and World Foam Gate 1
  feeder/reference scripts import the loader directly. PowerFoam Direct now
  wraps the same loader and prunes back to its historical key set, removing its
  duplicate data-loading body without changing its public result shape.
- `powerfoam_direct_config.py` now owns PowerFoam Direct defaults and
  `resolve_config(...)`. `powerfoam_direct_trainer.py` now owns the Direct
  trainer implementation and `run_training(...)`; the historical
  `train_powerfoam_direct.py` file is a thin CLI wrapper that imports only
  `run_training(...)`. The registry routes `powerfoam_direct` to the owner
  module, matching the Metal/Dynamic PowerFoam owner-module pattern. A
  one-step `src/train/train.py` smoke passed on CPU with 4 frames, 32 cells,
  64px render, `/tmp` output, disabled W&B, and final checkpoint write.
- `powerfoam_direct.py` now owns Direct PowerFoam render-option construction
  through `direct_powerfoam_render_options(...)`, colocated with
  `PowerFoamRenderOptions`. The Direct trainer no longer has a local
  config-to-render-options helper.
- Focused PowerFoam tests now import Direct defaults from
  `powerfoam_direct_config.py` and the shared schedule from
  `powerfoam_objectives.py` instead of using `train_powerfoam_direct.py` as a
  helper namespace. The Direct trainer imports only `resolve_config(...)` from
  its config module.
- `powerfoam_eval_artifacts.py` now owns PowerFoam eval artifact assembly:
  no-grad render calls, fixed-background compositing, optional eval color
  calibration, reconstruction metrics, aux/drift metrics, preview PNGs,
  optional MP4s, and W&B eval payloads. The Metal trainer keeps the old
  `log_artifacts` alias, but no longer embeds this media/metrics block.
- `train_cli.py` now owns the repeated one-config CLI boundary:
  `run_config_arg(...)` for `sys.argv[1]` script entrypoints and
  `run_config_or_path(...)` for public `main(config_or_path)` functions, plus
  `run_path_arg(...)` for path-dispatch entrypoints such as `src/train/train.py`.
  Token-GS, precomputed-feature, multicam, mixed same-heldout, PowerFoam, STAR
  UVT train/probe modules, and the registry CLI use it, so trainer files no
  longer import `sys`/`load_config_file` just for their local `main(...)`
  boilerplate.
- `train_cli.parse_csv_ints(...)` now owns train-probe comma-separated integer
  list parsing. The colorize init/matrix probes use it for `--seeds`, replacing
  their local split/int list comprehensions while keeping the same accepted CLI
  format.
- `train_logging.finish_wandb_run(...)` now owns the repeated W&B finish guard
  used by Token-GS, PowerFoam-family, and STAR UVT train/probe scripts. Those
  modules still own their experiment-specific train/probe loops, but no longer
  open-code local `wandb_run.finish()` or `wandb.finish()` branches.
- The STAR/dynamic alpha-background ablation now uses `finish_wandb_run(...)`
  too, removing its ablation-local `wandb.finish()` branch while preserving
  trainer lifecycle and stdout capture.
- `finish_wandb_run(...)` now also supports the global active `wandb.run` case,
  which lets benchmark/probe scripts share the same finish guard when they do
  not retain an explicit run object. General `src/benchmarks` CLIs and V-JEPA
  performance CLIs now use it instead of importing W&B only to finish a run.
- `wandb_run_lifecycle(...)` now wraps the shared init/finish pair for current
  PowerFoam/Gauge owners, so exceptions during the training body still trigger
  the same finish guard. This does not introduce a train-loop framework.
- PowerFoam/Gauge owner `__all__` surfaces are now narrow: Direct and Gauge
  trainer owners export only `run_training(...)`, and PowerFoam Metal exports
  only the structural model/run surface. Config defaults, raster builders,
  geometry, data loading, objectives, and artifact helpers stay public from
  their owning helper modules.
- `train_devices.py` now owns repeated `auto` device selection and device
  synchronization policy. PowerFoam-family trainers preserve MPS-then-CPU auto
  behavior with `auto_cuda=False`; Dynamic Gauge preserves its CUDA fallback
  with `auto_cuda=True`; STAR UVT preserves explicit requested-device
  validation through `validate_requested=True`; Token-GS profile timing now uses
  the shared synchronization helper.
- `train_devices.clear_torch_device_cache(...)` now owns the low-level Python
  GC plus MPS/CUDA cache-clear primitive. `video_feature_cache` calls it after
  feature baking, while benchmark memory sampling keeps its higher-level
  wrapper and sync policy local.
- Colorize probes, V-JEPA performance benchmarks, and the STAR alpha-background
  ablation orchestrator now use `train_devices` too. These are not trainer
  entrypoints, but they exercise the same training/profiling paths and should
  not carry a separate MPS/CUDA timing primitive.
- PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic Gauge
  Foam now call `resolve_torch_device(...)` directly instead of keeping local
  one-line `resolve_device(...)` wrappers. DUSt3R video export also uses the
  shared resolver while preserving its MPS-then-CUDA auto fallback.
- `external_paths.py` now owns the shared Dynaworld/third-party root and
  `sys.path` insertion primitives for train-local entrypoints. PowerFoam Metal,
  Dynamic PowerFoam Metal, DUSt3R video export, STAR UVT runtime, Taichi
  renderer bootstrap, and the two v12a objective helpers use it instead of
  rebuilding third-party roots locally. The v12a helpers also share the compiled
  bridge origin guard, preserving the old wrong-variant protection while
  removing the duplicated path/module check.
- The core Fast-Mac renderer wrapper now uses the same third-party path/origin
  helpers. `renderers.fast_mac` has one variants root plus a small
  `_fast_mac_variant_path(...)` helper, and `_ensure_variant_on_path(...)`
  delegates the compiled-bridge origin guard to `external_paths.ensure_module_path(...)`.
- `sync_torch_device(...)` now safely skips unavailable MPS and CUDA devices,
  and the depth-aware DOF demo plus fast-mac benchmark probes route their local
  timing fences through that shared helper instead of carrying device-specific
  synchronize guards at each call site.
- Dynamic Foam 4K trainability generation and real-view raytrace alpha
  diagnostics now use `sync_torch_device(...)` for their timing fences too.
  These stay diagnostic scripts, not trainer entrypoints; the shared boundary
  is only device synchronization.
- `resolve_torch_device(...)` now has an explicit `auto_prefer_cuda` flag for
  legacy renderer benchmark CLIs whose `auto` policy historically preferred CUDA
  before MPS. `splat_renderer_benchmark.py`, `splat_renderer_accuracy.py`, and
  `trainer_phase_benchmark.py` now use the shared device/sync helpers without
  changing that policy.
- `src/benchmarks/renderer_benchmark_cli.py` now owns shared renderer-benchmark
  CLI primitives: resolution parsing, comma-separated list parsing, torch dtype
  lookup, config deep-merge, project-relative output paths, safe filename
  parts, save-target selection, row-target matching, and CHW tensor
  preview-image writes. The splat renderer benchmark and accuracy CLIs use it
  instead of carrying parallel parser/path/image-selection/image-write blocks.
- The STAR UVT direct-feature, feature-autograd, sparse hidden sigmoid-MSE, and
  sparse hidden target-area kernel benchmarks now use shared device
  synchronization plus report JSON writes. Kernel-specific parity/timing logic
  stays local.
- STAR UVT report/prototype artifact writes have another shared-helper slice:
  the V-JEPA bridge audit, dense-alpha failure diagnostic, dense feature-tube
  prototype, V-JEPA versus Gaussian comparison, alpha-only visibility profile,
  visibility support bridge, visibility birth/split gate, feature1 whole-graph
  profile, and target-cache budget now use `report_artifacts` for root-relative
  JSON/text writes. The report helper also inserts the Dynaworld root on
  `sys.path` for direct `research_experiments.*` imports, and the dense
  prototype delegates device resolution/synchronization through
  `star_uvt_runtime`.
- The rest of the STAR UVT feature-tube report/profile matrix surfaces have
  been routed through shared artifact helpers as well: target-grid render-mode
  matrices, sparse-forward scale and repeat runners, sparse-forward profile,
  direct-feature mode matrix manifests, tile-slot budget summaries, batched
  sparse-forward target/step profiles, support birth/split sweeps, sparse
  visual loss VJP profiles, compact visual VJP comparisons, and logit/target
  grid VJP bridge reports. A targeted scan now finds no direct `.write_text(...)`
  calls under `research_experiments/star_uvt_feature_tubes`; explicit log and
  CSV file handles remain local by design.
- The top-level STAR UVT backward-kernel matrix script now reuses
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py` for
  dual-mode bootstrap, optional JSON loading, manifest/CSV/markdown writes, and
  logged subprocess execution. It still owns v0/PRT case construction and row
  schemas because those are benchmark-specific.
- The top-level renderer scaling report now reuses
  `research_experiments/report_artifacts.py` for Dynaworld-root resolution,
  `src/train` bootstrap, JSONL/CSV/JSON reads, and CSV/markdown writes. The
  shared helper delegates writes to `train_artifacts.py`; renderer-family row
  normalization and table structure stay local to the report. The
  multicam train2/holdout1 split smoke uses the same helper for root `chdir`
  and config path resolution, so the README smoke command no longer needs a
  manual `PYTHONPATH=src/train` prefix.
- `train_artifacts.py` now owns repeated trainer artifact I/O primitives:
  resolved-config JSON writing and serialized JSONL metric-history appends.
  PowerFoam-family trainers use the shared helper instead of open-coding
  `output_dir.mkdir(...)` plus `resolved_config.json`; the PowerFoam Metal
  trainers no longer carry local `append_jsonl(...)` helpers.
- `train_artifacts.py` now also owns generic benchmark/result artifact writes:
  `write_json(...)`, `write_jsonl(...)`, `write_csv(...)`, and
  `write_text(...)`. V-JEPA
  performance benchmarks and the STAR alpha-background ablation orchestrator use
  these helpers instead of local parent-mkdir plus sorted JSON/JSONL write
  loops. The fixed fast-mac variant matrix now uses the same helper while
  preserving row-at-a-time result recording through `append_jsonl(...)`. STAR
  UVT row-output helpers now use `write_json(...)` for file writes and keep only
  stdout formatting local. STAR UVT report/diagnostic markdown writers use
  `write_text(...)` for parent-safe text artifacts. The feature1 continuation
  report family now uses the same artifact helpers too.
- Gauge Fields JSON artifacts now keep their experiment-local
  `common.write_json(...)` name but delegate to `train_artifacts.write_json(...)`.
  Material-surfel training, splat-baseline training, cheat probes, run-matrix
  wall-clock reports, and summary JSON outputs share the parent-safe sorted
  JSON writer while keeping Gauge-specific schemas local.
- Gauge Fields run-matrix CLIs now share
  `common.parse_gauge_matrix_args(...)` and `common.run_gauge_matrix(...)`.
  The DeepView 3-cam holdout and incidence-matrix scripts keep only their
  matrix definitions plus descriptions/default output roots; shared CLI args,
  subprocess launch shape, wall-clock JSON, and nonzero-exit handling live in
  `common.py`.
- Gauge Fields single RGB MP4 writes now share `common.save_rgb_mp4(...)`.
  `smiley_smoke.py` and `cheat_probe_material_gauge.py` use the same OpenCV
  writer loop, matching the already-shared `save_side_by_side_mp4(...)`
  boundary for target-vs-render videos.
- Gauge Fields `*_columns.txt` sidecar writes now share
  `common.write_columns_legend(...)`. Gauge preview strips, smiley smoke media,
  and cheat-probe xmap/flow/probe strips keep their column names local while
  sharing the parent-safe sidecar writer.
- Gauge Fields script bootstrap now lives in `common.py` too. It owns
  experiment-dir plus `src/train` path ordering, and `cheat_probe_material_gauge.py`
  / `smiley_smoke.py` no longer copy the same preamble. The path helper
  deliberately keeps the Gauge experiment dir ahead of `src/train` so
  `from train` imports the Gauge-local trainer module, not `src/train/train.py`.
- Gauge Fields `common.resolve_device(...)` now delegates to
  `train_devices.resolve_torch_device(...)` with `auto_cuda=True` and
  `auto_prefer_cuda=True`, preserving Gauge's CUDA-first auto policy while
  removing the local CUDA/MPS/CPU branch copy.
- `fast_attn.pick_device()` now delegates to
  `train_devices.resolve_torch_device("auto", auto_cuda=True,
  auto_prefer_cuda=True)`, preserving the base Token-GS/browser-export
  CUDA-first auto policy without keeping a separate device branch.
- General timing/parity scripts under `src/benchmarks` now write optional JSON
  outputs through `train_artifacts.write_json(...)` as well: trainer-phase
  timing, fixed-render variant parity, camera-swap variant parity,
  fixed-render backward-mode parity, and train-step memory reports.
- `src/benchmarks/benchmark_bootstrap.py` now owns the shared Dynaworld/train
  path bootstrap for reusable trainer/variant benchmark CLIs. Trainer-phase,
  train-step memory, fixed-render variant parity, fixed-render backward-mode
  parity, and camera-swap parity scripts no longer repeat `Path(__file__)`
  root discovery, `sys.path` mutation, or import-order `# noqa` comments.
  The import smoke also fixed the stale train-step memory import of
  `sync_device` by routing it to `train_devices.sync_torch_device(...)`.
- The same benchmark bootstrap now exposes shared project/benchmark roots and a
  small `ensure_sys_path(...)` helper for benchmark-specific vendored paths.
  Depth-aware DOF, splat renderer benchmark/accuracy, and Mac renderer stack
  comparison now use it for project/train/benchmark path setup while keeping
  their renderer-specific third-party path lists explicit.
- The Fast-Mac project3d benchmark and v13 iteration matrix now use the same
  benchmark bootstrap too. `benchmark_bootstrap.py` exposes the repo-local
  `.venv` Python path for subprocess benchmark matrices, while project3d keeps
  only its variant list/build policy local.
- The raw-Metal MLX bridge used by renderer benchmarks now uses
  `benchmark_bootstrap.PROJECT_ROOT` and `ensure_sys_path(...)` instead of
  rebuilding benchmark/project roots locally. It still owns MLX import errors
  and raw-Metal settings because those are backend-specific.
- The WorldFoam Gate0 paired benchmark now uses the benchmark bootstrap for
  `research_experiments/world_foam_lane2` path setup and `train_artifacts.write_json(...)`
  for optional JSON output. It still owns its comparison/report math locally.
- Complete-table benchmark CSV/JSONL outputs now use the shared artifact
  helpers too: `splat_renderer_benchmark.py` routes optional JSONL/CSV results
  through `write_jsonl(...)` / `write_csv(...)`,
  `mac_renderer_stack_compare.py` routes CSV through `write_csv(...)`, and
  `depth_aware_dof_demo.py` routes its JSON summary through `write_json(...)`.
- The renderer benchmark pair also shares save-image CLI override application
  and comma-separated string parsing through `renderer_benchmark_cli.py`, so
  `--save-images`, `--no-save-images`, renderer lists, and overlap-variant
  lists no longer carry parallel mutation/parsing blocks. The same string-list
  parser now covers `mac_renderer_stack_compare.py --renderers` and
  `fast_mac_v13_iteration_matrix.py --versions`, plus project3d benchmark
  `--cases` comma tokenization while leaving the case record schema local.
- `profile_fast_mac_render_phases.py` now calls the shared fast-mac projected
  RGB/feature rasterizer helpers instead of importing a stale v5-only feature
  config helper. The profiler now follows active `rgb_variant` and
  `feature_variant` dispatch for projected-raster probes.
- `build_clip_dataset.py` now uses `train_artifacts` for per-clip summary JSON,
  full/split manifest JSONL files, and `dataset.json`, so reusable dataset
  construction no longer carries a local JSONL writer or direct JSON text writes.
- `train_artifacts.write_jsonl(..., compact=True)` now covers compact manifest
  rows too. `build_single_video_pretrain_manifest.py` uses that shared mode,
  preserving its compact JSONL format while deleting the local writer.
- Browser-bundle export, DUSt3R video export, Dynamic PowerFoam Metal metric
  JSONs, and PowerFoam Metal eval/best-metric JSONs now use `write_json(...)`.
  Binary tensor exports, NumPy arrays, checkpoints, images, and videos remain
  local to their owning modules.
- `research_experiments/dynamic_foam/report_artifacts.py` now owns simple
  Dynamic Foam report JSON reads/writes. Heldout-error, topology-edge,
  support-gap, camera-perturbation, color-affine, CUDA-vs-Metal,
  motion-vs-repaint, video-motion ranking, raytrace-support, and
  PowerFoam-vs-splats report scripts share sorted parent-safe JSON output while
  PLY writers, Modal staging files, config dumps, and row-at-a-time logs stay
  local. Its public `write_report_json(...)` wrapper now delegates to
  `train_artifacts.write_json(...)`, so Dynamic Foam keeps its report API while
  using the project-wide artifact writer.
- The same Dynamic Foam report helper now owns `PROJECT_ROOT` and
  `relative_to_project(...)` for report path display. Routed report scripts no
  longer carry local `ROOT`/`rel(...)` helpers just to serialize config,
  checkpoint, output, or panel paths. The ALIKED Modal geometry orchestrator
  also uses this display helper, while its local/remote repo-root fallback and
  Modal staging remain local.
- The Dynamic Foam report helper now owns frame-index list parsing and optional
  range validation too. Feature-triangulation, known-pose pycolmap, ALIKED
  geometry orchestration, and section diagnostics use
  `parse_frame_indices(...)` / `validate_frame_indices(...)`; each script keeps
  its own CLI defaults, `all` support, Modal staging, and command semantics.
- `research_experiments/dynamic_foam/experiment_paths.py` now owns Dynamic Foam
  repo-root and `src/train` bootstrap. Smoke dataset export plus the multiview
  plane-sweep, feature-triangulation, EX4DGS anchor, and known-pose pycolmap
  point-cloud builders import that path boundary through `report_artifacts`
  instead of copying `ROOT`/`TRAIN_SRC`/`sys.path.insert(...)` preambles. Data
  and PLY writers remain local because they are not the shared report contract.
- Dynamic Foam verifier/runner scripts use that same path boundary now: 4K
  benchmark/trainability verifiers, clean-init coverage, section diagnostics,
  completion audit, paper acceptance, CUDA smoke result verification,
  external-blocker orchestration, and the CUDA smoke runner. The Modal ALIKED
  geometry launcher keeps its custom root detector because it must handle both
  the local repo and `/root/dynaworld` staging inside Modal.
- Strict Dynamic Foam report-object readers now share
  `report_artifacts.load_report_json(...)` across motion-vs-repaint,
  raytrace support-gap, dynamic-geometry verification, CUDA-smoke verification,
  4K trainability, paper acceptance, clean-init coverage, and the completion
  audit. The import path is dual-mode so direct CLIs and package-imported tests
  both work. JSONL, JSON-list artifacts, copied remote artifacts, and embedded
  upstream runner internals remain local by design.
- PowerFoam-vs-splats comparison, raytrace support-gap diagnosis, and
  external-blocker orchestration no longer keep one-line JSON object reader
  wrappers; they call `report_artifacts.load_report_json(...)` directly. The
  support-gap diagnostic also uses the shared Dynamic Foam path bootstrap before
  importing train-local modules, fixing direct `--help` execution.
- Strict Dynamic Foam JSONL object histories now share
  `report_artifacts.load_report_jsonl(...)` with line-number object validation
  and explicit `missing_ok` handling. CUDA-vs-Metal smoke comparison and the
  PowerFoam paper-acceptance verifier use it for train/eval metric histories;
  the verifier still owns metrics-schema interpretation.
- Dynamic Foam point-cloud builders and 4K trainability generation now share
  `report_artifacts.write_report_json(...)` for their adjacent JSON summaries:
  multiview plane sweep, multiview feature triangulation, feature-triangulation
  failure diagnostics, known-pose pycolmap, merged ASCII PLY, EX4DGS anchor
  prep, and generated 4K trainability artifacts. The actual PLY/data generation
  paths stay local. The known-pose pycolmap builder also defers its optional
  `pycolmap` import until execution, so CLI help no longer requires the
  dependency.
- PowerFoam CUDA smoke `summary.json`, lane settings JSON, and Modal
  `modal_return.json` writes now share `report_artifacts.write_report_json(...)`;
  lane metrics reads now use `load_report_json(...)`. Copied remote JSON files
  and the embedded upstream smoke entry stay local because they are
  inputs/outputs for the cloned upstream PowerFoam checkout, not reusable
  Dynaworld report artifacts.
- ALIKED/Colmap Modal geometry report files now share
  `report_artifacts.write_report_json(...)`: `plan.json`, local probe/full
  result JSONs, `onnx_check.json`, `colmap_cli_onnx_check.json`, and generated
  remote config JSONC. Remote JSONL manifests and copied returned artifacts
  remain local or byte-preserving by design.
- PowerFoam parity fixture builders now use `write_report_json(...)` for their
  generated local/official fixture JSON files. Tensor payload construction and
  local/CUDA render math stay in the fixture scripts; only the parent-safe,
  sorted-newline JSON object write is shared.
- External-blocker generated training configs now also use
  `report_artifacts.write_report_json(..., sort_keys=False)` for the complete
  patched JSON config artifact. Dry-run stdout and Modal/remote input files
  stay local because they are not the same artifact contract.
- Dynamic Foam checkpoint/report diagnostics now share the train checkpoint
  loader boundary as well. Heldout-error, color-affine, raytrace-support,
  section/topology, camera-perturbation, real-view-alpha, start-support,
  official-parity, CUDA-smoke, runner, and external-blocker scripts use
  `checkpoint_utils.load_checkpoint_mapping(...)`,
  `model_state_dict_from_checkpoint(...)`, and report-artifact JSON helpers for
  report-shaped objects while leaving upstream settings, copied remote files,
  row-list artifacts, PLY metadata, and ffprobe output local.
- `research_experiments/star_uvt_feature_tubes/report_artifacts.py` now owns the
  local STAR report `src/train` bootstrap and root-relative artifact path
  resolution for those report scripts. This keeps one-off reports runnable
  directly while avoiding a copied `sys.path` preamble in every file. It also
  owns shared report JSON loading and table-cell formatting for the feature1
  report family.
- The STAR report helper now bootstraps the STAR UVT variant root too. The
  first-class scale report no longer rebuilds Dynaworld/train/variant roots or
  imports `train_artifacts` directly; it uses the report helper for JSON object
  loads and root-relative JSON/markdown writes.
- Sparse-forward and target-grid VJP profile scripts now use the same
  `report_artifacts` bootstrap plus `summary_stats(...)` timing-summary helper
  instead of reaching into `star_uvt_feature1_wholegraph_profile._stats`.
  This also removes a stale private-helper import after the stats helper moved.
- Batched sparse-forward target-VJP and step-benchmark reports now use the same
  `report_artifacts` bootstrap plus `distribution_stats(...)`, instead of
  rebuilding Dynaworld/train/STAR-UVT roots and mutating `sys.path` locally.
- Sparse hidden sigmoid-MSE and target-area native kernel benchmarks now use the
  same STAR report bootstrap for Dynaworld/train/STAR-UVT path setup. They keep
  their kernel parity/timing logic local, but no longer carry local
  `Path(__file__)` root discovery, `sys.path` mutation, or import-order
  `# noqa` comments.
- Alpha-only visibility, dense-alpha failure, sparse visual VJP, and logit
  handoff RGB VJP profiles now use the same STAR report bootstrap too. The two
  profiles with checked-in default config/output paths import `ROOT` from
  `report_artifacts` instead of rebuilding their own Dynaworld root.
- Existing STAR matrix/sweep scripts that already used the shared bootstrap now
  dropped stale import-order `# noqa: E402` comments. `report_artifacts` remains
  the explicit first local import because it establishes the train and STAR UVT
  variant paths before downstream local imports.
- First-class backward breakdown, feature1 whole-graph profile, V-JEPA bridge
  audit, and alpha-background ablation orchestration now rely on the same report
  bootstrap too. Their report JSON/text writes use the report helper wrapper
  where the scripts produce STAR report artifacts, while their benchmark math,
  config audit logic, and trainer orchestration stay local.
- The same STAR report helper now owns logged subprocess execution and the
  standard STAR UVT feature-overfit trainer subprocess wrapper. Sparse-forward
  timing repeat, sparse-forward scale matrix, and target-grid render-mode matrix
  reports use it for the common `PYTHONPATH`/`TMPDIR`/timeout/log/status/elapsed
  contract.
- Support birth/split sweeps now use the same subprocess helper as well. The
  trainer case path keeps its `WANDB_MODE=offline` default and
  `STAR_UVT_TILE_CAPACITY` override through helper env/default parameters, and
  the dense-support diagnostic uses the generic logged subprocess wrapper with a
  custom command.
- The target-grid analytic VJP trainer report, Gate 4 quality bracket report,
  and logit-handoff reducer report now call the same STAR report helper for JSON
  loading plus JSON/markdown artifact writes. Their report-specific CSV parsing,
  comparisons, and table formatting remain local.
- STAR result-summary reports now route report-shaped JSON object loads through
  `report_artifacts.load_report_json(...)` as well: target-cache budget,
  first-class scale summary, V-JEPA versus Gaussian comparison, and feature1
  whole-graph reference timing. Tolerant audit loaders that intentionally return
  `_load_error` payloads remain local.
- `report_artifacts.load_optional_report_json_or_error(...)` now owns the
  tolerant STAR report-object read for loaders that preserve
  `{"_load_error": ...}` instead of failing. The V-JEPA bridge audit uses that
  helper, and dense-alpha failure diagnostics use the shared checkpoint mapping
  loader instead of direct `torch.load(...)`.
- The STAR report helper now also owns optional report JSON loading plus basic
  comma-separated string/int/float parsing. Direct-feature mode matrices,
  target-grid render-mode matrices, sparse-forward scale/repeat reports, and
  support birth/split sweeps use those primitives instead of local `_read_json`
  or CSV split helpers. The tile-slot accumulator budget now uses the same int
  CSV parser. The direct-feature mode matrix also uses the shared logged
  subprocess wrapper for its benchmark launches; its summary CSV writer remains
  local because it streams one table with script-specific columns.
- The same STAR CSV helper boundary now covers direct feature-kernel,
  sparse-hidden sigmoid-MSE, sparse-hidden target-area, dense-alpha failure,
  background-cheat, and first-class backward-breakdown scripts. Feature-dim
  lists, raw-opacity bias lists, alpha sweeps, and backward mode lists share
  the typed report parsers; script-specific alpha validation, benchmark math,
  diagnostic rows, and report schemas stay local.
- The same report-matrix cluster now imports `ROOT`, `TRAIN_ROOT`, and
  `STAR_UVT_ROOT` from `report_artifacts` where needed instead of rebuilding
  local root constants and mutating `sys.path` before importing
  `config_utils`. This keeps direct script execution working while centralizing
  the path bootstrap.
- STAR UVT report/prototype modules now follow the Dynamic Foam import pattern:
  package imports use `.report_artifacts`, and direct script execution falls
  back to `report_artifacts`. The package-global alias version was rejected
  after focused pytest showed cross-package pollution: Dynamic Foam could occupy
  the top-level `report_artifacts` module before STAR imports needed
  `write_report_text`.
- STAR UVT profiling scripts now follow the same optional-path helper boundary:
  `star_uvt_logit_handoff_rgb_vjp_profile.py` and
  `star_uvt_feature1_wholegraph_profile.py` import
  `config_utils.path_or_none(...)` instead of carrying local `_path_or_none`
  helpers.
- Background-cheat diagnostics, compact visual VJP comparison, and the STAR
  V-JEPA-vs-Gaussian comparison now use the same STAR report artifact boundary
  for report-shaped JSON/text/root handling. The background diagnostic imports
  `report_artifacts` before train-local objective modules so direct script
  imports do not need an external `PYTHONPATH` bootstrap.
- `report_artifacts.write_report_csv(...)` now wraps the shared
  `train_artifacts.write_csv(...)` for root-relative STAR report CSVs while
  preserving first-seen column order. `direct_feature_mode_matrix.py` uses it
  for `summary.csv` instead of a local `csv.DictWriter` helper.
- `report_artifacts.read_report_csv(...)` now owns root-relative STAR report
  CSV reads. `logit_handoff_reduce_report.py` and
  `gate4_quality_bracket_report.py` use it instead of local `csv.DictReader`
  blocks while keeping row filtering and report-specific type conversion local.
- `report_artifacts.mean_timing_without_first(...)` now owns the repeated
  `step_timings_ms[1:]` mean helper used by target-grid render-mode matrices,
  sparse-forward repeat/scale reports, support birth/split sweeps, and the
  feature1 LR reset/schedule reports. Reports still own which timing keys become
  columns.
- `report_artifacts.summary_stats(...)` and
  `report_artifacts.distribution_stats(...)` now own the two repeated STAR
  profile timing-stat contracts: zero-empty `{samples, mean, min, max}` summaries
  and count/stdev distributions with `None` empty values. Logit-handoff,
  feature1 whole-graph, sparse-visual VJP, sparse-forward repeat, and batched
  sparse-forward profile/step reports use those shared helpers instead of local
  `_stats(...)` copies.
- `star_uvt_feature_tube_model.py` now owns the reusable STAR UVT feature-tube
  model/config/dense-render CPU contract. Train modules import
  `FeatureTubeRenderConfig` and `FeatureScreenTimeTubeModel` from `src/train`
  instead of reaching into `research_experiments/.../dense_feature_tube_prototype.py`.
  The dense prototype remains as a compatibility re-export plus gate runner for
  older benchmark scripts.
- `sequence_data.ManifestSequenceSampler` now owns same-view manifest entry
  loading, eager/lazy sampling, cycle/random order, and optional one-worker
  prefetch. The base token-GS trainer and the mixed same-heldout trainer use it,
  so lazy manifest mechanics no longer live in trainer-specific cursors.
- `mixed_data_scheduler.py` now owns the first typed mixed-scheduler boundary:
  `SameViewBatch`, `NovelViewBatch`, `MixedStepBatch`,
  `sample_mixed_step_batch(...)`, `scheduled_loss_kinds(...)`, and shared
  multicam `sample_view_indices(...)`. It keeps `same_view_recon` and
  `heldout_view_recon` as separate names and supports `both` and `alternate`
  scheduling. The active multicam trainer uses the shared view sampler, and the
  mixed optimizer-step bridge now consumes this scheduler directly.
- The mixed optimizer-step bridge no longer reimplements the schedule branch in
  the trainer. `sample_mixed_step_batch(...)` accepts a lazy same-view sequence
  provider and only calls it on scheduled same-view steps, preserving lazy
  manifest behavior while keeping the schedule decision in one module.
- `mixed_same_heldout_trainer.py` now owns the first mixed optimizer-step
  bridge. It subclasses the existing multicam precomputed-feature
  trainer, samples same-view and heldout-view batches through the scheduler,
  logs separate aux keys, and is dispatched through `src/train/train.py` with
  `arch=mixed_same_heldout_precomputed_feature_implicit_camera`. The historical
  `train_mixed_same_heldout_implicit_dynamic.py` file is now a thin CLI wrapper
  that imports only `run_training(...)`. The checked-in smoke config
  `src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`
  passed offline on MPS; it is a trainer plumbing/convergence trace, while a
  real W&B-enabled benchmark contract is still pending.
- 2026-05-21: the checked-in mixed smoke now sets
  `logging.always_log_last_step=true`, so the cheap 10-step run logs final-step
  preview/media. The current rerun passed at
  `wandb/offline-run-20260521_222750-9yvznqiq`, with `Loss/same_view_recon` and
  `Loss/heldout_view_recon` present in the W&B record plus TrainView0,
  TrainView1, and Heldout0 rendered/GT videos. This is stronger interface
  evidence than the scalar-only smoke, but still not a baseline or quality
  claim.
- 2026-05-22: reran the checked-in mixed smoke after the report/artifact helper
  cleanup at `wandb/offline-run-20260522_004727-ka4lm8g5`. The run exercised
  the current `train.py -> trainer_registry -> MixedSameHeldoutPrecomputedFeatureTrainer`
  path, W&B offline logging, cached RGB-pyramid features, alternate same-view
  and heldout-view steps, and final validation media. The W&B record contains
  `Loss/same_view_recon`, `Loss/heldout_view_recon`, train-view/heldout PSNR
  metrics, final `Render_GT_vs_Pred`, and TrainView0/TrainView1/Heldout0
  rendered+GT videos. This is the current runtime smoke for the shared
  scheduler/objective/logging/media stack; still not a benchmark or quality
  claim.
- 2026-05-22: split the mixed optimizer-step bridge into
  `mixed_same_heldout_trainer.py`, leaving
  `train_mixed_same_heldout_implicit_dynamic.py` as a CLI wrapper that imports
  only `run_training(...)`. The registry now routes the mixed arch to the owner
  module, direct tests import
  the owner module, and a 1-step `src/train/train.py` smoke passed at
  `wandb/offline-run-20260522_130229-5p9y6mrq`.
- `MixedBackwardResult` and `MixedStepAccumulator` now keep mixed-step
  aggregation typed in the mixed trainer. They preserve separate
  `same_view_recon` and `heldout_view_recon` aux keys while centralizing
  weighted recon accumulation, bank-rate term merge, camera-loss accumulation,
  preview selection, and final `StepResult` payload assembly.
- `MulticamPrecomputedFeatureImplicitTrainer._recon_loss_for_views(...)` now
  removes the duplicated train-view versus heldout-view reconstruction loops.
  The two public loss methods still preserve distinct names and target banks,
  but they share background sampling, alpha/background checks, preview capture,
  render profiling, and reconstruction-loss accumulation.
- `MulticamPrecomputedFeatureImplicitTrainer._rendered_view_recon_loss(...)`
  now shares the rendered-view loss/preview mechanics across multicam
  train-view, heldout-view, and camera-swap paths. Camera-swap still owns
  source grouping, relpose residuals, cycle loss, and bank-rate aggregation.
- `MulticamPrecomputedFeatureImplicitTrainer._step_result(...)` now shares
  StepResult assembly across multicam initial eval, normal train, and
  camera-swap train/eval branches. This keeps detach policy and zero camera
  regularizer payloads in one place while preserving branch-specific losses.
- `MulticamPrecomputedFeatureImplicitTrainer.multicam_validation_payload_from_renders(...)`
  now shares the validation payload assembly used by base multicam and
  relative-pose trainers. Each branch still owns its render-generation path;
  target resizing, W&B media payload construction, camera-rig metrics, fps,
  `gt_video_logged`, and best-heldout bookkeeping are shared.
- `pipeline.diagnostics.decoded_temporal_payload_from_sequence(...)` now shares
  the full-sequence decoded Gaussian temporal metric assembly used by base
  multicam external-view, base multicam oracle-relative, and full relative-pose
  validation render paths. Those branches still own render generation, but no
  longer rebuild the same decoded field buffer dict inline.
- `Trainer.temporary_render_size(...)` now shares the render-size context used
  by relative-pose multires training and validation logging. Relative-pose still
  owns token-detail-aware `_activate_render_size(...)`, while grid-cache reuse
  and size restore mechanics live in the base trainer.
- `runtime_types.build_step_result(...)` now shares result-payload assembly
  across base token-GS, known-camera, multicam, and mixed same-view/heldout
  trainer paths. This is intentionally payload-only: trainer-local math,
  backward strategy, optimizer stepping, and media choices remain local.
- `pipeline.validation_media.training_preview_payload(...)` now shares
  per-step preview image and optional feature-PCA image payload assembly across
  base and relative-pose `val_log` paths. `Trainer.log_gate_flags(...)` now
  shares the scalar/image/video cadence decision too. Trainer branches still
  own scalar payloads, validation videos, and render-size context.
- Base Token-GS `should_log_scalars/images/videos(...)` now delegate to
  `train_logging.should_log_scalar/image/video(...)`, so the only remaining
  trainer-local cadence policy is the explicit `log_initial_media` decision for
  image/video step zero.
- Base Token-GS and relative-pose `val_log(...)` submit through
  `train_logging.log_wandb_run_payload(...)` with the stored `wandb_run` handle.
  The relative-pose override now has the same disabled-W&B early return as the
  base trainer, so a disabled run cannot accidentally call global `wandb.log`.
- Base `Trainer.scalar_payload(...)` now uses `result.render_size` when a
  branch attaches one. Relative-pose multires logging keeps its branch-specific
  scalar fields, but no longer duplicates the generic `RenderSize` and
  `Render/BaseSize` payload writes.
- `pipeline.diagnostics.camera_state_summary_metrics(...)` and
  `camera_state_payload(...)` now share the camera fov/radius/rotation/
  translation scalar math used by train-step scalar payloads, progress strings,
  and full-sequence eval payloads. Callers keep their existing key shapes by
  choosing the payload key prefix.
- PowerFoam implicit-camera compact metrics now reuse
  `camera_state_summary_metrics(...)` for the shared fov/radius/rotation/
  translation math, while retaining PowerFoam-only origin/forward/global/
  active-frame scalars under the existing `state_camera_*` metric names.
  Token-GS progress strings call the shared helper directly instead of routing
  through a one-line trainer wrapper.
- `powerfoam_diagnostics.powerfoam_parameter_delta_metrics(...)` now shares the
  common PowerFoam parameter-drift scalar payload: center, xy/z, radius,
  density, feature, normal, and texel-site deltas. PowerFoam Metal and both
  Dynamic PowerFoam Metal model variants keep their temporal/camera/token and
  surface-specific extras local.
- `wandb_media.build_rgb_alpha_validation_video_payload(...)` and
  `build_rgb_alpha_eval_media_payload(...)` now share the repeated
  preview/render/side-by-side/GT/alpha W&B media payload for RGB+alpha eval
  paths. Direct PowerFoam, shared PowerFoam eval artifacts, Dynamic PowerFoam
  Metal, and Dynamic Gauge Foam use it; branch-specific scalars and Gauge's
  depth video stay local.
- `video_io.rgb_alpha_preview(...)`, `save_rgb_alpha_preview(...)`,
  `save_render_side_by_side_videos(...)`, and
  `save_rgb_alpha_eval_media(...)` now share the matching file artifact pattern
  for RGB+alpha eval paths. Direct PowerFoam, shared PowerFoam eval artifacts,
  Dynamic PowerFoam Metal, and Dynamic Gauge Foam keep the same
  preview/render/side-by-side filenames while dropping local triptych and
  side-by-side assembly blocks.
- `train_logging.mapped_metric_payload(...)` now shares the required/optional
  metric-to-W&B-key copy loop. Direct PowerFoam, shared PowerFoam eval
  artifacts, Dynamic PowerFoam Metal, and Dynamic Gauge Foam keep their
  branch-specific metric names as explicit local tables, but no longer
  duplicate the assignment loop or optional metric guards.
- PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic
  Gauge train-loop scalar W&B payloads now use the same key-map helper and
  null-safe explicit-run submit boundary. Metric names stay local to each
  trainer, but the disabled-run branch and hand-built copy dictionaries are no
  longer repeated in those scalar loops.
- `video_io.video_fps_from_config(...)` and
  `wandb_media.make_step_preview_image(...)` now share the small media defaults
  around those same eval paths: `video_fps` falls back to `4.0` in one place,
  and the W&B preview caption is no longer repeated at each trainer call site.
- `powerfoam_eval_render.powerfoam_eval_batch_size(...)` now shares the
  `train.frames_per_step` to eval-render batch-size policy for Direct
  PowerFoam, shared PowerFoam eval artifacts, and Dynamic PowerFoam Metal. The
  helper preserves the existing `max(1, ...)` clamp; train-time random frame
  sampling still reads `frames_per_step` locally.
- `powerfoam_training.powerfoam_train_batch_indices(...)` now shares the
  train-loop random index draw for Direct PowerFoam, PowerFoam Metal, Dynamic
  PowerFoam Metal, and Dynamic Gauge Foam. Each trainer still owns how sampled
  indices map to frame ids, targets, rays, staged camera counts, or colorizer
  inputs.
- `Trainer.run_training_loop(...)` and `print_training_header(...)` now share
  the base/known-camera train loop. `KnownCameraTrainer` keeps only the
  branch-specific banner/camera/completion/export hooks; the step loop, step-0
  diagnostic, profile-print hook, sequence-prefetch cleanup, and W&B finish are
  common.
- `Trainer.training_preamble_messages(...)` and
  `after_training_complete(...)` now handle the remaining lifecycle-only
  wrappers. `PrecomputedFeatureImplicitTrainer` uses the preamble hook for
  feature-cache metadata, and `MulticamRelativePoseImplicitTrainer` uses the
  post-success hook for optional checkpoint saves, so both inherit the shared
  `run(...)` method.
- `Trainer.model_eval_mode(...)` now shares eval/train restoration around
  initial diagnostics for base token-GS, known-camera, and multicam trainers.
  The helper keeps model-mode state handling out of each branch while leaving
  branch-specific clip selection, decode, and loss math local.
- `star_uvt_checkpoints.py` now shares STAR UVT training checkpoint save/load,
  target-grid RGB probe checkpoint save/load, optimizer LR helpers, and
  rendered-feature probe model-only resume metadata plus checkpoint saves. This
  keeps feature overfit, target-grid RGB probe, and rendered-feature RGB probe
  on one checkpoint state contract without introducing a broader checkpoint
  framework. The feature-overfit RGB-probe checkpoint loader wrapper also lives
  there now, so profiling scripts no longer import it from the trainer.
- STAR UVT checkpoint save paths now go through
  `checkpoint_utils.atomic_torch_save(...)`: training checkpoints, feature RGB
  probe checkpoints, and rendered-feature RGB probe checkpoints share the same
  temporary-file replace behavior instead of local `torch.save(...)` or
  `atomic_torch_save(...)` blocks.
- STAR UVT checkpoint load paths now use
  `checkpoint_utils.load_checkpoint_mapping(...)` for mapping-payload
  validation. STAR-specific required-key checks remain in
  `star_uvt_checkpoints.py` and `star_uvt_common.py`, so the helper boundary
  stays narrow.
- `star_uvt_colorizers.py` now shares STAR UVT `FeatureToColor` construction
  across the feature overfit trainer, RGB probes, checkpoint loaders, and
  feature-overfit diagnostic/profiling scripts. Config-based diagnostics no
  longer hand-copy `hidden_dim`/activation/pre-norm/init kwargs.
- `star_uvt_render_configs.py` now shares STAR UVT feature-tube render config
  construction. The feature overfit trainer and feature-overfit diagnostics use
  the same helper for `FeatureTubeRenderConfig` plus `UVTRenderConfig`, so the
  `data`/`feature_uvt` field mapping no longer lives in every script.
- `star_uvt_models.py` now shares STAR UVT feature-tube model construction.
  Feature overfit, rendered-feature RGB probe, and config-based
  diagnostics/profilers build the prototype model through the same helper,
  including `seed_section="probe"` for probe-local initialization.
- `star_uvt_outputs.py` now shares STAR UVT prediction-media and row-output
  mechanics. RGB overfit, feature overfit, target-grid RGB probe, and
  rendered-feature RGB probe keep their row schemas and W&B output keys local,
  but contact-sheet writing, side-by-side FPS fallback, output path handling,
  and sorted JSON result persistence no longer repeat in each script.
- `star_uvt_outputs.log_star_uvt_row_outputs(...)` now also shares the STAR W&B
  row-output wrapper: the standard contact-sheet/side-by-side media keys are
  defaults, and feature overfit passes only its extra RGB-probe media key
  override. The STAR trainer/probe files no longer carry local
  `_log_wandb_outputs(...)` wrappers.
- `star_uvt_timing.py` now shares STAR UVT timing summaries. Feature overfit,
  target-grid RGB probe, and rendered-feature RGB probe all compute mean timing
  tables through one helper, and feature overfit uses the same module for
  first/last/min/max timing trace summaries.
- `star_uvt_config_keys.py` now shares common STAR UVT config-section key
  contracts. RGB STAR overfit, feature overfit config normalization,
  target-grid RGB probe, and rendered-feature RGB probe use the same helpers
  for repeated `data`, `colorize`, `output`, and `logging` validation while
  keeping branch-specific train/probe/UVT keys local.
- `star_uvt_sparse_visual_sampling.py` now shares sparse-visual pixel-source
  enums, VJP-mode enums, stratified/patch pixel-id selection, patch phase
  cycling, local-frame selection, and loss sample-count math. Feature overfit,
  sparse-visual VJP profiling, and tests now import the sampling contract
  directly rather than through the overfit trainer.
- `star_uvt_sparse_visual_losses.py` now shares sparse-visual RGB composition,
  autograd/manual colorizer VJP helpers, target-area loss helpers,
  alpha/black-hole losses, and native target-area VJP mode mapping. The feature
  overfit trainer and sparse-visual VJP profiler now depend on that explicit
  loss contract instead of exporting loss internals from the trainer module.
- `star_uvt_rendered_feature_probe_objective.py` now shares the rendered-feature
  RGB probe sparse objective boundary. The rendered-feature probe trainer calls
  that module for target-grid pixel ids, stratified-grid pixel ids, target RGB
  gathers, sparse RGB composition, and local feature/alpha VJPs; the helper
  delegates to `star_uvt_sparse_grid.py`, `star_uvt_sparse_visual_sampling.py`,
  and `star_uvt_sparse_visual_losses.py` instead of keeping a parallel
  stratified sampling and sparse colorizer-loss implementation in the trainer.
- `star_uvt_feature_targets.py` now also shares the public grid-RGB probe
  adapter/loss boundary: `FEATURE_TARGET_GRID_ADAPTERS`,
  `adapt_rgb_to_grid(...)`, `upsample_grid_rgb(...)`, and
  `mean_rgb_grid_loss(...)`. The target-grid RGB probe consumes these names
  directly instead of exporting adapter aliases and a local `_mean_loss(...)`
  from the trainer, and the rendered-feature probe objective uses the same
  adapter set for target-grid sparse-pixel validation.
- `star_uvt_feature_rgb_probe_config.py` now shares target-grid RGB probe config
  validation. `star_uvt_feature_rgb_probe_trainer.py` imports
  `resolve_config(...)` from that module instead of owning required
  sections/keys, adapter validation, positive step/LR checks, and target-grid
  materialization checks inline. The historical
  `train_star_uvt_feature_rgb_probe.py` file is now a thin CLI wrapper for
  `run_probe(...)`, and the registry routes `star_uvt_feature_rgb_probe` to the owner
  module. A tiny 4-frame, 64px, 1-step `src/train/train.py` smoke passed with
  disabled W&B and `/tmp` outputs.
- `star_uvt_rendered_feature_probe_config.py` now shares rendered-feature RGB
  probe config validation. The rendered probe trainer imports
  `resolve_config(...)` from that module instead of owning STAR section/key
  checks, sparse pixel-source/grid-adapter validation, trainable-scope defaults,
  resume requirements, sample-grid bounds, frame-chunk checks, and feature-dim
  checks inline. The historical
  `train_star_uvt_rendered_feature_rgb_probe.py` file is now a thin
  CLI wrapper for `run_probe(...)`, and the registry routes
  `star_uvt_rendered_feature_rgb_probe` to
  `star_uvt_rendered_feature_rgb_probe_trainer.run_probe`. A tiny 4-frame,
  64px, 1-step `src/train/train.py` smoke passed through the registry on MPS
  with the 1500-step STAR checkpoint, disabled W&B, and `/tmp` row output.
- `star_uvt_video_overfit_config.py` now shares RGB STAR video overfit config
  validation. `star_uvt_video_trainer.py` imports `resolve_config(...)` from
  that module instead of owning data/train/UVT/per-frame/output/logging required
  section and key checks inline. The historical
  `train_star_uvt_video_overfit.py` file is now a thin CLI wrapper for
  `run_training(...)`, and the registry routes `star_uvt_video_overfit` to the owner module. A tiny
  4-frame, 64px, 128-tube, 1-step `src/train/train.py` smoke passed with
  disabled W&B and `/tmp` outputs.
- `star_uvt_visibility_support.py` now shares visibility-proxy target sampling
  and loss math plus support-birth-split target selection and tube placement.
  The feature overfit trainer still owns config validation and when to run the
  split, but it no longer exports those geometry helpers to tests.
- `star_uvt_schedules.py` now shares feature-target weight schedules, optimizer
  LR schedules, schedule serialization, and the enabled/RGB-weight predicates.
  Feature overfit and STAR profiling scripts now depend on that explicit
  schedule contract instead of importing schedule helpers from the trainer.
- `star_uvt_feature_losses.py` now shares STAR UVT feature-target loss/VJP
  mechanics: dense and sparse target-grid VJPs, RGB-probe grid gradients,
  trainable colorizer grid gradients, sparse image VJP packing, and VJP result
  records. The feature overfit trainer still owns warm-path orchestration, but
  it is no longer the export point for those reusable loss helpers.
- `star_uvt_feature_config.py` now shares STAR UVT feature-overfit config
  normalization and validation. Diagnostics, profilers, and tests import
  `resolve_config` from that module, leaving `run_training` as the only
  intentional feature-overfit trainer import in the profiling surface.
- `star_uvt_feature_overfit_trainer.py` now owns STAR UVT feature-overfit run
  orchestration and `run_training(...)`. The historical
  `train_star_uvt_feature_overfit.py` file is a thin CLI wrapper for
  `run_training(...)`, and the registry routes `star_uvt_feature_overfit`
  to the owner module. STAR report/audit helpers now route through
  `src/train/train.py`, `star_uvt_feature_overfit_trainer.py`, and
  `trainer_registry.py` rather than reading or launching the wrapper as the
  implementation owner. A tiny 8-frame, 64px, 512-tube, 1-step
  `src/train/train.py` smoke passed on MPS with direct-atomic rendering,
  disabled W&B, `/tmp` output, gradient flow through STAR parameters and the
  colorizer, and zero tile overflow.
- `star_uvt_tile_stats.py` and `star_uvt_render_modes.py` now share STAR UVT
  tile-load reporting and the feature-render-mode contract: mode order,
  validation set, backward dispatch, effective-mode reporting, fallback
  reporting, and the feature-gradcache cap. STAR profiling scripts and config
  validation use those explicit helper modules instead of importing or
  duplicating statistics/constants/mode maps from the feature-overfit trainer.
- `Trainer.train_step_context(...)` and `optimizer_step(...)` now share the
  zero-grad, step-total profiling, optimizer-step profiling, and
  timing-finalization envelope across base token-GS, known-camera, multicam,
  and mixed same-view/heldout steps. Branches still own sampling, decode,
  backward, and payload assembly.
- `Trainer.initial_clip_indices(...)` and `initial_clip_for_sequence(...)`
  now share first-window diagnostic clip setup across base token-GS,
  known-camera, and multicam initial paths. Branches still own camera-specific
  extraction, decode, and loss assembly.
- `KnownCameraTrainer.known_cameras_for_indices(...)` now shares indexed
  known-camera tuple extraction between initial eval and full-sequence eval, and
  the train step reuses `sample_clip(...)` instead of carrying a third camera
  validation shape.
- `KnownCameraTrainer.sample_clip(...)` no longer overrides the base
  `Trainer.sample_clip(...)` with a four-value return. The known-camera
  train-only helper is now `sample_known_clip(...)`, keeping the base
  `sample_clip(...)` interface stable.
- `Trainer.initial_recon_step_result(...)` now shares the initial eval
  render/reconstruction/V-JEPA/payload path between implicit-camera and
  known-camera trainers. Branches still own camera validation and forward
  decode, while the helper owns first preview render, recon loss, aux-loss
  payload, and `StepResult` construction.
- `KnownCameraTrainer` now inherits the base `render_full_sequence(...)`
  implementation. Its `_eval_decode_clip(...)` override supplies known cameras,
  so validation video rendering no longer has a duplicate full-sequence wrapper.
- `MulticamRelativePoseImplicitTrainer` now reuses the inherited
  `_rendered_view_recon_loss(...)` helper for relative-pose camera-swap
  rendered views. The branch still owns relpose residual/cycle/bank-rate math,
  but alpha/background guarding, recon-loss profiling, and preview capture are
  shared with the multicam trainer.
- 2026-05-22: `multicam_relative_pose_trainer.py` now owns the relative-pose
  trainer implementation and helper defaults/normalizers. The historical
  `train_multicam_relative_pose_implicit_dynamic.py` file is now a thin
  CLI wrapper that imports only `run_training(...)`, and the registry routes
  `multicam_relative_pose_implicit_camera` to the owner module. A 1-step
  `src/train/train.py` smoke passed at
  `wandb/offline-run-20260522_130926-infb3j96`; this run baked missing V-JEPA
  cache entries, exercised the real registry path, and kept checkpoint save
  disabled.
- 2026-05-22: a registry smoke matrix under `/tmp/dynaworld_registry_smokes`
  reran representative warm-path routes through `src/train/train.py` after the
  logging and owner/wrapper cleanup: Token-GS F=3, multicam RGB-pyramid,
  mixed same/heldout RGB-pyramid, multicam relative-pose RGB-pyramid, Direct
  PowerFoam, PowerFoam Metal, Dynamic PowerFoam RBF, Dynamic PowerFoam token/F32,
  and Dynamic Gauge. All passed with 1-step temporary configs. The Direct
  PowerFoam offline-W&B variant also passed and produced local media plus
  checkpoint artifacts under
  `/tmp/dynaworld_registry_smokes/powerfoam_direct_wandb_offline/outputs`, with
  W&B offline run `wandb/offline-run-20260522_152129-r22iyau1`.
- Validation stance: these cleanup smokes prove the refactored trainer plumbing
  executes, not that the training math is solved. The next proof has to be a
  longer W&B trace with media, loss curves, and benchmark/baseline rows, not
  more unit-test surface.
- Current-state correction: earlier sections below reference deleted or moved
  legacy files (`dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`, typo
  camera shims). They are kept as historical audit context, but `rg --files
  src/train` no longer finds them. Active LTX/WAN-VACE feature configs dispatch
  through `src/train/train.py` to `train_precomputed_feature_implicit_dynamic`.
- 2026-05-21 recheck: the current `src/train/` tree has `fast_attn.py`,
  `pipeline.diagnostics.eval_metric_payload(...)`, shared log cadence, typed
  render payloads, and registry/CLI/artifact helpers. Remaining cleanup should
  target live files only; do not resurrect old delete tasks for files that are
  already gone.

---

## Historical Trainer Inventory (stale audit context)

This table is the original audit snapshot, not the current live file map. Keep
it only as provenance for why the helper cleanup started. For current work, use
the progress section above plus `rg --files src/train`.

| File | LoC | Class hierarchy | Owns | Overrides | Status | Configs |
|------|-----|-----------------|------|-----------|--------|---------|
| `train_video_token_implicit_dynamic.py` | 1899 | `Trainer` (root); `KnownCameraTrainer(Trainer)` | Single-cam video-token train loop, shared run/header/lifecycle hooks, `step`, `recon_backward`, initial eval payloads, full-sequence validation, alpha-aware objective composition, random-per-step bg, F=32 colorize MLP wiring, F-PCA logging, alpha/composite validation media, and model-variant dispatch. `KnownCameraTrainer` now supplies only known-camera decode/step/init branches plus banner/export policy. | n/a (root) | **Active**. The hub of the system. | 35 configs with `arch=tokengs_video_implicit_camera`, plus `arch=tokengs` (2 configs), plus `arch=tokengs_video_known_camera` (1 config). |
| `train_precomputed_feature_implicit_dynamic.py` | 182 | `PrecomputedFeatureImplicitTrainer(Trainer)` | Feature-cache prebake, feature-cache preamble metadata, and `model_input_for_clip` override. Adds `FEATURE_OPTION_DEFAULTS` and `on_sequences_loaded` to attach a `VideoFeatureCache`. | `resolve_config`, `on_sequences_loaded`, `model_input_for_clip`, `training_preamble_messages`. | **Active**. Inherits the shared run loop, alpha/bg/log/composite work, and lifecycle hooks. | 4 configs with `arch=precomputed_feature_implicit_camera`. |
| `train_multicam_precomputed_feature_implicit_dynamic.py` | 1107 | `MulticamPrecomputedFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer)` | Multicam sampling, learnable camera rig, multi-view recon loss, heldout view eval, camera-swap support, multicam-specific validation video payload, and shared multicam rendered-view loss/StepResult helpers. | `resolve_config`, `__init__`, `load_train_sequences`, `load_eval_sequences`, `on_sequences_loaded`, `step`, `initial_step_result`, `scalar_payload`, `validation_video_payload`, `export_browser_bundle`. | **Active**. Recent cleanup moved alpha/background composition through `RGBReconObjective.render_view(...)` and shared rendered-view guards; visual/convergence proof still needs W&B runs. | 4 configs with `arch=multicam_precomputed_feature_implicit_camera` (incl. `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc`, the new F=32-alpha multicam target). |
| `train_ltx_feature_implicit_dynamic.py` | 32 | `LTXFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer)` | Backward-compat alias only. Empty body. | none | **Legacy alias**. Folder rename never happened. | 1 config (`arch=ltx_feature_implicit_camera`). The other "LTX" config (`arch=wan_vace_feature_implicit_camera`) also dispatches via `train_precomputed_feature_implicit_dynamic.py` directly through its shell script. |
| `train_camera_implicit_dynamic.py` | 417 | procedural `run_training` (no `Trainer` class) | Image-implicit single-frame training: own `resolve_config`, own `LOSS_OPTION_DEFAULTS`, own `eval_metric_payload`, own inline train loop, own W&B init, own image preview path. Does not call `Trainer`. | n/a | **Active for the image baselines**, but completely disjoint from the video trainer's vocabulary. Predates `Trainer`. | 2 configs (`arch=tokengs_image_implicit_camera`). |
| `train_image_encoder_implicit_camera_baseline.py` | 4 | shim → `train_camera_implicit_dynamic.main` | Hard-coded path to one config. | n/a | **Dead-ish**. Shim with hard-coded JSONC arg. Not referenced by any shell script. | none directly. |
| `train_camera_implict_dynamic.py` (sic — typo of "implicit") | 4 | shim → `train_camera_implicit_dynamic.main` | Hard-coded path to one config. | n/a | **Dead**. Same hard-coded path as the file above. Typo in filename. | none. |
| `dynamicTokenGS.py` | 731 | procedural `run_training` (no `Trainer` class). Also exports `pick_device`, `configure_fast_attn`, `fast_attn_context` used by `Trainer`. | Known-camera, prebaked, image-only training (one render per frame). Own `LOSS_OPTION_DEFAULTS`, own optimizer-group builder, own LR schedule, own debug-metrics path, own train loop, own W&B logging. Inline train loop with no `Trainer` class. | n/a | **Mixed**. Provides `pick_device`/fast_attn helpers that other files import — those cannot be deleted. The `run_training` body is legacy. | 9 configs (`arch=tokengs_prebaked_camera*`). Used by `train_full_dynamic_with_camera_prebake_all_frames.sh`. Probably still alive but only as a "compare to known-camera baseline" path. |

Notes:
- `tokenGS.py` (no "Dynamic") and `tokenGS_tiled.py`/`dynamicTokenGS_tiled.py`/`dynamicTokenGS_shared.py` are even older single-image trainers + 1-line shims, dispatched by `arch=tokengs_single_image*`. They are out of scope for this audit but if a sweep happens, they should be folded together with `dynamicTokenGS.py`.
- Configs whose `arch` is `gauge_fields_material_surfel` or `splat_baseline_*` do NOT dispatch via these trainers; they live under `research_experiments/gauge_fields/` (16 configs total). They should not be unified with the main trainer — they are an explicitly separate experiment surface.

---

## Historical Duplication Audit

The duplications below are concrete code blocks I found in more than one
file with only minor variation in the original audit. Some rows have since been
landed, corrected, or superseded; prefer the current progress section and the
status column before scheduling work from this table.

| Concern | Sites | Aggregate lines | Drift status | Unifiable? |
|---------|-------|-----------------|--------------|------------|
| **Compose rendered RGB from features + alpha + bg** (the `α · colorize(features) + (1-α) · bg` recipe) | `train_video_token_implicit_dynamic.py` lines ~1346-1361 (`recon_backward`, **uses random per-step bg**), ~1408-1416 (`initial_step_result`, **uses white bg**), ~1599-1609 (`Trainer.render_full_sequence`, **uses white bg**), ~1969-1979 (`KnownCameraTrainer.render_full_sequence`, **uses white bg**); MISSING in `train_multicam_precomputed_feature_implicit_dynamic.multicam_recon_loss` (~ln 197-205) | ~50 lines duplicated, plus 1 missing site that should compose | The eval site uses white bg, the train site uses random bg, the multicam site does no compose at all. Classic 3-way drift. | YES — single helper. Highest-ROI extraction. |
| **Build the validation-video W&B payload** (GT, render, alpha mask, feature PCA, composite columns) | `train_video_token_implicit_dynamic.Trainer.validation_video_payload` lines ~1649-1737; `train_multicam_precomputed_feature_implicit_dynamic.MulticamPrecomputedFeatureImplicitTrainer.validation_video_payload` lines ~320-361; partial copy in `train_camera_implicit_dynamic.run_training` lines ~367-398; partial copy in `dynamicTokenGS.run_training` lines ~667-710 | ~250 lines, four variants | The single-cam trainer has the new alpha mask + PCA + composite columns; the multicam trainer logs per-view rendered+GT but no alpha/PCA/composite; the older two have neither. | YES, but split: a lower-level `validation_video_logger(gt, rendered, features?, alpha?)` helper for the video-token paths. The two procedural trainers can call the same helper at the cost of giving them a `feature_sequence=None, alpha_sequence=None` no-op signature. |
| **Compute per-clip eval metrics** (L1, MSE, SSIM, DSSIM, recon-loss, PSNR) | `train_video_token_implicit_dynamic.eval_metric_payload` lines ~602-629; `train_camera_implicit_dynamic.eval_metric_payload` lines ~133-160 (literal copy-paste, including the same `1.0e-12` floor) | ~55 lines, two near-identical copies | Truly identical math. | YES — trivial dedup. Same goes for `temporal_similarity_payload` and `decoded_temporal_payload`, both currently single-source but inside the big trainer file. |
| **Render dispatch helper around `render_gaussian_frame[s]`** | Current active token-GS path uses `pipeline.render.render_clip_sequence(...) -> RasterizedClip`; older procedural wrappers named in earlier notes are not active under current `src/train/` paths. | mostly historical | The active alpha-aware path is typed; `render_gaussian_frames_alpha_aware(...)` tuple compatibility is gone. | PARTIAL — keep convergence around `RasterizedClip`; do not reintroduce tuple APIs. |
| **Optimizer construction and mechanical step sequence** | Active base token-GS and relative-pose-only scope shared the same Adam fused flag policy; PowerFoam, Dynamic PowerFoam, Dynamic Gauge Foam, STAR UVT probes/trainers, and gauge research scripts have distinct optimizer contracts. PowerFoam/Gauge trainer loops also repeated the same zero-grad/backward/optional-clip/step sequence. | narrow active duplicates removed | `train_optim.adam_with_device_fused(...)` owns the repeated Token-GS fused policy. `train_optim.optimizer_backward_step(...)` owns the shared PowerFoam/Gauge mechanical step sequence. | PARTIAL — keep optimizer construction, parameter groups, LR multipliers, and STAR/probe semantics local; only share the mechanical step sequence where it is identical. |
| **One-config CLI loading and train-probe CSV ints** | Active train/probe modules repeated `len(sys.argv) != 2`, usage strings, and `load_config_file(sys.argv[1])`; Token-GS-family public `main(config_or_path)` functions repeated the same path-vs-dict dispatch; the registry CLI had the last manual path-argument arity check; colorize probes repeated comma-separated seed parsing. | narrow duplicate removed | `train_cli.run_config_arg(...)`, `run_config_or_path(...)`, `run_path_arg(...)`, and `parse_csv_ints(...)` own the boundary. Focused test: `tests/test_train_cli.py`. | DONE for current `src/train/train_*.py` entrypoints, `src/train/train.py`, and colorize probe `--seeds`; the registry CLI still dispatches by config path, but no longer owns local `sys.argv` boilerplate. |
| **Trainer-as-helper imports** | Diagnostics and benchmark scripts imported `train_video_token_implicit_dynamic.py` just to call `resolve_config` or `trainer_class_for_config`; STAR feature-tube scripts imported a concrete STAR trainer just to run patched configs; benchmark probes imported concrete precomputed/multicam trainer classes just to instantiate by config; the multicam rig visualizer imported the multicam trainer only for `resolve_config`. | narrow duplicate removed | `trainer_registry.resolve_config_for_arch(...)`, `trainer_class_for_config(...)`, `instantiate_trainer_for_config(...)`, and `run_config_dict(...)` own the arch lookup, class lookup, instantiation, and in-memory dispatch. Focused tests/checks: `tests/test_trainer_registry.py`, `visualize_multicam_rig.py --help`, and a multicam config-resolution smoke. Import scan now leaves only structural subclass/test imports plus PowerFoam model-class diagnostics. | DONE for accidental helper imports; remaining direct trainer imports are structural subclass/test imports or diagnostics that need the model class itself. |
| **W&B init + final `wandb.finish()`** | Active routed trainers use `init_wandb_run(...)`; Token-GS, PowerFoam-family, STAR UVT, reusable `src/benchmarks`, and V-JEPA performance CLIs repeated local finish guards. Older rows in this doc reference files no longer present under current `src/train`. | active duplicate removed | `train_logging.finish_wandb_run(...)` owns the finish guard and handles run-object `.finish()` plus global active-run fallback. `wandb_run_lifecycle(...)` wraps init/finally-finish for current PowerFoam/Gauge owners. | DONE for current routed trainers and reusable benchmark/probe CLIs; keep this as a tiny safety wrapper, not a broader train-loop abstraction. |
| **Explicit W&B run-object log submit** | PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, Dynamic Gauge Foam, and PowerFoam eval artifacts called `wandb_run.log(...)` directly after local payload assembly. | active duplicate removed | `train_logging.log_wandb_run_payload(...)` owns the explicit-run submit call and handles the disabled-run no-op. | DONE for current `src/train`; keep payload construction local because metric/media schemas differ by trainer family. |
| **Lazy W&B media/eval payload submit** | Token-GS, multicam relative-pose, PowerFoam Direct, shared PowerFoam eval artifacts, Dynamic PowerFoam Metal, and Dynamic Gauge eval paths guarded expensive image/video payload construction with local `if wandb_run is not None` branches. | active duplicate removed | `train_logging.log_wandb_run_payload_lazy(...)` owns the disabled-run guard while accepting a payload factory, so disabled W&B does not build images/videos. | DONE for current expensive eval/media payload paths; scalar-only train-loop payloads can stay direct unless a future pass needs to skip reductions too. |
| **RGB+alpha W&B validation videos** | Direct PowerFoam, shared PowerFoam eval artifacts, Dynamic PowerFoam Metal, and Dynamic Gauge Foam repeated the same preview image plus `build_validation_video_payload(...)` + `GT_Video` + RGB-expanded `Alpha_Video` block. | active duplicate removed | `wandb_media.build_rgb_alpha_validation_video_payload(...)` and `build_rgb_alpha_eval_media_payload(...)` own preview image, render video, render/GT side-by-side video, GT video, and alpha video. | DONE for current PowerFoam/Gauge RGB+alpha eval paths; keep branch-specific scalar payloads and Gauge depth video local. |
| **RGB+alpha eval file artifacts** | The same PowerFoam/Gauge eval paths repeated preview triptych construction and render/side-by-side MP4 writes with stable filenames. | active duplicate removed | `video_io.rgb_alpha_preview(...)`, `save_rgb_alpha_preview(...)`, `save_render_side_by_side_videos(...)`, and `save_rgb_alpha_eval_media(...)` own the file-artifact pattern. | DONE for current PowerFoam/Gauge RGB+alpha eval paths; keep final-summary references and depth videos local. |
| **Metric-to-W&B payload key maps** | Direct PowerFoam, shared PowerFoam eval artifacts, Dynamic PowerFoam Metal, and Dynamic Gauge Foam repeated local `payload[...] = metrics[...]` blocks plus optional `if key in metrics` guards. | active duplicate removed | `train_logging.mapped_metric_payload(...)` owns required/optional key-map copying. | DONE for current PowerFoam/Gauge eval scalar payloads; keep the actual metric maps local because the schemas differ by trainer. |
| **PowerFoam/Gauge train scalar W&B copies** | Direct PowerFoam, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic Gauge train loops repeated local `if wandb_run is not None` plus hand-built train-metric payload dictionaries. | active duplicate removed | Trainer-local key-map tuples plus `mapped_metric_payload(...)` own the copy loop; `log_wandb_run_payload(...)` owns disabled-run no-op. | DONE for current PowerFoam/Gauge scalar train logs; Dynamic PowerFoam keeps a lazy factory only where alpha/feature tensor reductions should stay disabled when W&B is off. |
| **Media defaults and preview captions** | The PowerFoam/Gauge eval paths repeated `float(cfg.get("video_fps", 4.0))` and `step {step}: GT | render` W&B preview captions. | narrow duplicate removed | `video_io.video_fps_from_config(...)` and `wandb_media.make_step_preview_image(...)` own those media-policy literals. | DONE for current PowerFoam/Gauge RGB+alpha eval paths; cadence and payload ownership stay local. |
| **PowerFoam eval render batch size** | Direct PowerFoam, shared PowerFoam eval artifacts, and Dynamic PowerFoam Metal repeated `max(1, int(cfg["train"]["frames_per_step"]))` for artifact/eval rendering. | narrow duplicate removed | `powerfoam_eval_render.powerfoam_eval_batch_size(...)` owns the eval-render batch-size policy. | DONE for eval/artifact render paths; train-time random frame sampling remains local. |
| **PowerFoam train batch index sampling** | Direct PowerFoam, PowerFoam Metal, Dynamic PowerFoam Metal, and Dynamic Gauge Foam repeated `torch.randint(..., (int(cfg["train"]["frames_per_step"]),), device=device)` for train batches. | narrow duplicate removed | `powerfoam_training.powerfoam_train_batch_indices(...)` owns the random index draw. | DONE for current PowerFoam-family train loops; target/ray/stage/colorizer batch assembly stays local. |
| **PowerFoam loss-weight schedule** | Direct PowerFoam and Metal PowerFoam both used the same exponential auxiliary-weight schedule shape, but Direct had an extra `rgb_mse_sum_weight` term and Metal had normal-map/start-step keys. | active duplicate removed | `powerfoam_objectives.scheduled_loss_weights(...)` owns the shared schedule and returns optional `rgb_mse_sum_weight` when present. | DONE for Direct/Metal schedule plumbing; actual loss terms stay local to each trainer family. |
| **Device resolution/synchronization/cache clearing** | PowerFoam Direct, PowerFoam Metal, Dynamic PowerFoam Metal, Dynamic Gauge Foam, STAR UVT runtime, colorize probes, V-JEPA performance benchmarks, reusable renderer/fast-mac/depth benchmark probes, STAR alpha-background ablation, V-JEPA feature cache, and benchmark-memory helpers carried local `auto`, sync, or accelerator cache-clear branches. | active duplicate removed while preserving policy differences | `train_devices.resolve_torch_device(...)`, `sync_torch_device(...)`, and `clear_torch_device_cache(...)` own the primitives. Callers choose `auto_cuda`, `auto_prefer_cuda`, requested-device validation, and whether cache clearing should sync. | DONE for current PowerFoam-family + STAR runtime paths, Token-GS profile timing, colorize probes, V-JEPA benchmarks, selected reusable benchmark probes, STAR alpha-background orchestration, feature-cache clearing, and benchmark-memory cache clearing. Leave deep one-off WorldFoam kernel probes alone unless they become reusable train/benchmark entrypoints. |
| **Third-party path bootstrap** | PowerFoam Metal, Dynamic PowerFoam Metal, DUSt3R export, STAR UVT runtime, Taichi renderer bootstrap, v12a objective helpers, and the Fast-Mac renderer wrapper rebuilt Dynaworld or variant roots and mutated `sys.path` locally. | active duplicate removed | `external_paths.py` owns `PROJECT_ROOT`, `THIRD_PARTY_ROOT`, `third_party_path(...)`, `ensure_sys_path(...)`, `ensure_third_party_path(...)`, and `ensure_module_path(...)`. | DONE for train-local third-party path setup listed above. Keep renderer-specific variant selection and heavy imports local. |
| **Dataset script repo/train bootstrap** | `build_single_video_pretrain_manifest.py` and `visualize_multicam_rig.py` both rebuilt the repo root, inserted `src/train` into `sys.path`, carried repo-relative path text helpers, and the manifest builder carried a compact JSONL writer. | narrow duplicate removed | `src/dataset_scripts/script_paths.py` owns `REPO_ROOT`, `ensure_train_path()`, `repo_path(...)`, and `repo_text(...)`. Complete JSON/text artifact writes use `train_artifacts.write_json(...)` / `write_text(...)`; compact manifest rows use `train_artifacts.write_jsonl(..., compact=True)`. | DONE for active dataset CLIs; direct `--help` smokes pass for both scripts. |
| **Renderer benchmark CLI parsing/images** | `splat_renderer_benchmark.py`, `splat_renderer_accuracy.py`, `mac_renderer_stack_compare.py`, `fast_mac_v13_iteration_matrix.py`, `temporal_raster_overlap_profile.py`, and `depth_aware_dof_demo.py` repeated resolution CSV parsing, int/string/float CSV parsing, renderer/version string parsing, dtype lookup, config deep-merge, safe filename normalization, project-relative output path resolution, save-target selection, row-target matching, and CHW tensor preview-image conversion. | active duplicate removed | `src/benchmarks/renderer_benchmark_cli.py` owns the generic parser/path/merge/image-selection/image-write helpers. | DONE for the active splat renderer benchmark CLIs, Mac renderer stack renderer/version selection, temporal overlap int/float list parsing, and depth-aware DOF demo safe output stems; keep renderer-specific output schemas, subprocess commands, positive/nonempty validation, and filename conventions local. |
| **Gauge CSV/JSONC helpers** | `research_experiments/gauge_fields/make_sweep_configs.py` carried local CSV parsers, JSON-roundtrip config cloning, and generated-JSONC file writing while other Gauge scripts already route artifact/path helpers through `common.py`; matrix `--only` and summary `--columns` still split comma lists locally. | narrow duplicate removed | `research_experiments/gauge_fields/common.py` owns `parse_csv_strings/ints/floats/bools(...)`, `clone_jsonable(...)`, and `write_generated_jsonc(...)`. | DONE for the active Gauge sweep generator, matrix run selection, and summary-column parsing; keep sweep slug/tag/config mutation, run schemas, markdown layout, and metric sorting local. Compile/help smokes passed. |
| **V-JEPA benchmark splat/logging config helpers** | V-JEPA throughput, render-phase, and quality scripts repeated total-splat-count divisibility checks, effective splat count formulas, render-size/clip-length/step patching, and trainer media/log suppression settings; multicam V-JEPA repeated the step/media suppression subset. | active duplicate removed | `research_experiments/vjepa_performance/vjepa_benchmark_common.py` owns `set_total_splat_count(...)`, `effective_splat_count(...)`, `apply_video_benchmark_shape(...)`, and `quiet_training_logging(...)`. | DONE for current V-JEPA benchmark/profiling CLIs; keep case-specific config mutation, run names, and benchmark math local. Compile/help smokes passed. |
| **Artifact file primitives** | PowerFoam-family trainers repeated `output_dir.mkdir(...)` plus `resolved_config.json`; PowerFoam Metal and Dynamic PowerFoam Metal each carried local `append_jsonl(...)` helpers; V-JEPA performance, general `src/benchmarks`, and STAR alpha-background orchestration hand-wrote sorted JSON/JSONL output files. | active duplicate removed | `train_artifacts.write_resolved_config(...)`, `write_json(...)`, `write_jsonl(...)`, and `append_jsonl(...)` own those file-output primitives. | DONE for current PowerFoam-family resolved config/history calls plus reusable benchmark JSON artifacts, V-JEPA benchmark artifacts, and STAR alpha-background result artifacts. Keep checkpoint saving/local streaming local because payloads and atomicity differ. |
| **Log cadence gates** | Token-GS, PowerFoam-family, Dynamic Gauge Foam, and STAR UVT probe/overfit surfaces had local modulo/last-step checks. | active duplicate removed | `train_logging.should_log_step(...)`, `should_log_scalar(...)`, `should_log_image(...)`, and `should_log_video(...)` own the cadence primitive. | DONE for current routed trainer/probe surfaces; keep payload names and artifact choices local. |
| **Config-defaults application** | Active trainers still add their own defaults in their `resolve_config` paths. | deliberate specialization | Per-trainer additions are deliberate because each section has different required keys and backward-compatible defaults. | NO — do not centralize defaults into a giant shared dictionary; keep normalization close to the owning trainer/config contract. |
| **Init diagnostics call sites** | `init_diagnostics.py` is a single-source helper. It's invoked from `Trainer.__init__` (only conditionally — feature_pca_log) and not from any other trainer. | n/a | Not duplicated, just under-used elsewhere. | NO — already unified at the helper layer. |

---

## What's NOT duplicated and is genuinely different

These cases need to stay separate. Refactoring them out would create
useless wrapper abstractions.

1. **Multicam sampling** (`MulticamPrecomputedFeatureImplicitTrainer.sample_multicam_clip`,
   `sample_views`). Multiple cameras per step with a learnable rig is
   structurally different from single-clip sampling. The shape of
   `(views, clip_indices, clip_frames, clip_times)` will never match
   the single-clip shape `(sequence_data, clip_frames, clip_times)`.

2. **Precomputed-feature loading** (`PrecomputedFeatureImplicitTrainer.model_input_for_clip`
   bypasses the encoder forward entirely). Live encoder forward and
   precomputed-cache lookup return different things (clip frames vs.
   per-layer feature dict). The current `model_input_for_clip` hook is
   the right boundary; do not collapse it.

3. **Known-camera vs. implicit-camera gradient paths**. Known-camera
   skips the `compute_camera_losses` / `build_camera_loss` block and the
   `camera_state` is `None` throughout. The `KnownCameraTrainer` override
   is a legitimate fork.

4. **The `dynamicTokenGS.py` known-camera train loop** has one feature
   none of the others have: rich LR-multiplier groups, debug metrics
   (`debug_metrics.py`), gradient-clipping with non-finite detection.
   These are real, useful, and not in the new trainer. If we ever delete
   `dynamicTokenGS.py`, these need to migrate first.

5. **`train_camera_implicit_dynamic.py`'s per-frame render-then-loss
   loop** (image-implicit baseline, no temporal model). It iterates
   per-camera and does single-image renders. Folding it into the video
   trainer would require pretending a single image is a length-1 clip,
   which works mechanically but loses the simplicity of the baseline.
   Acknowledge it as "different baseline, leave separate."

---

## Proposed unification

Five small modules. No new abstract base class. The session's evidence
is that big class hierarchies (`Trainer` → `PrecomputedFeatureImplicitTrainer`
→ `MulticamPrecomputedFeatureImplicitTrainer`) become silent breakage
points: the multicam trainer overrode `step` and silently bypassed every
fix landed on the parent. Helpers do not have this problem because the
caller has to invoke them explicitly.

### 1. `compose_rendered_rgb(features, alpha, colorize, *, random_bg, training)`

Pure function. Single source of truth for the
`α · colorize(features) + (1-α) · bg` recipe.

Signature sketch:
```text
compose_rendered_rgb(
    features: Tensor[T, F, H, W],
    alpha: Tensor[T, H, W] | None,
    colorize: FeatureToColor | None,
    cameras: tuple[CameraSpec, ...],
    *,
    random_bg: Tensor[1,3,1,1] | None,   # caller decides train-vs-eval; None = white
    input_size: int,
    render_size: int,
) -> Tensor[T, 3, H, W]
```

Caller responsibilities:
- training step: pass `random_bg = torch.rand(3, ...).view(1,3,1,1)` (sampled once per step).
- eval / `initial_step_result`: pass `random_bg = None` → helper uses white background.
- multicam loop: same call with `random_bg = step_random_bg` (sampled once for the whole step, broadcast across views).

Trainers it deduplicates: 4 sites in `train_video_token_implicit_dynamic.py` + 1 missing site in `train_multicam_precomputed_feature_implicit_dynamic.py`. Net: ~50 lines collapse to ~10 (1 helper call per site).

Risk: low. Pure function with clear inputs. Add one numerical-tolerance test that asserts `compose_rendered_rgb(features, alpha=ones, colorize=identity, random_bg=anything) == features` (alpha=1 should erase the bg term).

### 2. `validation_video_logger(...)` built on `wandb_media.py`

Purpose: build the `Render_Video`, `Render_GT_Video`, `Alpha_Mask_Video`,
`Feature_PCA_Video`, `Render_Composite_Video` wandb payload from a
fixed argument vocabulary.

Signature sketch:
```text
build_validation_payload(
    *,
    gt_sequence: Tensor[T, 3, H, W],
    rendered_sequence: Tensor[T, 3, H, W],
    feature_sequence: Tensor[T, F, H, W] | None,
    alpha_sequence: Tensor[T, H, W] | None,
    fps: float,
    log_gt_video: bool,         # caller toggles based on `gt_video_logged`
) -> dict[str, Any]
```

Trainers it deduplicates: `Trainer.validation_video_payload` (the alpha-mask, feature-PCA, composite logic, ~80 lines). The multicam trainer's `validation_video_payload` calls it per-view with `feature_sequence=None, alpha_sequence=None` initially, then once we plumb alpha through the multicam path it calls it with the full set per view. The procedural older trainers (`train_camera_implicit_dynamic`, `dynamicTokenGS`) call the same helper with `alpha_sequence=None, feature_sequence=None` and lose nothing.

Risk: medium. The composite-column ordering and the `gt_video_logged` flag are easy to get wrong on the first attempt. One end-to-end fixture test (10 fake frames, all four optional inputs present, assert the dict has the five expected keys) is enough to lock the contract.

### 3. `RenderedClipBundle` dataclass

Replace the proliferating tuples passed around the trainer. Today
`render_clip_sequence` returns `tuple[Tensor, Tensor | None]` and three
unrelated `(features, alpha)` unpackings happen at call sites. Add:

```text
@dataclass(frozen=True)
class RenderedClipBundle:
    features: Tensor          # [T, F, H, W]
    alpha: Tensor | None      # [T, H, W]
    # later: depth, normals, etc — additive only
```

`render_clip_sequence` returns `RenderedClipBundle`. Call sites destructure
`bundle.features, bundle.alpha`. Adding `depth` (or similar) becomes a
one-line dataclass change instead of a search-and-update across every
unpacking site.

Risk: low. Mechanical refactor. Reduces tuple-arity bugs (the kind that
silently drop alpha at a call site). One fixture test that constructs a
bundle and checks shapes.

Current status: implemented under clearer names in `runtime_types.py`.
`RasterizedClip` is the features-plus-alpha raster payload returned by
`pipeline.render.render_clip_sequence`; `RenderedClip` is the stitched
full-sequence validation payload. `pipeline.render` now imports these runtime
payload dataclasses instead of defining them locally.

### 4. `colorize_module_from_config(cfg, feature_dim, device)` factory

Today the colorize-module construction is inlined in `Trainer.__init__`
(ln ~1010-1024) — a 14-line block with five `cfg.get` calls and a
post-condition error message. The multicam trainer inherits this through
`super().__init__` so it works there. **But** if anyone ever adds a new
colorize knob (the session already had three: `pre_norm`, `weight_init`,
`weight_init_gain`, plus `view_condition`), every subclass that
re-implements `__init__` has to remember to pass it through.

Move to a small factory in `colorize.py`. Both trainers call:
```text
self.colorize, self.colorize_view_condition = colorize_module_from_config(
    self.cfg.get("colorize"), feature_dim=self.feature_dim, device=self.device
)
```

Risk: very low.

Current status: implemented as `model_factories.build_colorizer(...)` with the
typed `ColorizeFactoryResult(module, view_condition, detach_view_condition)`.
The base Token-GS trainer and `probe_colorize_init.py` use that boundary. STAR
UVT keeps a separate `star_uvt_colorizers.build_feature_colorizer(...)` because
its colorizer config contract is narrower and required-key based.

### 5. (Optional) `run_step_loop(trainer, *, total_steps, log_intervals)` helper

The `for step in pbar: ... pbar.set_description(...) ... self.val_log(step, result)` outer loop is ~15 lines and identical between the single-cam and known-camera trainers. The procedural older trainers have a copy too. This is the only reason `Trainer.run` and `KnownCameraTrainer.run` differ at all — the run banner. A shared `run_step_loop(self)` plus a per-trainer `print_run_banner()` would let `KnownCameraTrainer` lose its `run` override entirely.

Risk: low. This is the one place a small base-class method (`run`) is fine, because the only reason for the override is a banner string.

### What is NOT proposed (deliberately)

- No new abstract `BaseTrainer` with virtual methods. The current
  `Trainer → PrecomputedFeatureImplicitTrainer → MulticamPrecomputedFeatureImplicitTrainer`
  chain already exists and is the source of the bug. Adding more
  inheritance layers would compound it.
- No broad framework registry beyond the current explicit `arch` map. Keep
  `trainer_registry.py` as the thin dispatch/config-resolution boundary; do not
  turn it into a base-trainer framework or hide experiment-specific CLIs behind
  vague dynamic loading.
- No model-architecture refactor. The 35-config `learned_time_orbit_path`
  fanout is a model concern, not a trainer concern.
- No `RenderConfig` dataclass refactor (proposed in `Clean_up_and_unify_interfaces.md`).
  That's still a good idea but is independent of the alpha/multicam
  unification and adds churn risk.

---

## What to delete

| File | Safety check | Verdict |
|------|--------------|---------|
| `src/train/train_camera_implict_dynamic.py` (sic — typo) | `rg --files src/train` and direct `rg` checks no longer find it. | Already absent; no action. |
| `src/train/train_image_encoder_implicit_camera_baseline.py` | `rg --files src/train` and `src/train_scripts` checks no longer find it. | Already absent; no action. |
| `src/train/train_ltx_feature_implicit_dynamic.py` | `rg --files src/train` no longer finds it; `ltx_feature_implicit_camera` dispatches through `trainer_registry.py` to `train_precomputed_feature_implicit_dynamic`. | Already folded; keep the registry route. |
| `src/train/dynamicTokenGS.py`, `src/train/tokenGS*.py`, `src/train/dynamicTokenGS_*.py` | `rg --files src/train` no longer finds these older procedural files in the current tree. | Already absent from live `src/train`; keep old references only as historical audit context. |

---

## Migration strategy

The original phase list below is historical audit context. The current tree has
already landed the composition, diagnostics, cadence, render-payload, CLI,
registry, device, artifact, and mixed-scheduler helpers described in the
progress section above. Use this section only to avoid reintroducing old forks;
for new work, start from the live-file checks in the progress section.

### Phase 1 (unblocks current pain)

1. **Extract `compose_rendered_rgb`.** Landed as `objective.compose_rgb` /
   `RGBReconObjective.render_view(...)`.
2. **Extract feature colorize/background composition.** Landed in the objective
   layer and STAR UVT feature composition helpers.
3. **Plumb alpha through multicam paths.** Landed through shared
   `_rendered_view_recon_loss(...)` and objective-bound background guards.

### Phase 2 (deduplicates the remaining noise)

4. **Extract validation media helpers.** Landed in `pipeline.validation_media`
   for active single-cam/multicam paths.
5. **Add typed render payloads.** Landed as `runtime_types.RasterizedClip` and
   `runtime_types.RenderedClip`.
6. **Lift `eval_metric_payload` to a shared module.** Landed in
   `pipeline.diagnostics`.
7. **Lift log cadence helpers to `train_logging.py`.** Landed as
   `should_log_step`, `should_log_scalar`, `should_log_image`, and
   `should_log_video`.

### Phase 3 (cleanup, optional)

8. **Delete dead shims.** Already absent in the current `src/train/` tree.
9. **Move `pick_device`/`fast_attn_context` out of `dynamicTokenGS.py`.** Already
   landed as `src/train/fast_attn.py`.
10. **Keep future cleanup live-file driven.** The next useful structural
    slices are no longer train-file owner moves; the routed warm-path trainers
    now dispatch through owner modules and thin wrappers, and STAR report/audit
    helpers route through `src/train/train.py` plus owner modules. The next
    useful experiment slices remain STAR UVT alpha-background ablation execution
    and real W&B-enabled smoke/quality evidence.

### What breaks if you do nothing

- The old alpha/composition drift called out here is now fixed for active
  single-cam and multicam paths. The remaining risk is subtler: new experiment
  scripts can still reintroduce local JSON, device, registry, or background
  composition forks if they bypass the shared helpers.
- Interface cleanup alone does not prove the training objective. Keep requiring
  W&B-enabled smokes, media checks, and benchmark rows before calling a trainer
  lane solved.

---

## Out of scope

- Changes to model architectures, the rasterizer (`v5_features`), or `gs_models/`.
- Changes to JSONC config schemas or to `pyproject.toml`.
- Single-image trainers (`tokenGS.py`, `tokenGS_tiled.py`, `dynamicTokenGS_tiled.py`).
- The `gauge_fields` and `splat_baseline` trainer paths (separate experiment surface under `research_experiments/`).
- The per-baseline `KnownCameraTrainer/ImageImplicitCameraTrainer/VideoImplicitCameraTrainer` split proposed in `Clean_up_and_unify_interfaces.md` Phase 2 — orthogonal cleanup, not blocked by this work.

---

## Cross-references

- `agent_notes/loose_notes/2026-04-29_22-00-00_feature_splatting_alpha_aware_composition_session.md` — origin of the alpha-aware composition + random-per-step-bg + F=32 colorize work that this audit unifies.
- `agent_notes/loose_notes/2026-04-30_00-00-00_feature_splatting_speedup_handoff_analysis.md` — kernel-side speedups (orthogonal to trainer unification, but informs the F-cap planning).
- `TODO/alpha_mask_white_background_cheating.md` — open issue; the fixes proposed there land at the `compose_rendered_rgb` site, so unifying that helper makes the experiments cheaper.
- `TODO/Clean_up_and_unify_interfaces.md` — earlier interface-cleanup plan. Phase 1 (runtime types, render dispatch, implicit-camera math) was implemented and stays valid. Phase 2 ("split each baseline into its own trainer class") is now superseded by this doc — the lesson is that more class boundaries make drift worse, not better.
