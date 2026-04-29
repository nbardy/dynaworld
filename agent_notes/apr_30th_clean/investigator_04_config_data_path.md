# Investigator 4: config schema and data pipeline

## TL;DR

- There is no centralized config schema. Three independent trainers each ship their own `resolve_config()` and their own DEFAULTS dicts; configs are JSONC files duck-typed into whichever trainer the launcher script invokes.
- The top-level `arch` field is **not read by any production trainer code**. Dispatch is by (a) which `python <trainer.py> <config.jsonc>` the shell script calls, (b) `model.variant` inside `build_model_from_config`. `arch` is a comment for humans.
- 96 train configs in `src/train_configs/` collapse to 4 broad families. ~31 of those configs target a fully separate "gauge fields" trainer in `research_experiments/` with no shared schema with the main pipeline.
- The main video-token trainer's `resolve_config()` only applies defaults to 5 of 9 sections (`data`, `model`, `camera`, `losses`, `export`). `render`, `train`, `logging`, `colorize`, `features` are required-or-`KeyError`, with a few one-off `if "x" not in cfg[…]` patches at the bottom of `resolve_config`.
- `build_model_from_config` does direct dict lookups for ~9 model-arch knobs (`bottleneck_dim`, `num_heads`, `mlp_ratio`, `tubelet_size_t`, `patch_compression`, `encoder_self_attn_layers`, `bottleneck_self_attn_layers`, `cross_attn_layers`, `time_fourier_bands`, `time_max_frequency`) that are NOT in `MODEL_OPTION_DEFAULTS`. Several model variants then swallow these via `**_unused`. Configs cargo-cult the values to satisfy the trainer.

## Files in scope

- `src/train/config_utils.py` — JSONC stripping + `apply_defaults` + `path_or_none` helpers (108 lines, no schema enforcement beyond required-section presence).
- `src/train/train_video_token_implicit_dynamic.py` (2072 lines) — main trainer + `build_model_from_config` + `Trainer`/`KnownCameraTrainer` + `resolve_config`.
- `src/train/train_precomputed_feature_implicit_dynamic.py` (166 lines) — subclass; adds `features` section + `FEATURE_OPTION_DEFAULTS`.
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` (386 lines) — sub-subclass; adds `DATA_MULTICAM_DEFAULTS`, `CAMERA_RIG_DEFAULTS`, `TRAIN_MULTICAM_DEFAULTS`.
- `src/train/dynamicTokenGS.py` — older single-clip / known-camera trainer (separate `MODEL_OPTION_DEFAULTS` etc.; not in modern path).
- `src/train/train_camera_implicit_dynamic.py` — older "image encoder + implicit camera" baseline.
- `src/train/sequence_data.py`, `src/train/multicam_video_data.py`, `src/train/multicam_val_data.py` — data loaders.
- `src/train/build_clip_dataset.py` — emits `manifest.jsonl` consumed by trainer when `data.manifest_path` is set.
- `src/dataset_pipeline/*.py` — multi-stage CLIs that materialize external datasets to `data/multicam_val/...`.
- `src/dataset_configs/*.jsonc` — inputs to dataset_pipeline scripts; **not** read by trainer.
- `src/train_configs/*.jsonc` — 96 files; trainer inputs.
- `src/train_scripts/*.sh`, `src/dataset_scripts/*.sh` — launchers.
- `research_experiments/gauge_fields/train.py`, `train_splat_baseline.py` — completely separate trainers consuming the `gauge_fields_*` and `splat_baseline_*` configs.

## Train-config schema (main video-token trainer)

The trainer requires the top-level dict to contain these sections: `data, model, camera, render, train, losses, logging`. Anything else (`arch`, `colorize`, `export`, `features`) is optional or ignored. Each leaf below is annotated:

- **R** = required, no default (`KeyError`/`ValueError` if absent)
- **D** = has a default in some `*_OPTION_DEFAULTS`
- **r** = read with `dict.get(key, inline_default)` (silent default at the use site)
- **U** = silently ignored / swallowed by `**_unused` for at least one model variant

### `arch` (top-level)
- `arch` (str, **never read** by production trainer code; the only places it appears are `init_diagnostics.py` and `probe_init_diagnostics.py`. Pure documentation.)

### `data`
- `sequence_dir` (Path|None) — **R** (required if `manifest_path` is None; `resolve_config` raises `ValueError`).
- `frames_dir` (Path|None) — **R** (no default; `cfg["data"]["frames_dir"]` is read directly at line 180 — KeyError if absent).
- `video_path` (Path|None) — **R** (read at line 181; required if `frame_source == "explicit_video"`).
- `frame_source` (str) — **R** (read at line 424, line 1097; values: `summary_sampled`, `all_frames`, `summary_video`, `explicit_video`, `camera_json`, plus `multicam_val` used by gauge_fields trainer).
- `max_frames` (int) — **R** (read at line 425, line 1084; 0 means "no limit").
- `frame_indices` (list[int]|None) — **r** (only consumed by multicam path; read with `data_cfg.get("frame_indices")`).
- `manifest_path` (Path|None) — **D** (default None, then `path_or_none`).
- `split` (str) — **D** (default "train").
- `eval_manifest_path` (Path|None) — **D** (default None).
- `eval_split` (str) — **D** (default "test").
- `eval_max_sequences` (int) — **D** (default 1).
- `camera_json` (Path|None) — **D** (default None).
- `camera_image_size` (int) — **D** (default 224).
- `camera_focal_mode` (str) — **D** (default "median"; lowercased).
- `multicam_*` keys — added only by `MulticamPrecomputedFeatureImplicitTrainer.resolve_config` via `DATA_MULTICAM_DEFAULTS`:
  - `multicam_manifest` (Path) — **D** (`data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl`).
  - `multicam_split` (str) — **D** ("val").
  - `multicam_sample_id` (str|None) — **D** (None).
  - `multicam_sample_index` (int) — **D** (0).
  - `multicam_train_cameras` (list[str]|None) — **D** (None).
  - `multicam_heldout_cameras` (list[str]|None) — **D** (None).
  - `multicam_heldout_camera` (str|None) — **D** (None).
  - `multicam_anchor_camera` (str|None) — **D** (None).
  - `multicam_condition_camera` (str|None) — **D** (None).

### `model`
- `variant` (str) — **D** (default "learned_time_orbit_path"; lowercased; dispatch key in `build_model_from_config`).
- `feature_dim` (int) — **D** (default 3).
- `size` (int) — **R** (line 432, 850 — required; image-side resolution).
- `train_frame_count` (int) — **R** (line 757, 849 — required; clip length).
- `tokens` (int) — **R** (line 280, 376, 851 — required for token bank size).
- `model_dim` (int) — **R** (line 853 → `feat_dim`).
- `gaussians_per_token` (int) — **R** (line 376, 857).
- `scene_extent` (float) — **R** (line 253, 858; `xy_extent`/`z_min`/`z_max` derive from this if absent).
- `bottleneck_dim` (int) — **R, U** (line 854; required by `build_model_from_config`, swallowed via `**_unused` in `UnconditionedTokenGSImplicitCamera`, `FreeGaussianBankImplicitCamera`).
- `num_heads` (int) — **R, U** (line 855; same).
- `mlp_ratio` (float) — **R, U** (line 856; same).
- `tubelet_size_t` (int) — **R, U** (line 876; same).
- `patch_compression` (int) — **R, U** (line 877–878; same).
- `encoder_self_attn_layers` (int) — **R, U** (line 880; same).
- `bottleneck_self_attn_layers` (int) — **R, U** (line 881; same).
- `cross_attn_layers` (int) — **R, U** (line 892; same).
- `time_fourier_bands` (int) — **D, U** (line 105, 915; default 8; only used when variant ∈ {`sinusoidal_time_path_mlp`, `token_to_pose_to_plucker`}).
- `time_max_frequency` (float) — **D, U** (line 106, 916; default 128.0; same).
- `xy_extent` (float|None) — **D** (line 79, default None → `scene_extent`).
- `z_min` (float|None) — **D** (default None → `-scene_extent`).
- `z_max` (float|None) — **D** (default None → `+scene_extent`).
- `scale_init` (float) — **D** (default 0.05).
- `scale_init_log_jitter` (float) — **D** (default 0.0).
- `opacity_init` (float|None) — **D** (default None).
- `query_token_init_std` (float) — **D** (default 0.02).
- `head_hidden_dim` (int) — **D** (default 64).
- `head_hidden_layers` (int) — **D** (default 1).
- `head_output_init_std` (float|None) — **D** (default None).
- `position_init_extent_coverage` (float) — **D** (default 0.0).
- `rotation_init` (str) — **D** (default "random").
- `rgb_init` (str|None) — **D** (default None; only meaningful for `feature_dim==3`).
- `rgb_init_min`, `rgb_init_max` (float) — **D** (default 0.01, 0.99).
- `video_encoder_backend` (str) — **D** (default "local"; values `local|vjepa_hf|vjepa_torchhub|precomputed|precomputed_ltx|none`).
- `vjepa_*` (multi-key) — **D** (all default None or sensible).
- `video_feature_layers` (list[str]|None) — **D** (default None).
- `video_feature_channels` (dict[str,int]|None) — **D** (default None).
- `ray_condition_grid_size` (int) — **D, U** (default 16; only used by `token_to_pose_to_plucker`).
- `static_tokens` (int|None) — **D** (default None; if not None, triggers `use_static_dynamic_split` derived flag).
- `dynamic_tokens` (int|None) — **D** (default None).
- `dynamic_time_basis_count`, `dynamic_time_max_frequency`, `dynamic_motion_extent`, `dynamic_rotation_degrees`, `dynamic_alpha_logit_extent`, `dynamic_coeff_output_init_std` — **D**.
- `free_*`, `residual_*` — **D, U** (gated by `model.variant`; swallowed by other variants).
- `use_static_dynamic_split` (bool) — **derived** by `resolve_config` itself. Configs MUST NOT set it.

### `camera`
- `global_head` (str) — **D** (default "legacy_orbit"; lowercased; values `legacy_orbit|legacy_pinhole|simple_pinhole|central_lens`).
- `lens_model` (str) — **D** (default "pinhole"; values `pinhole|radial_tangential|opencv_fisheye`).
- `base_fov_degrees` (float) — **D** (default 60.0).
- `base_radius` (float) — **D** (default 3.0).
- `max_fov_delta_degrees` (float) — **D** (default 15.0).
- `max_radius_scale` (float) — **D** (default 1.5).
- `max_aspect_log_delta` (float) — **D** (default 0.0).
- `max_principal_point_delta` (float) — **D** (default 0.0).
- `distortion_max_abs` (float) — **D** (default 0.0).
- `base_distortion` (None|list[float]) — **D** (default None).
- `max_rotation_degrees` (float) — **D** (default 5.0).
- `max_translation_ratio` (float) — **D** (default 0.2).
- `rig_*` — **D** in multicam subclass via `CAMERA_RIG_DEFAULTS`.
- `aist_translation_scale`, `n3d_translation_scale`, `vivo_translation_scale`, `multicam_pose_source` — **r** (read with `camera_cfg.get(...)` inside `multicam_video_data.load_multicam_video_bundle`; never declared in defaults).

### `render`
The whole section is **R** (required as a section). Per-key:
- `renderer` (str) — **R** (line 378).
- `render_size` (int) — **R** (line 380, 967).
- `auto_dense_limit` (int) — **R** (line 382).
- `tile_size` (int) — **R** (line 576).
- `bound_scale` (float) — **R** (line 577).
- `alpha_threshold` (float) — **R** (line 578).
- `near_plane` (float) — **D-inline** (line 333: `if "near_plane" not in cfg["render"]: cfg["render"]["near_plane"] = 1.0e-4`).
- `camera_projection` (str) — **D-inline** (line 335: default "auto"; lowercased; aliases `legacy → legacy_pinhole`).
- `fast_mac` (dict|None) — **D-inline** (line 347: default None). When non-null, the dict has its own internal schema (`tile_size`, `max_fast_pairs`, `alpha_threshold`, `transmittance_threshold`, `background`, `feature_background`, `enable_overflow_fallback`, `batch_strategy`, `batch_launch_limit_tiles`, `batch_launch_limit_gaussians`) consumed in `rendering.py` — none of which has a Python-side default.

### `train`
The whole section is **R**. None of the per-key defaults are declared via `apply_defaults`:
- `steps` (int) — **R**.
- `lr` (float) — **R**.
- `amp` (bool) — **R**.
- `recon_backward_strategy` (str) — **R** (validated in `Trainer.__init__`; values `framewise|microbatch|batched`).
- `temporal_microbatch_size` (int) — **R**.
- `train_views_per_step` (int) — **D** in multicam subclass via `TRAIN_MULTICAM_DEFAULTS` (default 0).
- `camera_rig_lr` (float|None) — **D** in multicam subclass (default None → falls back to `train.lr`).

### `losses`
- `type` (str) — **D** (default "l1_mse"; values `standard_gs|l1_mse|l1|mse`).
- `l1_weight`, `mse_weight`, `dssim_weight`, `ssim_window_size`, `ssim_c1`, `ssim_c2` — **D**.
- `camera_motion_weight`, `camera_temporal_weight`, `camera_global_weight` — **D**.
- `static_alpha_rate_weight`, `dynamic_alpha_rate_weight`, `dynamic_motion_rate_weight`, `dynamic_rotation_rate_weight`, `dynamic_alpha_time_rate_weight` — **D**.

### `logging`
The section is **R**. None of the per-key defaults are declared:
- `log_every`, `image_log_every`, `video_log_every`, `always_log_last_step` — **R**.
- `wandb_project`, `wandb_run_name` — **R**.
- `wandb_tags` — **r** (read with `.get`).
- `feature_pca_log` (bool) — **r** (read with `.get(..., False)`; only present in 5 of 96 configs).

### `colorize` (optional, top-level)
Entirely **r** — every key consumed via `colorize_cfg.get(key, default)` at the use site. Required when `model.feature_dim != 3`.
- `hidden_dim` (int|None) — **r** (no default declared; passed through).
- `activation` (str) — **r** (default "sigmoid").
- `pre_norm` (bool) — **r** (default False).
- `weight_init` (str) — **r** (default "kaiming"; lowercased).
- `weight_init_gain` (float) — **r** (default 1.0).
- `view_condition` (str) — **r** (default "none"; values `none|camera_center_ray|pixel_ray`).
- `detach_view_condition` (bool) — **r** (default True).

### `export` (optional, top-level)
- `enabled` (bool) — **D** (default False).
- `output_root` (Path) — **D** (default `outputs/browser_exports`; raises if null).
- `id` (str|None) — **D** (default None).
- `sequence_index` (int) — **D** (default 0).
- `window_start` (int) — **D** (default 0).
- May be a bare `bool` (collapsed to `{enabled: bool}`) or a dict.

### `features` (added by `PrecomputedFeatureImplicitTrainer`)
Exists ONLY when running the precomputed-feature subclass. `FEATURE_OPTION_DEFAULTS` defines ~35 keys including `extractor`, `model_id`, `pipeline`, `layers`, `cache_dir` (raises if null), `sample_cache_key`, `force_rebake`, `vjepa_*`, LTX/Wan-VACE-specific knobs (`prompt`, `guidance_scale`, `flow_shift`, `num_inference_steps`, etc.). All have defaults via `apply_defaults`.

## Required-but-not-defaulted keys

These are the trip-wires. A new variant that doesn't actually use these still has to set them or `build_model_from_config` raises `KeyError`:

| Key | Defaulted? | Used by | Swallowed-by-`**_unused` for |
|---|---|---|---|
| `model.size` | NO | all variants | – |
| `model.train_frame_count` | NO | all variants | – |
| `model.tokens` | NO | most variants | – |
| `model.model_dim` (`feat_dim`) | NO | most variants | – |
| `model.gaussians_per_token` | NO | all variants | – |
| `model.scene_extent` | NO | all variants | – |
| `model.bottleneck_dim` | NO | encoder-bearing variants | `unconditioned_tokens`, `unconditioned_residual_free_bank`, `free_splats`, `free_linear_time_splats` |
| `model.num_heads` | NO | encoder-bearing variants | same |
| `model.mlp_ratio` | NO | encoder-bearing variants | same |
| `model.tubelet_size_t` | NO | encoder-bearing variants | same |
| `model.patch_compression` | NO | encoder-bearing variants | same |
| `model.encoder_self_attn_layers` | NO | encoder-bearing variants | same |
| `model.bottleneck_self_attn_layers` | NO | encoder-bearing variants | same |
| `model.cross_attn_layers` | NO | encoder-bearing variants | same |
| `data.frames_dir` | NO | uncalibrated path | – |
| `data.video_path` | NO | uncalibrated path | – |
| `data.frame_source` | NO | uncalibrated path | – |
| `data.max_frames` | NO | uncalibrated path | – |
| `render.renderer` | NO | all | – |
| `render.render_size` | NO | all | – |
| `render.auto_dense_limit` | NO | all | – |
| `render.tile_size` | NO | all | – |
| `render.bound_scale` | NO | all | – |
| `render.alpha_threshold` | NO | all | – |
| `train.steps`, `train.lr`, `train.amp`, `train.recon_backward_strategy`, `train.temporal_microbatch_size` | NO | all | – |
| `logging.log_every`, `logging.image_log_every`, `logging.video_log_every`, `logging.always_log_last_step`, `logging.wandb_project`, `logging.wandb_run_name` | NO | all | – |

Bottom line: 30+ keys are required at use-site but never declared as defaults. Configs cargo-cult them and have written explanatory comments about the swallowed keys (e.g. the comment block in `local_mac_unconditioned_tokens_fast.jsonc:45-49` flagging which keys are inert).

## Arch / variant dispatch

There are **two** dispatch axes that don't agree:

### 1. Trainer file dispatch — by launcher script (the `arch` field is unread)

| Trainer file | Trainer class | Launcher scripts | Matching `arch` strings in configs |
|---|---|---|---|
| `train_video_token_implicit_dynamic.py` | `Trainer` / `KnownCameraTrainer` (chosen by `trainer_class_for_config(config)` at line 2048 from `model.variant`) | `train_full_dynamic_with_video_token_implicit_camera_all_frames.sh`, `train_implicit_camera_128_4fps_fast_mac_baseline.sh`, `train_smoke_dynamic_with_video_token_implicit_camera.sh`, `train_compare_vjepa2_fpc16_256_16f_single_overfit.sh`, `train_local_mac_30_clip_baseline.sh`, `train_local_mac_30_clip_vjepa2_256_baseline.sh`, `train_video_temporal_ablation_suite.sh`, parts of `train_static_dynamic_vjepa_features_ablation.sh` | `tokengs_video_implicit_camera`, `tokengs`, `tokengs_video_known_camera` |
| `train_precomputed_feature_implicit_dynamic.py` | `PrecomputedFeatureImplicitTrainer` | `train_ltx_feature_implicit_camera_128_4fps_fast_mac.sh`, `train_precomputed_vjepa2_1_torchhub_vitb_384.sh`, `train_wan_vace_feature_implicit_camera_128_4fps_fast_mac.sh`, parts of `train_static_dynamic_vjepa_features_ablation.sh` | `precomputed_feature_implicit_camera`, `ltx_feature_implicit_camera`, `wan_vace_feature_implicit_camera` |
| `train_multicam_precomputed_feature_implicit_dynamic.py` | `MulticamPrecomputedFeatureImplicitTrainer` | `train_multicam_static_dynamic_vjepa_features.sh` | `multicam_precomputed_feature_implicit_camera` |
| `dynamicTokenGS.py` (legacy) | (separate Trainer in same file) | `train_full_dynamic_with_camera_prebake_all_frames.sh` | `tokengs_prebaked_camera`, `tokengs_prebaked_camera_tiled` |
| `train_camera_implicit_dynamic.py` (legacy) | own Trainer | `train_full_dynamic_with_implicit_camera_all_frames.sh`, `train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh` (alias) | `tokengs_image_implicit_camera`, `tokengs_single_image`, `tokengs_single_image_tiled` |
| `research_experiments/gauge_fields/train.py` | gauge-fields Trainer | not launched from `src/train_scripts/`; called via `make_sweep_configs` and direct `python ...` | `gauge_fields_material_surfel` (31 configs use this string) |
| `research_experiments/gauge_fields/train_splat_baseline.py` | splat-baseline Trainer | direct `python ...` | `splat_baseline_static_3dgs`, `splat_baseline_free_dynamic_3dgs` |

The `arch` field in the train config is purely advisory: nothing in `src/train/` reads it. The trainer is chosen by which `.sh` script you run. `init_diagnostics.py` and `probe_init_diagnostics.py` are the only places that even `config.get("arch")`, and only for logging.

### 2. Model class dispatch — by `model.variant` inside `build_model_from_config`

| `model.variant` value | Model class |
|---|---|
| `learned_time_orbit_path` (default) | `DynamicVideoTokenGSImplicitCamera` |
| `free_splats`, `free_gaussian_bank` | `FreeGaussianBankImplicitCamera` |
| `free_linear_splats`, `free_linear_time_splats`, `linear_free_splats` | `LinearTimeFreeGaussianBankImplicitCamera` |
| `residual_free_bank`, `residual_free_video`, `residual_free_bank_video_tokens` | `ResidualFreeBankVideoTokenGSImplicitCamera` |
| `known_camera`, `known_camera_video_token` | `DynamicVideoTokenGSKnownCamera` (also routes to `KnownCameraTrainer` via `trainer_class_for_config`) |
| `sinusoidal_time_path_mlp` | `DynamicVideoTokenGSImplicitCameraSinusoidalTime` |
| `token_to_pose_to_plucker` | `DynamicVideoTokenGSImplicitCameraPoseToPlucker` |
| `unconditioned_tokens`, `token_decoder_unconditioned` | `UnconditionedTokenGSImplicitCamera` |
| `unconditioned_residual_free_bank`, `residual_free_bank_unconditioned_tokens` | `UnconditionedResidualFreeBankImplicitCamera` |

Configs that target the gauge_fields or splat_baseline trainers do not set `model.variant` at all — those trainers don't read it.

### Concrete inconsistencies

- `local_mac_overfit_video_token_smoke.jsonc` has `"arch": "tokengs"` (a leftover string seen nowhere else in the trainer dispatch; works only because `arch` is never read).
- `gauge_fields_material_surfel` is the `arch` value on every gauge-fields config including the multicam, motion, and rank-adaptive variants — even though those drive structurally different model and loss code paths inside the gauge-fields trainer.
- `local_mac_overfit_image_implicit_camera_separated.jsonc` sets `"variant": "separated_camera"` — that string isn't in `build_model_from_config`'s if-elif ladder, so it would crash if launched against `train_video_token_implicit_dynamic.py`. It is launched only through `train_camera_implicit_dynamic.py` (a different trainer), which has its own dispatch.
- `train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh` is an alias that just `exec bash` → `train_full_dynamic_with_implicit_camera_all_frames.sh`. Not load-bearing; safe to delete.

## Launcher scripts (triage)

| Script | Trainer | Status |
|---|---|---|
| `train_full_dynamic_with_video_token_implicit_camera_all_frames.sh` | `train_video_token_implicit_dynamic.py` | canonical default |
| `train_full_dynamic_with_implicit_camera_all_frames.sh` | `train_camera_implicit_dynamic.py` (legacy) | legacy single-image-encoder path |
| `train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh` | alias of the above | dead / can be deleted |
| `train_full_dynamic_with_camera_prebake_all_frames.sh` | `dynamicTokenGS.py` (legacy) | legacy known-camera path |
| `train_implicit_camera_128_4fps_fast_mac_baseline.sh` | `train_video_token_implicit_dynamic.py` | one-off entry point |
| `train_smoke_dynamic_with_video_token_implicit_camera.sh` | `train_video_token_implicit_dynamic.py` | smoke variant |
| `train_compare_vjepa2_fpc16_256_16f_single_overfit.sh` | `train_video_token_implicit_dynamic.py` | mode-multiplexer (12+ named modes) over a fixed set of compare/* configs |
| `train_local_mac_30_clip_baseline.sh` | `train_video_token_implicit_dynamic.py` | requires `data/youtube_scene_distinct/clip_sets/...` manifest |
| `train_local_mac_30_clip_vjepa2_256_baseline.sh` | `train_video_token_implicit_dynamic.py` | mode={local,vjepa,both} |
| `train_video_temporal_ablation_suite.sh` | `train_video_token_implicit_dynamic.py` | mode-multiplexer over `*_ablate_time_*` configs |
| `train_static_dynamic_vjepa_features_ablation.sh` | both `train_video_token_implicit_dynamic.py` and `train_precomputed_feature_implicit_dynamic.py` (chosen per-mode); sets `PYTHONPATH=src/train` | mode-multiplexer; the only script that explicitly sets `PYTHONPATH` |
| `train_ltx_feature_implicit_camera_128_4fps_fast_mac.sh` | `train_precomputed_feature_implicit_dynamic.py` | hardcoded config |
| `train_precomputed_vjepa2_1_torchhub_vitb_384.sh` | `train_precomputed_feature_implicit_dynamic.py` | hardcoded config |
| `train_wan_vace_feature_implicit_camera_128_4fps_fast_mac.sh` | `train_precomputed_feature_implicit_dynamic.py` | hardcoded config |
| `train_multicam_static_dynamic_vjepa_features.sh` | `train_multicam_precomputed_feature_implicit_dynamic.py` | hardcoded default config |

The `*build_*` and `get_camera.sh` scripts are dataset-prep, not training. The legacy three (`dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`, plus the alias) are still wired but the configs they consume are 3-5 small overfit examples. Most current research effort is on `train_video_token_implicit_dynamic.py` ± its precomputed-feature subclasses.

There is no centralized launcher; "which trainer runs" is encoded in 16 bash files. None of them set `WANDB_MODE` or other env overrides; only `train_static_dynamic_vjepa_features_ablation.sh` sets `PYTHONPATH=src/train` (the others rely on the `python src/train/...` invocation putting `src/train/` first via the script's working-directory layout).

## Config family analysis

96 train configs cluster into a small number of families with heavy duplication.

| Family | Representative | Config count | Trainer | Key variations within family |
|---|---|---|---|---|
| Gauge-fields material/screen-disk surfel + multicam variants | `local_mac_gauge_fields_material_surfel_128_16f_512el.jsonc` | ~31 | `research_experiments/gauge_fields/train.py` | `support_mode`, `num_elements`, `num_basis`, `init_*`, multicam keys, `arap_weight`/`smooth_weight` |
| Splat-baseline (static-3DGS / free-dynamic-3DGS) | `local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_static_3dgs_128_16f_2048splats.jsonc` | 3 | `research_experiments/gauge_fields/train_splat_baseline.py` | `splat_mode`, `num_splats`, `init_depth` |
| Compare encoder/conditioning matrix (16f, 128px, fast-mac, 8192 splats) | `local_mac_compare_local_video_encoder_16f_implicit_camera_128_fast_mac_8192splats.jsonc` | 11 | `train_video_token_implicit_dynamic.py` | `model.variant`, `model.video_encoder_backend`, `vjepa_*`, "strong-init" toggle |
| Time/cross-attention ablations (16f, 128px, "ablate_time_*") | `local_mac_ablate_time_crossattn1_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc` | ~10 | `train_video_token_implicit_dynamic.py` | `cross_attn_layers`, `time_*` keys, `static/dynamic_tokens` variants, 250-vs-1000-step variants |
| Static/dynamic 96/32 V-JEPA precomputed (128px, 16f, 8192 splats) | `local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_..._1000step.jsonc` | 4 | `train_precomputed_feature_implicit_dynamic.py` | `camera_clamp` flag, step count |
| Multicam DeepView 3cam (train2/test1 or 4cam train2/holdout2) | `local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc` | 3 | `train_multicam_precomputed_feature_implicit_dynamic.py` | `multicam_train_cameras`, `multicam_heldout_camera(s)`, splat count |
| F=3 unconditioned-tokens fast (200/400 step) | `local_mac_unconditioned_tokens_fast.jsonc` | 2 | `train_video_token_implicit_dynamic.py` | `train.steps` |
| F=32 unconditioned-tokens features (LN+kaiming, LN+orth, alpha) | `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc` | 4 | `train_video_token_implicit_dynamic.py` | `colorize.weight_init`, `colorize.weight_init_gain`, `colorize.pre_norm`, `feature_pca_log` |
| Local-mac scene-distinct-30 (256px, V-JEPA-ViTL) | `local_mac_scene_distinct_30_local_encoder_256_fast_mac_2048splats.jsonc` | 2 | `train_video_token_implicit_dynamic.py` | `video_encoder_backend` (`local` vs `vjepa_torchhub`) |
| Pre-baked camera (legacy) | `local_mac_overfit_prebaked_camera_*.jsonc` | 9 | `dynamicTokenGS.py` | resolution, frame count, taichi-vs-fast_mac |
| Single-image overfit (legacy) | `local_mac_overfit_single_image*.jsonc`, `local_mac_overfit_image_implicit_camera*.jsonc` | 4 | `train_camera_implicit_dynamic.py` | tile vs full |
| Single overfit video-token | `local_mac_overfit_video_token_*.jsonc`, `local_mac_overfit_video_token_implicit_camera_*.jsonc` | 5 | `train_video_token_implicit_dynamic.py` | sinusoidal-time variant, plucker variant, smoke variant |
| LTX/Wan-VACE feature precomputed (overfit) | `local_mac_overfit_ltx_feature_implicit_camera_..., local_mac_overfit_wan_vace_feature_implicit_camera_...` | 2 | `train_precomputed_feature_implicit_dynamic.py` | extractor flavor |
| One-off ablate | `local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc` | 1 | `train_video_token_implicit_dynamic.py` | – |
| Multicam 3cam RGB pyramid smoke | `local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc` | 1 | – | – |
| F32 ultimate alpha multicam | `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc` | 1 | – | – |

Within the largest "compare" family, every config is ~3.0-3.7K lines of JSONC, structurally identical for `data`/`camera`/`render`/`train`/`losses`/`logging` sections, with only 5-12 keys actually differing. Same for the time-ablation family. The "F32 features" family added `colorize`, `feature_dim=32`, and `feature_pca_log:true` simultaneously to 5 of the unconditioned-tokens configs and forgot to backport `feature_pca_log` to the feature variants without alpha — silent drift between siblings.

There is **no inheritance/composition mechanism**. Every config is a complete top-level dict.

## Dataset configs

`src/dataset_configs/*.jsonc` are inputs to `src/dataset_pipeline/*.py` CLIs, which materialize external datasets to disk under `data/`. They share **no** schema with train configs.

Schema by file (top-level fields are dataset-pipeline-script-specific):

- `multicam_val_v1_128_4fps_16f.jsonc` (consumed by `dataset_pipeline/multicam_val.py`):
  - `dataset_name` (str), `root_dir` (str), `clip_frames` (int), `fps` (float), `target_size` (int), `preview_size` (int), `preview_fps` (float), `materialize_metric_frames` (bool), `overwrite` (bool).
  - Per-source blocks: `aist`, `neural_3d_video`, `vivo`, `deepview_video`, each with `enabled` plus source-specific knobs (URLs, cameras_zip_url, scene names, source/target camera lists, scenes lists). Output: `data/multicam_val/clip_sets/<dataset_name>/manifest.jsonl`.

- `youtube_curated_spans_64_4fps_16f.jsonc` (consumed by `dataset_pipeline/youtube_curated_spans.py`):
  - `dataset_name`, `root_dir`, `source_manifests`, `defaults`, `records[]` (with `clip_id`, `url`, `start_time`, `end_time`, `start_seconds`, `end_seconds`, `notes`), `download` (yt-dlp options), `clip_dataset` (clip_frames/fps/target_size).

- `youtube_scene_distinct_30_64_4fps_16f.jsonc` (consumed by `dataset_pipeline/youtube_ingest.py`):
  - `dataset_name`, `root_dir`, `search.queries[]`, `download`, `segment` (motion/scene-cut thresholds), `clip_dataset` (target_count/train_count/test_count + fps/clip_frames/target_size).

- `neural_3d_video_seed.jsonc`, `deepview_video_seed.jsonc`, `vivo_seed.jsonc`, `ex4dgs_pretrained_val_seed.jsonc`, `youtube_high_camera_motion_seed.jsonc`, `youtube_scene_distinct_30_256_4fps_16f.jsonc`: download/extract/inventory configs for the corresponding dataset_pipeline script.

The connection to train configs is **by string**: a train config writes `data.multicam_manifest = "data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl"`, which is the path the dataset_pipeline materializes given `dataset_name = "multicam_val_v1_128_4fps_16f"`. There is no foreign-key enforcement; rebuild a dataset under a different name and your train configs silently break.

For single-clip / scene-distinct training, `build_clip_dataset.py` (run by `build_*.sh` train_scripts) writes a `manifest.jsonl` whose entries have `{clip_id, split, sequence_dir, frames_dir, frame_source: "summary_sampled", frame_count, fps, target_size, source_path, ...}`. The trainer's `load_manifest_sequence()` reads the entry and falls back to `data_cfg["frame_source"]`/`data_cfg["frames_dir"]`/`data_cfg["max_frames"]` for any key the entry doesn't override.

## Data-loading paths (which `data.*` keys feed which trainer)

| Data path | Trainer | Sample-loading function | `data.*` keys consumed |
|---|---|---|---|
| Single uncalibrated video, single clip | `Trainer.load_single_sequence_data` | `sequence_data.load_uncalibrated_sequence` (or `load_camera_sequence` if `frame_source=="camera_json"`) | `sequence_dir`, `frames_dir`, `video_path`, `frame_source`, `max_frames`, `camera_json`, `camera_image_size`, `camera_focal_mode` |
| Manifest of single-cam clips (e.g. youtube/scene-distinct) | `Trainer.load_train_sequences` → `load_manifest_sequences` → `load_manifest_sequence` per entry | same as above per entry; entry keys override `data_cfg` for `frames_dir`, `video_path`, `frame_source`, `max_frames`, `camera_json`, `camera_image_size`, `camera_focal_mode` | `manifest_path`, `split`, plus per-entry overrides |
| Eval manifest | `Trainer.load_eval_sequences` | same as train manifest | `eval_manifest_path` (falls back to `manifest_path`), `eval_split`, `eval_max_sequences` |
| Multicam from `multicam_val` manifest | `MulticamPrecomputedFeatureImplicitTrainer.load_train_sequences` → `load_multicam_video_bundle` | `multicam_video_data.load_multicam_video_bundle` | `multicam_manifest`, `multicam_split`, `multicam_sample_id`, `multicam_sample_index`, `multicam_train_cameras`, `multicam_heldout_cameras`, `multicam_heldout_camera`, `multicam_anchor_camera`, `multicam_condition_camera`, `frame_indices`, `max_frames`. Plus `camera.rig_init` to pick the per-dataset adapter (`deepview`, `aist`, `neural_3d_video`, `vivo`, `orthogonal_origin`). |

The same `data` dict therefore plays three different roles depending on which trainer reads it; many of the keys are inert in the other modes. Configs simply set everything in the union, plus `null`s for the unused half (e.g. multicam configs set `video_path: null`, `frames_dir: null`).

## Smells / problems / cleanup candidates

1. **`arch` field is dead documentation.** Nothing in `src/train/` reads it. Trainer choice is by which `.sh` script you run. Two configs with the same `arch` can route to different trainers (e.g. `tokengs_video_implicit_camera` may be launched by 6 different bash scripts). Configs with the same `model.variant` can target three different trainers (single-cam Trainer, PrecomputedFeatureImplicitTrainer, MulticamPrecomputedFeatureImplicitTrainer). The actual trainer-bearing field is the launcher path — it isn't in the config at all.

2. **30+ required-but-not-defaulted keys.** `MODEL_OPTION_DEFAULTS` defines ~50 keys but the trainer's `build_model_from_config` does direct dict lookups for `bottleneck_dim`, `num_heads`, `mlp_ratio`, `tubelet_size_t`, `patch_compression`, `encoder_self_attn_layers`, `bottleneck_self_attn_layers`, `cross_attn_layers` etc. with no fallback. Several model variants then `**_unused`-swallow them. So a "knobless" variant like `unconditioned_tokens` still must declare `bottleneck_dim:192, num_heads:8, mlp_ratio:4.0, tubelet_size_t:4, patch_compression:16, encoder_self_attn_layers:1, bottleneck_self_attn_layers:2, cross_attn_layers:0` — none of which it uses. The configs even document this explicitly:
   > "bottleneck_dim / num_heads / mlp_ratio / tubelet_size_t / patch_compression / encoder_self_attn_layers / bottleneck_self_attn_layers / cross_attn_layers are required by the trainer's build_model_from_config (direct dict lookup, not in MODEL_OPTION_DEFAULTS), but UnconditionedTokenGSImplicitCamera swallows them via **_unused — so they're inert at the model layer."
   The same applies to `train.steps/lr/amp/recon_backward_strategy/temporal_microbatch_size`, `render.renderer/render_size/auto_dense_limit/tile_size/bound_scale/alpha_threshold`, all of `logging.*`, `data.frames_dir/video_path/frame_source/max_frames`. These should either land in a defaults dict or the variant should declare via the type system that it ignores them.

3. **Two backgrounds in `render.fast_mac`.** `background` (3-vector, RGB) is the legacy F=3 path; `feature_background` (scalar broadcast across F channels) is the F!=3 path. Configs that use `feature_dim != 3` must add both. Five configs add `feature_background`; the others have `feature_dim=3` and don't. The renderer uses one or the other depending on what's present, but `resolve_config` doesn't validate the pair against `model.feature_dim`.

4. **Three layered `resolve_config()` chains.** `MulticamPrecomputedFeatureImplicitTrainer.resolve_config` calls `super().resolve_config()` (= `PrecomputedFeatureImplicitTrainer.resolve_config`), which calls `super().resolve_config()` (= the bare `Trainer.resolve_config`, which is a thin wrapper around the module-level `resolve_config`). `PrecomputedFeatureImplicitTrainer` mutates `raw["model"]["video_encoder_backend"]` to "precomputed" before delegating. Adding a new field requires editing whichever subclass is "responsible" for it, with subtle ordering constraints.

5. **`colorize` section is entirely `.get(key, default)` at the use site.** No declared defaults, no validation in `resolve_config`. The hard-coded defaults (`activation="sigmoid"`, `weight_init="kaiming"`, `weight_init_gain=1.0`, `view_condition="none"`, `detach_view_condition=True`) live inline in `Trainer.__init__` (lines 1014–1023). This violates the project's stated "config style" rule that defaults should live in one place.

6. **`logging.feature_pca_log` is read with `.get(..., False)` even though it's a hard requirement.** Setting `feature_pca_log=True` with `feature_dim=3` raises `ValueError`. But the key lives in only 5 of 96 configs; for the rest it implicitly defaults to False. Same drift pattern as the F=32 family — it was added late and cargo-culted unevenly.

7. **`render` section has inline fixups instead of a `RENDER_OPTION_DEFAULTS` block.** Lines 333-348 of `train_video_token_implicit_dynamic.py` patch `near_plane`, `camera_projection`, `fast_mac` one key at a time with `if "x" not in cfg["render"]:`. Same anti-pattern flagged elsewhere in the project's CLAUDE.md (the "scattered .get(default) at use sites" rule).

8. **96 configs, many ~95% identical.** No inheritance. New ablation = copy a representative + tweak 5 fields. Bug surface: when a key is added to one file in a family (e.g. `feature_pca_log`, `feature_background`), siblings drift. The agent_notes record several silent disagreements between sibling configs that took a session to debug.

9. **`data` schema is a union over modes.** `sequence_dir`, `frames_dir`, `video_path`, `frame_source`, `manifest_path`, `multicam_manifest`, `multicam_sample_id`, `frame_indices`, `camera_json` are all in the same flat dict, mostly `null` for the mode you aren't using. This makes "is this config a single-cam or multicam config?" a runtime question rather than a type-level one.

10. **Two parallel-but-incompatible "splat" trainers.** `train_video_token_implicit_dynamic.py` and `research_experiments/gauge_fields/train.py` both consume "data + model + camera + render + train + losses + logging" configs but with totally different field schemas. They have separate `MODEL_DEFAULTS`/`DATA_DEFAULTS`, separate frame-source vocabularies (gauge_fields adds `multicam_val` as a `frame_source` value the main trainer doesn't accept), separate loss families. Roughly a third of `src/train_configs/` targets the gauge-fields trainer. There is no signpost in the directory layout: a `gauge_fields_*.jsonc` config and a `local_mac_ablate_*.jsonc` config sit side by side in the same `src/train_configs/` directory.

11. **`use_static_dynamic_split` is a derived flag stored on the config.** `resolve_config` writes `cfg["model"]["use_static_dynamic_split"]` based on whether `static_tokens` or `dynamic_tokens` was set. Consumers then check the derived flag instead of asking "are these set?". This is fine as a normalization pass but the derived flag also leaks into `serialize_config_value(self.cfg)` which goes to W&B — so dashboards include a key the user never wrote.

12. **`PYTHONPATH=src/train`-vs-not is inconsistent across launchers.** Only one launcher script (`train_static_dynamic_vjepa_features_ablation.sh`) sets it. Others rely on Python's behavior of prepending the script's parent directory to `sys.path`, which works because `src/train/train_video_token_implicit_dynamic.py` and `src/train/multicam_video_data.py` live in the same dir. The sub-modules `src/train/gs_models/`, `src/train/sequence_data.py` are imported as bare names (`from sequence_data import ...`, `from gs_models import ...`), assuming the script's parent is the working dir. Move the entry point and you break imports.

13. **Optional-flag drift in `colorize`**: 7 configs declare it, 19 set `feature_dim != 3`-implying-need (only 7 have feature_dim=32+colorize). Some configs inadvertently kept old `colorize` from a copy when they actually want `feature_dim=3` — the trainer doesn't reject that, it just unused-ly builds the FeatureToColor module.

## Open questions for proposers

1. Should the trainer dispatch be explicit in the config? E.g. a top-level `trainer: "multicam_precomputed_feature_implicit_camera"` field that the launcher script reads, eliminating both the `arch`-is-unread footgun and the "which `.sh` do I run" lookup.

2. Should `MODEL_OPTION_DEFAULTS` be reorganized so that "required-but-not-defaulted" model-arch keys (the 8 `**_unused`-swallowed ones plus the 6 always-required `size/train_frame_count/tokens/model_dim/gaussians_per_token/scene_extent`) are one of: (a) explicitly typed as required and missing-here-means-error-here, (b) given sensible defaults so that a `unconditioned_tokens` config doesn't have to invent fake values for `cross_attn_layers`?

3. Should config inheritance / merge be added? Concrete proposal sketches: `extends: "common/F3_unconditioned_tokens.jsonc"` merged at load time; or per-section default files (`src/train_configs/_defaults/render_fast_mac.jsonc`); or a small Python-side `make_config()` family-builder that emits the full dict from a small overrides dict. The win is biggest in the "compare matrix" and "ablate time" families.

4. Should the `data` schema be split into `single_cam` / `manifest` / `multicam` discriminated-union variants, instead of a flat dict where 60% of the keys are null in any given mode?

5. Should `gauge_fields` and `splat_baseline` configs live in a separate directory (`src/train_configs/gauge_fields/`, `src/train_configs/splat_baseline/`) to mirror the trainer split, since they have entirely separate schemas? Right now `src/train_configs/` is structurally three different config languages sharing a directory.

6. Should `colorize`, `features`, `export` be hoisted into the "main" `apply_defaults` chain at the appropriate trainer subclass level, eliminating the `.get(key, default)` pattern at the use sites? (The project AGENTS.md explicitly bans this pattern: "Repeated `.get(..., default)` at use sites is a smell unless the value is truly optional and `None` has semantic meaning.")

7. Should `arch` be either (a) made the dispatch field (and `trainer_class_for_config` honored it) or (b) deleted from every config since it is dead? The current state — 16 distinct `arch` strings, 5 of which (e.g. `tokengs`, `tokengs_image_implicit_camera`, `tokengs_single_image`) appear nowhere in code — is the worst of both worlds.

agent_notes/apr_30th_clean/investigator_04_config_data_path.md
