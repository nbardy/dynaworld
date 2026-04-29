# Investigator 01 — Trainer Landscape Audit (Apr 30th)

> Scope: every trainer file under `src/train/`, the launcher scripts under
> `src/train_scripts/`, and config-to-trainer dispatch in `src/train_configs/`.
> This is a state-of-the-codebase audit only — no recommendations, no edits.

## TL;DR

- The active trainer landscape is **one 2 072-line monolith**
  (`train_video_token_implicit_dynamic.py`) with **two trainer classes inside it
  in a single file** (`Trainer`, `KnownCameraTrainer`) and **two thin
  subclasses** in sibling files (`PrecomputedFeatureImplicitTrainer`,
  `MulticamPrecomputedFeatureImplicitTrainer`). Plus one outright
  backward-compat alias (`LTXFeatureImplicitTrainer`).
- There are **three other "trainer" files** still wired by configs:
  `dynamicTokenGS.py` (legacy prebaked-camera path, 731 lines, function-style),
  `train_camera_implicit_dynamic.py` (image-encoder implicit-camera baseline,
  417 lines, function-style), and `tokenGS.py` (single-image overfit, 146
  lines). None of these inherit from `Trainer`. Each duplicates its own
  optimizer / sampling / render / log loop.
- The **alpha-aware composition + per-step random background** that the most
  recent loose notes call load-bearing exists **only in
  `Trainer.recon_backward` and `Trainer.initial_step_result`**. It is **completely
  absent from `MulticamPrecomputedFeatureImplicitTrainer.multicam_recon_loss`
  and `MulticamPrecomputedFeatureImplicitTrainer.initial_step_result`**, and
  the multicam path also **drops the alpha tuple silently** by storing
  `render_clip_sequence(...)` (which returns `(features, alpha)`) into a
  tensor-typed variable named `rendered`. Same shape of bug appears in
  `KnownCameraTrainer.initial_step_result` (line 1897). This is the load-bearing
  duplication the proposers will need to address.
- The `gauge_fields_material_surfel` / `splat_baseline_*_3dgs` configs do
  **not** dispatch to anything in `src/train/` — they are served by trainers
  under `research_experiments/gauge_fields/`, which are out of scope for this
  audit but explain why those configs do not appear in any `src/train_scripts/`
  shell launcher.
- Two trainer files are pure shims (`train_image_encoder_implicit_camera_baseline.py`
  and `train_camera_implict_dynamic.py` — note the typo spelling) that just
  re-export `train_camera_implicit_dynamic.main`. The typo file is dead.

## Trainer inventory table

| File | Top-level class | Inherits from | Dispatched-to by which configs (by `arch`) | Status | Methods overridden | Methods uniquely added |
|---|---|---|---|---|---|---|
| `src/train/train_video_token_implicit_dynamic.py` (2 072 lines) | `Trainer` (L944) | (none — root) | `tokengs`, `tokengs_video_implicit_camera` (29 configs) | active, primary | (root class — see "Methods uniquely added") | `__init__`, `on_sequences_loaded`, `colorize_features`, `load_single_sequence_data`, `load_train_sequences`, `load_eval_sequences`, `validate_train_sequences`, `autocast_context`, `sample_sequence`, `sample_clip`, `export_window_indices`, `export_browser_bundle`, `model_input_for_clip`, `forward_clip`, `compute_camera_losses`, `build_camera_loss`, `build_bank_rate_loss`, `temporal_recon_chunk_size`, `recon_backward`, `render_decoded_clip`, `initial_step_result`, `step`, `camera_metrics`, `progress_message`, `should_log_scalars`, `should_log_images`, `should_log_videos`, `scalar_payload`, `render_preview_image`, `render_full_sequence`, `validation_video_payload`, `val_log`, `run` |
| `src/train/train_video_token_implicit_dynamic.py` (same file, L1815) | `KnownCameraTrainer` | `Trainer` | `tokengs_video_known_camera` (1 config: `local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc`) | active, single-config | `validate_train_sequences` (calls super, adds camera-presence check), `sample_clip` (fully replaces — 4-tuple instead of 3-tuple), `step` (fully replaces), `initial_step_result` (fully replaces), `render_full_sequence` (fully replaces), `run` (fully replaces) | `forward_known_clip` |
| `src/train/train_precomputed_feature_implicit_dynamic.py` (165 lines) | `PrecomputedFeatureImplicitTrainer` (L55) | `Trainer` (`VideoTokenImplicitTrainer` alias) | `precomputed_feature_implicit_camera` (4 configs), `wan_vace_feature_implicit_camera` (1 config) | active | `resolve_config` (calls super, adds `FEATURE_OPTION_DEFAULTS` and validates `video_encoder_backend`), `on_sequences_loaded` (fully replaces — builds `VideoFeatureCache` and prebakes), `model_input_for_clip` (fully replaces — returns precomputed features), `run` (calls super, prepends print) | (none) |
| `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` (386 lines, **untracked in git**) | `MulticamPrecomputedFeatureImplicitTrainer` (L73) | `PrecomputedFeatureImplicitTrainer` | `multicam_precomputed_feature_implicit_camera` (4 configs) | active, fresh | `resolve_config` (calls super, adds 3 default sets), `__init__` (calls super, adds rig param group), `load_train_sequences` (fully replaces — loads multicam bundle), `load_eval_sequences` (fully replaces — returns `[]`), `on_sequences_loaded` (calls super, builds `LearnableCameraRig`), `initial_step_result` (fully replaces — multicam path), `step` (fully replaces — multicam path), `scalar_payload` (calls super, adds rig metrics), `export_browser_bundle` (fully replaces — raises) | `sample_views`, `sample_multicam_clip`, `_decode_clip`, `render_view_clip`, `multicam_recon_loss`, `rig_regularization_loss`, `render_full_external_views`, `validation_video_payload` (also overrides parent) |
| `src/train/train_ltx_feature_implicit_dynamic.py` (32 lines) | `LTXFeatureImplicitTrainer` (L11) | `PrecomputedFeatureImplicitTrainer` | `ltx_feature_implicit_camera` (1 config) | backward-compat alias only | (none — empty body) | (none) |
| `src/train/train_camera_implicit_dynamic.py` (417 lines) | none — function-based `run_training` | n/a | `tokengs_image_implicit_camera` (2 configs) | active, parallel codebase | n/a | `LOSS_OPTION_DEFAULTS`, `resolve_config`, `load_sequence_data`, `pick_renderer_mode`, `build_model_from_config`, `render_implicit_frame`, `eval_metric_payload`, `render_full_sequence`, `run_training`, `main` |
| `src/train/train_image_encoder_implicit_camera_baseline.py` (4 lines) | none — shim | n/a | (none — invoked manually) | shim, hardcoded config | n/a | `from train_camera_implicit_dynamic import main` |
| `src/train/train_camera_implict_dynamic.py` (4 lines, **typo**) | none — shim | n/a | (none — typo of `implicit`) | dead | n/a | `from train_camera_implicit_dynamic import main` |
| `src/train/dynamicTokenGS.py` (731 lines) | none — function-based `run_training` | n/a | `tokengs_prebaked_camera`, `tokengs_prebaked_camera_tiled` (11 configs total) | active legacy (prebaked-camera path) | n/a | `pick_device`, `resolve_sequence_dir`, `normalize_lr_schedule`, `normalize_optimizer_config`, `normalize_loss_config`, `normalize_render_config`, `resolve_config`, `resolve_camera_json_path`, `load_sequence_data`, `pick_renderer_mode`, `learning_rate_for_step`, `set_optimizer_lr`, `_weight_decay_exempt_parameter`, `build_optimizer_param_groups`, `build_optimizer`, `build_model_from_config`, `print_key_values`, `gaussian_sequence_nonfinite_counts`, `raise_for_nonfinite_decoded`, `render_one_frame`, `render_frame_batch`, `render_full_sequence`, `run_training`, `main`. Also exports `pick_device`, `configure_fast_attn` (re-export from `fast_attn`), `fast_attn_context`, `select_window_indices` (re-export from `sequence_data`) — all four still imported by `train_camera_implicit_dynamic.py` and `train_video_token_implicit_dynamic.py` |
| `src/train/dynamicTokenGS_tiled.py` (4 lines) | none — shim | n/a | (invokes `dynamicTokenGS.main`, hardcoded `local_mac_overfit_prebaked_camera_tiled.jsonc`) | shim | n/a | `from dynamicTokenGS import main` |
| `src/train/dynamicTokenGS_shared.py` (4 lines) | none — shim | n/a | (none — re-export only) | shim | n/a | `from gs_models import DynamicTokenGS; from image_utils import fetch_image` |
| `src/train/tokenGS.py` (146 lines) | none — function-based `run_training` | n/a | `tokengs_single_image` (1 config) | active legacy (single-image overfit) | n/a | `pick_device`, `resolve_config`, `pick_renderer_mode`, `render_single_frame`, `run_training`, `main` |
| `src/train/tokenGS_tiled.py` (12 lines) | none — shim | n/a | `tokengs_single_image_tiled` (1 config; default-arg fallback) | shim | n/a | `from tokenGS import main` |

(Both `dynamicTokenGS.py` and `train_camera_implicit_dynamic.py` re-import
several symbols from each other and from `dynamicTokenGS` — see "Trainer-by-trainer
notes" for the cross-import topology.)

## Class hierarchy diagram

```
Trainer  (src/train/train_video_token_implicit_dynamic.py:944)          [class]
├── KnownCameraTrainer  (same file:1815)                                [class]
└── PrecomputedFeatureImplicitTrainer
        (src/train/train_precomputed_feature_implicit_dynamic.py:55)    [class]
    ├── MulticamPrecomputedFeatureImplicitTrainer
            (src/train/train_multicam_precomputed_feature_implicit_dynamic.py:73)
                                                                         [class, untracked]
    └── LTXFeatureImplicitTrainer
            (src/train/train_ltx_feature_implicit_dynamic.py:11)         [empty body, alias]

(no inheritance — function-style trainers)
run_training  (src/train/dynamicTokenGS.py:453)                         [function]
run_training  (src/train/train_camera_implicit_dynamic.py:219)          [function]
run_training  (src/train/tokenGS.py:60)                                 [function]

shims (no behavior, hardcoded config or re-export):
- src/train/train_image_encoder_implicit_camera_baseline.py
- src/train/train_camera_implict_dynamic.py    (note typo)
- src/train/dynamicTokenGS_tiled.py
- src/train/dynamicTokenGS_shared.py
- src/train/tokenGS_tiled.py
- src/train/tokenGS_shared.py

Module-level dispatcher:
trainer_class_for_config (train_video_token_implicit_dynamic.py:2048)
  - if model.variant in {"known_camera", "known_camera_video_token"}: KnownCameraTrainer
  - else: Trainer
  - (PrecomputedFeatureImplicitTrainer / MulticamPrecomputedFeatureImplicitTrainer
    / LTXFeatureImplicitTrainer are NOT chosen by this dispatcher; they are
    chosen by `python <trainer_module>.py` invocation in the launcher script.)
```

## What's duplicated

For each major training-loop concept, here are the trainers that have their
own copy.

### 1. `resolve_config` / config defaulting

- `train_video_token_implicit_dynamic.resolve_config` (L174–370): the canonical
  one. Calls `resolved_config`, applies `DATA_OPTION_DEFAULTS`,
  `MODEL_OPTION_DEFAULTS`, `CAMERA_OPTION_DEFAULTS`, validates many enum-like
  fields, normalizes static/dynamic split, normalizes `render.fast_mac` /
  `render.camera_projection`, normalizes `export`. ~200 lines.
- `train_precomputed_feature_implicit_dynamic.PrecomputedFeatureImplicitTrainer.resolve_config`
  (L57–103): calls super, then applies `FEATURE_OPTION_DEFAULTS` and validates
  `video_encoder_backend in {precomputed, precomputed_ltx}`.
- `train_multicam_precomputed_feature_implicit_dynamic.MulticamPrecomputedFeatureImplicitTrainer.resolve_config`
  (L75–103): calls super, then applies `DATA_MULTICAM_DEFAULTS`,
  `CAMERA_RIG_DEFAULTS`, `TRAIN_MULTICAM_DEFAULTS`.
- `train_camera_implicit_dynamic.resolve_config` (L41–66): module-level
  function. Has its own private `LOSS_OPTION_DEFAULTS` (also defined in
  `train_video_token_implicit_dynamic.py` and again in `dynamicTokenGS.py` —
  three copies). Validates `model.variant in {joint_attention, separated_camera}`
  (different vocabulary from the parent file).
- `dynamicTokenGS.resolve_config` (L193–218): module-level. Its own
  `MODEL_OPTION_DEFAULTS`, `RENDER_OPTION_DEFAULTS`, `OPTIMIZER_DEFAULTS`,
  `TRAIN_OPTION_DEFAULTS`, `LOGGING_OPTION_DEFAULTS`, `LOSS_OPTION_DEFAULTS` —
  six dict literals, all duplicated subsets of the implicit trainer's
  versions. Has unique `lr_schedule` / optimizer-param-group / `clip_grad_norm`
  knobs that the implicit trainer doesn't expose at all.
- `tokenGS.resolve_config` (L26): trivial passthrough.

### 2. `pick_device`

- `dynamicTokenGS.pick_device` (L135): the canonical implementation.
- `tokenGS.pick_device` (L22): same body, copy-pasted.
- `train_camera_implicit_dynamic.py` and `train_video_token_implicit_dynamic.py`
  both `from dynamicTokenGS import pick_device` — they re-use it. So the
  duplication is between `dynamicTokenGS` and `tokenGS`.

### 3. `pick_renderer_mode_from_config`

- `train_video_token_implicit_dynamic.pick_renderer_mode_from_config` (L373) —
  uses `render_cfg["render_size"]`.
- `train_camera_implicit_dynamic.pick_renderer_mode` (L89) — uses
  `model_cfg["size"]`.
- `dynamicTokenGS.pick_renderer_mode` (L245) — uses `model_cfg["size"]`.
- `tokenGS.pick_renderer_mode` (L30) — uses `model_cfg["size"]`.

Four copies, one of them deliberately diverges on a `render_size` vs `size`
distinction.

### 4. `build_model_from_config`

- `train_video_token_implicit_dynamic.build_model_from_config` (L809–941): the
  big variant-dispatch (10 model classes, ~130 lines).
- `train_camera_implicit_dynamic.build_model_from_config` (L103–112): small
  pinned-variant dispatch (2 classes: `DynamicTokenGSImplicitCamera` vs
  `DynamicTokenGSSeparatedImplicitCamera`).
- `dynamicTokenGS.build_model_from_config` (L345–350): single class
  (`DynamicTokenGS`).
- `tokenGS.run_training` builds `TokenGS(...)` inline.

### 5. Sample clip / batch indices

Five distinct flavors:

- `Trainer.sample_clip` (L1155–1163) — implicit-camera + video token, one
  sequence, one clip:
```python
def sample_clip(self):
    sequence_data = self.sample_sequence()
    clip_indices = select_window_indices(sequence_data.frame_count,
                                         self.model_cfg["train_frame_count"],
                                         device=self.device)
    clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
    return sequence_data, clip_frames, clip_times
```
- `KnownCameraTrainer.sample_clip` (L1823–1834) — adds `clip_cameras` from the
  precomputed sequence cameras as a fourth tuple element.
- `MulticamPrecomputedFeatureImplicitTrainer.sample_multicam_clip` (L163–171) —
  one sequence, one clip, plus a list of view indices via `sample_views()`.
- `train_camera_implicit_dynamic.run_training` inlines `select_window_indices`
  at L280 with `frames_per_step` instead of `train_frame_count`.
- `dynamicTokenGS.run_training` inlines `select_window_indices` at L542 with
  `frames_per_step`.

### 6. Forward / decode

- `Trainer.forward_clip` (L1220–1222):
```python
def forward_clip(self, model_input, clip_times):
    with fast_attn_context(self.device), self.autocast_context():
        return self.model(model_input, decode_times=clip_times)
```
- `KnownCameraTrainer.forward_known_clip` (L1836–1843) — adds
  `cameras=clip_cameras`.
- `MulticamPrecomputedFeatureImplicitTrainer._decode_clip` (L173–175) —
  calls `model_input_for_clip` then `forward_clip`. Just routing.
- `train_camera_implicit_dynamic.py` inlines a `model(batch_frames,
  frame_times=batch_times)` call at L286 (no `decode_times` keyword — different
  model API entirely).
- `dynamicTokenGS.py` inlines `model(batch_frames, camera=batch_cameras,
  frame_times=batch_times)` at L550.

### 7. Render + colorize + composite (the load-bearing alpha-aware path)

This is THE duplication question. Five different shapes:

- `Trainer.recon_backward` (L1309–1372) — **alpha-aware path with random per-step bg**:
```python
random_bg = torch.rand(3, device=clip_frames.device, dtype=clip_frames.dtype).view(1, 3, 1, 1)
for chunk_start in range(0, frame_count, chunk_size):
    chunk_features, chunk_alpha = render_clip_sequence(...)
    if self.colorize is not None:
        splat_rgb = self.colorize_features(chunk_features, ...)
        if chunk_alpha is not None:
            alpha_expanded = chunk_alpha.unsqueeze(1)
            chunk_renders = alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * random_bg
        else:
            chunk_renders = splat_rgb
    else:
        chunk_renders = chunk_features
```
  Random bg sampled once per step, broadcast across all chunks/frames/pixels.
  This is the canonical "no degenerate manifold" trick.
- `Trainer.initial_step_result` (L1387–1437) — **same idea but composites
  against a hardcoded white bg of value `1.0`** (line 1412). Subtle: it's
  alpha-aware but does NOT use `random_bg`. Unclear if this is intentional or
  drift.
- `Trainer.render_full_sequence` (L1549–1647) — same hardcoded white bg of
  `1.0` (line 1604). Validation path; uses white deliberately.
- `KnownCameraTrainer.render_full_sequence` (L1925–2009) — same hardcoded
  white bg of `1.0` (line 1974). Copy-paste of the parent path with the
  known-camera input.
- `MulticamPrecomputedFeatureImplicitTrainer.multicam_recon_loss` (L189–205) —
  **NO ALPHA HANDLING AT ALL**:
```python
for view in views:
    rendered = self.render_view_clip(decoded, view=int(view), clip_indices=clip_indices)
    target = resize_images(self.multicam_bundle.train_frames[int(view), clip_indices], self.render_size)
    recon_loss = recon_loss + reconstruction_loss_per_image(rendered, target, self.loss_cfg).mean()
```
  Worse: `render_view_clip` calls `render_clip_sequence` whose return type is
  `tuple[torch.Tensor, torch.Tensor | None]` — so `rendered` here is a TUPLE,
  not a tensor. `reconstruction_loss_per_image(rendered, target, ...)` where
  `rendered` is a 2-tuple will crash. This trainer is broken on any execution
  path that exercises `multicam_recon_loss` unless something is upstream
  monkey-patching the return type. Same shape of mismatch in
  `MulticamPrecomputedFeatureImplicitTrainer.initial_step_result` (L222–227)
  and `MulticamPrecomputedFeatureImplicitTrainer.render_full_external_views`
  (L294–310). No `colorize`, no `random_bg`, no alpha composite. The multicam
  trainer would either crash on first `step()` or accidentally work only when
  the renderer mode is non-`fast_mac` AND something silently unwraps the
  tuple — neither is a stable contract.
- `KnownCameraTrainer.initial_step_result` (L1879–1923) — has the same
  tuple/tensor mismatch at L1897:
```python
rendered_features = self.render_decoded_clip(decoded)   # returns (Tensor, Tensor|None)
preview_features = rendered_features[0].detach() ...    # treats tuple[0] as a tensor
```
  This would fail silently or crash at startup of the known-camera config.
- `train_camera_implicit_dynamic.py` (L292–298) — inlines
  `render_implicit_frame` per local index, no alpha, no colorize.
- `dynamicTokenGS.py` (L553–567) — inlines `render_frame_batch` with optional
  `return_aux`, no alpha, no colorize.

### 8. Camera regularization losses

- `Trainer.compute_camera_losses` / `build_camera_loss` (L1224–1264) — three
  weighted terms (motion / temporal / global).
- `MulticamPrecomputedFeatureImplicitTrainer` deliberately replaces them with
  zeros + a separate `rig_regularization_loss()` (L207–208).
- `train_camera_implicit_dynamic.run_training` (L304–321) — inlines an
  identical-shape camera-loss computation.
- `KnownCameraTrainer.step` (L1845–1877) zeros these out (cameras are known).
- `dynamicTokenGS.py` has no camera regularization (camera is prebaked and
  fully fixed).

### 9. Backward / optimizer step

- `Trainer.recon_backward` does **per-chunk backward with `retain_graph=True`
  except on last chunk** (L1370). One optimizer.step at the end of `Trainer.step`
  (L1460).
- `KnownCameraTrainer.step` reuses `Trainer.recon_backward` (per-chunk
  backward).
- `MulticamPrecomputedFeatureImplicitTrainer.step` does a single
  `loss.backward()` (L261) followed by `self.optimizer.step()` — fully
  batched, ignores `recon_backward_strategy`.
- `train_camera_implicit_dynamic.run_training` does a single `loss.backward()`
  (L327) — also ignores `recon_backward_strategy`.
- `dynamicTokenGS.run_training` does a single `loss.backward()` plus optional
  gradient-clip (L601, L625–638) — has `clip_grad_norm` knob the implicit
  trainer doesn't honor.

### 10. Optimizer construction

- `Trainer.__init__` (L1037–1041) — flat `torch.optim.Adam(model + colorize
  parameters, lr=train_cfg["lr"], fused=...)`.
- `MulticamPrecomputedFeatureImplicitTrainer.__init__` (L106–114) — calls
  super then `optimizer.add_param_group` for the rig at a possibly-different
  LR.
- `train_camera_implicit_dynamic.run_training` (L255) — flat `Adam`, single
  param group, no colorize.
- `dynamicTokenGS.build_optimizer` (L318–342) — variant: `adam` or `adamw`,
  weight-decay-aware param groups, fused fallback. Most sophisticated path
  but only the prebaked-camera trainer uses it.

### 11. Validation video payload

- `Trainer.validation_video_payload` (L1649–1737) — multi-sequence average,
  builds `Render_GT_Video`, optional `Alpha_Mask_Video`, optional
  `Feature_PCA_Video`, optional `Render_Composite_Video` (concatenated columns),
  per-sequence `Eval/*` metric stats.
- `MulticamPrecomputedFeatureImplicitTrainer.validation_video_payload`
  (L320–361) — fully replaces. Per-train-view + per-heldout-view metrics
  (`TrainView{i}/Eval/*`, `Heldout{i}_{name}/Eval/*`), per-view rendered video,
  per-view GT video. **Does not produce `Alpha_Mask_Video`,
  `Feature_PCA_Video`, or `Render_Composite_Video` — those panels do not
  exist for any multicam config.**
- `train_camera_implicit_dynamic.run_training` inlines a
  `build_validation_video_payload(rendered_sequence, gt_sequence,
  sequence_data.video_fps)` call (L380–386) — no alpha / PCA / composite at
  all.
- `dynamicTokenGS.run_training` inlines the equivalent at L700–709, plus
  hand-rolled `Eval/L1`, `Eval/MSE`, `Eval/SSIM`, `Eval/DSSIM`, `Eval/Loss`,
  `Eval/PSNR` — three trainers compute these metrics, all in slightly
  different spellings.

### 12. Image preview

- `Trainer.render_preview_image` (L1543–1547) — small wrapper around
  `make_preview_image`.
- `train_camera_implicit_dynamic.py` and `dynamicTokenGS.py` inline
  `make_preview_image(batch_frames[0], renders[0], caption=...)` directly in
  `run_training`.
- `tokenGS.py` does the same inline.

### 13. WandB logging cadence (`should_log_scalars/images/videos`)

- `Trainer.should_log_scalars/images/videos` (L1499–1512) — three small
  methods.
- All other trainers inline the same `step % max(1, logging_cfg["log_every"])
  == 0 or (always_log_last_step and step == steps)` pattern, three times each,
  per trainer. The pattern is duplicated **at least four times** across the
  trainer files.

## What's NOT duplicated and is legitimately different

Cases where the trainers genuinely diverge in shape and unification would do
harm:

1. **Multicam vs single-cam sampling.** Single-cam trainers iterate one clip,
   one view. The multicam trainer iterates one clip, multiple views, with
   an explicit `train_views_per_step` knob. The view loop in
   `multicam_recon_loss` is fundamentally different from the single-cam
   reconstruction path because it needs to render each view separately
   (different camera per view) but share the decoded `GaussianSequence`. A
   shared "render+loss" abstraction would either need to lift "iterate views"
   to the shared layer or accept a strategy plug-in.

2. **Heldout views.** Only the multicam trainer has the concept of heldout
   cameras (`render_full_external_views`, `Heldout{i}_{name}/Eval/*` metrics).
   The single-cam trainers eval on the *same* train sequence (or eval-split
   manifest) — there is no per-camera-held-out concept.

3. **Camera rig parameters.** Only the multicam trainer adds a separate
   parameter group to the optimizer (`LearnableCameraRig`), and only the
   multicam trainer has `Rig/RegularizationWeight` in its scalar payload.

4. **Precomputed-feature loading + caching vs live encoder forward.** The
   `PrecomputedFeatureImplicitTrainer` overrides `model_input_for_clip` to
   return cached features, and `on_sequences_loaded` to prebake the cache
   and release the extractor. Single-cam live-encoder training does not have
   any analogue. This is genuinely a different lifecycle.

5. **Known-camera vs implicit-camera gradient paths.** The
   `KnownCameraTrainer` zeroes out `camera_motion_loss`, `camera_temporal_loss`,
   `camera_global_loss` — all of these exist as zero-tensors in the
   `StepResult`. There is no camera state to regularize when cameras are
   precomputed. The implicit trainer needs them. Sharing a step body would
   require either dummy zero-weight terms or an explicit "no camera reg"
   branch.

6. **Prebaked-camera (`dynamicTokenGS.py`) vs the rest.** The legacy file
   uses a fundamentally different model API: `model(batch_frames,
   camera=batch_cameras, frame_times=batch_times)` (camera is a positional
   prebaked input). The implicit-camera model API uses `decode_times=...`. The
   model classes are different (`DynamicTokenGS` vs `DynamicVideoTokenGS*`).
   `dynamicTokenGS.py` also has the only `clip_grad_norm`, `lr_schedule`, and
   weight-decay-aware optimizer in the codebase. This is legitimately a
   different shape, but it is also the only trainer that uses any of those
   features — the implicit trainers chose not to import them.

7. **Single-image overfit (`tokenGS.py`).** Loads one PNG, no
   `SequenceData`, no temporal anything. Tiny, special-purpose, hardcoded
   `default_camera`. The shape of "one image, one camera, one optimizer" is
   intrinsically different from the temporal trainers.

8. **Image-encoder (per-frame) implicit-camera baseline
   (`train_camera_implicit_dynamic.py`).** Conceptually different in that the
   model is called per-batch with `frame_times` (no clip-level cross-frame
   attention) and uses `DynamicTokenGSImplicitCamera /
   ...SeparatedImplicitCamera`, neither of which is in
   `train_video_token_implicit_dynamic.build_model_from_config`. This is the
   "image encoder baseline before video tokens" path. It is structurally
   different from the video-token implicit path even though they share the
   "implicit camera" idea.

## Trainer-by-trainer notes

### `src/train/train_video_token_implicit_dynamic.py` — `Trainer` + `KnownCameraTrainer`

The 2 072-line monolith. Module-level helpers up to L943 (resolve_config,
manifest loaders, `prepare_clip`, `viewport_cameras`,
`colorize_view_dirs_for_features`, `gaussian_sequence_slice`,
`render_clip_sequence`, `eval_metric_payload`, `temporal_similarity_payload`,
`init_decoded_frame_buffers`, `fill_decoded_frame_buffers`,
`decoded_temporal_payload`, `camera_temporal_payload`,
`render_full_sequence` — module-level legacy version, then
`build_model_from_config`).

Two trainer classes:
- `Trainer` (L944) — the implicit-camera video-token path. Owns the alpha-aware
  composition, the random-bg trick (only inside `recon_backward`), the colorize
  MLP wiring, the W&B side-by-side composite video panel, the static/dynamic
  bank-rate losses, and the export hook.
- `KnownCameraTrainer(Trainer)` (L1815) — replaces `sample_clip` to add
  `clip_cameras`, replaces `step`, `initial_step_result`, `render_full_sequence`,
  and `run`. Notably does NOT pick up the alpha-aware random-bg recon path —
  its `step` calls `recon_backward` (so it gets random bg) but its
  `initial_step_result` uses a non-tuple-unpacked render (line 1897) which is
  a latent bug.

`trainer_class_for_config` (L2048) is the only place where these two are
discriminated. The four subclassing trainers (precomputed, multicam, ltx,
multicam-precomputed) bypass this dispatcher entirely by being launched as
`python <subclass_module>.py <config>`.

What would need to change for feature splatting + alpha-aware composition +
random bg + new W&B logs to work in this trainer:

- **Already works for `Trainer`.** This is the reference implementation.
  `recon_backward` (L1309) has the alpha + random_bg path. `validation_video_payload`
  (L1649) has the W&B side-by-side composite, alpha-mask video, feature-PCA
  video.
- **`KnownCameraTrainer.initial_step_result`** at L1897 has a tuple-unpack bug
  — it calls `render_decoded_clip` (which returns `(features, alpha)`) and
  assigns the tuple to `rendered_features`. Step 0 init logs would either
  crash or silently log a tuple object. Needs the same alpha-aware unpacking
  the parent uses.
- **`KnownCameraTrainer.render_full_sequence`** at L1925 already does the
  alpha-aware unpack — its body is hand-rolled and works. The bug is only
  in `initial_step_result`.

### `src/train/train_precomputed_feature_implicit_dynamic.py` — `PrecomputedFeatureImplicitTrainer`

165 lines. A clean subclass that overrides three things:

- `resolve_config` — adds `FEATURE_OPTION_DEFAULTS` and validates
  `video_encoder_backend in {precomputed, precomputed_ltx}`.
- `on_sequences_loaded` — builds `VideoFeatureCache`, prebakes features for
  every train and eval sequence, infers feature channels from the cache,
  optionally releases the extractor.
- `model_input_for_clip` — returns cached features instead of clip_frames.

Inherits everything else (sample, forward, recon_backward, validation, run).
This trainer **automatically gets the alpha-aware random-bg path** because
`step` is inherited from `Trainer`.

What would need to change for feature splatting + alpha-aware composition +
random bg + new W&B logs to work in this trainer:

- **Nothing structural.** It already inherits all of it.
- The only feature-trainer-specific change would be if any new W&B log keys
  needed to know about the precomputed-feature lifecycle (e.g. cache hit
  rate); those would slot into `scalar_payload`.

### `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` — `MulticamPrecomputedFeatureImplicitTrainer`

386 lines. **Untracked in git as of the snapshot.** This is the newest and
most divergent subclass. Inherits from `PrecomputedFeatureImplicitTrainer`
but overrides almost the entire training loop:

- `resolve_config`, `__init__`, `load_train_sequences`, `load_eval_sequences`,
  `on_sequences_loaded`, `step`, `initial_step_result`, `scalar_payload`,
  `validation_video_payload`, `export_browser_bundle`.
- Adds `sample_views`, `sample_multicam_clip`, `_decode_clip`,
  `render_view_clip`, `multicam_recon_loss`, `rig_regularization_loss`,
  `render_full_external_views`.

What would need to change for feature splatting + alpha-aware composition +
random bg + new W&B logs to work in this trainer:

- **`render_view_clip` (L177–187)** returns `render_clip_sequence(...)` which
  is `(features, alpha)`. Currently the multicam trainer treats the return as
  a single tensor `rendered`. Must:
  - unpack: `features, alpha = render_view_clip(...)`
  - run `colorize_features(features, cameras)` to get `splat_rgb` (current
    code skips colorize entirely).
  - composite against `random_bg` like `Trainer.recon_backward` does, OR
    against `1.0` like validation paths do — pick deliberately.
  - feed `chunk_renders` into `reconstruction_loss_per_image`.
- **`multicam_recon_loss` (L189–205)** must consume the alpha-aware tuple
  and apply colorize + random_bg + composite. Currently it does
  `recon_loss = recon_loss + reconstruction_loss_per_image(rendered, ...)`
  where `rendered` is a tuple — this is an active bug.
- **`initial_step_result` (L210–246)** does the same on its eval clip:
  calls `multicam_recon_loss` (same bug), no alpha unpacking.
- **`render_full_external_views` (L287–318)** calls `render_view_clip` and
  `render_clip_sequence` directly with no tuple unpack — same bug.
- **`validation_video_payload` (L320–361)** does not log
  `Alpha_Mask_Video`, `Feature_PCA_Video`, or `Render_Composite_Video`. To
  match parity with the single-cam trainer, would need:
  - `feature_pca_log` plumbing into the multicam validation path
  - alpha mask collection per view per frame
  - composite-column logic
- The multicam `step` does a single full-batch `loss.backward()` (L261) —
  it does not honor `recon_backward_strategy`. Per-chunk-backward logic from
  `Trainer.recon_backward` would have to be ported, or the multicam path
  would have to declare `recon_backward_strategy=batched` as the only
  supported value.
- Three (different default) feature-channel-related config keys
  (`features.cache_dir`, `features.sample_cache_key`, `features.cache_version`)
  must be respected; this comes for free via `super().on_sequences_loaded()`.

This is the trainer with the heaviest migration cost.

### `src/train/train_ltx_feature_implicit_dynamic.py` — `LTXFeatureImplicitTrainer`

32 lines. Empty subclass body. Pure backward-compatibility alias for one
launcher script (`train_ltx_feature_implicit_camera_128_4fps_fast_mac.sh` —
which actually invokes `train_precomputed_feature_implicit_dynamic.py` with
the LTX config, not this file).

What would need to change for feature splatting + alpha-aware composition +
random bg + new W&B logs to work in this trainer:

- **Nothing — it inherits everything from `PrecomputedFeatureImplicitTrainer`,
  which inherits everything from `Trainer`.** The class body is `pass`. The
  only reason this file exists is the docstring "Backward-compatible
  LTX-named entrypoint."

### `src/train/train_camera_implicit_dynamic.py` — function-style image-encoder baseline

417 lines. Function-style (`run_training` + `main`) — does NOT inherit from
`Trainer`. Has its own `LOSS_OPTION_DEFAULTS`, `resolve_config`,
`load_sequence_data`, `pick_renderer_mode`, `build_model_from_config`,
`render_implicit_frame`, `eval_metric_payload`, `render_full_sequence`,
`run_training`. Uses `DynamicTokenGSImplicitCamera /
...SeparatedImplicitCamera` model classes that the video-token trainer does
NOT know about. Inlined per-frame loop:

```python
for local_index, camera in enumerate(decoded.cameras):
    render = render_implicit_frame(renderer_mode, cfg, dense_grid, camera, decoded.frame(local_index))
    target = batch_frames[local_index]
    renders.append(render)
    recon_losses.append(reconstruction_loss_per_image(render.unsqueeze(0), target.unsqueeze(0), loss_cfg)[0])
```

Imports `pick_device`, `configure_fast_attn`, `fast_attn_context`,
`select_window_indices` from `dynamicTokenGS`.

What would need to change for feature splatting + alpha-aware composition +
random bg + new W&B logs to work in this trainer:

- This trainer **does not call `render_clip_sequence`** at all. It calls
  `render_implicit_frame` (the legacy non-alpha-aware single-frame path) one
  frame at a time. Adding alpha-aware composition would require:
  - swap the inner loop to `render_gaussian_frames_alpha_aware` (or build
    a clip-level analogue).
  - add a `colorize` / `FeatureToColor` instance — currently there is no
    `colorize` config wiring at all.
  - add `random_bg` sampling per step.
- `eval_metric_payload` (L133–160) is a near-duplicate of the parent file's
  module-level version (L602–629). Different default loss weights but same
  structure.
- No `validation_video_payload` — inlines the same logic into `run_training`.
- No static/dynamic bank-rate loss support.
- No `colorize` support.
- No feature-PCA logging support.
- No precomputed-feature support.

In short: this file is a completely parallel mini-trainer with its own
config schema and would need a near-rewrite to inherit from `Trainer`.

### `src/train/train_image_encoder_implicit_camera_baseline.py`

4 lines, hardcoded config:
```python
from train_camera_implicit_dynamic import main
if __name__ == "__main__":
    main("src/train_configs/local_mac_overfit_image_implicit_camera.jsonc")
```
Pure shim. The launcher script
`src/train_scripts/train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh`
does NOT invoke this file — it execs
`train_full_dynamic_with_implicit_camera_all_frames.sh` which then invokes
`train_camera_implicit_dynamic.py` directly with whatever config the user
passes. So this file is currently invoked only via direct
`python <file>` calls, if at all.

### `src/train/train_camera_implict_dynamic.py`

4 lines, **typo of `implicit`**. Identical body to
`train_image_encoder_implicit_camera_baseline.py`. No script or doc that I
can find references this filename. Dead.

### `src/train/dynamicTokenGS.py` — legacy prebaked-camera trainer

731 lines, function-style. Owns prebaked-camera training (camera comes from
`per_frame_cameras.json` via `load_camera_sequence`). Has the most
sophisticated optimizer plumbing in the codebase
(`build_optimizer_param_groups` with weight-decay exemptions, fused fallback,
LR schedules), but the implicit trainers don't use any of it. Also owns
`debug_metrics` integration: `dense_render_diagnostics`, `optimizer_diagnostics`,
`render_aux_diagnostics`, finite/non-finite checks at every step, configurable
fail-fast.

Critically: `pick_device`, `configure_fast_attn`, `fast_attn_context`, and
`select_window_indices` are imported FROM this file (or re-exported from
modules it imports) by the implicit trainer. So even the implicit-trainer code
path is structurally tied to `dynamicTokenGS.py`.

What would need to change for feature splatting + alpha-aware composition +
random bg + new W&B logs to work in this trainer:

- A near-rewrite. This trainer has:
  - no `colorize` config wiring
  - no static/dynamic split awareness
  - no feature-PCA logging
  - no alpha-aware render path (calls `render_gaussian_frame` /
    `render_gaussian_frames` — both the non-alpha versions)
  - no clip / `train_frame_count` notion (uses `frames_per_step` instead)
- It also has structural features the implicit trainers don't:
  - `lr_schedule.cosine` with `final_lr_scale`
  - weight-decay-aware param groups
  - per-step `clip_grad_norm`
  - `metric_cfg` integration
  - `taichi_options` renderer wiring (which the alpha-aware path doesn't
    handle either way — `render_gaussian_frames_alpha_aware` falls through to
    the non-fast_mac renderer when mode != "fast_mac").
- Would need to choose: (a) bring this trainer under `Trainer` (massive
  migration), (b) port the four cross-cutting helpers to a shared module
  and let `dynamicTokenGS.py` keep its own loop, (c) deprecate this trainer
  in favor of an implicit-trainer config that passes prebaked cameras.

### `src/train/dynamicTokenGS_tiled.py` and `dynamicTokenGS_shared.py`

Pure shims. `dynamicTokenGS_tiled.py` invokes `dynamicTokenGS.main` with the
hardcoded tiled config. `dynamicTokenGS_shared.py` re-exports
`DynamicTokenGS` and `fetch_image`.

### `src/train/tokenGS.py` — single-image overfit

146 lines, function-style. Loads one PNG, builds one default camera, runs
`TokenGS(...)`. Fundamentally different from temporal trainers: no
`SequenceData`, no clip sampling, no validation, no alpha awareness.

What would need to change for feature splatting + alpha-aware composition +
random bg + new W&B logs to work in this trainer: a near-rewrite.

### `src/train/tokenGS_tiled.py` and `tokenGS_shared.py`

Shims. `tokenGS_tiled.py` invokes `tokenGS.main` with default-arg fallback to
`local_mac_overfit_single_image_tiled.jsonc`. `tokenGS_shared.py` re-exports
`TokenGS` and `fetch_image`.

## Configs grouped by trainer

### `train_video_token_implicit_dynamic.py` (`Trainer` and `KnownCameraTrainer`)

`Trainer` (`tokengs`, `tokengs_video_implicit_camera`):

- `local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_crossattn1_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_crossattn2_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_sinusoidal_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc`
- `local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_camera_clamp_video_implicit_128_fast_mac_8192splats_1000step.jsonc`
- `local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc`
- `local_mac_compare_free_linear_time_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_local_video_encoder_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_local_video_encoder_64f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_local_video_encoder_strong_init_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_residual_free_bank_local_video_encoder_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_residual_free_bank_vjepa2_vitl_fpc16_256_frozen_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_unconditioned_residual_free_bank_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_vjepa2_1_vitb_384_frozen_64f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_vjepa2_vitl_fpc16_256_frozen_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_compare_vjepa2_vitl_fpc16_256_frozen_strong_init_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `local_mac_overfit_video_token_full.jsonc`
- `local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`
- `local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats_pose_to_plucker.jsonc`
- `local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats_sinusoidal_time.jsonc`
- `local_mac_overfit_video_token_implicit_camera_vjepa2_1_torchhub_vitb_384.jsonc`
- `local_mac_overfit_video_token_smoke.jsonc`
- `local_mac_scene_distinct_30_local_encoder_256_fast_mac_2048splats.jsonc`
- `local_mac_scene_distinct_30_vjepa2_vitl_fpc16_256_frozen_256_fast_mac_2048splats.jsonc`
- `local_mac_tiny_30_video_token_smoke.jsonc`
- `local_mac_unconditioned_tokens_fast.jsonc`
- `local_mac_unconditioned_tokens_fast_400step.jsonc`
- `local_mac_unconditioned_tokens_features_F32.jsonc`
- `local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4.jsonc`
- `local_mac_unconditioned_tokens_features_F32_LN_orth_g3.jsonc`
- `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc`

`KnownCameraTrainer` (`tokengs_video_known_camera`):

- `local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc`

### `train_precomputed_feature_implicit_dynamic.py` (`PrecomputedFeatureImplicitTrainer`)

`precomputed_feature_implicit_camera`:

- `local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_camera_clamp_video_implicit_128_fast_mac_8192splats_1000step.jsonc`
- `local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`
- `local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc`
- `local_mac_overfit_precomputed_vjepa2_1_torchhub_vitb_384.jsonc`

`wan_vace_feature_implicit_camera`:

- `local_mac_overfit_wan_vace_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`

### `train_ltx_feature_implicit_dynamic.py` (`LTXFeatureImplicitTrainer`)

`ltx_feature_implicit_camera`:

- `local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`

(NB: the launcher script
`train_ltx_feature_implicit_camera_128_4fps_fast_mac.sh` actually invokes
`train_precomputed_feature_implicit_dynamic.py` directly, not
`train_ltx_feature_implicit_dynamic.py`.)

### `train_multicam_precomputed_feature_implicit_dynamic.py` (`MulticamPrecomputedFeatureImplicitTrainer`)

`multicam_precomputed_feature_implicit_camera`:

- `local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc`
- `local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc`
- `local_mac_multicam_deepview_4cam_train2_holdout2_overlap_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc`
- `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc`

### `train_camera_implicit_dynamic.py` (function-style)

`tokengs_image_implicit_camera`:

- `local_mac_overfit_image_implicit_camera.jsonc`
- `local_mac_overfit_image_implicit_camera_separated.jsonc`

### `dynamicTokenGS.py` (function-style, legacy)

`tokengs_prebaked_camera`:

- `local_mac_overfit_prebaked_camera.jsonc`
- `local_mac_overfit_prebaked_camera_64_4fps.jsonc`
- `local_mac_overfit_prebaked_camera_128_4fps.jsonc`
- `local_mac_overfit_prebaked_camera_128_4fps_wide_depth_1024splats.jsonc`
- `local_mac_overfit_prebaked_camera_128_4fps_wide_depth_8frames.jsonc`
- `local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_8192splats.jsonc`
- `local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_65536splats.jsonc`
- `local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi.jsonc`
- `local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi_8192splats.jsonc`

`tokengs_prebaked_camera_tiled`:

- `local_mac_overfit_prebaked_camera_tiled.jsonc`

### `tokenGS.py` (function-style, legacy single-image)

`tokengs_single_image`:

- `local_mac_overfit_single_image.jsonc`

`tokengs_single_image_tiled`:

- `local_mac_overfit_single_image_tiled.jsonc`

### Configs that do NOT dispatch to any in-scope trainer

These all have an `arch` field but their trainer lives outside `src/train/`,
under `research_experiments/gauge_fields/`:

- `gauge_fields_material_surfel` (29 configs, all `local_mac_gauge_fields_*.jsonc`).
- `splat_baseline_static_3dgs`
  (`local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_static_3dgs_128_16f_2048splats.jsonc`).
- `splat_baseline_free_dynamic_3dgs`
  (`local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_free_dynamic_3dgs_128_16f_2048splats.jsonc`,
  `local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc`).

These are out-of-scope but worth flagging so proposers know `src/train/` is
NOT the only home for trainers in the repo.

## Status / deletion candidates

| File | Status | Safety check | Recommendation |
|---|---|---|---|
| `src/train/train_camera_implict_dynamic.py` | **dead — typo of `implicit`** | No script, no doc, no config references this filename. Identical body to `train_image_encoder_implicit_camera_baseline.py`. | Delete after `git grep` confirms zero callers. Trivially safe. |
| `src/train/train_image_encoder_implicit_camera_baseline.py` | shim, hardcoded config | Launcher script `train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh` execs a different launcher and bypasses this file. The shim is reachable only via `python <file>`. | Verify no human/automation calls this file directly; if not, delete. |
| `src/train/dynamicTokenGS_shared.py` | shim, two-line re-export | Imported by anyone? `git grep "from dynamicTokenGS_shared"` should be the proof. | Likely safe to delete after grep check. |
| `src/train/tokenGS_shared.py` | shim, two-line re-export | Same check. | Likely safe to delete after grep check. |
| `src/train/dynamicTokenGS_tiled.py` | shim, hardcoded config | The tiled prebaked-camera config can be invoked via `dynamicTokenGS.py <config>`. This shim adds nothing. | Keep only if a human workflow depends on the bare-filename invocation; otherwise delete. |
| `src/train/tokenGS_tiled.py` | shim, hardcoded config fallback | Same shape as above; only difference is a fallback default arg. | Same recommendation. |
| `src/train/tokenGS.py` | active legacy single-image overfit | One config dispatches here (`local_mac_overfit_single_image.jsonc`, `local_mac_overfit_single_image_tiled.jsonc` via the shim). No `agent_notes/` references it as canonical. Used as a smoke for renderer/model wiring. | Keep, still load-bearing as a smoke. Could collapse into a 30-line script invoked from `Trainer` with a `single_image` data source — but that's a Wave 2 question. |
| `src/train/dynamicTokenGS.py` | active legacy prebaked-camera trainer | 11 configs dispatch here. Also re-exports `pick_device`, `configure_fast_attn`, `fast_attn_context`, `select_window_indices` consumed by `train_camera_implicit_dynamic.py` and `train_video_token_implicit_dynamic.py`. | **Keep — load-bearing.** It is the only home for those four helpers and the only trainer that uses prebaked-camera JSON. Deletion would orphan 11 configs and break the import chain in the implicit trainers. The four helpers should be moved out before any deletion can be considered. |
| `src/train/train_camera_implicit_dynamic.py` | active parallel image-encoder baseline | 2 configs dispatch here. Distinct model classes (`DynamicTokenGSImplicitCamera /SeparatedImplicitCamera`) not reachable through `train_video_token_implicit_dynamic.build_model_from_config`. | **Keep — distinct-model load-bearing.** No other trainer can build those two model variants. Migration would require porting the model variants into the central `build_model_from_config`. |
| `src/train/train_ltx_feature_implicit_dynamic.py` | empty backward-compat alias | One config dispatches to it. Launcher script bypasses this file and invokes `train_precomputed_feature_implicit_dynamic.py` directly. | Delete after verifying that the LTX config's `arch=ltx_feature_implicit_camera` does not gate on this exact class name. The class body is `pass`. The only behavioral effect is the print prefix in the parent's `run`. |
| `src/train/train_video_token_implicit_dynamic.py` | active primary trainer | 36+ configs dispatch here. | Keep — the canonical trainer. |
| `src/train/train_precomputed_feature_implicit_dynamic.py` | active feature-cache subclass | 5 configs dispatch here. | Keep. |
| `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` | active multicam subclass, untracked | 4 configs dispatch here. **Has the alpha/feature/colorize gap.** | Keep, but it is the most dangerous trainer in the codebase right now — it has at least three call sites that drop the alpha tuple silently and would crash or misbehave on `fast_mac` mode. |

## Open questions for the proposers

1. **One trainer hierarchy or two?** The current shape has `Trainer` →
   `PrecomputedFeatureImplicitTrainer` → `MulticamPrecomputedFeatureImplicitTrainer`
   (three-deep) plus a sibling `KnownCameraTrainer`. There are two
   completely-disjoint legacy function-style trainers (`dynamicTokenGS.py`,
   `train_camera_implicit_dynamic.py`). Should the legacy trainers be
   pulled into the hierarchy, or kept as parallel paths? Note
   `key_learnings.md:18` explicitly warns "A single shared `BaseTrainer`
   would hide real differences between known-camera, image-implicit, and
   video-token implicit training. Shared payload contracts are cleaner than
   shared trainer inheritance." That note pre-dates the multicam trainer.

2. **Multicam-vs-single-cam: strategy plug-in or two trainers?** The
   multicam `step` body is genuinely different (loops over views, has
   heldout cameras, has a rig optimizer group), but the "render + colorize +
   composite + alpha-aware random bg" sub-step is identical in shape. Should
   that sub-step be lifted to a shared helper that both trainers call, or
   should multicam absorb the single-cam path as the
   `train_views_per_step <= 1` special case?

3. **What is the right home for the "alpha-aware composition + random bg"
   path?** Currently it's inlined in `Trainer.recon_backward`. To get it
   into the multicam trainer, either:
   (a) lift it to a free function `composite_alpha_aware(features, alpha,
       cameras, *, colorize, random_bg) -> rendered`, called by both;
   (b) change `recon_backward` to take a per-step "render function"
       callable; or
   (c) push the alpha-aware path into `render_clip_sequence` itself (pre-
       composited tensors, alpha kept as a side-channel for logging only).
   The proposers must pick one.

4. **Is `KnownCameraTrainer.initial_step_result` the only shared bug?**
   The audit found two latent tuple-vs-tensor bugs (multicam render path,
   known-camera initial step). Worth a follow-up grep for any other place
   that calls `render_decoded_clip` or `render_clip_sequence` and treats
   the return as a tensor.

5. **What to do with `dynamicTokenGS.py`'s optimizer features?**
   `lr_schedule` (cosine, final_lr_scale), weight-decay-aware param groups,
   per-step `clip_grad_norm`, `metric_cfg` integration: all live only in
   the legacy prebaked-camera trainer. They never propagated to the
   implicit trainer family. If the implicit trainer family ever needs
   them, they have to be ported. If they don't, we should explicitly
   declare them frozen-by-design.

6. **What about the four helpers re-exported through `dynamicTokenGS.py`?**
   `pick_device`, `configure_fast_attn` (re-export from `fast_attn`),
   `fast_attn_context`, `select_window_indices` (re-export from
   `sequence_data`) are all imported by the implicit trainer files from
   `dynamicTokenGS`. The legacy trainer cannot be deleted until those four
   imports move to a neutral shared module.

7. **`tokengs_single_image_tiled` vs `tokengs_single_image`.** Two configs
   dispatch to `tokenGS.py` — the trainer doesn't know the difference; the
   `arch` value is informational. Is the tiled-vs-untiled split still
   meaningful? Same question for the prebaked-camera-tiled config.

8. **The `LTXFeatureImplicitTrainer` empty subclass.** Does any external
   caller depend on the class name? If not, the file is pure noise.

9. **The unaligned `LOSS_OPTION_DEFAULTS` triplet.** Three trainer files
   (`train_video_token_implicit_dynamic.py`,
   `train_camera_implicit_dynamic.py`, `dynamicTokenGS.py`) each define
   their own `LOSS_OPTION_DEFAULTS` dict literal with overlapping but
   non-identical fields. Should this be one shared constant?

End of report.
