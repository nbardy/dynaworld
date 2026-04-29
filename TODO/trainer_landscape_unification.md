# Trainer Landscape Unification

## TL;DR

- Seven trainer files exist; only **two** carry real logic
  (`train_video_token_implicit_dynamic.py`, ~2070 lines, and
  `train_multicam_precomputed_feature_implicit_dynamic.py`, ~390 lines).
  Two are 4-line shims, two more (`train_precomputed_feature_implicit_dynamic.py`,
  `train_ltx_feature_implicit_dynamic.py`) are thin subclasses that only
  add config defaults and a feature-cache hook.
- The composition recipe `α · colorize(features) + (1-α) · bg` is
  duplicated in **three** places inside the single-cam trainer alone
  (`recon_backward`, `initial_step_result`, `render_full_sequence`),
  and is **absent** in the multicam trainer because
  `multicam_recon_loss` short-circuits to `render_clip_sequence` and skips
  colorize/alpha entirely. This is the immediate breakage the user hit.
- The single largest legacy file (`dynamicTokenGS.py`, ~730 lines, the
  prebaked-camera path) and the older image-implicit
  trainer (`train_camera_implicit_dynamic.py`, ~420 lines) carry their
  own inline copies of the train loop, render dispatch, and W&B logging.
  These two predate the `Trainer` class; new features (alpha, random bg,
  feature_pca_log, composite video) never reach them.
- Proposal: extract four small pure helpers
  (`compose_rendered_rgb`, `validation_video_logger`, `RenderedClipBundle`,
  `colorize_module_from_config`). No new abstract base class. The multicam
  trainer becomes ~30 lines lighter and gets the alpha+bg+composite logs
  for free; the single-cam trainer drops three near-identical composition
  blocks.
- Delete-or-fold candidates: the two 4-line shim entrypoints
  (`train_camera_implict_dynamic.py` (sic — typo), `train_image_encoder_implicit_camera_baseline.py`),
  the LTX subclass, and the legacy `dynamicTokenGS.py` if no
  `tokengs_prebaked_camera*` configs are still in active use.
- This planning doc supersedes Phase 2 of `TODO/Clean_up_and_unify_interfaces.md`,
  which proposed splitting trainers into per-baseline classes; the present
  problem is the opposite — overrides have already split too much.

---

## Trainer inventory

| File | LoC | Class hierarchy | Owns | Overrides | Status | Configs |
|------|-----|-----------------|------|-----------|--------|---------|
| `train_video_token_implicit_dynamic.py` | 2072 | `Trainer` (root); `KnownCameraTrainer(Trainer)` | Single-cam video-token train loop, `Trainer.run/step/recon_backward/initial_step_result/render_full_sequence/validation_video_payload`, alpha-aware composition, random-per-step bg, F=32 colorize MLP wiring, F-PCA logging, alpha mask video, composite video, all model-variant dispatch. `KnownCameraTrainer` overrides `step`, `initial_step_result`, `render_full_sequence` for the cameras-known path. | n/a (root) | **Active**. The hub of the system. | 35 configs with `arch=tokengs_video_implicit_camera`, plus `arch=tokengs` (2 configs), plus `arch=tokengs_video_known_camera` (1 config). |
| `train_precomputed_feature_implicit_dynamic.py` | 165 | `PrecomputedFeatureImplicitTrainer(Trainer)` | Feature-cache prebake + `model_input_for_clip` override. Adds `FEATURE_OPTION_DEFAULTS` and `on_sequences_loaded` to attach a `VideoFeatureCache`. | `resolve_config`, `on_sequences_loaded`, `model_input_for_clip`, `run` (only adds a print). | **Active**. Inherits all the alpha/bg/log/composite work. | 4 configs with `arch=precomputed_feature_implicit_camera`. |
| `train_multicam_precomputed_feature_implicit_dynamic.py` | 386 | `MulticamPrecomputedFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer)` | Multicam sampling, learnable camera rig, multi-view recon loss, heldout view eval, multicam-specific validation video payload. | `resolve_config`, `__init__`, `load_train_sequences`, `load_eval_sequences`, `on_sequences_loaded`, `step`, `initial_step_result`, `scalar_payload`, `validation_video_payload`, `export_browser_bundle`. | **Active, broken w.r.t. recent fixes**. Overriding `step` and `initial_step_result` bypasses `recon_backward`, so the alpha-aware composition, random-per-step bg, and F=32 colorize composition added in the 2026-04-29 session never run here. `multicam_recon_loss` calls `render_clip_sequence` directly and skips `self.colorize`. | 4 configs with `arch=multicam_precomputed_feature_implicit_camera` (incl. `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc`, the new F=32-alpha multicam target). |
| `train_ltx_feature_implicit_dynamic.py` | 32 | `LTXFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer)` | Backward-compat alias only. Empty body. | none | **Legacy alias**. Folder rename never happened. | 1 config (`arch=ltx_feature_implicit_camera`). The other "LTX" config (`arch=wan_vace_feature_implicit_camera`) also dispatches via `train_precomputed_feature_implicit_dynamic.py` directly through its shell script. |
| `train_camera_implicit_dynamic.py` | 417 | procedural `run_training` (no `Trainer` class) | Image-implicit single-frame training: own `resolve_config`, own `LOSS_OPTION_DEFAULTS`, own `eval_metric_payload`, own inline train loop, own W&B init, own image preview path. Does not call `Trainer`. | n/a | **Active for the image baselines**, but completely disjoint from the video trainer's vocabulary. Predates `Trainer`. | 2 configs (`arch=tokengs_image_implicit_camera`). |
| `train_image_encoder_implicit_camera_baseline.py` | 4 | shim → `train_camera_implicit_dynamic.main` | Hard-coded path to one config. | n/a | **Dead-ish**. Shim with hard-coded JSONC arg. Not referenced by any shell script. | none directly. |
| `train_camera_implict_dynamic.py` (sic — typo of "implicit") | 4 | shim → `train_camera_implicit_dynamic.main` | Hard-coded path to one config. | n/a | **Dead**. Same hard-coded path as the file above. Typo in filename. | none. |
| `dynamicTokenGS.py` | 731 | procedural `run_training` (no `Trainer` class). Also exports `pick_device`, `configure_fast_attn`, `fast_attn_context` used by `Trainer`. | Known-camera, prebaked, image-only training (one render per frame). Own `LOSS_OPTION_DEFAULTS`, own optimizer-group builder, own LR schedule, own debug-metrics path, own train loop, own W&B logging. Inline train loop with no `Trainer` class. | n/a | **Mixed**. Provides `pick_device`/fast_attn helpers that other files import — those cannot be deleted. The `run_training` body is legacy. | 9 configs (`arch=tokengs_prebaked_camera*`). Used by `train_full_dynamic_with_camera_prebake_all_frames.sh`. Probably still alive but only as a "compare to known-camera baseline" path. |

Notes:
- `tokenGS.py` (no "Dynamic") and `tokenGS_tiled.py`/`dynamicTokenGS_tiled.py`/`dynamicTokenGS_shared.py` are even older single-image trainers + 1-line shims, dispatched by `arch=tokengs_single_image*`. They are out of scope for this audit but if a sweep happens, they should be folded together with `dynamicTokenGS.py`.
- Configs whose `arch` is `gauge_fields_material_surfel` or `splat_baseline_*` do NOT dispatch via these trainers; they live under `research_experiments/gauge_fields/` (16 configs total). They should not be unified with the main trainer — they are an explicitly separate experiment surface.

---

## What's actually duplicated across trainers

The duplications below are concrete code blocks I found in more than one
file with only minor variation. Estimated line counts are for the
aggregate of all sites.

| Concern | Sites | Aggregate lines | Drift status | Unifiable? |
|---------|-------|-----------------|--------------|------------|
| **Compose rendered RGB from features + alpha + bg** (the `α · colorize(features) + (1-α) · bg` recipe) | `train_video_token_implicit_dynamic.py` lines ~1346-1361 (`recon_backward`, **uses random per-step bg**), ~1408-1416 (`initial_step_result`, **uses white bg**), ~1599-1609 (`Trainer.render_full_sequence`, **uses white bg**), ~1969-1979 (`KnownCameraTrainer.render_full_sequence`, **uses white bg**); MISSING in `train_multicam_precomputed_feature_implicit_dynamic.multicam_recon_loss` (~ln 197-205) | ~50 lines duplicated, plus 1 missing site that should compose | The eval site uses white bg, the train site uses random bg, the multicam site does no compose at all. Classic 3-way drift. | YES — single helper. Highest-ROI extraction. |
| **Build the validation-video W&B payload** (GT, render, alpha mask, feature PCA, composite columns) | `train_video_token_implicit_dynamic.Trainer.validation_video_payload` lines ~1649-1737; `train_multicam_precomputed_feature_implicit_dynamic.MulticamPrecomputedFeatureImplicitTrainer.validation_video_payload` lines ~320-361; partial copy in `train_camera_implicit_dynamic.run_training` lines ~367-398; partial copy in `dynamicTokenGS.run_training` lines ~667-710 | ~250 lines, four variants | The single-cam trainer has the new alpha mask + PCA + composite columns; the multicam trainer logs per-view rendered+GT but no alpha/PCA/composite; the older two have neither. | YES, but split: a lower-level `validation_video_logger(gt, rendered, features?, alpha?)` helper for the video-token paths. The two procedural trainers can call the same helper at the cost of giving them a `feature_sequence=None, alpha_sequence=None` no-op signature. |
| **Compute per-clip eval metrics** (L1, MSE, SSIM, DSSIM, recon-loss, PSNR) | `train_video_token_implicit_dynamic.eval_metric_payload` lines ~602-629; `train_camera_implicit_dynamic.eval_metric_payload` lines ~133-160 (literal copy-paste, including the same `1.0e-12` floor) | ~55 lines, two near-identical copies | Truly identical math. | YES — trivial dedup. Same goes for `temporal_similarity_payload` and `decoded_temporal_payload`, both currently single-source but inside the big trainer file. |
| **Render dispatch helper around `render_gaussian_frame[s]`** | `dynamicTokenGS.render_one_frame` (lns ~378-395), `dynamicTokenGS.render_frame_batch` (lns ~398-415), `train_camera_implicit_dynamic.render_implicit_frame` (lns ~115-130), `train_video_token_implicit_dynamic.render_clip_sequence` (lns ~556-582). Three of these wrap the same underlying call. | ~70 lines, four wrappers | The video-token wrapper now calls `render_gaussian_frames_alpha_aware` (returns tuple). The other three call non-alpha variants. | PARTIAL — `Clean_up_and_unify_interfaces.md` already proposed `RenderConfig` + `render_gaussian_sequence`. We do not need a full refactor here, but at minimum these four wrappers should converge on the alpha-aware tuple-returning shape. |
| **Optimizer construction** | `train_video_token_implicit_dynamic.Trainer.__init__` lines ~1037-1041 (Adam, fused, includes colorize params); `train_multicam_precomputed_feature_implicit_dynamic.__init__` lines ~107-114 (adds rig param group on top); `train_camera_implicit_dynamic.run_training` line ~255 (Adam, fused); `dynamicTokenGS.build_optimizer` lines ~318-342 (rich AdamW + LR-multiplier groups). | ~80 lines across four sites | The `dynamicTokenGS` builder is significantly richer (LR multipliers, weight-decay exclusion, fused-fallback). The other three are bare `Adam(params, lr=...)`. | MAYBE — only worth unifying if/when the video-token trainer gains LR-multiplier groups. Today the `dynamicTokenGS` builder is over-engineered for the simpler trainers. Leave alone unless a need arises. |
| **W&B init + final `wandb.finish()`** | `Trainer.__init__` ~997-1002, `Trainer.run` finally-block ~1810; same pattern in `train_camera_implicit_dynamic.run_training` ~246-251 + ~401, in `dynamicTokenGS.run_training` ~485-490 + ~716, and `MulticamPrecomputedFeatureImplicitTrainer` inherits. | ~30 lines | Identical, modulo `serialize_config_value` only present in the newer ones. | LOW value — extracting saves ~5 lines per site at the cost of a layer of indirection. Skip. |
| **`should_log_scalars` / `should_log_images` / `should_log_videos`** logic | `Trainer` lines ~1499-1512; same boolean patterns inlined in `train_camera_implicit_dynamic.run_training` lines ~338-346 and `dynamicTokenGS.run_training` lines ~645-653 | ~25 lines, three copies | All three implement the same `step % every == 0 or (always_log_last_step and step == last)` check. | YES — small win. Move to `train_logging.py` as `should_log(step, every, *, total_steps, always_log_last_step)`. |
| **Config-defaults application** | `Trainer.resolve_config` lines ~174-370 (~200 lines of `apply_defaults` + cast + validate); `MulticamPrecomputedFeatureImplicitTrainer.resolve_config` adds 3 more `apply_defaults` blocks (~30 lines); `PrecomputedFeatureImplicitTrainer.resolve_config` adds another (~50 lines); `train_camera_implicit_dynamic.resolve_config` (~25 lines); `dynamicTokenGS.resolve_config` (~25 lines). | ~330 lines total | Per-trainer additions are deliberate (each adds the section it cares about). The pattern is right; the one issue is that big monolithic `resolve_config`s in the video trainer are 200 lines of defaults — that's fine and not a duplication smell. | NO — these are not duplicates; they are per-trainer specializations. Leave as is. |
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

### 2. `validation_video_logger(...)` in `train_logging.py`

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

### 5. (Optional) `run_step_loop(trainer, *, total_steps, log_intervals)` helper

The `for step in pbar: ... pbar.set_description(...) ... self.val_log(step, result)` outer loop is ~15 lines and identical between the single-cam and known-camera trainers. The procedural older trainers have a copy too. This is the only reason `Trainer.run` and `KnownCameraTrainer.run` differ at all — the run banner. A shared `run_step_loop(self)` plus a per-trainer `print_run_banner()` would let `KnownCameraTrainer` lose its `run` override entirely.

Risk: low. This is the one place a small base-class method (`run`) is fine, because the only reason for the override is a banner string.

### What is NOT proposed (deliberately)

- No new abstract `BaseTrainer` with virtual methods. The current
  `Trainer → PrecomputedFeatureImplicitTrainer → MulticamPrecomputedFeatureImplicitTrainer`
  chain already exists and is the source of the bug. Adding more
  inheritance layers would compound it.
- No registry-based trainer dispatch. The current `arch` string +
  shell-script-per-trainer is fine.
- No model-architecture refactor. The 35-config `learned_time_orbit_path`
  fanout is a model concern, not a trainer concern.
- No `RenderConfig` dataclass refactor (proposed in `Clean_up_and_unify_interfaces.md`).
  That's still a good idea but is independent of the alpha/multicam
  unification and adds churn risk.

---

## What to delete

| File | Safety check | Verdict |
|------|--------------|---------|
| `src/train/train_camera_implict_dynamic.py` (sic — typo) | 4-line shim. No shell script references it. No config references it. Delete-search returns nothing. | **Delete.** The typo in the filename suggests it was an accidental commit. |
| `src/train/train_image_encoder_implicit_camera_baseline.py` | 4-line shim with hard-coded config path. Referenced by `src/train_scripts/train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh` (one line, just runs the shim). | **Delete the shim**, redirect the shell script to call `train_camera_implicit_dynamic.py` with the same config path. One-line shell-script edit. |
| `src/train/train_ltx_feature_implicit_dynamic.py` | 32 lines, declares `LTXFeatureImplicitTrainer = PrecomputedFeatureImplicitTrainer` with empty body. Referenced by `train_ltx_feature_implicit_camera_128_4fps_fast_mac.sh` (which calls **`train_precomputed_feature_implicit_dynamic.py` directly**). The "wan_vace" config also goes through the precomputed entrypoint. | **Delete.** The shim is unreferenced — its shell script already calls the parent file directly. The one config (`arch=ltx_feature_implicit_camera`) only needs to keep working with the precomputed trainer's resolver. |
| `src/train/dynamicTokenGS.py` (the procedural `run_training`) | 9 active configs (`arch=tokengs_prebaked_camera*`). Shell script `train_full_dynamic_with_camera_prebake_all_frames.sh` is wired. Multiple imports from other trainers (`pick_device`, `configure_fast_attn`, `fast_attn_context`). The known-camera training is a real baseline. | **Do not delete the file.** Shrink it: lift `pick_device` and the fast-attn helpers into a tiny `device_utils.py` so other trainers can stop importing from a 730-line module. The `run_training` body itself stays (real users) but the optimizer-builder, LR schedule, and debug-metrics block could become standalone helpers. Out of scope for this audit; flag for follow-up. |
| `src/train/tokenGS.py`, `src/train/tokenGS_tiled.py`, `src/train/dynamicTokenGS_tiled.py`, `src/train/dynamicTokenGS_shared.py`, `src/train/tokenGS_shared.py` | Single-image baselines. Out of scope for this audit. Several configs (`arch=tokengs_single_image*`, `tokengs_prebaked_camera_tiled`) reference them. | **Hold.** Audit separately. |

---

## Migration strategy

Order is highest-ROI-first. The first three items are the path that
unblocks the multicam F=32 alpha config without re-porting the same code
twice.

### Phase 1 (unblocks current pain)

1. **Extract `compose_rendered_rgb`.** Touches: `train_video_token_implicit_dynamic.py` (4 sites), `train_multicam_precomputed_feature_implicit_dynamic.py` (1 new call site). Test: numerical tolerance check + unchanged-loss smoke run on `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc`. Effort: ~60 minutes for the helper + 4 site edits + 1 test. Risk: silent regression if the helper's `random_bg` semantics differ from the inline code; mitigated by a deterministic seeded smoke run at 1 step.
2. **Extract `colorize_module_from_config` factory.** Touches: `Trainer.__init__`, plus a new direct call in the multicam trainer's `__init__` if we want the multicam trainer to apply alpha-aware composition (which means it now needs a colorize too — same code path). Effort: ~30 minutes. Risk: very low.
3. **Plumb alpha through the multicam trainer.** Once steps 1-2 land, `multicam_recon_loss` is rewritten as: render → `compose_rendered_rgb` → `reconstruction_loss_per_image` per view. The `step` and `initial_step_result` overrides shrink. Effort: ~90 minutes for the rewrite + a 1-step run on the 4 multicam configs to confirm shapes. Risk: medium — multicam train hadn't seen alpha at all before, so the first run will be the first time the rig + colorize + alpha combination has been exercised end-to-end.

### Phase 2 (deduplicates the remaining noise)

4. **Extract `validation_video_logger`.** Touches: `Trainer.validation_video_payload` (replace ~80 lines), `MulticamPrecomputedFeatureImplicitTrainer.validation_video_payload` (replace per-view inner loop), then optionally `train_camera_implicit_dynamic` and `dynamicTokenGS.run_training` if we want the same column structure on those baselines. Effort: ~2 hours. Risk: medium — composite column ordering and the `gt_video_logged` flag are easy to break. Add one fixture test.
5. **Add `RenderedClipBundle`.** Touches: `render_clip_sequence` return type plus every unpacking site (~6 sites across both trainers). Effort: ~45 minutes, mostly mechanical search-and-replace. Risk: low.
6. **Lift `eval_metric_payload` to a shared module.** Touches: delete the literal copy in `train_camera_implicit_dynamic.py`, import from `train_video_token_implicit_dynamic.py` (or move to `losses.py`/`debug_metrics.py`). Effort: ~15 minutes.
7. **Lift `should_log_scalars/images/videos` to `train_logging.py`.** Touches: 3 sites. Effort: ~15 minutes.

### Phase 3 (cleanup, optional)

8. **Delete the dead shims** (`train_camera_implict_dynamic.py`, `train_ltx_feature_implicit_dynamic.py`). Touches: file deletes + 1 shell-script line edit. Effort: ~15 minutes.
9. **Move `pick_device`/`fast_attn_context` out of `dynamicTokenGS.py`.** Touches: 1 new tiny file, 3 imports. Effort: ~30 minutes. Risk: low.
10. **Consider whether `train_camera_implicit_dynamic.py`'s procedural train loop should adopt the helpers from Phases 1-2.** If yes, the per-frame render loop calls `compose_rendered_rgb(..., alpha=None, colorize=None, random_bg=None)` and gets the same logging helper. If no, leave it alone — it's a baseline that hasn't been touched in months. Recommend "no" until someone asks for alpha on the image baseline.

### What breaks if you do nothing

- The new F=32 + alpha + random-bg + composite-video pipeline lands ONLY for the single-cam configs. The 4 multicam configs (incl. the ultimate one) silently train without alpha-aware composition, with no random bg, and with no alpha-mask/composite W&B logs. The training-loss curves will look fine, but the underlying geometry will degenerate the same way the white-background runs did before the alpha fix.
- Every future feature added to `Trainer.recon_backward` or `Trainer.validation_video_payload` will need a parallel re-port into `MulticamPrecomputedFeatureImplicitTrainer.multicam_recon_loss` and `validation_video_payload`. The longer this goes, the more surface area drifts.

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
