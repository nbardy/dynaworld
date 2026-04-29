# Investigator 02 — Loss Path & Alpha Composition

Scope: every site in `dynaworld/src/train/` that computes a reconstruction
loss, applies the colorize MLP, runs alpha-aware composition, samples a
per-step random background, and propagates gradients back through the
rasterizer. The report is concrete (`file:line`) and oriented to make the
seam for a shared helper visible to the wave-2 proposers.

## TL;DR

- Loss kernel is centralized in `src/train/losses.py` (~80 LOC, four
  recon-loss types incl. `standard_gs = L1 + DSSIM`). That part is clean and
  shared.
- Colorize + alpha-aware composition is duplicated **four times** inside
  `train_video_token_implicit_dynamic.py` with three different background
  policies: per-step `random_bg` (training), fixed `1.0` white (eval), and
  no-composition (legacy F=3). Same six-line pattern, three sites with
  different bg semantics, one site with a latent tuple-vs-tensor bug.
- The multicam trainer
  (`train_multicam_precomputed_feature_implicit_dynamic.py`) **never calls
  `colorize_features`, never reads alpha, and treats `render_clip_sequence`'s
  return value as a `Tensor` despite its true return type being
  `tuple[Tensor, Tensor | None]`**. Both training (`render_view_clip` ->
  `multicam_recon_loss`) and eval (`render_full_external_views`) hit the bug.
  This is the broken seam.
- `KnownCameraTrainer.initial_step_result` (line 1897) shares the same
  tuple-as-tensor latent bug, plus it skips alpha composition even on the
  alpha-aware F!=3 path — its eval-time render goes through colorize but never
  composites against white.
- Random background is sampled in exactly one place (`recon_backward` line
  1330) with a per-step shape of `(1, 3, 1, 1)`. Eval paths use a hardcoded
  scalar `1.0` (white). There is no sampling helper, no seed control, no
  config knob.
- The legacy module-level `render_full_sequence` (line 743, dead code now —
  the trainer-class method shadows it) explicitly throws away alpha and
  documents itself as "legacy path: no colorize, no alpha composition." It is
  reachable only from `train_camera_implicit_dynamic.py`, an older trainer
  that uses the F=3-only path.

## Loss-path call graph

### Single-cam (Trainer in `train_video_token_implicit_dynamic.py`)

```
Trainer.step (line 1439)
 ├─ self.optimizer.zero_grad
 ├─ self.sample_clip                  -> sequence_data, clip_frames, clip_times
 ├─ self.model_input_for_clip
 ├─ self.forward_clip                 -> decoded: GaussianSequence (incl. cameras, camera_state)
 ├─ self.build_camera_loss            -> camera_loss + (motion, temporal, global)   [line 1250]
 ├─ self.build_bank_rate_loss         -> bank_rate_loss + per-term dict             [line 1266]
 ├─ self.recon_backward(              # does the actual forward+backward of recon
 │     clip_frames,
 │     decoded,
 │     regularizer_loss = camera_loss + bank_rate_loss,    # added on the LAST chunk
 │     keep_preview)
 │   ├─ random_bg = torch.rand(3, ...)                    [line 1330]
 │   └─ for each chunk in [microbatch | framewise | batched]:
 │       ├─ chunk_features, chunk_alpha = render_clip_sequence(...)   [line 1335]
 │       ├─ if self.colorize is not None:
 │       │   ├─ splat_rgb = self.colorize_features(chunk_features, cameras)
 │       │   └─ if chunk_alpha is not None:
 │       │       chunk_renders = α · splat_rgb + (1-α) · random_bg     [line 1357]
 │       │     else:
 │       │       chunk_renders = splat_rgb                              [line 1359]
 │       │ else:
 │       │   chunk_renders = chunk_features                             [line 1361]
 │       ├─ chunk_losses = reconstruction_loss_per_image(chunk_renders, target, loss_cfg)
 │       ├─ chunk_recon_loss = chunk_losses.sum() / frame_count
 │       ├─ backward_loss = chunk_recon_loss + (regularizer_loss if last_chunk else 0)
 │       └─ backward_loss.backward(retain_graph=not last_chunk)         [line 1370]
 ├─ self.optimizer.step
 └─ return StepResult
        loss = recon_loss + camera_loss.detach() + bank_rate_loss.detach()
```

Notes:
- The regularizer (`camera_loss + bank_rate_loss`) is added to **only the last
  chunk** so it doesn't get applied N times; the recon loss is normalized by
  `frame_count` per chunk so chunked summation matches the unchunked mean.
- `recon_backward` calls `.backward()` itself (multiple times for multi-chunk).
  The caller does NOT `loss.backward()`.
- `chunk_alpha is None` covers two cases: (a) F=3 fast_mac (legacy v5 RGB
  path always returns `alpha=None`), (b) any non-fast_mac mode (dense, taichi,
  tiled).

### Multicam (`MulticamPrecomputedFeatureImplicitTrainer.step`)

```
MulticamPrecomputedFeatureImplicitTrainer.step (line 248)
 ├─ self.optimizer.zero_grad
 ├─ self.sample_multicam_clip            -> clip_indices, clip_frames, clip_times, views
 ├─ self._decode_clip                    -> decoded
 ├─ self.build_bank_rate_loss            -> bank_rate_loss + terms
 ├─ self.rig_regularization_loss         -> rig_loss
 ├─ self.multicam_recon_loss(decoded, clip_indices, views, keep_preview):
 │   └─ for each view in views:
 │       ├─ rendered = self.render_view_clip(decoded, view, clip_indices)   # line 200
 │       │              └─ render_clip_sequence(...) -> (features, alpha)   # tuple, NOT tensor
 │       ├─ target = resize_images(self.multicam_bundle.train_frames[view, clip_indices], render_size)
 │       └─ recon_loss += reconstruction_loss_per_image(rendered, target, loss_cfg).mean()
 │              # ^^ rendered is a TUPLE here. Will crash or silently compare
 │              # the wrong dtype/shape.
 ├─ loss = recon_loss + bank_rate_loss + rig_loss
 ├─ loss.backward()                      # single-shot backward, unlike single-cam's chunked backwards
 └─ self.optimizer.step
```

The contrast IS the load-bearing finding:

- **No `self.colorize_features(...)` anywhere.** F!=3 multicam configs (e.g.
  `local_mac_overfit_wan_vace_feature_implicit_camera_*`) decode to F-channel
  feature splats but compute the recon loss against an F-channel rendered
  buffer, with GT being 3-channel RGB. Loss math doesn't even broadcast.
- **No alpha-aware composition.** Even if the colorize call were inserted,
  there is no `α · splat_rgb + (1-α) · bg` step; views inherit the legacy
  pre-alpha behavior.
- **No random per-step background.** Multicam loss path doesn't see
  `random_bg` and can't, because there is no composition step to drop it
  into.
- **Single backward call.** Multicam does `loss.backward()` directly on the
  combined tensor; single-cam does `chunk_recon_loss.backward(retain_graph=...)`
  per chunk inside `recon_backward`. They use different gradient flow
  strategies for the same conceptual operation. Any unification has to
  decide whether to keep the chunked strategy (memory) or the single-shot
  one (simplicity).

## Per-call-site table for `render_clip_sequence` and friends

`render_clip_sequence` is defined at
`train_video_token_implicit_dynamic.py:556` and returns
`tuple[torch.Tensor, torch.Tensor | None]` (rendered features, alpha mask).

| Caller (file:line) | Method | Expects | Has | Status |
|---|---|---|---|---|
| `train_video_token_implicit_dynamic.py:778` (`render_full_sequence` module-level) | `render_clip_sequence` | tuple | tuple unpack `_alpha_unused` | OK (alpha intentionally discarded) |
| `train_video_token_implicit_dynamic.py:1335` (`Trainer.recon_backward`) | `render_clip_sequence` | tuple | tuple unpack | OK |
| `train_video_token_implicit_dynamic.py:1377` (`Trainer.render_decoded_clip`) | `render_clip_sequence` | tuple (passes through) | tuple | OK as a wrapper |
| `train_video_token_implicit_dynamic.py:1406` (`Trainer.initial_step_result`) | `self.render_decoded_clip` | tuple | tuple unpack `(rendered_features, alpha_clip)` | OK |
| `train_video_token_implicit_dynamic.py:1584` (`Trainer.render_full_sequence` method) | `render_clip_sequence` | tuple | tuple unpack `(rendered_features, alpha_clip)` | OK |
| `train_video_token_implicit_dynamic.py:1897` (`KnownCameraTrainer.initial_step_result`) | `self.render_decoded_clip` | tuple | **assigns to single Tensor `rendered_features`** | **BUG** |
| `train_video_token_implicit_dynamic.py:1954` (`KnownCameraTrainer.render_full_sequence`) | `render_clip_sequence` | tuple | tuple unpack | OK |
| `train_multicam_precomputed_feature_implicit_dynamic.py:179` (`render_view_clip`) | `render_clip_sequence` | declares `-> torch.Tensor` | tuple | **BUG** |
| `train_multicam_precomputed_feature_implicit_dynamic.py:301` (`render_full_external_views`) | `render_clip_sequence(...).detach().cpu()` | tuple does not have `.detach()` | tuple | **BUG** |
| `probe_colorize_matrix.py:90` | `render_clip_sequence` | tuple | tuple unpack `_alpha_unused` | OK |
| `probe_colorize_init.py:85` | `render_clip_sequence` | tuple | tuple unpack | OK |

`render_gaussian_frames` (legacy single-tensor return) callers:

| Caller (file:line) | Status |
|---|---|
| `dynamicTokenGS.py:401` (`render_frame_batch`) | OK (legacy F=3 path; `render_gaussian_frames` strips alpha for them) |
| `train_video_token_implicit_dynamic.py:35` import only | n/a |

`render_gaussian_frames_alpha_aware` callers:

| Caller (file:line) | Status |
|---|---|
| `train_video_token_implicit_dynamic.py:569` (inside `render_clip_sequence`) | OK (correct tuple return) |

So three concrete bugs all rooted in the same April-29 tuple-arity change:
- `KnownCameraTrainer.initial_step_result:1897` (eval-only diagnostic, masked
  by `@torch.no_grad`; visible only when running known-camera config with
  feature splatting)
- `multicam_precomputed.render_view_clip:179` (TRAINING path; F!=3 multicam
  is currently non-functional)
- `multicam_precomputed.render_full_external_views:301` (eval path; same
  arity bug as above)

## Composition formula instances table

Composition pattern: `final_rgb = α · splat_rgb + (1 - α) · bg`, expressed as
`alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * bg`. Found at four
sites in single-cam, zero sites in multicam.

| File:line | Background | Random/Fixed | Context |
|---|---|---|---|
| `train_video_token_implicit_dynamic.py:1357` | `random_bg` (per-step `torch.rand(3,...)`) | random | training (`Trainer.recon_backward`) — gradient flows |
| `train_video_token_implicit_dynamic.py:1412` | `1.0` (scalar white broadcast) | fixed | eval (`Trainer.initial_step_result`, `@torch.no_grad`) |
| `train_video_token_implicit_dynamic.py:1604` | `1.0` (scalar white broadcast) | fixed | eval (`Trainer.render_full_sequence`, `@torch.no_grad`) |
| `train_video_token_implicit_dynamic.py:1974` | `1.0` (scalar white broadcast) | fixed | eval (`KnownCameraTrainer.render_full_sequence`, `@torch.no_grad`) |

Notably absent (composition would belong here, but is missing):

| Should-have-been site | Why missing |
|---|---|
| `train_video_token_implicit_dynamic.py:~1900` (`KnownCameraTrainer.initial_step_result`) | After colorize, just assigns `rendered_clip = self.colorize_features(rendered_features, ...)` with no alpha branch. Even if the tuple-arity bug at line 1897 were fixed, no composition would be done here — eval renders skip the white bg and feed raw colorized features into recon-loss. |
| `train_multicam_precomputed_feature_implicit_dynamic.py:multicam_recon_loss (~200)` | Multicam never adopted alpha. F!=3 here will crash; F=3 silently runs with wrong-shape data. |
| `train_multicam_precomputed_feature_implicit_dynamic.py:render_full_external_views (~295-309)` | Same — feeds raw renderer output into eval metrics. |

## Loss assembly per trainer

Each entry is "what scalar tensor gets used for the gradient and the
`StepResult.loss` field":

- **`Trainer` (single-cam, implicit camera)** —
  `recon_backward()` does N `chunk_recon_loss.backward()` calls, last chunk
  also includes `(camera_loss + bank_rate_loss)`. Reported total
  `loss = recon_loss + camera_loss.detach() + bank_rate_loss.detach()`. The
  `recon_loss` itself is the sum of L1 + DSSIM (or L1 + MSE, depending on
  `loss_cfg.type`) per-image, averaged across frames.
- **`KnownCameraTrainer` (subclasses Trainer)** — same as above but
  `camera_loss = 0` because cameras are fixed; only `recon_loss + bank_rate_loss`
  contribute. Inherits `recon_backward`.
- **`PrecomputedFeatureImplicitTrainer` (subclasses Trainer)** — no
  step/loss override; uses parent's `step` and `recon_backward`. Same loss
  shape as Trainer.
- **`MulticamPrecomputedFeatureImplicitTrainer`** — own `step` and own
  `multicam_recon_loss` (line 189). Does NOT call `recon_backward`. Loss is
  `recon_loss + bank_rate_loss + rig_loss` where `recon_loss` is averaged
  across views (sum of per-view `reconstruction_loss_per_image(...).mean()`
  divided by `len(views)`). Single `loss.backward()` call.
- **`run_training` in `train_camera_implicit_dynamic.py`** (older
  per-frame-loop trainer; not class-based, uses module-level
  `render_implicit_frame`) — Loss is
  `recon_loss + camera_motion_weight·camera_motion + camera_temporal_weight·camera_temporal + camera_global_weight·camera_global`,
  no bank_rate. Uses `render_gaussian_frame` (single-frame) and `loss.backward()`.

Where each scalar comes from:

- `recon_loss`: `reconstruction_loss_per_image(prediction, target, loss_cfg).mean()`.
  `loss_cfg.type` selects from `{mse, l1, l1_mse, standard_gs}`. `standard_gs`
  is the canonical 3DGS recipe `l1_weight·L1 + dssim_weight·DSSIM` with the
  DSSIM window/c1/c2 from config.
- `camera_motion_loss`: `camera_state.rotation_delta` and
  `translation_delta/radius` concatenated, `.pow(2).mean()`. Built in
  `Trainer.compute_camera_losses` (line 1224).
- `camera_temporal_loss`: difference of consecutive `camera_state.motion_features()`
  rows, `.pow(2).mean()`. Same function. Skipped (=0) if `clip_times.shape[1] == 1`.
- `camera_global_loss`: `camera_state.global_residuals.pow(2).mean()`. Same
  function.
- `bank_rate_loss`: in `Trainer.build_bank_rate_loss` (line 1266); requires
  five auxiliary keys (`static_opacities`, `dynamic_opacities`, `dynamic_A_mu`,
  `dynamic_A_rot`, `dynamic_A_alpha`). Returns the weighted sum
  `static_alpha_rate_weight·mean(static_op) + dynamic_alpha_rate_weight·mean(dyn_op) + dynamic_motion_rate_weight·|A_mu|.mean() + dynamic_rotation_rate_weight·|A_rot|.mean() + dynamic_alpha_time_rate_weight·|A_alpha|.mean()`,
  or zero if the auxiliary dict is missing keys. All weights default to 0.0.
- `rig_loss` (multicam only): `cfg["camera"]["rig_regularization_weight"] *
  camera_rig.regularization_loss()` where `regularization_loss` (`camera_rig.py:241`)
  is `global_rot.square().mean() + global_trans.square().mean() + bounded_rot.square().mean() + bounded_trans.square().mean()`.

There is **no TV loss, no opacity-entropy loss, no sparsity loss, no
smoothness/density-grid loss** anywhere in `src/train/`. Confirmed by
`grep tv_loss|tv_weight|opacity_entropy|sparsity|entropy_weight|smoothness_loss`
returning empty.

## Backward strategies

`recon_backward_strategy` is a config knob with values `{batched, microbatch,
framewise}`. Validation at `train_video_token_implicit_dynamic.py:958-963`.
Dispatches inside `Trainer.temporal_recon_chunk_size` (line 1302):

```
if strategy == "batched":   chunk = frame_count            # one big chunk
if strategy == "framewise": chunk = 1                      # T chunks
else:                       chunk = min(temporal_microbatch_size, frame_count)
```

This affects only `Trainer.recon_backward` (the chunk loop at line 1332).
The choice trades memory (smaller chunk = less peak activation memory) for
backward overhead (smaller chunk = more `.backward(retain_graph=True)`
calls). All three strategies use exactly the same alpha composition
(line 1357) — only the loop granularity changes.

Coverage:
- **`batched`** — single chunk; equivalent to one `.backward()` of the full
  recon loss. Used when memory permits.
- **`microbatch`** — chunks of `temporal_microbatch_size`. Used as a
  middle ground.
- **`framewise`** — one frame at a time; max time, min memory.

All three paths support feature splatting + alpha because the composition
lives inside the chunk loop. The multicam trainer does NOT participate in
this dispatch; it bakes in a per-view loop with single-shot backward.

## The colorize MLP application sites

`self.colorize` is an `Optional[FeatureToColor]` constructed at
`Trainer.__init__:1007-1024`. Constructed when either `cfg.colorize` is
present, or `feature_dim == 3` with the legacy parity case (the conv is
identity-initialized). Otherwise None.

`self.colorize_features(features, cameras)` wrapper at line 1060-1071 packs
the optional view conditioning, calls `self.colorize(features, view_dirs)`.

Sites that call `self.colorize` or `self.colorize_features` or read
`self.colorize is None`:

| File:line | Context | Notes |
|---|---|---|
| `train_video_token_implicit_dynamic.py:1007-1036` | `Trainer.__init__` constructor | Builds module |
| `train_video_token_implicit_dynamic.py:1060-1071` | `Trainer.colorize_features` wrapper | view-cond logic |
| `train_video_token_implicit_dynamic.py:1346-1361` | `Trainer.recon_backward` | training; alpha-aware composition with random bg |
| `train_video_token_implicit_dynamic.py:1408-1416` | `Trainer.initial_step_result` | eval; alpha-aware composition with white bg |
| `train_video_token_implicit_dynamic.py:1599-1609` | `Trainer.render_full_sequence` | eval; alpha-aware composition with white bg |
| `train_video_token_implicit_dynamic.py:1899-1902` | `KnownCameraTrainer.initial_step_result` | eval; **NO alpha composition** even when alpha would be available — just `colorize_features(rendered_features, decoded.cameras)` and assigns directly |
| `train_video_token_implicit_dynamic.py:1969-1979` | `KnownCameraTrainer.render_full_sequence` | eval; alpha-aware composition with white bg |
| `multicam.py` | — | **NO calls; trainer never touches colorize** |

The single-cam Trainer is six sites deep into a copy-paste of the same
seven-line block (rendered_features unpack -> colorize check -> alpha unpack ->
expand alpha -> compose with bg -> fall back to splat_rgb -> fall back to
features). Eval paths use white bg (`1.0`); training uses `random_bg`. The
shape of the duplication is uniform.

## The random per-step background

```
# train_video_token_implicit_dynamic.py:1325-1330
# Random per-step background (3DGS-canonical trick to remove the
# degenerate (alpha, splat_rgb) cheating manifold). Sampled ONCE per
# training step, broadcast across all chunks, frames, and pixels of
# this step. Different step → different bg → the only solution that
# works across iterations is alpha = 1 + splat_rgb = GT.
random_bg = torch.rand(3, device=clip_frames.device, dtype=clip_frames.dtype).view(1, 3, 1, 1)
```

Properties:
- Uses default global RNG. **No reproducibility seed.** Two runs with the
  same `torch.manual_seed` will diverge here unless the seed is reset every
  step; it is not.
- Sampled per `recon_backward` call (= per training step), not per chunk
  and not per frame. Broadcast `(1, 3, 1, 1)` across the chunk shape
  `(T, 3, H, W)`.
- Lives only inside `recon_backward`. No helper function, no shared module.
  Eval paths can't trivially call this; they just hardcode `1.0`.
- The choice of "training: random; eval: white" is asymmetric. The session
  note `2026-04-29_22-00-00_feature_splatting_alpha_aware_composition_session.md`
  explains the math: random bg structurally forces splats to cover the
  scene because no fixed bg lets the model cheat; eval uses white because
  the published metric is a fixed-bg deterministic render. Both are
  correct as currently written.

A sensible insertion point for a shared sampler:
- `compose_rendered_rgb(features, alpha, *, colorize, training: bool, generator: torch.Generator | None = None) -> Tensor`
  with internal `random_bg = torch.rand(3, ..., generator=generator)` for
  training and `bg = 1.0` for eval.

## Camera regularization losses

Defined at `Trainer.compute_camera_losses` (line 1224) and `build_camera_loss`
(line 1250):

- `camera_motion_loss`: how much each frame's pose deviates from the
  parametric path (rotation_delta, translation_delta/radius, squared mean).
- `camera_temporal_loss`: how much the per-frame motion features change
  between consecutive frames (squared mean of differences); zero for
  single-frame clips.
- `camera_global_loss`: how much the global camera residuals deviate from
  zero (squared mean).
- `camera_loss = motion_weight·motion + temporal_weight·temporal + global_weight·global`.

All three weights default to nonzero (`0.01`, `0.02`, `0.005` in
`LOSS_OPTION_DEFAULTS`).

The multicam trainer does not use these. Instead it has
`rig_regularization_loss` (line 207) =
`rig_regularization_weight * camera_rig.regularization_loss()`, which
penalizes the global SE3 transform's rotation+translation magnitudes plus
the per-view bounded SE3 deltas. Different concept (rig pose stability vs
implicit-camera path smoothness).

The older `train_camera_implicit_dynamic.py` (line 304-326) computes
camera_motion / camera_global / camera_temporal exactly like
`Trainer.compute_camera_losses` but inlined into `run_training` rather than
factored into helpers. This is dead code drift; the centralized
`compute_camera_losses` is the canonical implementation.

## Bank rate / TV / sparsity / opacity-entropy regularization

| Term | Where | Trainers | Knob |
|---|---|---|---|
| `static_alpha` rate | `build_bank_rate_loss:1287` | Trainer + subclasses | `loss_cfg.static_alpha_rate_weight` (default 0) |
| `dynamic_alpha` rate | line 1288 | same | `dynamic_alpha_rate_weight` (default 0) |
| `dynamic_motion` rate | line 1289 | same | `dynamic_motion_rate_weight` (default 0) |
| `dynamic_rotation` rate | line 1290 | same | `dynamic_rotation_rate_weight` (default 0) |
| `dynamic_alpha_time` rate | line 1291 | same | `dynamic_alpha_time_rate_weight` (default 0) |
| `rig_regularization` (multicam only) | `camera_rig.py:241` | MulticamPrecomputedFeatureImplicitTrainer | `cfg.camera.rig_regularization_weight` (default 1e-4) |

No TV, no opacity entropy, no sparsity, no other regularizers exist. The
bank-rate terms only fire when the model emits the five required auxiliary
keys (static/dynamic split models); other variants get a zero tensor and
zero terms.

## The `loss.backward()` call sites

Where the actual `.backward()` is invoked:

- `train_video_token_implicit_dynamic.py:1370` — `backward_loss.backward(retain_graph=not is_last_chunk)`
  inside `Trainer.recon_backward`. Called once per chunk; multi-call,
  retain_graph until the last chunk.
- `train_multicam_precomputed_feature_implicit_dynamic.py:261` — `loss.backward()`
  inside `MulticamPrecomputedFeatureImplicitTrainer.step`. Single call.
- `train_camera_implicit_dynamic.py:327` — `loss.backward()` inside the
  inline training loop of `run_training`. Single call.

Three different invocation styles in the same trainer family. Any unifier
needs to pick one (probably the chunked style, because it's the only one
that handles temporal microbatching) and have multicam join it.

## Where the alpha-aware composition needs to live for unification

The natural insertion point is a single helper that takes everything the
duplicated block needs and returns the composited render. Based on the
current four single-cam sites, one signature that covers training and eval:

```python
def compose_rendered_rgb(
    features: torch.Tensor,                # [T, F, H, W] from rasterizer
    alpha: torch.Tensor | None,            # [T, H, W] from v5_features, or None for F=3 / non-fast_mac
    *,
    colorize: FeatureToColor | None,       # may be None for legacy F=3 RGB-direct path
    cameras: tuple[Any, ...],              # for view-conditioned colorize
    background: torch.Tensor | float,      # 1.0 for eval; per-step random_bg tensor for train
    input_size: int,                       # for view-dirs camera scaling
    render_size: int,
    view_condition: str,
    detach_view_condition: bool,
) -> torch.Tensor:                         # [T, 3, H, W] RGB
    ...
```

This collapses lines 1346-1361, 1408-1416, 1599-1609, 1899-1902, 1969-1979
into a single call. The same helper, with `background=1.0`, fixes
`KnownCameraTrainer.initial_step_result`'s missing-composition bug.

A separate small helper handles the bg-sampling asymmetry:

```python
def sample_recon_background(
    *,
    training: bool,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,   # optional reproducibility
) -> torch.Tensor:
    if training:
        return torch.rand(3, device=device, dtype=dtype, generator=generator).view(1, 3, 1, 1)
    return torch.tensor(1.0, device=device, dtype=dtype)
```

The multicam trainer's `multicam_recon_loss` becomes:

```python
for view in views:
    rendered_features, alpha_clip = self.render_view_clip(decoded, view, clip_indices)
    rendered_rgb = compose_rendered_rgb(
        rendered_features, alpha_clip,
        colorize=self.colorize, cameras=self.camera_rig.cameras_for_view(view, clip_indices),
        background=random_bg,
        ...
    )
    target = resize_images(self.multicam_bundle.train_frames[view, clip_indices], render_size)
    recon_loss += reconstruction_loss_per_image(rendered_rgb, target, self.loss_cfg).mean()
```

This is exactly what the multicam trainer is missing today. The render
function returns the tuple; the helper does the composition; the existing
`reconstruction_loss_per_image` consumes the RGB.

I am NOT proposing the API; that's wave 2. I am only naming the seam. The
seam is the union of:
- `compose_rendered_rgb(features, alpha, ...)` (collapses 4 duplicates +
  fixes 3 bugs)
- `sample_recon_background(...)` (factors random vs fixed bg policy)
- a thin pre-rendering accessor `(rendered_features, alpha_clip) =
  self.render_clip(...)` so the tuple-unpack is owned by one function and
  no caller has to know the v5_features arity.

## Open questions for proposers

- **Where does `colorize` live?** Currently `self.colorize` on the trainer.
  Options: keep on trainer, move to model (so `decoded.colorize_features(...)`
  is a method on `GaussianSequence` or its parent), or make a free function
  that takes a colorize module. The view-conditioning logic complicates the
  free-function option (needs cameras + sizes from config).
- **Should bg sampling be deterministic per-step?** A `torch.Generator` per
  trainer would make `recon_backward` reproducible across reruns. Currently
  it is not.
- **Should training-time bg also be `random_bg` for the chunked path?**
  Yes (it is); but should it be fresh per chunk, per frame, or per step?
  Currently per step. The session note argues this is intentional but we
  may want to revisit (per-frame would give more bg diversity per step
  without changing asymptotic gradient).
- **How is the eval-time fixed bg exposed?** Hardcoded `1.0` everywhere
  today. Should it be a config knob (`render.eval_background = "white" |
  "black" | "gray" | float`) so feature-bg vs scene-bg cheating can be
  diagnosed?
- **Should the multicam trainer adopt `recon_backward`?** Or keep its
  per-view loop with single-shot backward? The chunked strategy adds
  complexity for a context where each view is already a chunk.
- **Should `KnownCameraTrainer.initial_step_result` be deleted or fixed?**
  It is `@torch.no_grad` and runs once at step 0; its tuple-arity bug is
  latent. If we fix `render_decoded_clip` to consistently return tuples,
  this site needs its assignment fixed too. Cleanest fix: route both
  trainers through the same shared helper.
- **Should `render_gaussian_frames` keep its alpha-stripping behavior?**
  It silently drops alpha for the F=3 path. Right now that's safe (F=3
  always has `alpha=None`). When v5_features supports F=3 alpha later,
  this helper becomes a foot-gun. Maybe deprecate in favor of
  `render_gaussian_frames_alpha_aware` everywhere.
- **What happens to `render_full_sequence` (module-level, line 743)?**
  It's reachable only from `train_camera_implicit_dynamic.py` (a legacy
  per-frame F=3 trainer) and explicitly comments itself "Module-level legacy
  path: no colorize, no alpha composition." Probably delete with the
  legacy trainer if/when that's retired.

## Concrete file paths

- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/losses.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/colorize.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/rendering.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/renderers/fast_mac.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_video_token_implicit_dynamic.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_precomputed_feature_implicit_dynamic.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_camera_implicit_dynamic.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/camera_rig.py` (regularization_loss)
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/agent_notes/loose_notes/2026-04-29_22-00-00_feature_splatting_alpha_aware_composition_session.md` (background context for the alpha rework that introduced the duplicates)
