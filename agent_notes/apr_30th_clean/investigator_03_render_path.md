# Investigator 03 — Render Path Audit

Scope: from `GaussianSequence` produced by the model to the rendered features
(or RGB) and per-pixel alpha. Files audited:

- `src/train/rendering.py` — single dispatch entry point and convenience wrappers.
- `src/train/renderers/common.py` — pinhole projection (single + batch), pixel-grid utility, `MIN_RENDER_DEPTH`.
- `src/train/renderers/projection.py` — full lens-model camera projection (`pinhole`, `radial_tangential`, `opencv_fisheye`).
- `src/train/renderers/dense.py` — pure-PyTorch reference rasterizer (`render_pytorch_3dgs[_batch]`).
- `src/train/renderers/tiled.py` — tile-binned PyTorch rasterizer (`render_pytorch_3dgs_tiled`); single-frame only.
- `src/train/renderers/taichi.py` — Taichi/Metal rasterizer (`render_taichi_3dgs[_batch]`).
- `src/train/renderers/fast_mac.py` — Metal rasterizer dispatch + projection-to-2D bridge (`render_fast_mac_3dgs[_batch]`, `project_for_fast_mac[_batch]`).
- `third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5/rasterize.py` — `gsplat_metal_v5` ops, F=3 RGB only, returns `Tensor`.
- `third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features/rasterize.py` — `gsplat_metal_v5_features` ops, F-channel + alpha, returns `tuple[Tensor, Tensor]`.

## TL;DR

- `render_gaussian_frames` (legacy, returns `Tensor`) and
  `render_gaussian_frames_alpha_aware` (new, returns `tuple[Tensor, Tensor | None]`)
  coexist; trainers half-migrated. The legacy entry point still strips alpha
  and exists only because callers were not updated. This is the central rough
  edge in the render layer.
- Alpha is exposed on exactly one path: `fast_mac` with `F != 3` (v5_features).
  All other paths return alpha=None (taichi/dense/tiled have no notion of an
  exposed accumulated-alpha output even though they compute one internally).
- The dispatch by `feature_dim = rgbs.shape[-1]` inside `fast_mac.py` is a
  silent structural branch: `F=3 -> v5` (no alpha), `F!=3 -> v5_features`
  (alpha). There is no way for an `F=3` config to opt into alpha without
  changing the channel count.
- Render-helper proliferation across trainers is large: nine distinct
  helpers, three of which are named `render_full_sequence` in different
  modules and two are *separate* methods on different `Trainer` subclasses
  in the same file. Most of them differ only in cosmetic plumbing
  (`viewport_cameras`, `render_size`, alpha handling).
- There is at least one live tuple-vs-tensor mismatch:
  `train_multicam_precomputed_feature_implicit_dynamic.render_view_clip` is
  annotated `-> torch.Tensor` but returns `tuple[Tensor, Tensor | None]`
  (because it forwards `render_clip_sequence`); its callers then call
  `.detach().cpu()` / `[0]` / pass it to `reconstruction_loss_per_image`
  on a tuple. The KnownCameraTrainer initial-step path
  (`train_video_token_implicit_dynamic.py:1897`) has the same shape: it
  binds the tuple result to `rendered_features` and treats it as a tensor
  on lines 1898–1903.

## 1. Renderer-mode dispatch table

The single dispatch is `render_gaussian_frames` (and its single-frame sibling
`render_gaussian_frame`) in `src/train/rendering.py`. `pick_renderer_mode`
(rendering.py:50) resolves `"auto"` to `dense` or `tiled` based on
`gaussian_count * H * W`. Modes:

| Mode | Backend module | Backend entry | Alpha exposed? | Batched native? | Used by configs |
|---|---|---|---|---|---|
| `dense` | `renderers/dense.py` | `render_pytorch_3dgs` / `render_pytorch_3dgs_batch` | No (alpha is computed but not returned; `return_aux` exposes `alpha_max`/`weight_sum` summaries only) | Yes (`render_pytorch_3dgs_batch`) | small/debug configs (e.g. `local_mac_overfit_image_implicit_camera*.jsonc`) |
| `tiled` | `renderers/tiled.py` | `render_pytorch_3dgs_tiled` | No | No (single-frame only; `render_gaussian_frames` falls back to a Python list-comprehension over `render_gaussian_frame`) | `local_mac_overfit_*_tiled.jsonc` |
| `taichi` | `renderers/taichi.py` | `render_taichi_3dgs` / `render_taichi_3dgs_batch` | No (background composited inside the renderer, alpha is not surfaced) | Yes (`render_taichi_3dgs_batch` -> `taichi_splatting.rasterizer.rasterize_batch`) | `local_mac_overfit_prebaked_camera_*_taichi*.jsonc` |
| `fast_mac` (F=3) | `renderers/fast_mac.py` -> `torch_gsplat_bridge_v5` | `rasterize_projected_gaussians` (v5) | **No** (returns Tensor only; `fast_mac.py` returns `(features, None)` to keep tuple-shape uniform) | Yes (native B,G,2) | RGB baseline configs (most `*_fast_mac_*.jsonc`) |
| `fast_mac` (F=32, F=64, …) | `renderers/fast_mac.py` -> `torch_gsplat_bridge_v5_features` | `rasterize_projected_gaussians` (v5_features) | **Yes** ([B,H,W] accumulated alpha = `1 - T_final`) | Yes | F32/F64 alpha configs (`*_features_F32*.jsonc`, e.g. `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc`) |

Notes on the dispatch itself:

- `rendering.py:308–339` (fast_mac batch branch) and `rendering.py:168–194`
  (fast_mac single-frame branch) both unpack the tuple and discard
  `_alpha`. The comments at lines 169–172 and 310–313 make this a
  deliberate "legacy strip" rather than a forgotten TODO.
- `rendering.py:340–360` (the `else` after the four typed modes) is a
  silent fallback that loops `render_gaussian_frame` per camera. It is
  currently only reachable via `mode == "tiled"` because the explicit
  `tiled` branch only exists in the single-frame path. So `tiled` batches
  are emulated by stacking single-frame renders.
- `pick_renderer_mode` (rendering.py:50) does not understand tile/F
  thresholds. The auto path only chooses dense vs tiled and is not
  alpha-aware.

## 2. The `(features, alpha)` tuple plumbing

Tuple convention introduced in `fast_mac.py`:

| Function | File:line | Returns |
|---|---|---|
| `render_fast_mac_3dgs` | `renderers/fast_mac.py:286` | `tuple[Tensor[3,H,W] \| Tensor[F,H,W], Tensor[H,W] \| None]` |
| `render_fast_mac_3dgs_batch` | `renderers/fast_mac.py:354` | `tuple[Tensor[B,3,H,W] \| Tensor[B,F,H,W], Tensor[B,H,W] \| None]` |

Trace from the entry-points in `rendering.py`:

| Entry point | File:line | Returns | Alpha behaviour |
|---|---|---|---|
| `render_gaussian_frame` | `rendering.py:106` | `Tensor[3 or F, H, W]` (or `(Tensor, dict)` when `dense + return_aux`) | strips alpha for fast_mac (line 172–194); other modes never had it |
| `render_gaussian_frames` | `rendering.py:234` | `Tensor[T, 3 or F, H, W]` (or `(Tensor, dict)` for dense+return_aux) | strips alpha for fast_mac (line 314–339) |
| `render_gaussian_frames_alpha_aware` | `rendering.py:363` | `tuple[Tensor[T, 3 or F, H, W], Tensor[T, H, W] \| None]` | only fast_mac route returns a real alpha; everything else returns `(features, None)` (line 416–434) |

Trainer-side wrappers:

| Wrapper | File:line | Returns | Alpha used? |
|---|---|---|---|
| `render_clip_sequence` (module-level) | `train_video_token_implicit_dynamic.py:556` | `tuple[Tensor, Tensor \| None]` | callers vary |
| `Trainer.render_decoded_clip` | `train_video_token_implicit_dynamic.py:1374` | `tuple[Tensor, Tensor \| None]` (annotated) | implicit-camera trainer uses alpha; KnownCameraTrainer (line 1897) treats the result as a Tensor — mismatch |
| `MulticamTrainer.render_view_clip` | `train_multicam_precomputed_feature_implicit_dynamic.py:177` | annotated `-> torch.Tensor`, actually returns `tuple[Tensor, Tensor \| None]` | callers (line 200, 295, 309) treat it as a Tensor — bug in the multicam path the moment the renderer returns a real tuple |

### Tuple call-site audit

| Caller | Line | Expected | Actual return | Status |
|---|---|---|---|---|
| `Trainer.recon_backward` (chunk_features, chunk_alpha = render_clip_sequence(...)) | `train_video_token_implicit_dynamic.py:1335` | tuple | tuple | OK |
| `Trainer.initial_step_result` (rendered_features, alpha_clip = self.render_decoded_clip(...)) | `train_video_token_implicit_dynamic.py:1406` | tuple | tuple | OK |
| `Trainer.render_full_sequence` (rendered_features, alpha_clip = render_clip_sequence(...)) | `train_video_token_implicit_dynamic.py:1584` | tuple | tuple | OK |
| `KnownCameraTrainer.initial_step_result` (rendered_features = self.render_decoded_clip(...)) | `train_video_token_implicit_dynamic.py:1897` | Tensor | tuple | **BUG**: subsequent `rendered_features[0]` and `self.colorize_features(rendered_features, …)` will operate on a tuple, not a Tensor |
| `KnownCameraTrainer.render_full_sequence` (rendered_features, alpha_clip = render_clip_sequence(...)) | `train_video_token_implicit_dynamic.py:1954` | tuple | tuple | OK |
| Module-level `render_full_sequence` (rendered_features_clip, _alpha_unused = render_clip_sequence(...)) | `train_video_token_implicit_dynamic.py:778` | tuple, alpha discarded | tuple | OK (legacy path, no colorize) |
| `MulticamTrainer.multicam_recon_loss` (rendered = self.render_view_clip(...)) | `train_multicam_precomputed_feature_implicit_dynamic.py:200` | Tensor | tuple | **BUG**: `reconstruction_loss_per_image(rendered, target, …)` and `rendered[0].detach()` both wrong on a tuple |
| `MulticamTrainer.render_full_external_views` (.detach().cpu() on `render_view_clip` and direct `render_clip_sequence` results) | `train_multicam_precomputed_feature_implicit_dynamic.py:295, 309` | Tensor | tuple | **BUG**: `.detach()` is called on the tuple, which will throw |
| `probe_colorize_init.py:85` (`_alpha_unused = render_clip_sequence(...)`) | — | tuple, alpha discarded | tuple | OK |
| `probe_colorize_matrix.py:90` (`_alpha_unused = render_clip_sequence(...)`) | — | tuple, alpha discarded | tuple | OK |

The bugs in the multicam trainer and `KnownCameraTrainer.initial_step_result`
exist *as soon as* `render_clip_sequence` is run on a fast_mac config with
F!=3. For F=3 paths the alpha is `None`, but the result is still a tuple,
so the failure is unconditional in current code — it just hasn't fired
because those code paths are not exercised on F!=3 alpha configs.

## 3. `feature_dim` dispatch in `fast_mac.py`

`render_fast_mac_3dgs` (line 286) and `render_fast_mac_3dgs_batch` (line
354) both inspect `feature_dim = rgbs.shape[-1]`. Branch:

```
if feature_dim == 3:
    -> torch_gsplat_bridge_v5.rasterize_projected_gaussians
       returns Tensor; we wrap as (features.clamp(0,1), None)
else:
    -> torch_gsplat_bridge_v5_features.rasterize_projected_gaussians
       returns (Tensor, Tensor); we forward (features, alpha)
```

Implications:

- The dispatch is *purely* by channel count. There is no config knob to
  force the alpha-aware backend at F=3 (e.g. for a baseline that wants
  alpha for visualization or for the alpha-composited training loss).
- The F=3 branch hard-clamps the output to `[0, 1]` (line 335, 403). The
  F!=3 branch does not clamp — features can be negative or very large
  before the colorize MLP. This asymmetry is correct given the semantics
  but is a structural branch.
- The two backends differ in `RasterConfig` schema (see section 6); the
  channel-count check is also the implicit selector for which config
  type to construct.

## 4. The `feature_background` knob

`FastMacRendererConfig` (renderers/fast_mac.py:46) carries two background
fields:

- `background: tuple[float, float, float]` — RGB-only, fed to v5
  `RasterConfig.background` (line 113).
- `feature_background: float | tuple[float, ...]` — fed to v5_features
  `RasterConfig.background` (line 142). Helper `_make_v5_features_config`
  (line 122) accepts a scalar (broadcast to F channels by the kernel via
  `_background_for_feature_dim` in the bridge, line 104) or an exact
  F-length tuple.

Schema parsing:

- `_normalize_rgb_background` (line 30) requires exactly 3 values.
- `_normalize_feature_background` (line 37) accepts a scalar (most common,
  `0.0`) or any non-empty tuple.

Trainer-side, the knobs are largely vestigial under alpha-aware composition:
`Trainer.recon_backward` (`train_video_token_implicit_dynamic.py:1330`)
samples a per-step random RGB background and composites against the
rasterizer output post-colorize using `chunk_alpha`. So the rasterizer's
own `background` setting is overwritten downstream for alpha-aware paths;
it only matters where the rasterizer composites internally (F=3 v5 path,
and the F!=3 path when `chunk_alpha` is None — which it shouldn't be).

The dual knob is necessary only because the two bridges have incompatible
`RasterConfig.background` shapes (3-tuple vs F-tuple). It could be
unified by a single `background: tuple[float, ...]` whose length is
checked against `feature_dim` at render time, removing the F=3-specific
shape constraint.

## 5. The projection chain

Two parallel APIs live in `renderers/`:

- `renderers/common.py:project_gaussians_2d[_batch]` — pinhole-only,
  fast (no per-camera lens model), takes raw `fx, fy, cx, cy` scalars or
  vectors. Sorts by camera-frame depth and returns
  `(means2d, inv_cov2d, cov2d, opacities, rgbs)`.
- `renderers/projection.py:project_gaussians_2d_camera[_batch]` —
  CameraSpec-aware, supports `pinhole`, `radial_tangential` (k1,k2,p1,p2,k3),
  and `opencv_fisheye` (k1..k4). Builds the analytic
  `d(pixel)/d(camera_xyz)` Jacobian in `_radial_tangential_project_normalized`
  / `_opencv_fisheye_project_normalized`, then composes
  intrinsics ⋅ lens ⋅ norm jacobians. Pinhole inputs short-circuit back to
  `project_gaussians_2d` for parity; mixed batches loop the single-frame
  variant.

`fast_mac.project_for_fast_mac[_batch]` (lines 173, 227) wraps both:

| Field | Value |
|---|---|
| Inputs | `means3d [G,3] or [B,G,3]`, `scales [G,3] or [B,G,3]`, `quats [G,4] or [B,G,4]` (WXYZ), `opacities [G,1] or [B,G,1]`, `rgbs [G,F] or [B,G,F]`, plus `fx/fy/cx/cy`, `camera`/`cameras`, `camera_to_world`, `near_plane`, `projection_mode` |
| Outputs | `means2d.contiguous()`, `conics = _conics_from_inv_cov(inv_cov2d)` (shape [..., 3] = (a, 0.5*(b+c), d)), `colors.contiguous()`, `opacities.squeeze(-1).contiguous()`, `depths = _rank_depths(...)` |
| Projection modes | `legacy_pinhole` (uses `fx/fy/cx/cy`, ignores camera lens model) — OR — `camera_model` (uses `CameraSpec.lens_model`, distortion, intrinsics; raises if camera missing) |
| `near_plane` wiring | passed through to projection helpers; defaults to `MIN_RENDER_DEPTH = 1e-4` (`common.py:3`); `_validate_near_plane` enforces positive |
| `bound_scale` wiring | NOT used by fast_mac. `bound_scale` is consumed only by `tiled.compute_gaussian_bounds` (tile mapper). All non-tiled paths ignore it. |

The `_rank_depths` trick (lines 162–170) replaces real depths with a
normalized rank index `[0, 1]`. This is correct because
`project_gaussians_2d` already sorts front-to-back; the kernel only needs
a non-negative monotone scalar to preserve order. The taichi path does
the same (`taichi.py:171–173` and `241–244`).

Mode resolution: `_resolve_camera_projection_mode` (rendering.py:88)
upgrades `auto` to `camera_model` when any input camera is non-pinhole,
and refuses `legacy_pinhole` for non-pinhole batches with a clear
ValueError.

## 6. v5 vs v5_features bridge surface

| Aspect | v5 (`torch_gsplat_bridge_v5`) | v5_features (`torch_gsplat_bridge_v5_features`) |
|---|---|---|
| Custom-op namespace | `torch.ops.gsplat_metal_v5` | `torch.ops.gsplat_metal_v5_features` |
| `RasterConfig.background` | `tuple[float, float, float]` (RGB only, 3-fixed) | `tuple[float, ...]` length 1 (broadcast) or F |
| `_make_meta` | 12-int / 7-float meta tensors (no feature_dim, no per-channel bg) | 12-int / 4+feature_cap-float meta tensors, last meta_i32 entry is `feature_dim`, meta_f32 carries padded F-channel bg |
| Forward bridge | `bin -> render_fast_forward_state -> [render_overflow_forward] -> render_fast_forward_eval` (eval path) | `bin -> render_fast_forward_state -> [render_overflow_forward] -> render_fast_forward_eval`, all returning a paired `(image, alpha)` |
| Forward signature (Python) | `rasterize_projected_gaussians(...) -> Tensor` | `rasterize_projected_gaussians(...) -> tuple[Tensor, Tensor]` |
| Backward signature | `_RasterizeProjectedGaussiansV5.backward(ctx, grad_out)` | `_RasterizeProjectedGaussiansV5Features.backward(ctx, grad_features, grad_alpha)` (handles `None` for either gradient) |
| Runtime caps | `GSP_TILE_SIZE`, `GSP_FAST_CAP`, `GSP_CHUNK` | adds `GSP_FEATURE_CAP` (default 64); enforces `feature_dim <= feature_cap` in `_runtime_validate` |
| `nn.Module` shim | `ProjectedGaussianRasterizer.forward -> Tensor` | `ProjectedGaussianRasterizer.forward -> Tensor` (annotation says `Tensor` but actually returns `tuple[Tensor, Tensor]` — minor inconsistency in the bridge) |

The two bridges share ~95% of their structure (sort/perm, overflow gather,
batch chunking, autograd Function, profile path). The only meaningful
differences are: (a) presence of alpha output and gradient, (b) feature
dimension carried through meta + dynamic background length, (c) op
namespace name.

## 7. Render context (camera + viewport)

Helpers (lifted from `rendering.py` and `train_video_token_implicit_dynamic.py`):

| Helper | File:line | Purpose |
|---|---|---|
| `camera_for_viewport` | `rendering.py:25` | Scale a single CameraSpec's intrinsics from `(source_h, source_w)` to `(target_h, target_w)`. Pose stays the same. |
| `viewport_cameras` | `train_video_token_implicit_dynamic.py:472` | Map a tuple of CameraSpecs through `camera_for_viewport`. Square-only (uses `input_size` for both H/W). |
| `pick_renderer_mode` | `rendering.py:50` | Resolve `"auto"` to `dense`/`tiled` based on `gaussian_count * H * W`. Does not consider F. |
| `build_or_reuse_grid` | `rendering.py:64` | Cache a [H,W,2] pixel grid for the dense path; returns the supplied grid if shape+device match. |
| `_camera_scalar_vector` | `rendering.py:219` | Stack `getattr(camera, field)` across a list of cameras into a `[T]` tensor on the supplied device, handling both scalar floats and 0-d tensors. Mirror copy in `renderers/projection.py:23` (with explicit dtype). |

`viewport_cameras` is the entry-funnel for every trainer path:
`render_clip_sequence` (line 568) calls it; the multicam trainer goes
through `render_view_clip -> render_clip_sequence` (line 179). The square
constraint (using `input_size` for both H/W) is a real assumption baked
into every config — it is fine for the current 128/256-square configs but
will break the moment we want a non-square render target.

## 8. Dense + return_aux

`render_pytorch_3dgs[_batch]` (dense.py:105, 170) is the only path that
honours `return_aux=True`. The aux dict is `{alpha_max, weight_sum}` per
spatial extent (dense.py:97). All other modes raise `ValueError` at
`rendering.py:144` and `:280` ("return_aux is only supported by the dense
renderer.") — this is enforced both for the single-frame and batch paths.

`render_gaussian_frames_alpha_aware` does not interact with `return_aux`:
its non-fast_mac path (line 416) calls `render_gaussian_frames` *without*
`return_aux`, and treats a tuple result as an error
(`raise RuntimeError("...does not support return_aux modes")`, line 433).
This means `dense + alpha_aware` cannot return both alpha and dense aux
in the current API — but neither does dense ever return alpha, so the
"no alpha" path (`features, None`) is the only outcome.

## 9. `_camera_scalar_vector`

Two near-identical copies live at:

- `rendering.py:219` — bare implementation, infers dtype as `torch.float32`.
- `renderers/projection.py:23` — accepts an explicit `dtype` argument,
  used by `project_gaussians_2d_camera_batch` (projection.py:283).

Both produce a 1-D `Tensor[T]` from a list of CameraSpec values. The
duplicate is a harmless minor smell; collapsing them would not change
behaviour.

## 10. Render-helper inventory

| Helper | File:line | Returns | Used by | Duplicated logic |
|---|---|---|---|---|
| `render_gaussian_frame` | `rendering.py:106` | `Tensor` (or `(Tensor, aux)` for dense+return_aux) | `dynamicTokenGS.render_one_frame`, `tokenGS.render_single_frame`, `train_camera_implicit_dynamic.render_implicit_frame`, `splat_renderer_benchmark.run_custom_case` (line 366), `research_experiments/gauge_fields/train_splat_baseline.py:251,393` | core dispatch lives here; thin wrappers re-pack render_cfg knobs |
| `render_gaussian_frames` | `rendering.py:234` | `Tensor` (or `(Tensor, aux)` for dense+return_aux) | `dynamicTokenGS.render_frame_batch` (line 401) | core dispatch; loops `render_gaussian_frame` for `tiled` |
| `render_gaussian_frames_alpha_aware` | `rendering.py:363` | `tuple[Tensor, Tensor \| None]` | `train_video_token_implicit_dynamic.render_clip_sequence` (line 569) | mostly forwards to `render_fast_mac_3dgs_batch`; reconstructs everything else via `render_gaussian_frames` and tacks on `None` |
| `render_clip_sequence` (module-level) | `train_video_token_implicit_dynamic.py:556` | `tuple[Tensor, Tensor \| None]` | called by 5 trainer methods (Trainer.recon_backward, Trainer.initial_step_result, Trainer.render_full_sequence, Trainer.render_decoded_clip, KnownCameraTrainer.render_full_sequence, module-level render_full_sequence, MulticamTrainer.render_view_clip + render_full_external_views, probe_colorize_init.py:85, probe_colorize_matrix.py:90) | wraps `render_gaussian_frames_alpha_aware` after `viewport_cameras` |
| `render_decoded_clip` | `train_video_token_implicit_dynamic.py:1374` (Trainer method) | `tuple[Tensor, Tensor \| None]` (annotated) | `Trainer.initial_step_result` (line 1406, OK), `KnownCameraTrainer.initial_step_result` (line 1897, BUG — treats as Tensor) | thin wrapper around `render_clip_sequence` with `decoded.cameras` injected |
| `render_view_clip` | `train_multicam_precomputed_feature_implicit_dynamic.py:177` (Multicam method) | annotated `Tensor`, actually `tuple[Tensor, Tensor \| None]` | `multicam_recon_loss` (line 200, BUG), `render_full_external_views` (line 295, BUG) | thin wrapper around `render_clip_sequence` with `camera_rig.cameras_for_view(view, clip_indices)` |
| `render_full_external_views` | `train_multicam_precomputed_feature_implicit_dynamic.py:288` (Multicam method) | `tuple[list[Tensor], list[Tensor], dict]` | `validation_video_payload` (line 321) | per-view loop calling `render_view_clip` (train) and `render_clip_sequence` (heldout) |
| `render_full_sequence` (module-level) | `train_video_token_implicit_dynamic.py:743` | `tuple[Tensor, CameraState, dict]` | not called from current code (legacy / standalone path); was the entry before the Trainer class form |
| `render_full_sequence` (Trainer method) | `train_video_token_implicit_dynamic.py:1550` | `tuple[Tensor, CameraState \| None, dict, Tensor \| None, Tensor \| None]` (rendered + state + decoded metrics + feature PCA buffer + alpha) | `Trainer.validation_video_payload` (line 1662) | per-clip loop with feature_pca_log + alpha accumulation |
| `render_full_sequence` (KnownCameraTrainer method) | `train_video_token_implicit_dynamic.py:1926` | same 5-tuple as above | `KnownCameraTrainer.validation_video_payload` | known-camera variant: uses `sequence_data.cameras` instead of `decoded.cameras`, no CameraState; otherwise structurally identical to the implicit version |
| `render_full_sequence` | `train_camera_implicit_dynamic.py:164` | `tuple[Tensor, list[CameraSpec], CameraState]` | `validation_video_payload` (line 368) | parallel implementation of the same loop without alpha/feature plumbing |
| `render_full_sequence` | `dynamicTokenGS.py:419` | `Tensor` | `validation_video_payload` (line 668) | original / legacy variant: no CameraState, no alpha, no feature buffer |

Three distinct same-name `render_full_sequence` functions live in three
modules; two more are subclass methods on `Trainer` and `KnownCameraTrainer`
in the same file. They differ in (a) what extra metadata they accumulate
across clips (CameraState, decoded metrics, feature PCA, alpha), (b)
whether cameras come from `decoded.cameras` vs `sequence_data.cameras`,
(c) whether they use `prepare_clip` vs raw slicing.

## What should the unified API look like (seams only)

Natural seams to refactor along:

- A single `RenderedClipBundle` (features, alpha, cameras, optional
  PCA-feature buffer, optional aux) flowing out of every renderer entry
  point. Fields that don't exist for a given backend are explicitly
  `None`, not stripped silently. This collapses the
  `render_gaussian_frames` / `render_gaussian_frames_alpha_aware` split.
- A single `render(seq, cameras, mode, *, render_size, render_cfg,
  return_aux=False) -> RenderedClipBundle`. The `tiled`-batch
  list-comprehension fallback can stay inside, but the API is uniform.
- `render_clip_sequence` (the trainer-side wrapper) is thin enough that
  it could simply *be* the public renderer entry, with `viewport_cameras`
  inlined or moved into `RenderConfig.from_mapping`.
- `render_view_clip`, `render_decoded_clip`, and `render_one_frame`/
  `render_implicit_frame`/`render_single_frame` are all
  configuration-renaming wrappers around the same dispatch. If
  `render_clip_sequence` is the public API and `render_cfg` is a typed
  object, none of them are needed.
- The five `render_full_sequence` variants share one structural skeleton
  (clip loop + per-frame buffer fill + camera-state merge + optional
  alpha/feature buffer). A single helper parameterized by
  `(sequence_iterator, decode_step, post_decode_hook,
  accumulate_camera_state)` would absorb all five.
- The `feature_dim==3` branch in `fast_mac.py` and the dual `background`
  / `feature_background` knobs collapse if v5_features can serve F=3
  (it currently can — its background is length-1-or-F). Whether to
  always route through v5_features is a product question, not an API one.

## Open questions for proposers

- Should `fast_mac` always route through `v5_features` (using `alpha`
  internally even at F=3) so that the structural F-dispatch in
  `fast_mac.py` is removed? The price is the additional alpha output
  channel and slightly different numerics (no `clamp(0,1)` post-write).
  The v5 fork would still exist as a fast RGB-only path but be unused
  from the dynaworld trainer.
- Should `render_gaussian_frame` (single-frame) survive the refactor?
  The benchmark (`splat_renderer_benchmark.py:366`) and the gauge-field
  research script (`research_experiments/gauge_fields/train_splat_baseline.py`)
  are the only remaining callers; trainers all use the batch entry.
- Should `depth` be added to the `RenderedClipBundle` now that alpha is
  there? The Metal kernels do not return depth today, but the dense
  reference path holds it transiently and could expose it cheaply.
  v5_features could be extended with a depth output without disturbing
  the alpha/forward contract; v5 is unlikely to be touched.
- Should `_camera_scalar_vector` (twice) and the `viewport_cameras` /
  `camera_for_viewport` pair migrate from `rendering.py` into a typed
  CameraBatch primitive (matching the `C2W`, `Intrinsics3x3` typed
  primitives elsewhere)?
- Is the `(features, None)` tuple convention from
  `render_gaussian_frames_alpha_aware`'s non-fast_mac branch (line 434)
  the right shape, or should non-fast_mac modes raise instead of
  silently returning `None` for alpha? The current shape lets a trainer
  written for alpha-aware composition fall back to non-alpha modes
  during debugging without crashing — at the cost of silently degrading
  to an opaque-only render.

agent_notes/apr_30th_clean/investigator_03_render_path.md
