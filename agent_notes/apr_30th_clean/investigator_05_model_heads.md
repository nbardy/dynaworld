# Investigator 05 — Model architecture and Gaussian heads

Scope: every model class that produces a `GaussianSequence`, every parameter
head that decodes tokens to splat parameters, the colorize MLP, view
conditioning, and how `feature_dim` threads through everything.

## TL;DR

- There are TWO parallel model lineages in the tree. The active line is
  `gs_models/dynamic_video_token_gs_implicit_camera.py` (12 classes, 2.5k
  lines, all dispatched by `train_video_token_implicit_dynamic.build_model_from_config`).
  The legacy line is `dynamicTokenGS.py` + `gs_models/{token_gs.py,
  dynamic_token_gs.py, dynamic_token_gs_implicit_camera.py,
  dynamic_token_gs_separated_implicit_camera.py}` and has its own trainers
  (`tokenGS.py`, `dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`).
  The legacy line still serves as a utility holder — `train_video_token_implicit_dynamic.py`,
  `train_camera_implicit_dynamic.py`, `export_dynaworld_browser_bundle.py`,
  and `probe_init_diagnostics.py` all import `pick_device`, `fast_attn_context`,
  `configure_fast_attn`, and `select_window_indices` from `dynamicTokenGS`.
  That coupling is why the legacy file has not been deleted yet.
- There are 3 head classes: `GaussianParameterHeads` (the canonical
  token→splat head, used by everything in the active line),
  `DynamicResidualGaussianBankHead` (the time-basis dynamic head, used only
  by the static/dynamic split path), and `ResidualFreeGaussianParameterHeads`
  (per-token base bank + bounded residuals, used by the residual-free-bank
  variants). The legacy `CanonicalGaussianParameterHeads` in
  `dynamic_token_gs_implicit_camera.py` is a separate, simpler third
  implementation with no `feature_dim` plumbing.
- `feature_dim` was threaded into 7 of the 9 active model classes plus 2
  active heads. **Two active classes are missing it**:
  `FreeGaussianBankImplicitCamera` and `LinearTimeFreeGaussianBankImplicitCamera`.
  Both have `**_unused` in the signature so the kwarg from
  `build_model_from_config` is silently dropped, and `raw_rgbs` is
  hardcoded to 3 channels (see `dynamic_video_token_gs_implicit_camera.py:1103`).
  Configs that select `variant: "free_splats"` or `"free_linear_time_splats"`
  with `model.feature_dim != 3` will silently train an RGB-3 splat bank
  while the trainer's `colorize` MLP expects an F-channel input — a
  shape error at first render rather than a logical bug, but the failure
  point is misleading.
- `FeatureToColor` (the colorize MLP) lives in `colorize.py` and is owned
  by the **Trainer**, not the model. The trainer also owns
  `colorize_view_dirs_for_features` (free function) and
  `Trainer.colorize_features` (method). The model returns whatever the
  head returned (`rgbs`), and the trainer decides whether to colorize by
  inspecting `self.colorize is None`. There is no model-side awareness of
  whether the output is RGB-3 or F-channel features beyond the F=3 sigmoid
  branch inside the heads.
- The `GaussianSequence.rgbs` field name is now load-bearing-misleading
  for F!=3 — the docstring acknowledges this and says it is kept "for
  cascade compatibility." Any rename has to happen at the same time as a
  field rename in `dynamic_video_token_gs_implicit_camera.py`,
  `runtime_types.py`, `train_video_token_implicit_dynamic.py`, and the
  rendering / debug paths.

## Model class inventory

All paths are absolute below the table; see file listing at end.

| Class | File:line | Parent | Used by which configs/variants | feature_dim threaded? | Status |
|---|---|---|---|---|---|
| `TokenGS` | `gs_models/token_gs.py:7` | `TokenGSBackbone` | `local_mac_overfit_single_image*.jsonc` via `tokenGS.py` | no (via Backbone, but trainer never passes it) | active legacy (single-image overfit smoke) |
| `DynamicTokenGS` | `gs_models/dynamic_token_gs.py:9` | `TokenGSBackbone` | `local_mac_overfit_prebaked_camera*.jsonc` via `dynamicTokenGS.py` | no (Backbone accepts it, `DynamicTokenGS.__init__` does not forward it) | active legacy (DUSt3R prebaked-camera baseline) |
| `DynamicTokenGSImplicitCamera` | `gs_models/dynamic_token_gs_implicit_camera.py:35` | `nn.Module` | `local_mac_overfit_image_implicit_camera.jsonc` via `train_camera_implicit_dynamic.py` | no | active legacy (image-encoder implicit camera baseline) |
| `DynamicTokenGSSeparatedImplicitCamera` | `gs_models/dynamic_token_gs_separated_implicit_camera.py:11` | `nn.Module` | `local_mac_overfit_image_implicit_camera_separated.jsonc` via `train_camera_implicit_dynamic.py` | no | active legacy (separated-camera image-encoder baseline) |
| `FreeGaussianBankImplicitCamera` | `gs_models/dynamic_video_token_gs_implicit_camera.py:1024` | `nn.Module` | `variant: free_splats`, `free_gaussian_bank` | **NO** (signature uses `**_unused` and hardcodes 3 channels at line 1103) | **bug — feature_dim silently dropped** |
| `LinearTimeFreeGaussianBankImplicitCamera` | `gs_models/dynamic_video_token_gs_implicit_camera.py:1207` | `FreeGaussianBankImplicitCamera` | `variant: free_linear_time_splats`, `free_linear_splats`, `linear_free_splats` | **NO** (inherits the bug) | **bug — feature_dim silently dropped** |
| `UnconditionedTokenGSImplicitCamera` | `gs_models/dynamic_video_token_gs_implicit_camera.py:1435` | `nn.Module` | `variant: unconditioned_tokens`, `token_decoder_unconditioned` | yes | active |
| `UnconditionedResidualFreeBankImplicitCamera` | `gs_models/dynamic_video_token_gs_implicit_camera.py:1684` | `UnconditionedTokenGSImplicitCamera` | `variant: unconditioned_residual_free_bank`, `residual_free_bank_unconditioned_tokens` | yes (forwards via `kwargs.get("feature_dim", 3)`) | active |
| `DynamicVideoTokenGSImplicitCamera` | `gs_models/dynamic_video_token_gs_implicit_camera.py:1746` | `nn.Module` | `variant: learned_time_orbit_path` (default) | yes | active (default) |
| `ResidualFreeBankVideoTokenGSImplicitCamera` | `gs_models/dynamic_video_token_gs_implicit_camera.py:2134` | `DynamicVideoTokenGSImplicitCamera` | `variant: residual_free_bank`, `residual_free_video`, `residual_free_bank_video_tokens` | yes (forwards via `kwargs.get("feature_dim", 3)`) | active |
| `DynamicVideoTokenGSKnownCamera` | `gs_models/dynamic_video_token_gs_implicit_camera.py:2198` | `nn.Module` | `variant: known_camera`, `known_camera_video_token` | yes | active |
| `DynamicVideoTokenGSImplicitCameraSinusoidalTime` | `gs_models/dynamic_video_token_gs_implicit_camera.py:2394` | `DynamicVideoTokenGSImplicitCamera` | `variant: sinusoidal_time_path_mlp` | yes (inherits from parent) | active |
| `DynamicVideoTokenGSImplicitCameraPoseToPlucker` | `gs_models/dynamic_video_token_gs_implicit_camera.py:2498` | `DynamicVideoTokenGSImplicitCameraSinusoidalTime` | `variant: token_to_pose_to_plucker` | yes (inherits) | active |

`build_model_from_config` (in `train_video_token_implicit_dynamic.py:809`) is
the dispatcher for every model in the active lineage. It always passes
`feature_dim=model_cfg["feature_dim"]` (line 852).

## Class hierarchy diagram

```
nn.Module
├── TokenGSBackbone                               [blocks.py:231]
│   ├── TokenGS                                   [token_gs.py:7]              (legacy)
│   └── DynamicTokenGS                            [dynamic_token_gs.py:9]      (legacy)
├── DynamicTokenGSImplicitCamera                  [.../dynamic_token_gs_implicit_camera.py:35]   (legacy)
├── DynamicTokenGSSeparatedImplicitCamera         [.../dynamic_token_gs_separated_implicit_camera.py:11]  (legacy)
├── FreeGaussianBankImplicitCamera                [.../dynamic_video_token_gs_implicit_camera.py:1024]
│   └── LinearTimeFreeGaussianBankImplicitCamera  [...:1207]
├── UnconditionedTokenGSImplicitCamera            [...:1435]
│   └── UnconditionedResidualFreeBankImplicitCamera  [...:1684]
├── DynamicVideoTokenGSImplicitCamera             [...:1746]
│   ├── ResidualFreeBankVideoTokenGSImplicitCamera   [...:2134]
│   └── DynamicVideoTokenGSImplicitCameraSinusoidalTime  [...:2394]
│       └── DynamicVideoTokenGSImplicitCameraPoseToPlucker  [...:2498]
└── DynamicVideoTokenGSKnownCamera                [...:2198]
```

## Head inventory

| Head class | File:line | Used by which models | What it outputs | feature_dim handling |
|---|---|---|---|---|
| `GaussianParameterHeads` | `blocks.py:88` | `TokenGSBackbone` (so `TokenGS`, `DynamicTokenGS`); `UnconditionedTokenGSImplicitCamera`; `DynamicVideoTokenGSImplicitCamera`; `DynamicVideoTokenGSKnownCamera`; embedded as `base_heads` in `DynamicResidualGaussianBankHead`; embedded as static head in static/dynamic split | `(xyz, scales, quats, opacities, rgbs)` per-batch tuple, where `rgbs` shape is `[B, G, feature_dim]` | constructor `feature_dim=3` default; `rgb_head` outputs `gaussians_per_token*feature_dim`; F=3 path applies `sigmoid` (legacy parity), F!=3 returns raw features. Init bias `rgb_init="uniform"` only applied when `feature_dim==3`. |
| `DynamicResidualGaussianBankHead` | `gs_models/dynamic_video_token_gs_implicit_camera.py:862` | `UnconditionedTokenGSImplicitCamera` (dynamic split path); `DynamicVideoTokenGSImplicitCamera` (dynamic split path) | `DynamicGaussianBank` dataclass: `(xyz0, scales, quats0, opacities0, rgbs, A_mu, A_rot, A_alpha)` — temporal Fourier coefficients on top of base values | constructor `feature_dim=3` default; forwards to `base_heads = GaussianParameterHeads(... feature_dim=...)`; the `A_*` heads are not feature-dim-aware (they only model xyz/rot/alpha). |
| `ResidualFreeGaussianParameterHeads` | `gs_models/dynamic_video_token_gs_implicit_camera.py:1273` | `UnconditionedResidualFreeBankImplicitCamera`; `ResidualFreeBankVideoTokenGSImplicitCamera` | `(xyz, scales, quats, opacities, rgbs)` like `GaussianParameterHeads`, but base values are per-token `nn.Parameter`s and the head outputs bounded residuals on top | constructor `feature_dim=3` default; `base_raw_rgbs` shape `(num_tokens, gpt, feature_dim)`; `rgb_residual_head` outputs `gaussians_per_token*feature_dim`; F=3 sigmoid branch matches `GaussianParameterHeads`. |
| `CanonicalGaussianParameterHeads` | `gs_models/dynamic_token_gs_implicit_camera.py:10` | `DynamicTokenGSImplicitCamera`; `DynamicTokenGSSeparatedImplicitCamera` | `(xyz, scales, quats, opacities, rgbs)` — output 3-channel sigmoid-RGB only, hard-coded `scene_extent`/0.05 scale init | **no `feature_dim` thread**; legacy. |

## Colorize MLP API (FeatureToColor)

`src/train/colorize.py`. The constructor knobs and helper-call shape:

```
FeatureToColor(
    feature_dim: int,
    hidden_dim: int | None = None,            # None -> single Conv2d(F+view_dim, 3); int -> Conv2d(F+view_dim, h)->GELU->Conv2d(h, 3)
    activation: "sigmoid" | "identity",       # post-conv non-linearity
    pre_norm: bool = False,                   # nn.LayerNorm over the F channel-dim per pixel before the conv
    weight_init: "kaiming" | "orthogonal",
    weight_init_gain: float = 1.0,            # scales last-layer weight; ~7-8 for raw raster, ~2 with pre_norm
    view_condition: "none" | "camera_center_ray" | "pixel_ray" | aliases,
)

forward(features, view_dirs=None) -> rgb                      # rgb in [0,1] when activation="sigmoid"
forward_with_logits(features, view_dirs=None) -> (rgb, logits) # diagnostic; logits is pre-sigmoid

_run(features, view_dirs)        # internal: handles 4D ([B,F,H,W]) AND 5D ([B,T,F,H,W]) with T-into-B fold
_prepare_channels(features, view_dirs) -> tensor   # applies pre_norm, validates+concatenates view_dirs
_is_legacy_parity_case() -> bool                   # F=3 + hidden_dim=None + kaiming + gain=1.0 + no pre_norm + view=none
_identity_init(conv)                               # zeros + diagonal RGB identity weights for parity case
```

Notes:
- The legacy parity case (`F=3 + hidden_dim=None + kaiming + gain=1.0 + no
  pre_norm + view=none`) zeroes the conv and writes identity weights so
  the forward becomes `sigmoid(features)` — bit-identical to what the
  pre-colorize trainer used to do directly.
- `view_condition="camera_center_ray"` appends 3 channels (the camera's
  optical-axis ray) broadcast across H,W. `pixel_ray` appends a per-pixel
  ray direction. The view dirs are computed in the trainer
  (`colorize_view_dirs_for_features` in `train_video_token_implicit_dynamic.py:501`)
  using `camera_center_ray_dirs` / `build_camera_rays_batch` and
  detached by default.
- `_run` accepts a 5D tensor and folds time into batch — but the trainer
  always feeds `[T, F, H, W]` (calls go through `Trainer.colorize_features`
  with the rendered features tensor; see `train_video_token_implicit_dynamic.py:1071`).
  The 5D path exists but I see no live caller.

## feature_dim thread audit (active models only)

| Model | Accepts `feature_dim`? | Passes to head? | Forward returns `[K,G,feature_dim]`? | Status |
|---|---|---|---|---|
| `FreeGaussianBankImplicitCamera` | NO (eaten by `**_unused`) | head bypassed; raw_rgbs is `nn.Parameter`, hardcoded 3 channels at line 1103 | NO — always `[K, G, 3]` | **BROKEN** |
| `LinearTimeFreeGaussianBankImplicitCamera` | NO (inherits) | inherits | NO | **BROKEN** |
| `UnconditionedTokenGSImplicitCamera` | yes (line 1482, default 3) | yes (line 1542 in `gaussian_head_kwargs`) | yes | OK |
| `UnconditionedResidualFreeBankImplicitCamera` | yes (`kwargs.get("feature_dim", 3)` at 1716) | yes (forwards to `ResidualFreeGaussianParameterHeads`) | yes | OK |
| `DynamicVideoTokenGSImplicitCamera` | yes (line 1808) | yes (line 1893) | yes | OK |
| `ResidualFreeBankVideoTokenGSImplicitCamera` | yes (`kwargs.get("feature_dim", 3)` at 2166) | yes | yes | OK |
| `DynamicVideoTokenGSKnownCamera` | yes (line 2247) | yes (line 2315) | yes | OK |
| `DynamicVideoTokenGSImplicitCameraSinusoidalTime` | yes (inherits) | yes | yes | OK |
| `DynamicVideoTokenGSImplicitCameraPoseToPlucker` | yes (inherits) | yes | yes | OK |

The two BROKEN classes' bug is silent because the constructor signature
ends in `**_unused`; the kwarg arrives, vanishes, and the model trains
with `feature_dim=3` regardless of `model.feature_dim` in the config. The
trainer downstream constructs a `FeatureToColor(feature_dim=feature_dim)`
that expects 32-channel (or whatever) input. The first forward will hit
the `_prepare_channels` shape check and raise. So the failure is loud but
the error message will not mention `FreeGaussianBankImplicitCamera`.

The legacy classes (`TokenGS`, `DynamicTokenGS`,
`DynamicTokenGSImplicitCamera`, `DynamicTokenGSSeparatedImplicitCamera`)
do not thread `feature_dim` and use 3-channel sigmoid RGB in their head
forwards. Their trainers (`tokenGS.py`, `dynamicTokenGS.py`,
`train_camera_implicit_dynamic.py`) never read `model.feature_dim`, and
none of the corresponding configs set it, so this is consistent — the
legacy line is RGB-only by construction.

## LearnedQueryTokenBank

Defined at `gs_models/dynamic_video_token_gs_implicit_camera.py:792`. It
is a thin wrapper around an `nn.Parameter` of shape `(1, total_tokens, dim)`,
exposed through `forward(batch_size)` that broadcasts to
`(batch_size, total_tokens, dim)`. Used directly by:

- `UnconditionedTokenGSImplicitCamera` (line 1518) with
  `total_tokens = num_tokens + 2` (one global camera token + one path token + N world tokens).
- `DynamicVideoTokenGSImplicitCamera` (line 1859) — same `+2` layout.
- `DynamicVideoTokenGSKnownCamera` (line 2281) with `total_tokens = num_tokens`
  (no camera tokens).

The inheriting subclasses
(`UnconditionedResidualFreeBankImplicitCamera`,
`ResidualFreeBankVideoTokenGSImplicitCamera`,
`DynamicVideoTokenGSImplicitCameraSinusoidalTime`,
`DynamicVideoTokenGSImplicitCameraPoseToPlucker`) inherit the same query
token bank from their parent and override only the head or the camera path.

`FreeGaussianBankImplicitCamera` does NOT use `LearnedQueryTokenBank`. It
keeps a `global_camera_token` and `path_camera_token` as plain
`nn.Parameter`s, and the world Gaussians are stored as raw parameter
tensors (`raw_xyz`, `raw_scales`, `raw_quats`, `raw_opacities`,
`raw_rgbs`) — the model has no token decoder at all.

## Transformer blocks

`TransformerBlock` (`gs_models/dynamic_video_token_gs_implicit_camera.py:103`)
and `QueryCrossAttentionBlock` (line 118) are RMSNorm + multi-head
self-attention + feed-forward (TransformerBlock) and cross-attention
variant (QueryCrossAttentionBlock). Both are batch-first.

Used by:

- `VideoEncoder.stage1_blocks` and `VideoEncoder.bottleneck_blocks`
  (lines 264, 271): TransformerBlock self-attn over patch tokens at two
  resolutions.
- `DynamicVideoTokenGSImplicitCamera.query_decoder_blocks` (line 1864):
  QueryCrossAttentionBlock layers that refine query tokens against the
  encoded video tokens.
- `DynamicVideoTokenGSKnownCamera.query_decoder_blocks` (line 2286): same.
- `PluckerRayTokenConditioner.ray_cross_attn` (line 153):
  QueryCrossAttentionBlock cross-attending GS tokens against a Plucker
  ray grid.

The unconditioned variants
(`UnconditionedTokenGSImplicitCamera`,
`UnconditionedResidualFreeBankImplicitCamera`) skip the cross-attention
stack entirely — they have no `query_decoder_blocks`. Their
`refine_queries` (line 1577) just applies a time embedding additive
offset to the parameter token bank and returns it directly. This is the
"unconditioned" in their name: no video, no transformer-block stack.

`FreeGaussianBankImplicitCamera` has neither encoder nor cross-attention.

## Runtime types

`runtime_types.py`:

- `GaussianFrame` (line 154): one renderable frame. Fields `xyz [G,3]`,
  `scales [G,3]`, `quats [G,4]`, `opacities [G,1]`, `rgbs [G,F]`. Has a
  `.float()` cast helper. Field name is `rgbs` even when `F!=3`; the
  docstring states this is "retained for cascade compatibility."
- `GaussianSequence` (line 187): `K`-frame batch. Same fields with `[K,G,*]`
  shape, plus optional `cameras: tuple[CameraSpec, ...]`, optional
  `camera_state: CameraState`, and an `auxiliary: Mapping[str, Any]` for
  per-step diagnostics (e.g., dynamic Fourier coefficients in the
  static/dynamic split path). `.frame(index)` returns a `GaussianFrame` view.
- `CameraState` (line 119): camera-head diagnostics
  (`fov_degrees`, `radius`, `global_residuals`, `rotation_delta`,
  `translation_delta`, optional `path_residuals`). `.from_mapping()`
  constructs from a dict; `.motion_features()` returns
  `[T, 6]` for camera regularizers. Used by both implicit-camera models
  and the legacy `DynamicTokenGSImplicitCamera`.
- `SequenceData` and `ClipBatch` (lines 47, 95): training data containers,
  not model output. Out of scope for this report but they are the input
  side of the model contract.

The reason `rgbs` was kept rather than renamed to `features`: every
renderer call, every cascade snapshot, every diagnostic that touches
`decoded.rgbs` would have to change in lock-step. The docstring marks
this as a deliberate naming compromise.

## Camera-side typed wrappers

`camera.py` defines `CameraSpec` (a dataclass containing
`fx/fy/cx/cy/camera_to_world/lens_model/distortion`) plus a family of
helpers (`make_camera_like`, `build_camera_rays`,
`build_plucker_ray_grid`, distortion inversion). The dataclass is used
verbatim by every model that touches a camera, by the renderers
(`renderers/projection.py` per the task brief), by the colorize view-
condition dirs, and by `LearnableCameraRig` in `camera_rig.py`.

The brief calls out `C2W`, `W2C`, `Intrinsics3x3`, `QuatWXYZ` as typed
wrappers. **They do not exist in this codebase.** `camera.py` exposes a
single dataclass `CameraSpec` and operates on raw tensors for everything
else. The richer typed-3D-primitives style ("typed wrappers around camera
tensors that validate at construction") that the task brief alludes to is
described in the parent project's CLAUDE.md (the gflow upstream), not in
dynaworld. The dynaworld camera convention is a single dataclass plus
free functions; quaternions are raw `[N,4]` tensors with `F.normalize`
applied at the head and then again inside `_axis_angle_to_quat` and
`_quat_multiply` (`gs_models/dynamic_video_token_gs_implicit_camera.py:835, 848`).

## `Trainer.colorize_features`

Lives at `train_video_token_implicit_dynamic.py:1060`. It is a method on
`Trainer`, not on any model class.

```
def colorize_features(self, features: Tensor, cameras: tuple[CameraSpec, ...]) -> Tensor:
    if self.colorize is None:
        return features                               # legacy F=3 pass-through
    view_dirs = colorize_view_dirs_for_features(      # helper, not a method
        features, cameras,
        view_condition=self.colorize_view_condition,
        input_size=self.model_cfg["size"],
        render_size=self.render_size,
        detach=self.colorize_detach_view_condition,
    )
    return self.colorize(features, view_dirs=view_dirs)
```

The colorize MLP itself is owned by `Trainer.__init__` (line 1007–1024):
constructed lazily from `cfg["colorize"]`; if `feature_dim != 3` and no
colorize section is present, the trainer raises with a friendly error. The
trainer adds the colorize MLP's parameters to its optimizer at line 1036.

There are five live call sites of `Trainer.colorize_features` in the
trainer (lines 1346, 1409, 1599, 1900, 1969). The pattern is always:
render F-channel features, then colorize to RGB-3 just before loss /
preview / video logging.

This is the wrong abstraction layer for the "one clean path" rule but
intentionally so: the model returns whatever the head produces, the
renderer carries the F-channel through, and the trainer picks the
colorize step to run only at the loss boundary. Putting it inside the
model graph would force the rasterizer to run twice (once for features,
once for RGB) or force the colorize MLP into the per-Gaussian inner
loop. Neither is acceptable. The current placement keeps the colorize
MLP one-shot per rendered frame, which is the only sensible attachment
point for an "RGB after rasterization" decoder.

## Legacy / dead files

| File | Last referenced by | Active? | Recommendation |
|---|---|---|---|
| `dynamicTokenGS.py` | `train_video_token_implicit_dynamic.py:15` (imports `pick_device`, `fast_attn_context`, `configure_fast_attn`); `train_camera_implicit_dynamic.py:10` (also imports `select_window_indices`); `export_dynaworld_browser_bundle.py:13`; `probe_init_diagnostics.py:76`. Plus its own active script `train_full_dynamic_with_camera_prebake_all_frames.sh`. | **active as both a trainer AND a utilities module** | Move `pick_device` / `fast_attn_context` / `configure_fast_attn` / `select_window_indices` into a small shared module (`train_runtime.py` or similar), then delete the trainer half. Cannot just delete today — would break four other files. |
| `dynamicTokenGS_shared.py` | nobody (no live importer) | dead | delete |
| `dynamicTokenGS_tiled.py` | live shell script `train_full_dynamic_with_camera_prebake_all_frames.sh`? — no, it does not call this file. No live importer found. | dead | delete |
| `tokenGS.py` | active script `train_full_dynamic_with_camera_prebake_all_frames.sh`? — no. Has its own `if __name__ == "__main__"` and is callable directly via `uv run python src/train/tokenGS.py ...`. | active in CLI form (single-image overfit smoke). The corresponding configs `local_mac_overfit_single_image*.jsonc` exist. | Keep until the F-channel MVP cuts over to the active line. Otherwise delete with `local_mac_overfit_single_image*.jsonc`. |
| `tokenGS_shared.py` | nobody | dead | delete |
| `tokenGS_tiled.py` | nobody (calls `tokenGS.main` but no script invokes this file) | dead | delete |
| `dynamic_token_gs.py` (`gs_models/`) | `dynamicTokenGS.py:27` | active legacy | keep as long as `dynamicTokenGS.py` is alive |
| `dynamic_token_gs_implicit_camera.py` (`gs_models/`) | `train_camera_implicit_dynamic.py:16`; configs `local_mac_overfit_image_implicit_camera.jsonc` and `local_mac_overfit_image_implicit_camera_separated.jsonc` | active legacy (image-encoder implicit-camera baselines) | keep if the image-encoder baseline is part of the matrix; otherwise delete with the configs. The "v" between this file and the active video-token version is the test of whether the image-only baseline is ever rerun. |
| `dynamic_token_gs_separated_implicit_camera.py` (`gs_models/`) | `train_camera_implicit_dynamic.py:16`; one config | active legacy | same fate as above |
| `token_gs.py` (`gs_models/`) | `tokenGS.py:13` and `tokenGS_shared.py:1` | active legacy | bound to `tokenGS.py` |

The cleanest cut: delete `dynamicTokenGS_shared.py`,
`dynamicTokenGS_tiled.py`, `tokenGS_shared.py`, `tokenGS_tiled.py`
unconditionally. The other legacy files survive because at least one
live importer (a trainer or a config) names them.

## Smells / cleanup candidates

- `FreeGaussianBankImplicitCamera`/`LinearTimeFreeGaussianBankImplicitCamera`
  silently drop `feature_dim` because of `**_unused` in the signature.
  This is an actual bug, not just a smell. The `**_unused` swallow was
  likely added before `feature_dim` existed, then never revisited when
  the F-channel work landed.
- `_init_free_gaussian_raw_params` does take `feature_dim` (default 3),
  but `FreeGaussianBankImplicitCamera` does not pass it through — so the
  helper's parameter is dead code in this call site. The
  `ResidualFreeGaussianParameterHeads` call site (line 1340) is the only
  one that actually exercises non-default `feature_dim` in this helper.
- Field name `rgbs` in `GaussianFrame`/`GaussianSequence` is accurate
  only for `feature_dim==3`. For F!=3 it carries raw splat features.
  The docstring acknowledges this.
- `colorize_features` lives on the trainer. Cross-cutting: any non-trainer
  consumer of the model (export, probes, init diagnostics, the cascade
  baking step) has to either re-implement view-condition handling or
  duplicate the trainer's `colorize` reference. Currently
  `export_dynaworld_browser_bundle.py` does not run colorize — the export
  bundle is the F-channel splat itself, and the in-browser renderer
  presumably re-applies its own colorize. Worth confirming.
- Heads have lots of duplicated constructor knobs (`rgb_init`,
  `rgb_init_min`, `rgb_init_max`, `head_hidden_dim`, `head_hidden_layers`,
  `head_output_init_std`, `position_init_extent_coverage`,
  `rotation_init`, `scale_init`, `scale_init_log_jitter`, `opacity_init`,
  `feature_dim`, `xy_extent`, `z_min`, `z_max`). These are copy-pasted
  across `GaussianParameterHeads`, `DynamicResidualGaussianBankHead`,
  `ResidualFreeGaussianParameterHeads`, plus every model class
  constructor that wraps them. The `gaussian_head_kwargs` dict pattern in
  `UnconditionedTokenGSImplicitCamera` and
  `DynamicVideoTokenGSImplicitCamera` is the cleanest; the residual-free
  classes that use `kwargs.get("feature_dim", 3)` to pull from their own
  kwargs are messier.
- `UnconditionedResidualFreeBankImplicitCamera` and
  `ResidualFreeBankVideoTokenGSImplicitCamera` both grab a snapshot of
  constructor args from `kwargs` *before* calling `super().__init__()`,
  then mutate `self.gaussian_heads` after. This pattern exposes
  internal head identity as load-bearing — if the parent constructor
  changes the attribute name `gaussian_heads` to `static_gaussian_heads`
  (it does, in the static/dynamic split path), the residual-free wrapper
  silently drops on the floor. The
  `ResidualFreeBankVideoTokenGSImplicitCamera` constructor explicitly
  raises if `use_static_dynamic_split` is true (line 2169), so it
  defends against this — but the underlying coupling is fragile.
- Two distinct `to_3dgs_r_t` / quaternion conversion idioms
  (`F.normalize(..., dim=-1)` after the rotation head versus
  `_axis_angle_to_quat` + `_quat_multiply` for the dynamic delta)
  exist side by side. They produce the same convention (WXYZ, unit
  norm), but a typed `QuatWXYZ` wrapper of the kind the brief mentions
  would catch the case where someone tries to feed a rotation head's
  `[B, G, 4]` output into `_quat_multiply` (which expects WXYZ
  ordering and silently returns garbage if you swap to XYZW).
- The legacy line and the active line diverge in their handling of the
  position parametrization. Legacy `CanonicalGaussianParameterHeads`
  uses `tanh(xyz) * scene_extent` and `0.05 * exp(scales)` directly;
  active `GaussianParameterHeads` splits xy/z, uses
  `xy_extent`/`z_min`/`z_max`, and applies `xyz_bias[:, :2/2:].copy_(...)`
  if `position_init_extent_coverage > 0`. Worth confirming the legacy
  trainers can never re-enter the active code path.

## Open questions for proposers

- Should the colorize MLP move from `Trainer.colorize` into the model
  graph (so the model returns RGB directly)? The render-once-then-MLP
  layout argues no, but a typed `ModelOutput = RGB | Features`
  discrimination at the model boundary might be cleaner.
- Should `GaussianSequence.rgbs` be renamed to `features` now that the
  F=3 path also goes through the same field? The cascade-compatibility
  excuse buys nothing once the cascade is updated. The cost is a global
  rename across `dynamic_video_token_gs_implicit_camera.py`,
  `runtime_types.py`, `train_video_token_implicit_dynamic.py`, and the
  rendering and debug paths.
- Should the model variants be collapsed via composition (a single
  `TokenGS` class with plug-ins for time / camera / video conditioning)
  instead of 7 separate classes plus 4 inheritance edges? The current
  variant logic in `build_model_from_config` is essentially already a
  registry; the constructors carry near-identical kwarg sets. The
  duplication is in the constructors, not the dispatch.
- Are the legacy `dynamicTokenGS.py` lineage files safe to delete?
  Answer: the trainer file itself is NOT — three live trainers and the
  export script depend on its top-level utility functions. The five
  trampoline files at the top level (`*_shared.py`, `*_tiled.py`)
  are safe to delete. The `gs_models/` legacy heads are bound to the
  configs that select them.
- Should `feature_dim` be lifted into a single typed `ModelOutputSpec`
  (similar to the brief's `Presence[T]` pattern) so that the F=3 sigmoid
  branch in the head forward becomes a type-directed dispatch rather
  than an `if self.feature_dim == 3:` check that the proposer has to
  remember to copy when adding a new head?

## File listing (absolute paths)

- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/__init__.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/blocks.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/dynamic_token_gs_implicit_camera.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/dynamic_token_gs_separated_implicit_camera.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/dynamic_token_gs.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/token_gs.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/time_conditioning.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/implicit_camera.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/colorize.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/runtime_types.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/camera.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/camera_rig.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/dynamicTokenGS.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/dynamicTokenGS_shared.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/dynamicTokenGS_tiled.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/tokenGS.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/tokenGS_shared.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/tokenGS_tiled.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_video_token_implicit_dynamic.py`
- `/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_camera_implicit_dynamic.py`
