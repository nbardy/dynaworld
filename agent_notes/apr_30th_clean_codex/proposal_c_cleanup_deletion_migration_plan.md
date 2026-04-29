# Proposal C: Cleanup, Deletion, And Staged Migration Plan

Date: 2026-04-30
Author role: Proposal Writer C
Scope: cleanup, deletion, compatibility, and migration sequencing for the Dynaworld trainer stack.

## Thesis

The trainer stack needs a cleanup pass, but it cannot be a rewrite that loses
the experimental trail. The right move is a staged consolidation:

1. Put a typed router and compatibility layer in front of the current scripts.
2. Extract shared functional adapters for data, decode, render/objective, loss,
   validation, smoke, and baseline provenance.
3. Migrate active families one at a time behind the router.
4. Keep old entrypoints as shims until parity smokes and W&B/BASELINES audits
   prove the route is equivalent.
5. Delete only the true duplicate shims immediately; retire legacy trainers only
   after their configs are either migrated or explicitly marked as archival.

The core failure pattern this plan prevents is exactly what just happened with
F32 feature splatting: the base single-camera trainer got alpha-aware feature
composition and random train backgrounds, but multicam had its own render/loss
override and silently bypassed the fix. A cleanup that only renames files will
not solve that. The invariant must become: every train/eval path calls one
shared objective boundary where `(features, alpha)` becomes final RGB and loss.

## Current State Summary

Observed current surface:

- 96 checked-in `src/train_configs/*.jsonc` files.
- 96 configs with an explicit top-level `arch` value.
- 17 shell scripts under `src/train_scripts/`.
- Several active trainer files plus procedural legacy trainers.
- Multiple trainer entrypoints are selected by "which Python file the shell
  script calls", not by a central `arch` dispatcher.
- `Trainer -> PrecomputedFeatureImplicitTrainer -> MulticamPrecomputedFeatureImplicitTrainer`
  inheritance currently hides a broken override surface: the multicam subclass
  bypasses the new base F32 render/loss path.

Important current run provenance:

- Single-camera F32 alpha-aware + random train background has a successful
  small baseline run:
  - Config: `src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc`
  - Command:
    ```bash
    PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py \
      src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc
    ```
  - W&B: `https://wandb.ai/nbardy/dynaworld/runs/9gr2dm3v`
  - Final observed loss: `Loss 0.0665 / recon 0.0660`
  - Media logged: `Alpha_Mask_Video`, `Feature_PCA_Video`,
    `Render_Composite_Video`
  - Caveat: random background is hardcoded in the trainer, not represented in
    config provenance.

## Non-Negotiable Cleanup Constraints

1. Do not lose current baselines.
   `BASELINES.md` is the canonical standings table. Cleanup work must append
   dated rows when reruns happen and must never silently overwrite old metrics.

2. Do not break old launch commands until a replacement route has passed parity.
   The old scripts should become shims that print their replacement route and
   call the central router.

3. Do not preserve inheritance just because it exists.
   Prefer typed data records plus functions. Inheritance is allowed only as a
   temporary compatibility wrapper.

4. Do not scatter config defaults.
   Normalize once at config load. Warm paths should receive typed sections or
   dictionaries that have already been validated.

5. Do not let render/loss logic fork again.
   Feature colorization, alpha-aware composition, random train background,
   validation background, alpha videos, feature PCA, RGB reconstruction loss,
   and metrics must live behind one shared objective interface.

6. Do not use source-view PSNR to rank cleanup success.
   For representation work, held-out-camera metrics are the selector. Source
   overfit smokes are only "does the route run" checks.

## Current Trainer File Inventory

### Active Hub: `src/train/train_video_token_implicit_dynamic.py`

Current roles:

- Main single-camera implicit-camera video-token trainer.
- Known-camera trainer selected internally by `model.variant`.
- Model factory for the modern token/implicit-camera variants.
- Config defaults and normalization for many current configs.
- Colorize MLP creation.
- F32 feature-splatting alpha-aware render/loss path.
- Random per-step train background in `Trainer.recon_backward`.
- Validation videos including feature PCA and alpha/composite media.

Important classes/functions:

```python
@dataclass
class StepResult: ...

def resolve_config(config: dict[str, Any]) -> dict[str, Any]: ...
def pick_renderer_mode_from_config(config: dict[str, Any]) -> tuple[str, int]: ...
def prepare_clip(sequence_data: SequenceData, clip_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
def render_clip_sequence(...) -> tuple[torch.Tensor, torch.Tensor | None]: ...
def build_model_from_config(config: dict[str, Any]) -> torch.nn.Module: ...

class Trainer:
    def recon_backward(...) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: ...
    def render_decoded_clip(self, decoded: GaussianSequence) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def initial_step_result(self) -> StepResult: ...
    def render_full_sequence(...) -> tuple[torch.Tensor, CameraState | None, dict[str, float], torch.Tensor | None, torch.Tensor | None]: ...
    def validation_video_payload(self) -> dict[str, Any]: ...
    def run(self) -> None: ...

class KnownCameraTrainer(Trainer): ...

def trainer_class_for_config(config: dict[str, Any]) -> type[Trainer]: ...
def run_training(config: dict[str, Any]) -> None: ...
def main(config: dict[str, Any] | str | Path) -> None: ...
```

Keep for now, but extract from it. This file should shrink into a thin recipe
adapter after the migration.

### Active Thin Adapter: `src/train/train_precomputed_feature_implicit_dynamic.py`

Current roles:

- Single-camera precomputed-feature trainer.
- Forces `model.video_encoder_backend = "precomputed"` unless caller already set
  the compatible precomputed backend.
- Owns `FEATURE_OPTION_DEFAULTS`.
- Builds `VideoFeatureCache`.
- Prebakes features.
- Mutates `model_cfg["video_feature_channels"]` and
  `model_cfg["video_feature_layers"]` after cache inspection.
- Overrides `model_input_for_clip`.
- Inherits base alpha/colorize/random-background path from `Trainer`.

Important signatures:

```python
class PrecomputedFeatureImplicitTrainer(VideoTokenImplicitTrainer):
    @classmethod
    def resolve_config(cls, config: dict[str, Any]) -> dict[str, Any]: ...
    def on_sequences_loaded(self) -> None: ...
    def model_input_for_clip(
        self,
        sequence_data: SequenceData,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
    ) -> Any: ...
    def run(self) -> None: ...

def run_training(config: dict[str, Any]) -> None: ...
def main(config: dict[str, Any] | str | Path) -> None: ...
```

Merge into a `FeatureProvider` adapter. Do not keep feature-cache setup as a
trainer subclass long term.

### Active But Broken After Tuple Return: `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`

Current roles:

- Multicam DeepView/manifest trainer.
- Uses `MulticamVideoBundle`.
- Builds `LearnableCameraRig`.
- Samples train views and held-out views.
- Computes train-view reconstruction loss.
- Logs train and held-out videos.

Important signatures:

```python
class MulticamPrecomputedFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer):
    @classmethod
    def resolve_config(cls, config: dict[str, Any]) -> dict[str, Any]: ...
    def load_train_sequences(self): ...
    def load_eval_sequences(self): ...
    def on_sequences_loaded(self) -> None: ...
    def sample_views(self) -> list[int]: ...
    def sample_multicam_clip(self): ...
    def _decode_clip(self, sequence_data, clip_frames: torch.Tensor, clip_times: torch.Tensor): ...
    def render_view_clip(self, decoded, *, view: int, clip_indices: torch.Tensor) -> torch.Tensor: ...
    def multicam_recon_loss(...) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def rig_regularization_loss(self) -> torch.Tensor: ...
    def initial_step_result(self) -> StepResult: ...
    def step(self, keep_preview: bool = False) -> StepResult: ...
    def scalar_payload(self, result: StepResult) -> dict[str, Any]: ...
    def render_full_external_views(self): ...
    def validation_video_payload(self) -> dict[str, Any]: ...
    def export_browser_bundle(self) -> None: ...
```

Current blockers:

- `render_view_clip()` returns the tuple from `render_clip_sequence(...)` but is
  annotated and consumed as a tensor.
- `multicam_recon_loss()` passes that tuple to `reconstruction_loss_per_image`.
- `render_full_external_views()` calls `.detach().cpu()` on tuple returns.
- Even after tuple unpacking, it still lacks shared colorize, alpha composition,
  random train background, feature PCA logging, alpha mask logging, and composite
  held-out videos.

This is the highest-priority migration target after a shared objective is
extracted.

### Empty Compatibility Alias: `src/train/train_ltx_feature_implicit_dynamic.py`

Current roles:

- Defines `LTXFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer)` with no
  overrides.
- Provides a backward-compatible named entrypoint.
- The current LTX shell script calls `train_precomputed_feature_implicit_dynamic.py`
  directly instead, so this file is not carrying unique behavior.

Important signatures:

```python
class LTXFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer): ...
def run_training(config: dict[str, Any]) -> None: ...
def main(config: dict[str, Any] | str | Path) -> None: ...
```

Delete or redirect after the central router supports
`arch = "ltx_feature_implicit_camera"` as an alias of the precomputed recipe.

### Legacy Image Implicit-Camera Trainer: `src/train/train_camera_implicit_dynamic.py`

Current roles:

- Procedural image/video implicit-camera baseline.
- Own data loading, config defaults, loss, rendering, W&B loop.
- Not on the modern `Trainer` path.
- Not alpha/colorize/F32 aware.

Important signatures:

```python
def resolve_config(config: dict[str, Any]) -> dict[str, Any]: ...
def load_sequence_data(...) -> SequenceData: ...
def pick_renderer_mode(config: dict[str, Any]) -> tuple[str, int]: ...
def build_model_from_config(model_cfg: dict[str, Any]): ...
def render_implicit_frame(renderer_mode, config, dense_grid, camera, frame: GaussianFrame): ...
def eval_metric_payload(...): ...
def render_full_sequence(...): ...
def run_training(config: dict[str, Any]): ...
def main(config: dict[str, Any] | str | Path) -> None: ...
```

Retire behind a legacy adapter unless this baseline is still scientifically
needed. Do not delete until the two image configs either migrate or are marked
archival.

### Duplicate Shims For Image Implicit-Camera

Files:

- `src/train/train_camera_implict_dynamic.py`
- `src/train/train_image_encoder_implicit_camera_baseline.py`

Both currently do this:

```python
from train_camera_implicit_dynamic import main

if __name__ == "__main__":
    main("src/train_configs/local_mac_overfit_image_implicit_camera.jsonc")
```

These are the "two similarly named implicit camera trainers/shims" cleanup
problem. One is a misspelled alias (`implict`), the other is a duplicated
baseline alias. They are immediate delete candidates after one compatibility
shim period, or immediate redirect candidates if old external commands may still
exist.

Recommended move:

1. Add central router support for `tokengs_image_implicit_camera`.
2. Replace both files with one-line deprecation shims for one release cycle.
3. Delete both after the route is listed in `train.py --explain-routing`.

### Legacy Known/Prebaked Camera Procedural Trainer: `src/train/dynamicTokenGS.py`

Current roles:

- Procedural known/prebaked-camera baseline.
- Launch target for `tokengs_prebaked_camera` and
  `tokengs_prebaked_camera_tiled`.
- Owns optimizer helpers, LR schedule helpers, render helpers, debug metrics.
- Newer trainers import some utilities from this legacy file.

Important signatures:

```python
def pick_device(): ...
def resolve_sequence_dir(data_cfg: dict[str, Any]) -> Path: ...
def normalize_lr_schedule(train_cfg: dict[str, Any]) -> None: ...
def normalize_optimizer_config(train_cfg: dict[str, Any]) -> None: ...
def normalize_loss_config(cfg: dict[str, Any]) -> None: ...
def normalize_render_config(cfg: dict[str, Any]) -> None: ...
def resolve_config(config: dict[str, Any]) -> dict[str, Any]: ...
def resolve_camera_json_path(sequence_dir: Path, camera_json: Path | None) -> Path: ...
def load_sequence_data(...) -> SequenceData: ...
def pick_renderer_mode(config: dict[str, Any]) -> tuple[str, int]: ...
def learning_rate_for_step(base_lr: float, train_cfg: dict[str, Any], step: int) -> float: ...
def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None: ...
def build_optimizer_param_groups(...) -> list[dict[str, Any]]: ...
def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any], device: torch.device) -> torch.optim.Optimizer: ...
def build_model_from_config(model_cfg: dict[str, Any]) -> DynamicTokenGS: ...
def render_one_frame(...): ...
def render_frame_batch(...): ...
def render_full_sequence(...): ...
def run_training(config: dict[str, Any]): ...
def main(config: dict[str, Any] | str | Path) -> None: ...
```

Do not delete yet. Extract first:

- `pick_device`
- fast attention configuration/context helpers if still imported from here
- optimizer normalization
- LR schedule
- common render helpers if any current configs still need them

After extraction, route prebaked-camera configs through a `LegacyAdapter` until
there is evidence the config family is no longer useful.

### Legacy Single-Image Token Trainer: `src/train/tokenGS.py`

Current roles:

- Procedural single-image baseline.
- Launch target for `tokengs_single_image` and likely old tiled variants.
- Owns minimal config/load/render/train loop.

Important signatures:

```python
def pick_device(): ...
def resolve_config(config: dict[str, Any]) -> dict[str, Any]: ...
def pick_renderer_mode(config: dict[str, Any]) -> tuple[str, int]: ...
def render_single_frame(renderer_mode, config, dense_grid, camera, decoded): ...
def run_training(config: dict[str, Any]): ...
def main(config: dict[str, Any] | str | Path) -> None: ...
```

Retire, not immediate delete. Single-image configs are not part of the current
multicam/V-JEPA/feature-splatting lane, but they are useful as historical
minimal baselines. They should be marked as archival and routed by the central
router through a legacy adapter.

### Tiled Legacy Trainers

Known current config arch values include:

- `tokengs_single_image_tiled`
- `tokengs_prebaked_camera_tiled`

These imply companion files such as `tokenGS_tiled.py` and
`dynamicTokenGS_tiled.py` or older tiled helpers. They should be audited with
the same policy as `tokenGS.py` and `dynamicTokenGS.py`: route and retire, not
delete blindly.

### Gauge-Field Stack

Current config families:

- `gauge_fields_material_surfel`
- `splat_baseline_free_dynamic_3dgs`
- `splat_baseline_static_3dgs`

Known entrypoints:

- `research_experiments/gauge_fields/train.py`
- `research_experiments/gauge_fields/train_splat_baseline.py`
- `research_experiments/gauge_fields/run_deepview_3cam_holdout.py`

This stack should stay separate for now. It is a research experiment harness
with its own incidence laws, support-mode ablations, and held-out-camera
baseline role. It should be visible in the central router and baseline audit,
but not merged into the video-token trainer cleanup until the video-token
trainer has stabilized.

Policy:

- Keep the gauge-field implementation separate.
- Add router entries that explain and delegate to the gauge-field entrypoints.
- Keep held-out-camera metrics in `BASELINES.md`.
- Do not force gauge-field configs through the new `RenderObjective` until there
  is a specific reason; the gauge stack has different primitive semantics.

## Config Arch Family Inventory

Current `arch` counts:

| Arch | Count | Bucket | Current route |
|---|---:|---|---|
| `tokengs_video_implicit_camera` | 35 | Keep and merge into modern recipe | `src/train/train_video_token_implicit_dynamic.py` |
| `gauge_fields_material_surfel` | 31 | Keep separate; expose through router | `research_experiments/gauge_fields/train.py` |
| `tokengs_prebaked_camera` | 9 | Retire behind legacy adapter | `src/train/dynamicTokenGS.py` |
| `precomputed_feature_implicit_camera` | 4 | Keep; merge into feature provider recipe | `src/train/train_precomputed_feature_implicit_dynamic.py` |
| `multicam_precomputed_feature_implicit_camera` | 4 | Keep; highest-priority migration | `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` |
| `tokengs_image_implicit_camera` | 2 | Retire behind legacy adapter | `src/train/train_camera_implicit_dynamic.py` |
| `tokengs` | 2 | Alias to video-token recipe | `src/train/train_video_token_implicit_dynamic.py` |
| `splat_baseline_free_dynamic_3dgs` | 2 | Keep separate; expose through router | `research_experiments/gauge_fields/train_splat_baseline.py` |
| `wan_vace_feature_implicit_camera` | 1 | Alias to precomputed feature recipe | `src/train/train_precomputed_feature_implicit_dynamic.py` |
| `ltx_feature_implicit_camera` | 1 | Alias to precomputed feature recipe | `src/train/train_precomputed_feature_implicit_dynamic.py` |
| `tokengs_video_known_camera` | 1 | Keep; fix stale paths | `KnownCameraTrainer` in video trainer |
| `tokengs_single_image` | 1 | Retire behind legacy adapter | `src/train/tokenGS.py` |
| `tokengs_single_image_tiled` | 1 | Retire behind legacy adapter | legacy tiled single-image trainer |
| `tokengs_prebaked_camera_tiled` | 1 | Retire behind legacy adapter | legacy tiled dynamic/prebaked trainer |
| `splat_baseline_static_3dgs` | 1 | Keep separate; expose through router | `research_experiments/gauge_fields/train_splat_baseline.py` |

Grouping:

- Active modern video-token family:
  - `tokengs_video_implicit_camera`
  - `tokengs`
  - `tokengs_video_known_camera`
- Active feature-conditioning family:
  - `precomputed_feature_implicit_camera`
  - `ltx_feature_implicit_camera`
  - `wan_vace_feature_implicit_camera`
- Active multicam feature family:
  - `multicam_precomputed_feature_implicit_camera`
- Research harness family:
  - `gauge_fields_material_surfel`
  - `splat_baseline_free_dynamic_3dgs`
  - `splat_baseline_static_3dgs`
- Legacy archival family:
  - `tokengs_prebaked_camera`
  - `tokengs_prebaked_camera_tiled`
  - `tokengs_image_implicit_camera`
  - `tokengs_single_image`
  - `tokengs_single_image_tiled`

## Train Script Inventory

Current shell scripts:

| Script | Current role | Cleanup action |
|---|---|---|
| `src/train_scripts/build_100_clip_dataset.sh` | Dataset builder | Keep; separate data CLI namespace later. |
| `src/train_scripts/build_local_mac_30_clip_dataset.sh` | Dataset builder | Keep; separate data CLI namespace later. |
| `src/train_scripts/get_camera.sh` | Camera extraction helper | Keep; not trainer cleanup. |
| `src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh` | Single-overfit comparison matrix | Redirect through router after parity. |
| `src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh` | Legacy prebaked camera via `dynamicTokenGS.py` | Retire behind legacy adapter. |
| `src/train_scripts/train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh` | Legacy image implicit-camera baseline | Retire behind legacy adapter. |
| `src/train_scripts/train_full_dynamic_with_implicit_camera_all_frames.sh` | Legacy/new implicit camera all-frames script | Redirect through router. |
| `src/train_scripts/train_full_dynamic_with_video_token_implicit_camera_all_frames.sh` | Modern video-token full dynamic | Redirect through router. |
| `src/train_scripts/train_implicit_camera_128_4fps_fast_mac_baseline.sh` | Modern video-token baseline | Redirect through router. |
| `src/train_scripts/train_local_mac_30_clip_baseline.sh` | Local 30-clip baseline | Redirect through router. |
| `src/train_scripts/train_local_mac_30_clip_vjepa2_256_baseline.sh` | V-JEPA 256 baseline | Redirect through router. |
| `src/train_scripts/train_ltx_feature_implicit_camera_128_4fps_fast_mac.sh` | LTX/precomputed feature route | Redirect through router alias. |
| `src/train_scripts/train_multicam_static_dynamic_vjepa_features.sh` | Multicam V-JEPA held-out route | Keep, but point to router after multicam parity. |
| `src/train_scripts/train_precomputed_vjepa2_1_torchhub_vitb_384.sh` | Precomputed V-JEPA route | Redirect through router. |
| `src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh` | Smoke | Replace with smoke harness wrapper. |
| `src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh` | Static/dynamic feature ablation | Redirect through router. |
| `src/train_scripts/train_video_temporal_ablation_suite.sh` | Temporal ablation suite | Redirect through router. |
| `src/train_scripts/train_wan_vace_feature_implicit_camera_128_4fps_fast_mac.sh` | Wan/precomputed feature route | Redirect through router alias. |

Script cleanup policy:

- Keep scripts as human-friendly named launchers.
- Make scripts call `src/train/train.py run CONFIG`.
- Add `src/train/train.py explain CONFIG` and require scripts to print the route
  when `DYNATRAIN_EXPLAIN=1`.
- Remove scripts only after a separate command inventory proves they are unused.

## Risky Bypasses To Fix Or Contain

### 1. Multicam F32 tuple/render/loss bypass

Problem:

```python
def render_view_clip(...) -> torch.Tensor:
    return render_clip_sequence(...)
```

`render_clip_sequence(...)` now returns:

```python
tuple[torch.Tensor, torch.Tensor | None]
```

Then:

```python
rendered = self.render_view_clip(...)
reconstruction_loss_per_image(rendered, target, self.loss_cfg)
```

That is wrong for F32 and likely crashes after the tuple-return change. Even if
unpacked, multicam still does not call the shared colorize/alpha/random-bg path.

Containment:

- Do not launch the "ultimate" multicam F32 config until the shared objective is
  wired into multicam.
- Add a route validation failure:
  `feature_dim != 3 and arch == "multicam_precomputed_feature_implicit_camera"`
  should fail unless `objective.version >= "feature_alpha_v1"`.

### 2. Known-camera initial validation stale path

Problem:

`KnownCameraTrainer.initial_step_result()` assigns:

```python
rendered_features = self.render_decoded_clip(decoded)
```

but `render_decoded_clip()` returns `(features, alpha)`.

Containment:

- Fix with shared objective before running known-camera F32 or known-camera
  feature configs.
- Add a smoke route for `tokengs_video_known_camera` even if it is not a primary
  current baseline.

### 3. Module-level `render_full_sequence` discards alpha

Problem:

There is a module-level `render_full_sequence(...)` in the video trainer that
explicitly discards alpha and skips colorize. It appears unused now, but it is a
future footgun.

Containment:

- Mark private/deprecated.
- Replace callers with `render_validation_views(...)`.
- Delete after `rg render_full_sequence` shows only class methods remain.

### 4. Random train background is not config-visible

Problem:

The single-camera F32 path now samples random background in code, but configs and
W&B do not record this as a knob.

Containment:

- Add normalized config:
  `losses.background.mode = "random_rgb"` for intentional random-bg runs.
- Log:
  `LossBackground/Mode`, `LossBackground/TrainMode`,
  `LossBackground/EvalMode`, `LossBackground/LastSampleRGB`.

### 5. `render.fast_mac.feature_background` is not the same as loss composition background

Problem:

The rasterizer feature background and the RGB composition background live at
different semantic layers. Reusing one name causes confusing configs.

Containment:

- Use `render.fast_mac.feature_background` only for rasterizer fill behavior.
- Use `losses.background` for alpha-aware RGB composition after colorize.

### 6. Feature cache mutates model config

Problem:

`PrecomputedFeatureImplicitTrainer.on_sequences_loaded()` mutates:

```python
self.model_cfg["video_feature_channels"]
self.model_cfg["video_feature_layers"]
```

after the feature cache inspects the data.

Containment:

- Replace mutation with a `FeatureProvider.describe()` result consumed by model
  factory.

## Keep / Merge / Delete / Retire Buckets

### Keep As Active

Keep these because they are current experiment surfaces:

- `src/train/train_video_token_implicit_dynamic.py`
  - Temporary active hub. Shrink over time.
- `src/train/train_precomputed_feature_implicit_dynamic.py`
  - Temporary adapter. Merge into `FeatureProvider`.
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  - Active but broken for F32. Migrate before ultimate run.
- `src/train/rendering.py`
  - Keep, but typed outputs should be wrapped by objective-level records.
- `src/train/renderers/fast_mac.py`
  - Keep; it owns F==3 vs F!=3 dispatch.
- `research_experiments/gauge_fields/*`
  - Keep separate.

### Merge Into Shared Functional Modules

Merge these responsibilities:

- Config loading and validation:
  - from every trainer's `resolve_config`
  - into `src/train/experiment_spec.py`
- Train routing:
  - from shell/Python-file selection
  - into `src/train/train.py`
- Model construction:
  - from `build_model_from_config`
  - into `src/train/model_registry.py`
- Data loading and sampling:
  - from `Trainer`, `KnownCameraTrainer`, `MulticamPrecomputedFeatureImplicitTrainer`
  - into `src/train/data_pipeline.py`
- Feature cache:
  - from precomputed trainer
  - into `src/train/feature_provider.py`
- Render/colorize/compose/loss:
  - from base trainer and multicam overrides
  - into `src/train/objective.py`
- Validation media:
  - from trainer class methods
  - into `src/train/validation.py`
- Smoke and route checks:
  - from ad hoc scripts
  - into `src/train/smoke.py`
- Baseline audit:
  - from manual notes
  - into `src/train/baseline_audit.py`

### Delete Immediately Or After One Shim Period

Immediate delete candidates, subject to one short compatibility shim period:

- `src/train/train_camera_implict_dynamic.py`
  - Misspelled duplicate shim.
- `src/train/train_image_encoder_implicit_camera_baseline.py`
  - Duplicate shim to the same image implicit config.

Recommended not to delete in the same patch that introduces the router. First
replace with deprecation shims that call the router and print the new command.
Delete after the smoke matrix passes and `rg train_camera_implict_dynamic` /
`rg train_image_encoder_implicit_camera_baseline` show no internal references.

### Delete After Extraction

- `src/train/train_ltx_feature_implicit_dynamic.py`
  - Delete once `arch = "ltx_feature_implicit_camera"` routes through the
    precomputed recipe and the LTX shell script uses the central router.
- Module-level `render_full_sequence` in
  `src/train/train_video_token_implicit_dynamic.py`
  - Delete once no callers remain and validation uses `render_validation_views`.

### Retire Behind Legacy Adapter

Do not delete yet:

- `src/train/dynamicTokenGS.py`
- `src/train/tokenGS.py`
- likely tiled companion trainers
- `src/train/train_camera_implicit_dynamic.py`

Retirement sequence:

1. Add legacy recipe entries.
2. Add route explain output.
3. Run one-step or minimal smokes for each legacy arch family.
4. Append a deprecation row in the cleanup note.
5. Only delete when configs are either migrated or marked archival and no script
   calls the old files directly.

## Proposed Public Interfaces

The cleanup should introduce a small public API. The API should be typed at
dispatch boundaries and dictionary-friendly inside config sections where that
keeps experimentation flexible.

### Router: `src/train/train.py`

Public CLI:

```bash
PYTHONPATH=src/train uv run python src/train/train.py run <config.jsonc>
PYTHONPATH=src/train uv run python src/train/train.py smoke <config.jsonc> --steps 1 --offline
PYTHONPATH=src/train uv run python src/train/train.py explain <config.jsonc>
PYTHONPATH=src/train uv run python src/train/train.py audit-baselines
PYTHONPATH=src/train uv run python src/train/train.py list-arch
```

Public functions:

```python
def main(argv: Sequence[str] | None = None) -> None: ...

def run_config(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    mode: Literal["train", "smoke"] = "train",
) -> None: ...

def explain_routing(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> "RoutingReport": ...

def list_arches() -> list["ArchRouteSummary"]: ...
```

### Normalized Experiment Spec

File: `src/train/experiment_spec.py`

```python
@dataclass(frozen=True)
class ExperimentSpec:
    config_path: Path
    arch: str
    raw: Mapping[str, Any]
    data: Mapping[str, Any]
    model: Mapping[str, Any]
    camera: Mapping[str, Any]
    render: Mapping[str, Any]
    train: Mapping[str, Any]
    losses: Mapping[str, Any]
    logging: Mapping[str, Any]
    export: Mapping[str, Any]
    features: Mapping[str, Any] | None
    views: "ViewSpec"
    compatibility: "CompatibilitySpec"


@dataclass(frozen=True)
class CompatibilitySpec:
    original_entrypoint: str | None
    legacy_family: str | None
    warnings: tuple[str, ...] = ()
    deprecation: "DeprecationNotice | None" = None


@dataclass(frozen=True)
class DeprecationNotice:
    old_command: str
    new_command: str
    removal_not_before: str
    reason: str


def load_experiment_spec(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentSpec: ...


def normalize_config(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
) -> ExperimentSpec: ...


def validate_spec(spec: ExperimentSpec) -> None: ...
```

Normalization policy:

- Defaults are applied once.
- Required keys fail loudly.
- Legacy aliases are normalized to active arch names with a deprecation notice.
- `feature_dim != 3` requires a valid colorize/objective section.
- F32 configs should explicitly set `losses.background.mode`.

### Route Registry

File: `src/train/train_registry.py`

```python
@dataclass(frozen=True)
class TrainRecipe:
    name: str
    arch_values: tuple[str, ...]
    status: Literal["active", "compat", "legacy", "external"]
    build_data: Callable[[ExperimentSpec, torch.device], "DataSource"]
    build_feature_provider: Callable[[ExperimentSpec, torch.device], "FeatureProvider | None"]
    build_model_program: Callable[[ExperimentSpec, "FeatureDescription | None"], "ModelProgram"]
    build_objective: Callable[[ExperimentSpec, torch.device], "RenderObjective"]
    build_loop: Callable[[ExperimentSpec, "TrainComponents"], "TrainLoop"]
    legacy_entrypoint: str | None = None


@dataclass(frozen=True)
class RoutingReport:
    config_path: Path
    arch: str
    recipe_name: str
    recipe_status: Literal["active", "compat", "legacy", "external"]
    old_command: str | None
    new_command: str
    model_class: str | None
    data_source: str
    feature_provider: str | None
    objective: str
    expected_smokes: tuple[str, ...]
    warnings: tuple[str, ...]


ARCH_REGISTRY: dict[str, TrainRecipe] = {...}


def resolve_recipe(spec: ExperimentSpec) -> TrainRecipe: ...
def route_report(spec: ExperimentSpec, recipe: TrainRecipe) -> RoutingReport: ...
```

Initial registry:

```python
ARCH_REGISTRY = {
    "tokengs_video_implicit_camera": VIDEO_TOKEN_RECIPE,
    "tokengs": VIDEO_TOKEN_RECIPE,
    "tokengs_video_known_camera": VIDEO_TOKEN_KNOWN_CAMERA_RECIPE,
    "precomputed_feature_implicit_camera": PRECOMPUTED_VIDEO_TOKEN_RECIPE,
    "ltx_feature_implicit_camera": PRECOMPUTED_VIDEO_TOKEN_RECIPE,
    "wan_vace_feature_implicit_camera": PRECOMPUTED_VIDEO_TOKEN_RECIPE,
    "multicam_precomputed_feature_implicit_camera": MULTICAM_PRECOMPUTED_RECIPE,
    "tokengs_image_implicit_camera": LEGACY_IMAGE_IMPLICIT_RECIPE,
    "tokengs_prebaked_camera": LEGACY_PREBAKED_CAMERA_RECIPE,
    "tokengs_prebaked_camera_tiled": LEGACY_PREBAKED_CAMERA_TILED_RECIPE,
    "tokengs_single_image": LEGACY_SINGLE_IMAGE_RECIPE,
    "tokengs_single_image_tiled": LEGACY_SINGLE_IMAGE_TILED_RECIPE,
    "gauge_fields_material_surfel": EXTERNAL_GAUGE_FIELDS_RECIPE,
    "splat_baseline_free_dynamic_3dgs": EXTERNAL_SPLAT_BASELINE_RECIPE,
    "splat_baseline_static_3dgs": EXTERNAL_SPLAT_BASELINE_RECIPE,
}
```

### Compatibility Shims

File: `src/train/compat.py`

```python
@dataclass(frozen=True)
class LegacyEntrypoint:
    file: Path
    old_command_template: str
    replacement_command_template: str
    arch_values: tuple[str, ...]
    removal_policy: str


def run_legacy_entrypoint(
    *,
    old_file: str,
    config: dict[str, Any] | str | Path,
    default_config: str | Path | None = None,
) -> None: ...


def print_deprecation_notice(notice: DeprecationNotice) -> None: ...


def legacy_main(
    old_file: str,
    argv: Sequence[str],
    *,
    default_config: str | Path | None = None,
) -> None: ...
```

Shim behavior:

- Resolve default config if no explicit config was historically accepted.
- Print old command, new command, and removal policy.
- Call `train.py run`.
- Preserve exit code.
- Never duplicate trainer logic.

Example replacement for misspelled shim:

```python
from compat import legacy_main

if __name__ == "__main__":
    legacy_main(
        "src/train/train_camera_implict_dynamic.py",
        sys.argv[1:],
        default_config="src/train_configs/local_mac_overfit_image_implicit_camera.jsonc",
    )
```

### Legacy Adapter

File: `src/train/legacy_adapter.py`

```python
@dataclass(frozen=True)
class LegacyAdapter:
    name: str
    entrypoint: Callable[[dict[str, Any]], None]
    config_normalizer: Callable[[dict[str, Any]], dict[str, Any]] | None
    smoke_config_patch: Callable[[dict[str, Any]], dict[str, Any]]
    supported_arches: tuple[str, ...]


def run_legacy_recipe(spec: ExperimentSpec, adapter: LegacyAdapter) -> None: ...
def smoke_legacy_recipe(spec: ExperimentSpec, adapter: LegacyAdapter, smoke: "SmokeSpec") -> "SmokeResult": ...
```

Use cases:

- `dynamicTokenGS.py`
- `tokenGS.py`
- `train_camera_implicit_dynamic.py`
- tiled legacy scripts
- external gauge-field delegates

The adapter makes legacy status explicit without blocking router adoption.

### Data And View Contracts

File: `src/train/view_batch.py`

```python
@dataclass(frozen=True)
class ConditioningInput:
    sequence_id: str
    source_path: Path | None
    camera_name: str | None
    frames: torch.Tensor | None
    features: Mapping[str, torch.Tensor] | torch.Tensor | None
    frame_indices: torch.Tensor
    frame_times: torch.Tensor
    cameras: tuple[CameraSpec, ...] | None
    video_fps: float
    feature_cache_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetView:
    role: Literal["train", "eval", "heldout", "source"]
    view_id: str
    camera_name: str | None
    frames: torch.Tensor
    frame_indices: torch.Tensor
    frame_times: torch.Tensor
    cameras: tuple[CameraSpec, ...] | None
    loss_weight: float = 1.0
    metrics_prefix: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewBatch:
    sample_id: str
    scene_id: str | None
    conditioning: ConditioningInput
    targets: tuple[TargetView, ...]
    decode_times: torch.Tensor
    frame_indices: torch.Tensor
    device: torch.device
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

The point is to make single-camera, known-camera, precomputed-feature, and
multicam training all look like:

```python
batch = data_source.sample_train_batch(...)
model_output = model_program.decode(batch)
loss_bundle = objective.loss(model_output.sequence, batch, phase="train")
```

### Feature Provider

File: `src/train/feature_provider.py`

```python
@dataclass(frozen=True)
class FeatureDescription:
    provider_name: str
    layers: tuple[str, ...]
    channels: Mapping[str, int]
    temporal_length: int | None
    spatial_size: tuple[int, int] | None
    dtype: torch.dtype
    cache_key: str | None


class FeatureProvider(Protocol):
    def prebake(self, sequences: Sequence[SequenceData]) -> None: ...
    def describe(self, sequence: SequenceData) -> FeatureDescription: ...
    def load(self, conditioning: ConditioningInput) -> Mapping[str, torch.Tensor] | torch.Tensor: ...
    def release(self) -> None: ...


def build_feature_provider(
    spec: ExperimentSpec,
    *,
    device: torch.device,
) -> FeatureProvider | None: ...
```

This removes model-config mutation from the precomputed trainer.

### Model Program

File: `src/train/model_program.py`

```python
@dataclass(frozen=True)
class ModelInput:
    condition: torch.Tensor | Mapping[str, torch.Tensor] | None
    input_times: torch.Tensor | None
    decode_times: torch.Tensor
    render_cameras: tuple[CameraSpec, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutput:
    sequence: GaussianSequence
    camera_owner: Literal["model", "batch", "external_rig", "none"]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ModelProgram(Protocol):
    feature_dim: int
    camera_owner: Literal["model", "batch", "external_rig", "none"]

    def make_input(self, batch: ViewBatch) -> ModelInput: ...
    def decode(self, model_input: ModelInput) -> ModelOutput: ...


def build_model_program(
    spec: ExperimentSpec,
    feature_description: FeatureDescription | None,
) -> ModelProgram: ...
```

Do not proliferate trainer subclasses for:

- known-camera vs implicit-camera
- precomputed vs online features
- multicam vs single-cam
- static/dynamic split

Represent those as data/model/objective capabilities.

### Render Objective

File: `src/train/objective.py`

```python
@dataclass(frozen=True)
class BackgroundSpec:
    mode: Literal["white", "black", "random_rgb", "fixed_rgb"]
    rgb: tuple[float, float, float] | None = None
    train_mode: Literal["white", "black", "random_rgb", "fixed_rgb"] | None = None
    eval_mode: Literal["white", "black", "fixed_rgb"] = "white"
    sample_scope: Literal["step", "view", "frame"] = "step"


@dataclass(frozen=True)
class RasterizedClip:
    features: torch.Tensor
    alpha: torch.Tensor | None
    cameras: tuple[CameraSpec, ...]
    view: TargetView


@dataclass(frozen=True)
class RenderedClip:
    rgb: torch.Tensor
    features: torch.Tensor
    alpha: torch.Tensor | None
    splat_rgb: torch.Tensor | None
    background_rgb: torch.Tensor | None
    view: TargetView


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    recon: torch.Tensor
    regularizers: Mapping[str, torch.Tensor]
    metrics: Mapping[str, float]
    previews: Mapping[str, RenderedClip]


class RenderObjective:
    def rasterize(
        self,
        decoded: GaussianSequence,
        view: TargetView,
    ) -> RasterizedClip: ...

    def colorize(
        self,
        raster: RasterizedClip,
    ) -> torch.Tensor | None: ...

    def sample_background(
        self,
        *,
        phase: Literal["train", "eval"],
        like: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | None: ...

    def compose(
        self,
        raster: RasterizedClip,
        splat_rgb: torch.Tensor | None,
        background_rgb: torch.Tensor | None,
    ) -> RenderedClip: ...

    def reconstruct_loss(
        self,
        rendered: RenderedClip,
        target_rgb: torch.Tensor,
    ) -> torch.Tensor: ...

    def render_view(
        self,
        decoded: GaussianSequence,
        view: TargetView,
        *,
        phase: Literal["train", "eval"],
        generator: torch.Generator | None = None,
    ) -> RenderedClip: ...

    def loss(
        self,
        decoded: GaussianSequence,
        batch: ViewBatch,
        *,
        phase: Literal["train", "eval"],
        regularizers: Mapping[str, torch.Tensor] | None = None,
        keep_preview: bool = False,
    ) -> LossBundle: ...
```

Required invariants:

- F=3 legacy path may use RGB directly.
- F!=3 path must have a colorize module.
- F!=3 fast-mac path should return alpha when available.
- Train background mode is sampled by objective and logged.
- Eval background defaults to white for stable visual comparison.
- Every view, including multicam heldout, uses the same compose logic.

### Training Loop

File: `src/train/train_loop.py`

```python
@dataclass
class TrainComponents:
    spec: ExperimentSpec
    device: torch.device
    data_source: "DataSource"
    feature_provider: FeatureProvider | None
    model_program: ModelProgram
    objective: RenderObjective
    optimizer: torch.optim.Optimizer
    logger: "TrainLogger"
    export_policy: "ExportPolicy"


@dataclass(frozen=True)
class StepOutput:
    step: int
    batch: ViewBatch
    decoded: GaussianSequence
    losses: LossBundle
    camera_state: CameraState | None
    scalar_payload: Mapping[str, float]


class TrainLoop(Protocol):
    def initial_step_result(self) -> StepOutput: ...
    def step(self, *, keep_preview: bool = False) -> StepOutput: ...
    def validate(self, *, step: int) -> "ValidationPayload": ...
    def run(self) -> None: ...
```

### Validation And Media

File: `src/train/validation.py`

```python
@dataclass(frozen=True)
class ValidationRender:
    view: TargetView
    rendered: RenderedClip
    target_rgb: torch.Tensor
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ValidationPayload:
    scalars: Mapping[str, float]
    videos: Mapping[str, Any]
    images: Mapping[str, Any]
    decoded_metrics: Mapping[str, float]


def render_validation_views(
    components: TrainComponents,
    *,
    phase: Literal["eval"] = "eval",
) -> tuple[ValidationRender, ...]: ...


def build_validation_payload(
    renders: Sequence[ValidationRender],
    *,
    log_gt: bool,
    video_fps: float,
    include_alpha: bool,
    include_feature_pca: bool,
    include_composite: bool,
) -> ValidationPayload: ...
```

Every active recipe should get:

- GT video
- rendered video
- render-vs-GT video
- alpha mask video when alpha exists
- feature PCA video when F!=3 and requested
- composite video: GT | pred | alpha | feature PCA
- train-view and heldout prefixes for multicam

### Smoke Harness

File: `src/train/smoke.py`

```python
@dataclass(frozen=True)
class SmokeSpec:
    name: str
    config_path: Path
    arch: str
    steps: int = 1
    wandb_mode: Literal["offline", "disabled"] = "offline"
    expected_media_keys: tuple[str, ...] = ()
    expected_metric_keys: tuple[str, ...] = ()
    timeout_seconds: int = 300
    patch: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmokeResult:
    name: str
    config_path: Path
    command: tuple[str, ...]
    exit_code: int
    wall_seconds: float
    log_tail: str
    media_keys_found: tuple[str, ...]
    metric_keys_found: tuple[str, ...]
    passed: bool


def make_smoke_config(
    spec: ExperimentSpec,
    smoke: SmokeSpec,
    *,
    output_path: Path,
) -> Path: ...


def run_smoke(smoke: SmokeSpec) -> SmokeResult: ...
def run_smoke_matrix(smokes: Sequence[SmokeSpec]) -> list[SmokeResult]: ...
```

Smoke principle:

- `py_compile` is not enough.
- Any return-shape/config/dataclass/override migration needs a runtime smoke
  that exercises the actual call graph.
- For trainer routes, one-step smoke must exercise both train and validation
  media paths.

### Baseline Audit

File: `src/train/baseline_audit.py`

```python
@dataclass(frozen=True)
class BaselineRow:
    tier: str
    category: str
    config_path: Path
    route: str
    wandb_run_id: str | None
    steps: int | None
    wall_seconds: float | None
    train_metrics: Mapping[str, float]
    heldout_metrics: Mapping[str, float]
    notes: str
    refreshed_at: str


@dataclass(frozen=True)
class BaselineAuditReport:
    known_configs: tuple[Path, ...]
    rows: tuple[BaselineRow, ...]
    missing_configs: tuple[Path, ...]
    stale_routes: tuple[str, ...]
    missing_wandb: tuple[Path, ...]
    missing_heldout_metrics: tuple[Path, ...]
    warnings: tuple[str, ...]


def parse_baselines(path: Path = Path("BASELINES.md")) -> BaselineAuditReport: ...
def audit_config_routes(specs: Sequence[ExperimentSpec], baselines: BaselineAuditReport) -> BaselineAuditReport: ...
def print_baseline_audit(report: BaselineAuditReport) -> None: ...
```

Baseline provenance policy:

- Run id identifies a result row.
- Cleanup reruns append dated rows.
- Config route changes must preserve old route in notes.
- For random background, W&B config must include the background mode.
- For multicam, heldout metrics must be first-class:
  - `Heldout*/Eval/PSNR`
  - `Heldout*/Eval/SSIM`
  - `Heldout*/Eval/L1`
  - `Heldout*/Eval/Loss`

### Deprecation Notice Format

File: `src/train/deprecation.py`

```python
@dataclass(frozen=True)
class DeprecationRecord:
    old_path: Path
    replacement: str
    status: Literal["redirected", "deprecated", "deleted", "archival"]
    first_warned_date: str
    removal_not_before: str
    reason: str
    parity_smokes: tuple[str, ...]


def render_deprecation_notice(record: DeprecationRecord) -> str: ...
def write_deprecation_index(records: Sequence[DeprecationRecord], path: Path) -> None: ...
```

This prevents deletion decisions from living only in commit messages.

## Migration Order

### Phase 0: Freeze And Audit Current Surface

Goal: create mechanical visibility without behavior change.

Work:

1. Add `src/train/train.py explain CONFIG`.
2. Add `ExperimentSpec`, `TrainRecipe`, `RoutingReport`.
3. Register every current arch value.
4. For each arch, report:
   - current Python entrypoint
   - current shell script if known
   - recipe status: active/compat/legacy/external
   - whether F32 feature splatting is supported
   - whether heldout eval is supported
   - required smoke
5. Add `train.py list-arch`.

Exit criteria:

- `train.py explain` works for all 96 configs.
- No trainer behavior changes.
- No scripts changed yet.

### Phase 1: Add Config-Visible Background And Feature-Splatting Route Checks

Goal: make current F32 behavior provable in W&B/config.

Work:

1. Add normalized `losses.background`:
   ```python
   {
       "mode": "random_rgb",
       "eval_mode": "white",
       "sample_scope": "step",
   }
   ```
2. Default old configs to existing behavior:
   - F=3: white/legacy unless explicitly set.
   - F!=3 F32 configs: migrate intentional alpha configs to random RGB.
3. Add route validation:
   - `feature_dim != 3` requires `colorize`.
   - `feature_dim != 3` requires objective supports feature alpha composition.
   - multicam F32 should fail with a clear message until Phase 4.
4. Log background mode/scalars to W&B.

Exit criteria:

- Existing single-camera F32 smoke passes.
- W&B config and scalar payload make random background visible.

### Phase 2: Extract Render Objective Without Changing Single-Camera Behavior

Goal: remove the most dangerous duplicated logic.

Work:

1. Add `RasterizedClip`, `RenderedClip`, `BackgroundSpec`, `RenderObjective`.
2. Move base `Trainer.recon_backward` render/colorize/compose/loss logic into
   objective functions.
3. Move base `initial_step_result` and `render_full_sequence` composition into
   objective functions.
4. Keep `Trainer` as a caller of objective, not the owner of objective logic.
5. Add a targeted one-step F3 and F32 smoke.

Exit criteria:

- Single-camera F3 smoke passes.
- Single-camera F32 smoke passes and logs alpha/PCA/composite videos.
- W&B run for F32 still has comparable first-step behavior to prior random-bg
  run, acknowledging stochastic background.

### Phase 3: Fix Known-Camera Stale Paths

Goal: make the known-camera branch safe before more router work.

Work:

1. Make `KnownCameraTrainer.initial_step_result()` call objective.
2. Make known-camera validation call objective.
3. Add a smoke for:
   - `src/train_configs/local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc`
4. Confirm no tuple is passed to colorize/loss.

Exit criteria:

- Known-camera one-step smoke passes train and validation.
- `rg "rendered_features = self.render_decoded_clip"` no longer finds stale
  tuple misuse.

### Phase 4: Migrate Multicam To Shared ViewBatch And Objective

Goal: unblock the "ultimate" multicam F32 V-JEPA baseline.

Work:

1. Add `ConditioningInput`, `TargetView`, `ViewBatch`.
2. Convert multicam sampler to emit `ViewBatch` with:
   - condition sequence
   - train target views
   - heldout target views for validation
   - stable `view_id` and camera names
3. Convert `LearnableCameraRig` output into `TargetView.cameras`.
4. Replace `render_view_clip()` and `multicam_recon_loss()` with objective loss.
5. Replace `render_full_external_views()` with `render_validation_views()`.
6. Add multicam media:
   - `TrainView*_Alpha_Mask_Video`
   - `TrainView*_Feature_PCA_Video`
   - `TrainView*_Render_Composite_Video`
   - `Heldout*_Alpha_Mask_Video`
   - `Heldout*_Feature_PCA_Video`
   - `Heldout*_Render_Composite_Video`
7. Add route validation for heldout metrics.

Exit criteria:

- Multicam F3 or RGB-pyramid smoke passes if available.
- Multicam F32 ultimate one-step smoke passes.
- Heldout media exists.
- No multicam code calls `render_clip_sequence(...).detach()` directly.

### Phase 5: Extract FeatureProvider

Goal: remove feature-cache mutation from trainer/model construction.

Work:

1. Add `FeatureProvider.describe()`.
2. Build model after feature description is known.
3. Remove mutation of `model_cfg["video_feature_channels"]`.
4. Make LTX, Wan, V-JEPA all aliases of the same precomputed feature provider.

Exit criteria:

- Precomputed V-JEPA single-camera smoke passes.
- LTX/Wan config routes explain correctly.
- Feature cache provenance is printed and logged.

### Phase 6: Redirect Scripts To Router

Goal: preserve human launchers while centralizing routing.

Work:

1. Update active scripts to call:
   ```bash
   PYTHONPATH=src/train uv run python src/train/train.py run "$CONFIG_PATH"
   ```
2. Keep script names.
3. Add `DYNATRAIN_EXPLAIN=1` path to print route.
4. Leave dataset/camera utility scripts alone.

Exit criteria:

- Script smoke suite passes.
- Old commands still work.
- Route reports match old entrypoints.

### Phase 7: Add Legacy Adapters

Goal: make old procedural trainers visible and contained.

Work:

1. Add `LegacyAdapter` entries for:
   - `dynamicTokenGS.py`
   - `tokenGS.py`
   - `train_camera_implicit_dynamic.py`
   - tiled legacy variants
2. Add smoke patches for their smallest configs.
3. Print deprecation notices for legacy families.
4. Extract shared utilities from `dynamicTokenGS.py`.

Exit criteria:

- Legacy configs can be routed.
- Legacy configs can be smoked or explicitly marked archival.
- New trainers no longer import utility functions from `dynamicTokenGS.py`.

### Phase 8: Delete True Duplicates

Goal: remove obvious clutter after compatibility warnings.

Delete:

- `src/train/train_camera_implict_dynamic.py`
- `src/train/train_image_encoder_implicit_camera_baseline.py`
- `src/train/train_ltx_feature_implicit_dynamic.py` if no script calls it and
  router handles `ltx_feature_implicit_camera`.

Delete later:

- module-level `render_full_sequence` after replacement.

Do not delete:

- `dynamicTokenGS.py`
- `tokenGS.py`
- tiled legacy trainers
- gauge-field stack

until the corresponding legacy adapter and baseline audit says they are unused
or archival.

### Phase 9: Run Baseline/Provenance Audit

Goal: prove cleanup did not erase standings.

Work:

1. Run `train.py audit-baselines`.
2. Check every active config has:
   - route
   - W&B run if it is a baseline
   - tier
   - heldout metrics if it claims novel-view value
3. Append cleanup rows only when reruns happen.
4. Add a migration note to each new W&B run:
   - old route
   - new route
   - objective version
   - background mode
   - feature_dim
   - renderer mode

Exit criteria:

- `BASELINES.md` remains append-only.
- Ultimate multicam F32 row stays TODO until the migrated route is actually run.
- No claim that a cleanup run beats a baseline without heldout metric context.

## Smoke And Test Matrix

### Required Route Smokes

| Smoke | Config | Purpose | Expected media |
|---|---|---|---|
| F3 single-camera video-token | fast F3 video-token config patched to 1 step | Base route, legacy RGB render/loss | GT, Render, Render_GT |
| F32 single-camera feature splat | `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc` patched to 1 step | Feature rasterizer, colorize, alpha, random train background | GT, Render, Render_GT, Alpha, Feature_PCA, Composite |
| Known-camera | `local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc` patched to 1 step | Known-camera stale tuple path | GT, Render, Render_GT |
| Precomputed V-JEPA single-camera | `local_mac_overfit_precomputed_vjepa2_1_torchhub_vitb_384.jsonc` patched to 1 step | FeatureProvider and inherited objective | GT, Render, Render_GT |
| Multicam RGB/F3 | smallest multicam config, patched to 1 step | ViewBatch/routing/camera rig without F32 complexity | TrainView and Heldout videos |
| Multicam F32 ultimate | `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc` patched to 1 step | Actual target route | Train/Heldout GT, Render, Alpha, Feature_PCA, Composite |
| Legacy image implicit | `local_mac_overfit_image_implicit_camera.jsonc` patched tiny | Legacy adapter sanity | Existing legacy media |
| Legacy prebaked camera | `local_mac_overfit_prebaked_camera_128_4fps.jsonc` patched tiny | `dynamicTokenGS.py` adapter | Existing legacy media |
| Gauge-field smoke | `local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc` | External route delegate | Gauge smoke output |
| Splat baseline smoke | smallest static/dynamic 3DGS config | External baseline route | Heldout metrics if route supports it |

### Assertions Beyond Exit Code

Each smoke result should record:

```python
expected = {
    "exit_code": 0,
    "step_count": 1,
    "has_train_loss": True,
    "has_validation_payload": True,
    "has_expected_media": True,
    "no_tuple_arity_exception": True,
    "no_shape_mismatch_feature_vs_rgb": True,
}
```

F32-specific assertions:

```python
expected_f32 = {
    "feature_dim": 32,
    "colorize_enabled": True,
    "alpha_available": True,
    "background_mode_logged": True,
    "alpha_mask_video_logged": True,
    "feature_pca_video_logged": True,
    "composite_video_logged": True,
}
```

Multicam-specific assertions:

```python
expected_multicam = {
    "train_view_metrics_logged": True,
    "heldout_metrics_logged": True,
    "heldout_render_video_logged": True,
    "heldout_alpha_video_logged_if_f32": True,
    "rig_metrics_logged": True,
}
```

### Tests Worth Adding

Add tests only where they catch real regressions:

```python
def test_arch_registry_covers_all_train_configs() -> None: ...
def test_feature_dim_gt_3_requires_colorize_and_objective_support() -> None: ...
def test_multicam_f32_route_fails_until_objective_is_enabled() -> None: ...
def test_legacy_aliases_emit_deprecation_notices() -> None: ...
def test_explain_routing_reports_old_and_new_commands() -> None: ...
```

Avoid brittle tests that assert implementation details such as "helper X called
torch.linalg.svd exactly once." Prefer smokes and artifact checks for trainer
routes.

## W&B And BASELINES Provenance Plan

### W&B Config Fields To Log

Every active run should include:

```python
{
    "Route/Arch": spec.arch,
    "Route/Recipe": recipe.name,
    "Route/OriginalEntrypoint": compatibility.original_entrypoint,
    "Objective/Version": objective.version,
    "Model/FeatureDim": spec.model["feature_dim"],
    "Renderer/Mode": spec.render["renderer"],
    "LossBackground/Mode": background.mode,
    "LossBackground/TrainMode": background.train_mode or background.mode,
    "LossBackground/EvalMode": background.eval_mode,
    "Data/ViewContract": type(batch).__name__,
}
```

F32 runs should additionally log:

```python
{
    "FeatureSplatting/Colorize": colorize.__class__.__name__,
    "FeatureSplatting/PreNorm": colorize_cfg["pre_norm"],
    "FeatureSplatting/WeightInit": colorize_cfg["weight_init"],
    "FeatureSplatting/WeightInitGain": colorize_cfg["weight_init_gain"],
    "FeatureSplatting/AlphaAvailable": alpha is not None,
}
```

Multicam runs should log:

```python
{
    "Multicam/TrainViewCount": train_view_count,
    "Multicam/HeldoutViewCount": heldout_view_count,
    "Multicam/ConditionCamera": condition_camera_name,
    "Multicam/AnchorCamera": anchor_camera_name,
    "Multicam/HeldoutCameras": heldout_camera_names,
}
```

### BASELINES Update Rule

Do not update `BASELINES.md` from a smoke unless the row is explicitly a smoke
row. For meaningful reruns:

1. Append a new dated row.
2. Include old route and new route if route changed.
3. Include W&B run id and URL.
4. Include step count, wall clock, device.
5. Include heldout metrics for Tier 2.
6. Preserve older rows.

### Migration Run Labels

Use W&B tags:

- `cleanup-router`
- `objective-v1`
- `feature-splatting`
- `random-bg`
- `multicam`
- `heldout-eval`
- `legacy-adapter` when applicable

## Deletion Decision Table

| File/family | Decision | Earliest action | Blocker |
|---|---|---|---|
| `train_camera_implict_dynamic.py` | Delete after shim period | Replace with deprecation shim now | Need router route for image implicit default config |
| `train_image_encoder_implicit_camera_baseline.py` | Delete after shim period | Replace with deprecation shim now | Need router route for image implicit default config |
| `train_ltx_feature_implicit_dynamic.py` | Delete/redirect | After precomputed alias route exists | Need `ltx_feature_implicit_camera` route |
| module-level `render_full_sequence` in video trainer | Delete | After validation helper replaces it | Need `rg` no callers |
| `dynamicTokenGS.py` | Retire behind adapter | Extract utilities first | Scripts/configs still call it |
| `tokenGS.py` | Retire behind adapter | Add legacy route first | Single-image configs still exist |
| `train_camera_implicit_dynamic.py` | Retire behind adapter | Add legacy route first | Two image implicit configs still exist |
| tiled legacy trainers | Retire behind adapter | Audit exact files first | Tiled configs still exist |
| gauge-field stack | Keep separate | Add external route only | Different research harness semantics |
| multicam trainer subclass | Merge into functional recipe | After shared objective exists | Current F32 route broken |
| precomputed trainer subclass | Merge into feature provider | After FeatureProvider exists | Current cache/model mutation |
| base `Trainer` class | Shrink, then maybe replace | After TrainLoop exists | Active stable path today |

## First Implementation Slice

The first slice should be intentionally boring:

1. Add `src/train/train.py explain`.
2. Add `ExperimentSpec`, `TrainRecipe`, and route registry.
3. Register all 96 configs.
4. Add no behavior-changing train logic.
5. Add tests:
   - registry covers every config arch
   - route report includes old and new command
6. Write no deletion patch yet.

Why this first:

- It gives immediate visibility.
- It does not risk current runs.
- It creates the mechanical map needed to redirect scripts safely.
- It exposes arch/config drift before the render objective refactor starts.

## Second Implementation Slice

Second slice should fix the active bug surface:

1. Add `BackgroundSpec` config normalization.
2. Extract `RenderObjective`.
3. Port base single-camera trainer to objective.
4. Fix known-camera initial/eval objective path.
5. One-step smoke:
   - F3 single-camera
   - F32 single-camera
   - known-camera

Why this second:

- It makes the F32 fix config-visible.
- It eliminates tuple/render/loss drift in the base and known-camera paths.
- It prepares the exact interface multicam needs.

## Third Implementation Slice

Third slice should unblock the actual desired baseline:

1. Add `ViewBatch`.
2. Convert multicam training and validation to `ViewBatch + RenderObjective`.
3. Add heldout alpha/PCA/composite videos.
4. Smoke ultimate F32 multicam for 1 step.
5. Only then launch the 1000-step ultimate baseline.
6. Append `BASELINES.md` row after the real run completes.

Why this third:

- Multicam heldout is the real selector.
- The current ultimate config is not trustworthy until this route is fixed.
- This is the point where cleanup becomes scientifically useful, not just tidy.

## Open Questions

1. Should legacy image/single-image/prebaked configs remain runnable forever, or
   should they become archival docs after one final smoke?

2. Should gauge-field routes live in `src/train/train.py`, or should the router
   print a delegated command and intentionally not import gauge-field modules?
   The safer first version is delegated command only.

3. Should random background default on for all `feature_dim != 3` configs, or
   only for configs that opt in? The migration-safe answer is opt-in with route
   warnings for F32 configs that leave it unset.

4. Should multicam validation always render all train and heldout views, or only
   the first train view plus all heldout views? For debugging F32 holes, all
   heldout alpha videos matter more than all train-view videos.

5. Should `rgbs` be renamed to `features` in `GaussianSequence`? Not during this
   migration. The field name is ugly, but renaming it now would enlarge the
   blast radius. Add `GaussianSequence.features` as a property later.

## Final Recommendation

Do not start by deleting files. Start by making route ownership explicit, then
make the render/objective boundary impossible to bypass, then port multicam.
Only after that should deletion begin.

Priority order:

1. Router/explain registry for all configs.
2. Config-visible random background and objective extraction.
3. Known-camera stale tuple fix.
4. Multicam `ViewBatch + RenderObjective` migration.
5. One-step smoke matrix.
6. Ultimate F32 multicam run.
7. Script redirection.
8. Legacy adapter containment.
9. Delete duplicate shims and empty aliases.

This preserves current baselines, makes the active F32 fix auditable, and
prevents the same bug class from recurring in multicam or known-camera routes.
