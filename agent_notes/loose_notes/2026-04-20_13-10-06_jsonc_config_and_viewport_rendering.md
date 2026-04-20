# JSONC Config And Viewport Rendering

This is a raw session note following the local `AGENTS.md` memory pattern:
chronology and decisions live here; dense reusable lessons go in
`agent_notes/key_learnings.md`.

## Starting Point

The video-token implicit-camera baseline had just gone through a broad trainer
interface cleanup. The live refactor surface already included:

- `src/train/runtime_types.py`
- `src/train/sequence_data.py`
- `src/train/rendering.py`
- `src/train/train_logging.py`
- `src/train/gs_models/implicit_camera.py`

The user then asked what it meant that "camera intrinsics are scaled from input
size to render size," whether this implied separate viewport and camera-ray
abstractions, and whether the config refactor aligned with the newer helper
modules.

The user also pushed back on a pattern that had started to appear: huge default
parameter maps in Python plus env-var and shell-script fanout for every knob.
The desired style is config-first and lean:

- define all experiment knobs once in checked-in configs
- avoid argparse and env-var mirrors for every hyperparameter
- pass config sections or kwargs through warm paths
- keep any unavoidable name mapping near the boundary that needs it

## Camera/View Abstraction Decision

We clarified that there are two related but separate concepts:

- input size: the video tensor size the encoder consumes
- render size / viewport: the raster target size used by the splat renderer and
  the resized GT frame used for reconstruction loss

The camera intrinsics are expressed in pixel units. If a camera was decoded for
a 384x384 input, but the renderer targets 192x192, then:

```text
fx, cx *= 192 / 384
fy, cy *= 192 / 384
```

This preserves FoV and ray directions because normalized image coordinates keep
the same value:

```text
(x - cx) / fx
```

The operation is a viewport resize, not a crop. A crop would alter the principal
point and/or visible FoV unless handled explicitly. This baseline should not
silently crop for render-size changes.

The clean abstraction is:

- keep model/camera prediction in the input-image coordinate system
- convert to render-viewport intrinsics at the render boundary
- resize GT frames to the render viewport for the loss
- keep pose unchanged

## Implementation

`src/train/rendering.py` gained:

- `camera_for_viewport(camera, source_height, source_width, target_height, target_width)`

That helper returns the same `CameraSpec.camera_to_world` with `fx`, `fy`, `cx`,
and `cy` scaled into the target viewport. This keeps viewport math in the render
facade instead of hand-rolled in the trainer.

`src/train/train_video_token_implicit_dynamic.py` changed to:

- require a config dict or config path
- load JSONC config files through `load_config_file(...)`
- remove `DEFAULT_CONFIG`
- remove `config_from_env(...)`
- remove env-var fanout as the default extension mechanism
- call `camera_for_viewport(...)` when rendering a frame
- build dense pixel grids through `rendering.build_or_reuse_grid(...)`

The JSONC parser supports line and block comments while preserving strings. The
checked-in configs intentionally avoid trailing commas.

The shell wrappers were reduced to thin launchers:

- `src/train_scripts/train_full_dynamic_with_video_token_implicit_camera_all_frames.sh`
- `src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh`

Each wrapper now accepts at most one optional config path and otherwise uses its
default JSONC config.

Added configs:

- `src/train_configs/video_token_implicit_camera_full.jsonc`
- `src/train_configs/video_token_implicit_camera_smoke.jsonc`

The full config currently uses:

- input/model size: 384
- render size: 192
- train frame count: 16
- 8 3DGS tokens
- 64 gaussians per token

The 192 render size is an intentional memory guard for MPS. Setting render size
back to 384 restores full-resolution render loss but can hit the dense-render
memory cliff.

The smoke config currently uses:

- input/model size: 32
- render size: 16
- train frame count: 16
- 4 3DGS tokens
- 16 gaussians per token

W&B remains enabled by default. We did not bake `WANDB_MODE=disabled` into the
smoke path because the user wants W&B runs unless explicitly disabled.

## Local Agent Guide Update

The local `AGENTS.md` now records the config philosophy:

- JSONC train configs under `src/train_configs/` are the source of truth
- no giant Python default maps
- no duplicated env-var/argparse schema
- shell scripts choose config files
- Python normalizes runtime details and fails loudly when required keys are
  missing

It also defines the note pattern used here:

- raw chronology: `agent_notes/loose_notes/{YYYY-MM-DD_HH-MM-SS}_{slug}.md`
- dense memory: `agent_notes/key_learnings.md`

## Verification

Static checks:

```bash
uv run python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/rendering.py \
  src/train/runtime_types.py \
  src/train/sequence_data.py \
  src/train/train_logging.py
```

Shell checks:

```bash
bash -n src/train_scripts/train_full_dynamic_with_video_token_implicit_camera_all_frames.sh
bash -n src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh
```

Config load check:

```bash
uv run python - <<'PY'
import sys
sys.path.insert(0, 'src/train')
from train_video_token_implicit_dynamic import load_config_file, resolve_config
for path in [
    'src/train_configs/video_token_implicit_camera_full.jsonc',
    'src/train_configs/video_token_implicit_camera_smoke.jsonc',
]:
    cfg = resolve_config(load_config_file(path))
    print(path, cfg['model']['size'], cfg['render']['render_size'], cfg['train']['steps'])
PY
```

Smoke run:

```bash
bash src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh
```

Result:

- completed 10 steps
- W&B run: `43w9me5p`
- input size: 32
- render size: 16
- train frame count: 16
- dense renderer
- final printed loss: about `0.1975`

## Follow-Up Context

The trainer is cleaner but still not fully typed end-to-end. `prepare_clip(...)`
and several trainer paths still use dict-style access through the temporary
`SequenceData.__getitem__` bridge. That is acceptable for this slice but should
eventually move to attribute access or `ClipBatch`.

`build_model_from_config(...)` remains the right place for the current
name-mapping layer because model constructor names do not exactly match config
keys. Avoid spreading that mapping into the shell scripts or multiple helper
functions.

If future configs need overrides, prefer creating another small JSONC file or a
deliberate config-composition mechanism. Do not rebuild the old env-var fanout
one knob at a time.
