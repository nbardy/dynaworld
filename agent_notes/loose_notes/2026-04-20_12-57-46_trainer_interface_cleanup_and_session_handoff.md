# Trainer Interface Cleanup And Session Handoff

This note is intentionally a journal entry, not a polished research summary. It
records what we did, what we thought, what turned out awkward, and what future
agents should know before continuing.

## Starting Point

The work started from a fast-moving `dynaworld` experiment inside
`gsplats_browser`. The user wanted a local `uv` project, fast Mac smoke runs,
DUSt3R camera prebake for a small video, and several dynamic TokenGS baselines.

Early repo/setup facts:

- `uv runpython ...` failed because `runpython` is not a `uv` subcommand.
- `uv run dynaworld/train_scripts/tokenGS.py` from the parent repo failed
  because the parent `pyproject.toml` had no `[project]` table at that time.
- The right fix was to make `dynaworld` its own `uv` project and run from that
  folder with `uv run python ...`.
- The user also wanted shared `uv` package caching. We discussed that `uv`
  already has a default global cache, and an explicit cache env var only matters
  if we want to override the default path.

We added project structure around `dynaworld`, then eventually made it a public
submodule repo at `git@github.com:nbardy/dynaworld.git`.

## Training Speed Thread

The first single-image `tokenGS.py` run on Mac used MPS and was slow enough that
the user asked for "lightning fast for testing." We explored:

- smaller token counts and lower render size for quick smoke tests
- mixed/lower precision ideas
- FasterGS-style renderer and optimizer speedups
- tile-based rasterization

Important conclusion from that thread:

- A naive tiled renderer with nested Python loops was slower than dense PyTorch
  for this small local setting.
- One "big tile" was surprisingly faster in the full loop, which suggested that
  some differences were overhead/path related rather than pure raster math.
- The useful immediate speed knob became decoupling input size from render/loss
  size for the video-token trainer, so the video encoder can still see larger
  frames while the expensive splat render/loss runs smaller.

The phrase "dense N x H x W rendering" came up because the simple dense renderer
evaluates all gaussians across all pixels. Tile-based rasterization is only
worth it if it actually culls/sorts useful subsets cheaply. A Python-loop tiled
implementation does not give the same win as a fused CUDA/Metal rasterizer.

## Camera And Video Data Thread

The user wanted dynamic training with camera data from video, so we pulled in
DUSt3R as `third_party/dust3r` and made camera prebake scripts.

Video preprocessing:

- Source video: `/Users/nicholasbardy/Downloads/14951282_2160_3840_30fps.mp4`
- Created `test_data/test_video_small.mp4` as a 256x256 center crop at 2 fps.
- Created `test_data/test_video_384_3fps.mp4` as a 384x384 center crop at 3 fps.
- Added `test_data/catalogue.md` after fixing the earlier typo-ish
  `text_data`/`test_data` confusion.

DUSt3R camera prebake:

- `src/train/run_dust3r_video.py` extracts sampled frames and runs DUSt3R pairs.
- `src/train_scripts/get_camera.sh` wraps the camera run.
- The user saw what looked like "two passes" in DUSt3R output. The actual
  explanation was that DUSt3R first ran inference over image pairs, then ran
  global alignment optimization. It was not duplicate frame extraction.
- A stale frame folder caused confusion: old runs left many PNGs behind. The
  code path used the explicit records in `per_frame_cameras.json`, not every
  stale PNG in the folder. Still, clearing old extracted frames before a new run
  was the safer UX.

Camera modeling decisions:

- For known-camera dynamic training, DUSt3R gives intrinsics and camera-to-world
  extrinsics per frame.
- We anchored poses relative to the first frame for a local sequence frame.
- The renderer needed to use extrinsics, not just intrinsics, otherwise the model
  could cheat by emitting frame-conditioned gaussians rather than a consistent
  world-space scene.
- DUSt3R focal estimates can jitter, so the known-camera loader supports median
  focal mode.
- Lens type, f-stop, shutter, and sensor metadata are not useful for the current
  pinhole renderer.

## Model/Trainer Baselines Before Cleanup

By the time of this cleanup there were three real dynamic baselines:

- Known/prebaked camera: `src/train/dynamicTokenGS.py`
- Image encoder implicit camera: `src/train/train_camera_implicit_dynamic.py`
- Video encoder implicit camera: `src/train/train_video_token_implicit_dynamic.py`

There are also single-image and convenience scripts:

- `src/train/tokenGS.py`
- `src/train/tokenGS_tiled.py`
- `src/train/dynamicTokenGS_tiled.py`
- `src/train/train_camera_implict_dynamic.py` is a typo shim.

The user asked whether the trainers should share one full trainer shape. The
answer we settled on was no: keep separate readable trainers because the loops
are meaningfully different, but share boundary contracts and helper functions.

That decision matters. A giant shared `BaseTrainer` would hide the real
differences between known cameras, image-implicit cameras, and video-token
implicit cameras. The useful cleanup is common data/model/render/log vocabulary.

## Interface Cleanup Plan

We wrote detailed proposed signatures in
`TODO/Clean_up_and_unify_interfaces.md`, including:

- `SequenceData`
- `ClipBatch`
- `CameraState`
- `GaussianFrame`
- `GaussianSequence`
- `StepLosses`
- `TrainStepResult`
- shared loader signatures
- render dispatch signatures
- implicit-camera model helper signatures
- model protocols
- separate trainer class outlines

That TODO was committed first as:

```text
40acb77 Detail trainer interface contracts
```

## Three-Agent Implementation

The user then explicitly asked to use three sub-agents and "do it." The task was
split into disjoint ownership:

1. Worker 1: runtime payload contracts and sequence loaders.
2. Worker 2: render dispatch and W&B logging helpers.
3. Worker 3: implicit-camera model internals.

Worker outputs:

- Added `src/train/runtime_types.py`.
- Added `src/train/sequence_data.py`.
- Added `src/train/rendering.py`.
- Added `src/train/train_logging.py`.
- Added `src/train/gs_models/implicit_camera.py`.
- Updated both implicit-camera model files to import shared camera heads/math.

Then the main integration pass migrated the three trainer scripts onto those
helpers while keeping their loops separate.

## Helper Modules Added

`src/train/runtime_types.py`:

- `SequenceData`
- `ClipBatch`
- `CameraState`
- `GaussianFrame`
- `GaussianSequence`
- `StepLosses`
- `TrainStepResult`
- renderer/source/backward-strategy literal types

One compromise: `SequenceData.__getitem__` exists as a temporary bridge so older
dict-style code can migrate gradually. It is intentionally not the final style.

`src/train/sequence_data.py`:

- FPS inference
- timestamp normalization
- video sequence loading
- frame folder loading
- DUSt3R camera JSON loading
- uncalibrated sequence loading for implicit-camera baselines
- contiguous window sampling
- `make_clip`

`src/train/rendering.py`:

- resize helper
- dense/tiled renderer selection
- pixel grid construction/reuse
- `render_gaussian_frame`

Later, during another agent's config migration, this also gained
`camera_for_viewport(...)` so intrinsics can be scaled cleanly when input size
and render size differ.

`src/train/train_logging.py`:

- W&B video conversion
- preview image construction
- validation render/GT side-by-side payload

`src/train/gs_models/implicit_camera.py`:

- zero-init head helper
- skew/axis-angle SE(3) math
- camera composition
- `GlobalCameraHead`
- `PathCameraHead`

## What Went Wrong Or Needed Fixing

The first system-Python import smoke failed because `python3` outside the `uv`
env did not have `torchvision`. The fix was to run import checks with:

```bash
PYTHONPATH=src/train uv run python ...
```

`python -m py_compile` also failed for workers because `python` was not on PATH
in their shell. `python3` and `uv run python` worked.

The smoke shell script was not executable:

```text
permission denied: ./src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh
```

We fixed that with `chmod +x`, matching the other train scripts.

The first cleanup pass still left old dead loader helpers inside
`train_camera_implicit_dynamic.py`. We noticed that during review and removed
the stale helpers so the image-implicit trainer uses `sequence_data.py` as the
single loader implementation.

The other active agent started moving the video-token trainer from env vars to
checked-in JSONC config files right after our interface cleanup. We explicitly
stopped expanding env-var fanout and avoided touching those config migration
edits.

## Verification Performed

Static/compile checks:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/runtime_types.py \
  src/train/sequence_data.py \
  src/train/rendering.py \
  src/train/train_logging.py \
  src/train/dynamicTokenGS.py \
  src/train/train_camera_implicit_dynamic.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/gs_models/implicit_camera.py \
  src/train/gs_models/dynamic_token_gs_implicit_camera.py \
  src/train/gs_models/dynamic_video_token_gs_implicit_camera.py
```

Import and contract smokes:

- trainer imports passed under `PYTHONPATH=src/train`
- known-camera loader loaded two DUSt3R frames
- uncalibrated video loader loaded two video frames
- image-token implicit camera forward produced `[2, 4, 3]` xyz
- video-token implicit camera forward produced `[4, 4, 3]` xyz

One-step offline W&B smoke runs passed for:

- known/prebaked-camera trainer
- image-implicit-camera trainer
- video-token implicit-camera trainer

## Commits Pushed To `dynaworld`

Trainer interface signature plan:

```text
40acb77 Detail trainer interface contracts
```

Implemented interface cleanup:

```text
ff5d99b Unify trainer interfaces
```

The parent `gsplats_browser` repo has not had its submodule pointer committed
for this latest dynaworld commit because the parent worktree has unrelated
active changes from other agents.

## Current Dirty State At This Note

After `ff5d99b`, another agent started the config migration. At the time this
note was written, the `dynaworld` worktree had uncommitted changes in:

- `AGENTS.md`
- `src/train/rendering.py`
- `src/train/train_video_token_implicit_dynamic.py`
- `src/train_scripts/train_full_dynamic_with_video_token_implicit_camera_all_frames.sh`
- `src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh`
- `src/train_configs/`

Those are not part of the interface cleanup commit. Do not overwrite them.

## Current Mental Model

The project should stay experiment-friendly. The clean shape is:

- configs choose experiments
- trainer scripts stay readable
- shared modules define vocabulary and remove copy-paste
- model files keep architecture-specific code
- `agent_notes/` records the messy chronology
- `research_notes/` records more durable research artifacts

The next cleanup after config migration should probably remove the temporary
dict-style `SequenceData.__getitem__` once all trainer call sites use attributes.
