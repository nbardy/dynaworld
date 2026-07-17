# Cleanup And Unification Next Report

Date: 2026-05-18 16:50 Asia/Ho_Chi_Minh

Prompt: after the dataset-loader audit, answer whether there is cleanup work to
do: simplify modules, unify helpers, delete excess, and decide whether a global
"get all" helper makes sense.

Scope: current on-disk `dynaworld` tree, especially `src/train` dataset loaders,
trainer integration, objective/render helpers, logging/media helpers, and stale
cleanup docs.

Status: planning/report only. No code cleanup was implemented in this pass.

## TL;DR

Yes, there is real cleanup work, but it should not be a single global "get all"
helper.

The right shape is:

1. Keep same-view and novel-view as separate first-class data modes.
2. Add small typed normalization helpers at loader boundaries.
3. Add typed batch/result payloads so tuple shapes stop drifting.
4. Extend existing helper modules instead of inventing a new base trainer.
5. Delete or mark stale docs/shims only after `rg` proves they are not active.

The current repo already moved past part of the old trainer-unification TODO:
`RGBReconObjective`, `objective.types.RenderedView`, `pipeline/render.py`,
`pipeline/validation_media.py`, and `train_logging.py` exist. The next cleanup
should build on those modules, not repeat the stale proposal that alpha/colorize
is missing from multicam.

The highest-leverage next slice is a data-boundary cleanup:

- normalize single-sequence manifest records before loading;
- normalize multicam camera splits before loading;
- return typed `SameViewBatch` / `NovelViewBatch` payloads from sampling;
- add a tiny mixed scheduler only after the two loader contracts are clean.

## Why Not One Global Get-All Helper?

The tempting helper is something like:

```text
get_all_training_data(cfg) -> dict[str, Any]
```

That would be a mistake.

It would hide the core distinction the repo is trying to preserve:

- same-view rows reconstruct the encoded camera;
- multicam rows test or train camera transfer;
- query-conditioned relpose is not the same thing as source-only heldout;
- true heldout metrics should not be mixed with train losses.

A global "get all" helper would probably return a bag with `sequences`,
`bundles`, `eval`, `heldout`, `cameras`, `features`, and config flags. That
bag would make call sites shorter for a week, then make claims less auditable.

The safer alternative is one narrow orchestration helper:

```text
resolve_training_data_sources(cfg) -> TrainingDataSources
```

where:

```text
TrainingDataSources.same_view: SameViewSource | None
TrainingDataSources.novel_view: NovelViewSource | None
TrainingDataSources.eval_sources: EvalSources
```

That helper can centralize file paths, manifest existence checks, and split
metadata. It should not load every tensor, merge schemas, or erase `loss_kind`.

Rule of thumb:

- "get all config paths and source descriptors" is fine.
- "get all tensors/cameras/loss targets in one untyped object" is bad.

## Current Good Foundations

The codebase already has several good helper boundaries. Do not replace them.

`objective.objective.RGBReconObjective` is now the right loss/render-composition
boundary. It owns rasterize -> colorize -> background compose -> reconstruction
loss. This means the old `compose_rendered_rgb` TODO is mostly obsolete.

`objective.types` already has `TargetView`, `RasterizedView`, `ColorizedView`,
`BackgroundSample`, `RenderedView`, and `ViewLoss`. These are good typed
contracts. The cleanup direction should be to use them more consistently.

`pipeline/render.py` already introduced `RasterizedClip` and `RenderedClip` to
avoid some tuple-arity drift in full-sequence rendering. The next step is to add
typed training batches, not another render bundle.

`pipeline/validation_media.py` already centralizes single-cam and multicam W&B
media payloads, including alpha, feature PCA, composites, and multicam grids.
The old note claiming multicam lacks this path is stale.

`train_logging.py` is tiny but useful. It should grow one small scheduling helper
for logging cadence instead of leaving each trainer to repeat modulo checks.

`runtime_types.SequenceData` is a good base payload for loaded single-camera
sequences. It should not be overloaded into a multicam bundle.

## Current Mess

### 1. Loader Inputs Are Still Raw Dicts Too Deep Into The Call Graph

`sequence_data.load_manifest_sequence()` receives a raw manifest row and a raw
`data_cfg`. It immediately reads `entry["sequence_dir"]`, even for
`explicit_video_window` rows where the contract says `video_path`,
`start_seconds`, and `duration_seconds` should be sufficient.

`multicam_video_data.load_multicam_video_bundle()` receives a raw selected
record and raw config fields. It prefers `data_cfg["multicam_train_cameras"]`
over record-level `train_cameras`, so rich manifest rows only work correctly
when the launcher patches them into config.

This is the main cleanup seam: raw JSON dictionaries should be normalized at
the boundary, then the rest of the loader should consume typed or at least
validated objects.

### 2. Sampling Return Shapes Are Inconsistent

Current examples:

- base `sample_clip()` returns `(sequence_data, clip_frames, clip_times)`;
- known-camera `sample_clip()` returns those plus `clip_cameras`;
- multicam `sample_multicam_clip()` returns `sequence_data`, `clip_indices`,
  `clip_frames`, `clip_times`, and `views`;
- camera-swap loss returns a long tuple carrying loss, terms, preview tensors,
  camera state, frames, and sequence data.

This is a cleanup problem more than a style problem. It is the exact class of
bug the project guide warns about: `py_compile` will not catch tuple-arity
drift.

### 3. Split Names And Loss Names Are Not Yet First-Class

The docs want `same_view_recon` and `heldout_view_recon` as conceptual lanes,
but the current code mostly logs generic `Loss/Reconstruction`,
`TrainView*/Eval/*`, and `Heldout*/Eval/*`.

That is fine while lanes are separate. It becomes dangerous the moment one
trainer samples both lanes. The mixed bridge needs explicit `loss_kind` and
split semantics before it trains.

### 4. Calibration Metadata Is Preserved At The Loader But Dropped Later

`MulticamVideoBundle` carries lens metadata. DeepView can produce fisheye
metadata. But downstream rig construction in `camera_rig.py` creates pinhole
`CameraSpec`s. This is not a "global helper" problem; it needs a small camera
metadata helper and a stricter camera-spec construction path.

### 5. Some Old Cleanup Docs Are Now Stale

`TODO/trainer_landscape_unification.md` still says the multicam trainer bypasses
alpha/colorize. Current code routes multicam rendering through
`RGBReconObjective` and validation media helpers. That TODO should be marked
superseded or rewritten before anyone follows it.

The old delete candidates in that file also appear partly resolved: files such
as `dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`, and old shim names
are not present under current `src/train` in this checkout. Do not spend time
deleting ghosts. First run `rg --files`.

## Proposed Helper Modules

This is the cleanup I would do next.

### A. `src/train/data_records.py`

Purpose: normalize raw JSON manifest rows into small validated records.

Sketch:

```text
SingleSequenceRecord
    record_type
    frame_source
    split
    frame_count
    fps
    image_crop_mode
    explicit_video: ExplicitVideoWindow | None
    frame_clip: FrameClipSource | None
    camera_json: CameraJsonSource | None

MulticamRecord
    dataset
    sample_id
    split
    source_camera
    target_camera
    train_cameras
    heldout_cameras
    anchor_camera
    condition_camera
    fps
    frame_count
```

Rules:

- `explicit_video_window` should not require `sequence_dir`.
- `summary_sampled` should require enough sampled paths unless compatibility
  mode is explicitly on.
- multicam records should prefer record-level train/heldout fields, with config
  overrides only when the config intentionally overrides them.
- all JSONL parse errors should include path and line number.

Expected payoff:

- Smaller `sequence_data.load_manifest_sequence()`.
- Smaller `multicam_video_data.load_multicam_video_bundle()`.
- Tests can target normalization without opening videos.
- Docs and code can share one row-contract vocabulary.

Risk:

- If the dataclasses become too detailed, they turn into a second schema system.
  Keep them as loader-normalization records, not global domain models.

### B. `src/train/data_batches.py`

Purpose: stop tuple return drift in training and sampling.

Sketch:

```text
SameViewBatch
    loss_kind = "same_view"
    sequence_data
    clip_indices
    clip_frames
    clip_times
    cameras | None

NovelViewBatch
    loss_kind = "novel_view"
    bundle
    source_sequence
    clip_indices
    clip_frames
    clip_times
    train_views
    target_views
    target_role  # train_novel, heldout_eval, query_conditioned_eval

CameraSwapBatch
    loss_kind = "camera_swap"
    pairs
    clip_indices
    source_sequence
```

Rules:

- Training code receives a batch object, not a positional tuple.
- Batch object names must make leakage obvious: `heldout_eval` is not a train
  target unless a config explicitly renames it.
- `clip_times` must state whether it came from normalized sequence time or
  selected original frame indices.

Expected payoff:

- Smaller `step()` bodies.
- Easier mixed scheduler later.
- Runtime smokes become more meaningful because the call graph uses named
  fields instead of long unpacking.

Risk:

- If introduced everywhere at once, it will be noisy. Start with multicam
  sampling and camera-swap return values, where tuple drift is worst.

### C. `src/train/data_sources.py`

Purpose: replace a hypothetical global "get all" helper with a narrow source
descriptor resolver.

Sketch:

```text
TrainingDataSources
    same_view: SameViewSource | None
    novel_view: NovelViewSource | None
    eval: EvalSourceSet

resolve_training_data_sources(cfg) -> TrainingDataSources
```

This helper should:

- resolve manifest paths;
- check that files exist;
- load row counts cheaply when useful;
- classify source modes;
- report whether eval is empty, train-preview fallback, or true heldout.

It should not:

- open videos;
- load tensors;
- create cameras;
- merge same-view and novel-view records into one list;
- choose losses.

Expected payoff:

- Cleaner trainer startup prints.
- One place to detect "eval is empty but videos will fall back to train".
- A better preflight for scale scripts.

### D. `src/train/multicam_splits.py`

Purpose: centralize camera split semantics.

Sketch:

```text
resolve_multicam_split(record, data_cfg) -> MulticamSplit
validate_multicam_split(split) -> None
```

`MulticamSplit`:

```text
train_cameras
heldout_cameras
anchor_camera
condition_camera
source_camera
target_camera
split_origin  # "record", "config_override", "source_target_fallback"
```

Expected payoff:

- Fixes record-level split fields being ignored.
- Makes launcher-patched config less magical.
- Makes W&B tags and baseline rows easier to label.

Risk:

- Need careful precedence. Recommended precedence:
  1. explicit config override, if provided;
  2. record-level rich fields;
  3. source/target fallback.

### E. `src/train/camera_metadata.py`

Purpose: preserve lens/distortion metadata through rig and render-target
construction.

Sketch:

```text
CameraMetadata
    name
    lens_model
    distortion

CameraViewSet
    names
    K
    w2c
    metadata

camera_specs_from_view_set(...)
```

Expected payoff:

- DeepView fisheye metadata no longer disappears after loading.
- Goodset relpose rows can report whether they are fisheye-preserving or
  pinhole-approximation.
- Camera-rig tests can assert lens preservation.

Risk:

- Renderer support may still be pinhole-only in some paths. If so, the helper
  should fail loudly or mark a deliberate approximation; it should not silently
  drop distortion.

### F. `src/train/logging_schedule.py` Or Extend `train_logging.py`

Purpose: one tiny helper for modulo logging gates.

Sketch:

```text
should_log(step, every, *, total_steps, always_log_last_step) -> bool
```

Expected payoff:

- Removes repeated modulo expressions in video-token, relpose, PowerFoam, and
  gauge trainers.
- Low risk, easy test.

This is small cleanup, not a headline project.

## What To Delete Or Mark Stale

Do not delete aggressively from a dirty tree. First classify.

### Mark Superseded

`TODO/trainer_landscape_unification.md` should get a short header saying it is
partly stale under the current objective/pipeline helper layout. Keep it for
history, but stop treating it as the current plan.

Suggested header:

```text
Status as of 2026-05-18: partially superseded.
Current code already has RGBReconObjective, objective RenderedView,
pipeline/render.py, and pipeline/validation_media.py. The alpha/colorize
missing-from-multicam diagnosis in this note is stale. Use
agent_notes/loose_notes/2026-05-18_16-50-18_cleanup_unification_next_report.md
for the next cleanup plan.
```

### Search Before Deleting

The old unification note lists files that do not appear in current `src/train`
under this checkout:

- `dynamicTokenGS.py`
- `train_camera_implicit_dynamic.py`
- `train_ltx_feature_implicit_dynamic.py`
- typo shim `train_camera_implict_dynamic.py`
- `train_image_encoder_implicit_camera_baseline.py`

Before deleting anything, run:

```bash
rg --files src/train | rg 'dynamicTokenGS|train_camera|train_ltx|train_image'
rg -n 'dynamicTokenGS|train_camera_implicit|train_ltx_feature|train_image_encoder' .
```

If no code references remain, the cleanup is doc cleanup, not file deletion.

### Possible Future Deletes

Potential excess is more likely in stale TODO notes, old generated outputs, and
obsolete config aliases than in current trainer source files. The current
trainer source files are large, but not obviously dead:

- `train_video_token_implicit_dynamic.py`: too large, but active.
- `train_multicam_precomputed_feature_implicit_dynamic.py`: active.
- `train_multicam_relative_pose_implicit_dynamic.py`: active.
- PowerFoam/STAR/Gauge trainers: separate experiment surfaces; do not fold into
  the dataset-loader cleanup.

## What Not To Unify

Do not make a single loader class for static/dynamic/same-view/multicam. The
data contract is about supervision availability, not whether frames are static
or dynamic.

Do not merge `SequenceData` and `MulticamVideoBundle`. A multicam bundle has a
rig and split semantics. A sequence does not.

Do not add a new abstract `BaseTrainer`. The current issue is not insufficient
inheritance; it is that specialized trainers override enough parent methods
that fixes can drift.

Do not hide query-conditioned relpose under generic `heldout`. It is a distinct
mode and should stay labeled.

Do not move full config defaults into helper dataclasses. Project style wants
JSONC configs as the source of experiment knobs and Python normalization at
load time.

## Concrete Cleanup Phases

### Phase 0: Documentation And Claim Hygiene

Goal: stop future agents from following stale cleanup instructions.

Write set:

- `TODO/trainer_landscape_unification.md`
- optionally `TODO/README.md`

Actions:

1. Mark `TODO/trainer_landscape_unification.md` partially superseded.
2. Add a pointer to this report and the dataset audit.
3. Note that current objective/render/media helpers already exist.

Validation:

```bash
git diff --check TODO/trainer_landscape_unification.md TODO/README.md
```

Effort: 15-30 minutes.

### Phase 1: Loader Normalization Helpers

Goal: fix small loader-contract bugs while shrinking branchy code.

Write set:

- `src/train/data_records.py` or `src/train/manifest_records.py`
- `src/train/sequence_data.py`
- `src/train/multicam_video_data.py`
- tests for same-view and multicam manifest records

Actions:

1. Add `normalize_single_sequence_record(entry, data_cfg, model_cfg)`.
2. Add `normalize_multicam_record(record, data_cfg)`.
3. Move `explicit_video_window` required-field logic into the normalizer.
4. Make rich multicam record split fields visible to the loader.
5. Add path:line JSONL read errors if practical.

Validation:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_sequence_data_single_frame.py \
  tests/test_multicam_video_data.py -q
```

Add or extend tests for:

- `explicit_video_window` without `sequence_dir`;
- multicam train2-holdout1 row without launcher-patched config overrides.

Effort: half day.

### Phase 2: Batch Payloads

Goal: stop tuple-shape drift before mixed training.

Write set:

- `src/train/data_batches.py`
- `src/train/train_video_token_implicit_dynamic.py`
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
- maybe `src/train/train_multicam_relative_pose_implicit_dynamic.py`

Actions:

1. Add `SameViewBatch`.
2. Add `NovelViewBatch`.
3. Convert multicam `sample_multicam_clip()` first.
4. Convert camera-swap return values second.
5. Leave base same-view `sample_clip()` for last, because it is simpler and
   has more call sites.

Validation:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_temporal_sampling.py \
  tests/test_camera_swap_sampling.py \
  tests/test_pipeline_helpers.py -q
```

Then run one-step runtime smokes, because signature refactors can pass imports
and fail in the call graph.

Effort: half day to one day, depending on how far relpose is included.

### Phase 3: Camera Metadata Preservation

Goal: stop silently downgrading DeepView/fisheye camera semantics.

Write set:

- `src/train/camera_metadata.py`
- `src/train/camera_rig.py`
- `src/train/multicam_video_data.py`
- multicam/relpose tests

Actions:

1. Represent lens model/distortion next to `K/w2c`.
2. Pass metadata into `LearnableCameraRig`.
3. Preserve metadata in source-relative camera construction.
4. If a renderer path cannot honor distortion, mark the approximation in
   metadata and logs.

Validation:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_multicam_video_data.py \
  tests/test_source_relative_cameras.py \
  tests/test_multicam_relative_pose_trainer.py -q
```

Effort: one day if renderer support is already compatible; longer if pinhole
assumptions are embedded deeper.

### Phase 4: Mixed Scheduler, Not Global Loader

Goal: implement the bridge only after the contracts are clean.

Write set:

- `src/train/data_sources.py`
- `src/train/mixed_data_scheduler.py`
- a new trainer or narrow extension point for mixed same-view plus novel-view
- train config JSONC
- tests and smoke script

Actions:

1. Resolve same-view and novel-view source descriptors.
2. Alternate or weight sample kinds explicitly.
3. Return typed batches with `loss_kind`.
4. Log separate losses and mixture weights.
5. Keep eval-heldout separate from train-novel unless intentionally renamed.

Validation:

- F=32 same-view one-step.
- F=32 multicam one-step.
- mixed one-step with both `loss_kind`s observed.
- validation/media path with `video_log_every=1`.

Effort: one to two days after Phases 1-2; riskier if done before them.

## Expected Line Count Effect

This cleanup will not delete thousands of lines immediately. The value is
contract sharpness and lower future drift.

Likely reductions:

- `sequence_data.py`: smaller manifest dispatch, but new normalizer module will
  absorb some lines. Net maybe neutral to -30.
- `multicam_video_data.py`: smaller split/setup block. Net maybe -20 to -60.
- multicam trainers: batch payloads may reduce unpacking and repeated local
  variables, but dataclasses add lines. Net neutral short-term, simpler long-term.
- logging cadence: tiny reduction across several trainer files.
- stale TODO docs: clearer, not necessarily shorter.

If the goal is raw LOC reduction, the largest files are not the safest first
targets. `train_video_token_implicit_dynamic.py` is big because it still owns
real trainer policy. Extracting random chunks out of it would make navigation
worse. Extract only where a helper enforces a contract used by multiple callers.

## Backtracks From Older Cleanup Thinking

Older assumption:
    Multicam lacks alpha/colorize composition and validation media.

Current status:
    Weakened or stale. Current code has `RGBReconObjective` and
    `pipeline/validation_media.py`, and multicam validation media uses rendered
    alpha/features when available.

Older assumption:
    Delete legacy trainer files like `dynamicTokenGS.py` and image implicit
    shims.

Current status:
    Stale for this checkout. Those files are not visible under current
    `src/train`. Deletion work should start with `rg --files`, not the old list.

Older assumption:
    The main cleanup is trainer class hierarchy design.

Current status:
    Weakened. The main cleanup is data-boundary and batch-contract design.
    More inheritance is not the fix.

## Falsification Tests

Hypothesis:
    Loader normalization helpers will simplify code and catch real contract bugs.

Cheap test:
    Implement only `normalize_single_sequence_record()` and the pure
    `explicit_video_window` test. If `sequence_data.py` gets longer and the test
    is awkward, the helper is overdesigned.

Hypothesis:
    Typed batch payloads reduce risk.

Cheap test:
    Convert only `sample_multicam_clip()` to return `NovelViewBatch`. If the
    resulting `step()` body is easier to read and tests pass, continue. If every
    call site immediately unwraps all fields into locals, the abstraction is not
    paying rent.

Hypothesis:
    One source resolver is useful without becoming a get-all blob.

Cheap test:
    Add a read-only `resolve_training_data_sources(cfg)` used by a `check`
    command or startup print. It should not require torch, OpenCV, or video
    loading. If it starts importing render/model code, stop.

Hypothesis:
    Camera metadata preservation matters for promoted DeepView rows.

Cheap test:
    Add an assertion that a DeepView fisheye row reaches the render target or
    rig camera as `opencv_fisheye`, not `pinhole`. If a renderer then errors,
    that is useful evidence: the current metric is a pinhole approximation and
    should be labeled.

## Recommended Next Commit Slice

Smallest useful cleanup commit:

1. Add a superseded header to `TODO/trainer_landscape_unification.md`.
2. Add `normalize_single_sequence_record()` for single-sequence rows.
3. Fix `explicit_video_window` so it does not require `sequence_dir`.
4. Add a unit test for a pure explicit-video-window row.
5. Add `resolve_multicam_split()` or equivalent helper.
6. Make `load_multicam_video_bundle()` honor record-level split fields.
7. Add a unit test for a train2-holdout1 row without config split overrides.

That commit is small enough to verify, and it directly pays down the dataset
audit's two most concrete loader-contract bugs.

Do not start with the mixed scheduler. The mixed scheduler is the right product
direction, but it should sit on clean source descriptors and typed batches.
