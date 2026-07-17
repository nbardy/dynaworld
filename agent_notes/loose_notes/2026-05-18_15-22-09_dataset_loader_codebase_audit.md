# Dataset Loader Codebase Audit

Date: 2026-05-18 15:22 Asia/Ho_Chi_Minh

Scope: dataset docs, dataset manifests, same-view loaders, multicam loaders,
feature-cache data paths, trainer integration, configs, tests, and baseline
bookkeeping in the current on-disk `dynaworld` tree.

Method: five read-only subagent reviews plus a local synthesis pass. No code was
changed before this note. No trainer smoke or pytest suite was run for this
audit. The worktree was already dirty; this note treats the current on-disk
state as the source of truth and does not attempt to normalize unrelated edits.

## Executive Summary

The current repo has two real, reusable dataset families:

- Same-view scale data through `src/train/sequence_data.py`.
- Calibrated multicam data through `src/train/multicam_video_data.py`.

The docs are mostly aligned on the important thesis: same-view reconstruction is
not evidence of world-token novel-view behavior, and multicam heldout cameras
are the actual cheap novel-view probe. The mixed same-view plus novel-view
scheduler/trainer is still not implemented. Claims should keep those modes
separate until a trainer actually owns both loaders and logs named
`same_view_recon` and `heldout_view_recon` or a deliberately renamed equivalent.

The biggest actionable gaps are not "we need more docs." They are contract
holes where code, manifests, and docs almost agree but not quite:

1. `load_manifest_sequence()` still requires `sequence_dir` before it dispatches
   `explicit_video_window`, even though the data contract only requires
   `video_path`, `start_seconds`, and `duration_seconds` for those rows.
2. The checked-in 1k same-view artifact is schema-stale (`v0`) relative to the
   current builder schema (`v1`), while the config still asks for 64 eval rows
   even though the generated eval manifest is empty.
3. The multicam loader ignores rich record-level train/heldout camera split
   fields unless a shell launcher patches them into config.
4. DeepView fisheye lens metadata is preserved in `MulticamVideoBundle` but
   dropped by the V-JEPA multicam trainer and `LearnableCameraRig`, so current
   DeepView relpose/V-JEPA rows should be described as pinhole approximations
   unless the camera path is fixed.
5. Multicam `frame_indices` select the right pixels but rebuild timestamps as
   dense `0..T-1`, so sparse temporal samples lose original timing.
6. `BASELINES.md` and some TODO/research docs still carry stale sample counts or
   old trainer assumptions. The canonical data contract is newer and should win.

## Current Contract

`research_notes/data_contract.md` is the most accurate current contract. Its
core distinction is:

- Same angle: one video window or prepared clip goes in, same camera/time window
  is reconstructed. This is the scale pretraining path.
- Novel angle: calibrated train cameras go in, heldout cameras score the
  rendered result. This is the actual world-token check.
- Mixed training: intended next bridge, not complete yet.

The stronger `research_notes/training_contract_v1.md` is stricter than the
current scale lane. Its predictive contract samples `(O, H, Q) ~ D_var` with
the hard rule that encoder input observations and loss target observations must
not overlap. That means today's same-view reconstruction is a useful dev and
pretraining signal, not the final predictive-quotient proof.

The most important wording to keep in future docs and W&B names:

- `same_view`: encoded source-view reconstruction.
- `calibrated_no_query_heldout`: source train camera(s) condition the model,
  heldout camera is rendered/scored without consuming heldout pixels as query
  features.
- `query_conditioned_relpose`: heldout/query camera features are available to
  the relative-pose head during eval. This is useful but is not the same
  benchmark as source-only heldout hallucination.

## Inventory

Canonical docs and state:

- `README.md`: public progress checklist; says mixed sampler/trainer is still
  open.
- `TODO/README.md`: active backlog; puts the mixed same-view plus novel-view
  bridge near the center of gravity.
- `BASELINES.md`: standings table, but currently stale on multicam-val sample
  counts and still has TODO rows for key Tier 1/Tier 2 reruns.
- `research_notes/data_contract.md`: canonical loader and manifest contract.
- `research_notes/training_contract_v1.md`: operational predictive-quotient
  sampler/loss contract.
- `agent_notes/key_learnings.md`: compressed lessons; current line 199 says
  the multicam V-JEPA/static-dynamic-token lane is real but not yet a 1k-item
  benchmark contract.

Same-view code:

- `src/train/sequence_data.py`: frame/image/video/camera-json loaders and
  single-sequence manifest dispatch.
- `src/train/runtime_types.py`: `SequenceData` and `ClipBatch` contracts.
- `src/dataset_scripts/build_single_video_pretrain_manifest.py`: no-copy
  single-video pretrain manifest builder.
- `src/train/train_video_token_implicit_dynamic.py`: base single-sequence
  trainer, lazy manifest sampling, source-view reconstruction, validation.
- `src/train/train_precomputed_feature_implicit_dynamic.py`: feature-cache
  conditioning wrapper for the same-view trainer.
- `src/train/video_feature_cache.py`: sample and feature fingerprinting,
  V-JEPA extraction, lazy/on-demand cache.

Multicam code:

- `src/train/multicam_val_data.py`: train-time JSONL reader and frame sampler
  for multicam pair records.
- `src/train/multicam_video_data.py`: calibrated bundle loader and camera rig
  adapter for DeepView, AIST, Neural3D, ViVo, and synthetic orthogonal rigs.
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`: multicam
  V-JEPA/static-dynamic-token trainer family.
- `src/train/train_multicam_relative_pose_implicit_dynamic.py`: relpose/query
  conditioned multicam trainer family.
- `src/train/camera_swap_sampling.py`: train/heldout camera-swap pair sampling.
- `src/train/camera_rig.py`: learnable camera rig wrapper used by multicam
  trainers.

Current important artifacts:

- `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl`
  has 1000 rows.
- `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/eval_manifest.jsonl`
  has 0 rows.
- The 1k train split is 955 `single_view_video_window`, 44
  `frame_clip_sequence`, and 1 `synthetic_camera_json_sequence`.
- `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl`
  has 8 pair records.
- `data/multicam_val/clip_sets/multicam_val_v1_chunked_128_4fps_16f/manifest.jsonl`
  has 14 pair records.
- `src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f_manifest.jsonl`
  has 5 smoke records with explicit train/heldout camera lists.

## Same-View Loader Audit

What is solid:

- `SequenceData` is a clear common payload: `frames [T,3,H,W]`,
  normalized `frame_times [T,1]`, optional per-frame cameras, source metadata,
  crop mode, frame counts, and paths.
- The video-window cache key includes resolved video path/stat, target size,
  start/duration, fps, frame count, and crop mode. That is the right class of
  identity for no-copy source-video windows.
- `video_feature_cache.py` includes source/frame fingerprints, frame source,
  crop mode, records, feature extractor settings, V-JEPA crop size, cache
  version, and sample cache key. That is strong enough to avoid most silent
  feature reuse across crop/source changes.
- Lazy training mode loads one bootstrap sequence and then samples manifest
  entries per step. This makes the 1k no-copy path practical locally.

High-priority findings:

1. The manifest contract is stricter in code than in docs.

   `load_manifest_sequence()` reads `entry["sequence_dir"]` before checking
   `frame_source`. A pure `explicit_video_window` row that follows the data
   contract would fail even though it has `video_path`, `start_seconds`, and
   `duration_seconds`. Current builder rows happen to include `sequence_dir`,
   so the bug is masked by current artifacts.

   Fix: either update `data_contract.md` to require `sequence_dir` for all
   single-sequence rows, or move the `sequence_dir` read into the branches that
   actually need it. The cleaner fix is branch-local reads.

2. The 1k artifact is schema-stale.

   The builder currently emits `dynaworld_single_video_pretrain_manifest_v1`,
   but `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/`
   records and `dataset.json` still report `v0`. Counts are fine; freshness is
   not. Any baseline that cites the 1k artifact should say it is the current
   1k v0 artifact, not a regenerated v1 contract.

   Fix: regenerate with the current builder, or document that v0 is intentionally
   frozen and add a v1 artifact path.

3. `summary_sampled` can silently degrade to `all_frames`.

   `_resolve_frame_paths()` tries summary-sampled paths, but if fewer than two
   exist it falls back to every PNG in the frame directory and returns
   `all_frames`. That is convenient for old local data but weak for a manifest
   contract. A row that says `summary_sampled` should probably fail if the
   summary sample is missing.

   Fix: make strict manifest rows error on unusable summary sampling. If
   backward compatibility is needed, gate the fallback behind an explicit
   config flag and log the downgrade.

4. Camera-json crop handling can invalidate intrinsics.

   `load_camera_sequence()` loads images with `image_crop_mode`, including
   `center_square`, but intrinsics are only scaled by target size over original
   camera image size. The code does not subtract crop offsets or account for
   non-square source dimensions. This can desync rendered camera geometry from
   loaded pixels.

   Fix: reject non-`resize` crop modes for camera-json rows until intrinsics are
   adjusted, or implement crop-aware `fx/fy/cx/cy` adjustment before scaling.

5. Video windows clamp past EOF instead of failing early.

   `load_video_window_sequence()` rounds target indices and clamps them to
   `total_frames - 1`. A bad row can duplicate the last frame while preserving
   requested timestamps. That is bad evidence for temporal training.

   Fix: validate requested end time and target frame indices against video
   duration/frame count before clamping. If a final-frame clamp is intentionally
   allowed, record it in metadata and exclude those rows from timing-sensitive
   benchmarks.

6. The eval config and artifact disagree.

   `src/dataset_configs/single_video_pretrain_1k_manifest.jsonc` still says
   `target_eval_items: 64`, but the generated eval manifest is empty and the
   scale train config explicitly disables eval. The data contract correctly
   records the actual empty eval state.

   Fix: set target eval to 0 until real non-leaking eval sources are emitted, or
   add the actual eval sources and rebuild. Do not let `target_eval_items: 64`
   imply validation exists.

7. Same-view video validation can fall back to train-source reconstruction.

   When `eval_max_sequences=0`, `load_eval_sequences()` returns `[]`, but
   `validation_video_payload()` uses `self.eval_sequences or [self.sequence_data]`.
   If video logging fires, the videos are train-source reconstruction, not eval.

   Fix: make the payload label this explicitly (`TrainPreview/*`) when there is
   no eval sequence, or skip eval-named video payloads when eval is disabled.

## Multicam Loader Audit

What is solid:

- `MulticamVideoBundle` exposes train and heldout tensors separately:
  `train_frames [V,T,3,H,W]`, `train_K`, `train_w2c`, camera names, optional
  lens metadata/distortion, and analogous heldout fields.
- Split validation rejects empty train/heldout sets, duplicate cameras, train
  and heldout overlap, and anchor/condition cameras outside the train set.
- The loader has explicit adapter code for DeepView, AIST, Neural3D, ViVo, and
  orthogonal synthetic rigs instead of pretending all datasets share one camera
  schema.
- Unit tests already cover basic train/heldout shapes and DeepView fisheye
  metadata preservation.

High-priority findings:

1. Rich record-level camera splits are ignored by default.

   The train2-holdout1 manifest records include `train_cameras`,
   `heldout_cameras`, `anchor_camera`, and `condition_camera`, and the data
   contract documents those fields. `load_multicam_video_bundle()` only uses
   config keys such as `multicam_train_cameras`; otherwise it falls back to
   `source_camera` and `target_camera`. The scale launcher compensates by
   patching each record into a temp config.

   Fix: make the loader prefer record-level split fields, then fall back to
   config overrides or source/target. Add a regression test that loads a
   train2-holdout1 row without launcher-side config patching.

2. Lens metadata is preserved, then dropped.

   DeepView fisheye metadata reaches `MulticamVideoBundle` and is tested there.
   The V-JEPA multicam trainer then calls `cameras_from_K_w2c()` without lens
   metadata, and `LearnableCameraRig._make_camera()` always returns
   `lens_model="pinhole"`. Source-relative camera swaps also omit target lens
   metadata.

   Fix: carry lens model and distortion through `LearnableCameraRig`,
   source-relative camera construction, and renderer targets. Until then, mark
   affected DeepView V-JEPA/relpose rows as pinhole approximations.

3. Sparse frame indices lose original timing.

   `select_configured_multiview_frames()` selects pixels by configured
   `frame_indices`, but `load_multicam_video_bundle()` rebuilds `frame_times` as
   dense `arange(T) / fps` after selection. A selection like `[0, 2]` becomes
   two adjacent normalized times instead of preserving the skipped frame gap.

   Fix: compute times from selected original frame indices divided by fps, then
   normalize. Add a sparse-index timing assertion to `tests/test_multicam_video_data.py`.

4. AIST scale defaults are unsafe.

   Loader comments say AIST translations are millimeters and `0.001` should be
   used for meters, but runtime default is `1.0`, and the current multicam scale
   config does not set `aist_translation_scale`.

   Fix: require explicit AIST scale in configs, or change the default with a
   migration note. Add a magnitude guard on relative baseline distances.

5. Resize policy diverges between materialized frames and train-time sampling.

   The dataset pipeline can pad to square while preserving aspect. Current
   train-time multicam sampling resizes directly to square through
   `multicam_val_data.py`. The current multicam-val config has
   `materialize_metric_frames: false`, so train-time direct video sampling is
   the path that matters today.

   Fix: centralize resize/crop policy and make bundle loading honor materialized
   frames when present. The contract should name whether square resize,
   center-square crop, or pad-to-square is the canonical camera/intrinsics mode.

6. Train-time JSONL diagnostics are weaker than pipeline diagnostics.

   `load_multicam_val_manifest()` calls `json.loads()` directly. The dataset
   pipeline has tested path:line error wrapping. This is not a modeling issue,
   but it will waste time when a generated manifest line is malformed.

   Fix: share the robust JSONL reader or mirror the path:line error contract in
   train-time manifest loading.

7. Known limitations should be labeled as limitations.

   ViVo extra cameras require timestamp offsets and currently error when those
   offsets are unavailable. ViVo `rotation_correction.json` is not supported.
   Those are acceptable constraints, but docs/config names should not imply ViVo
   arbitrary-camera training works yet.

## Trainer Integration Audit

Current reality:

- `src/train/train.py` routes same-view and multicam work to separate trainer
  families.
- `Trainer.load_train_sequences()` uses `sequence_data.py`.
- `MulticamPrecomputedFeatureImplicitTrainer.load_train_sequences()` replaces
  the base loader with one selected `MulticamVideoBundle`.
- There is no mixed scheduler that owns both the 1k same-view manifest and a
  multicam manifest.
- There are no literal scalar keys named `same_view_recon` or
  `heldout_view_recon`.

Loss and validation semantics:

- Same-view train loss renders decoded model cameras back against the same clip
  frames. It is source-view reconstruction.
- Multicam default train loss renders train views. Heldout-camera rendering is
  validation-only in the default path.
- Camera-swap train pairs are sampled from train cameras. Heldout pairs exist
  for eval/render paths.
- If a future trainer backprops into cameras named `heldout_*`, the split name
  must change or the config must make that leakage explicit. A train loss on
  heldout pixels is not a heldout metric anymore.

Integration risks:

1. Tuple return shape drift is the immediate refactor hazard.

   Base `sample_clip()` returns three values. Known-camera paths return four.
   Multicam sampling returns five. Camera-swap loss returns a much larger tuple.
   This is exactly the kind of drift `agent_notes/key_learnings.md` warned
   about: named runtime payloads are the right boundary.

   Fix: introduce typed `SameViewBatch`, `NovelViewBatch`, and result payloads
   before building the mixed trainer.

2. Precomputed-feature conditioning is full-sequence, not clip-local.

   `model_input_for_clip()` in the precomputed-feature trainer ignores
   `clip_frames` and `clip_times` and returns `feature_cache.load_or_bake` for
   the whole sequence. That may be fine, but the mixed contract must name
   whether each branch conditions on full windows or sampled clips.

3. Query-conditioned relpose must be reported separately.

   The relative-pose trainer can consume heldout/query features during eval.
   That is a real trainer mode and may be useful, but it is not source-only
   novel-view hallucination. `BASELINES.md`, config names, and W&B tags should
   split query-conditioned heldout from calibrated no-query heldout.

4. Best-heldout artifact preservation is still incomplete.

   Heldout metrics/videos and best-heldout scalar tracking exist, but the
   follow-up TODO still says best checkpoint artifact preservation is missing.
   For any promoted multicam run, cheap scalar heldout eval should be decoupled
   from video cadence, and the best-heldout checkpoint should be saved with the
   W&B row.

5. Some old trainer TODOs are now stale.

   `TODO/trainer_landscape_unification.md` says the multicam trainer bypasses
   alpha/colorize composition. Current code routes multicam rendering through
   `rgb_objective.render_view()` and explicitly errors if F-channel training
   gets `alpha=None`. That TODO should be refreshed before anyone follows its
   old "multicam alpha missing" diagnosis.

## Docs And Baseline Drift

The canonical docs are mostly in the right order now:

1. `README.md` progress.
2. `TODO/README.md` active backlog.
3. `BASELINES.md` reruns needed.
4. `research_notes/data_contract.md`.
5. `agent_notes/key_learnings.md`.

Known drift to fix:

- `BASELINES.md` says `multicam_val_v1_128_4fps_16f` has 4 multicam samples
  and targets 20. The current manifest/config describe 4 source datasets with
  2 target rows each, i.e. 8 pair records. Rewrite this as "4 source datasets /
  8 pair records" or update the table to the current artifact count.
- `BASELINES.md` still has TODO rows for important Tier 1/Tier 2/Tier 2b runs.
  Do not claim "beats baseline" until those rows are appended with real W&B ids
  or explicitly marked as missing.
- `research_notes/three_architectures_for_novel_view_synthesis.md` still says
  "No multi-view paired data" in a proposal context. That is historical now; the
  note should point forward to `data_contract.md` and label old assumptions.
- Architecture docs discuss diffusion/GAN losses as training ideas. The current
  `training_contract_v1.md` treats diffusion priors and similar teachers as
  diagnostics or escape hatches unless a future contract promotes them. Keep
  that distinction visible.
- Goodset relpose configs have sample-id/name drift where the sample id names
  one camera pair but the train/heldout camera lists name a different promoted
  split. Config names should match the actual camera split.

## Test Coverage Audit

Strengths:

- Same-view loader tests cover explicit video-window sampling, frame-cache
  reuse, center-square crop, and feature-cache key separation.
- Multicam tests cover train/heldout/condition shapes, camera split validation,
  and DeepView fisheye metadata preservation at the loader boundary.
- Camera-swap sampling and source-relative camera invariants have separate unit
  tests.
- Temporal sampling and lazy prefetch have focused unit tests.
- The 300-clip helper has the strongest practical gate: audit, load-check,
  resolve, probe, and prebake/status.

Gaps worth filling:

- Manifest dispatch tests for all same-view row types, especially a pure
  `explicit_video_window` row with no `sequence_dir`.
- A strict `summary_sampled` failure test when sampled paths are absent.
- Camera-json crop/intrinsics regression coverage.
- Out-of-range explicit-video-window failure coverage.
- Builder schema/count validation for regenerated 1k artifacts.
- Multicam record-level train/heldout split loading without launcher patching.
- Multicam sparse `frame_indices` timestamp coverage.
- AIST scale magnitude guard coverage.
- Train-time JSONL path:line error coverage.
- One-step runtime smokes for same-view F=3, same-view F=32, multicam F=32, and
  relpose goodset configs. Unit tests alone are not enough for tuple/dataclass
  return shape drift.

Brittle tests:

- Some relpose config tests assert exact ignored `outputs/.../checkpoint_final.pt`
  strings. Those prove resolver strings, not artifact existence or runtime
  behavior.
- Sequence tests fake OpenCV reads. That is good for unit speed but does not
  cover real codec/path/manifest failure modes.
- Lazy manifest prefetch tests use `object.__new__` and monkeypatching, which
  bypasses full config normalization and trainer construction.

## Recommended Fix Order

P0: Fix the claims surface before another baseline claim.

- Patch `BASELINES.md` multicam sample wording to "4 source datasets / 8 pair
  records" and leave target expansion separate.
- Mark query-conditioned relpose and calibrated no-query heldout as separate
  modes in baseline rows/tags.
- Do not claim mixed same-view plus novel-view training exists until a trainer
  does both and logs the mixture explicitly.

P1: Fix loader contracts that can silently load the wrong thing.

- Make `explicit_video_window` manifest dispatch not require `sequence_dir`.
- Make strict `summary_sampled` rows fail instead of falling back silently.
- Reject or correct camera-json crop modes that desync intrinsics.
- Validate explicit video-window end bounds before frame-index clamping.
- Prefer multicam record-level train/heldout split fields in the loader.
- Preserve sparse multicam frame timestamps after `frame_indices`.

P2: Fix calibration and geometry semantics.

- Preserve DeepView lens model/distortion through `LearnableCameraRig`,
  source-relative cameras, and render targets, or label current rows as pinhole
  approximations.
- Require explicit AIST translation scale and add a baseline-distance guard.
- Centralize the multicam resize/crop/pad policy and make the camera/intrinsics
  assumption explicit.

P3: Build the mixed bridge deliberately.

- Add typed `SameViewBatch` and `NovelViewBatch` payloads first.
- Build a scheduler with `same_view_manifest`, `multicam_manifest`,
  `same_view_weight`, and `novel_view_weight`.
- Log source id, camera ids, `loss_kind`, mixture weights, `same_view_recon`,
  and a clearly named novel-view train loss.
- Keep true heldout metrics eval-only unless the config intentionally consumes
  heldout pixels for training and renames the split.

P4: Add runtime gates.

- Keep the existing focused pytest gate for loader/camera/sampling helpers.
- Add a real `smoke` mode to `src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh`.
- Add a one-step relpose runtime smoke for the promoted goodset config family.
- Add a heldout/eval smoke separate from the 1k same-view smoke, because the
  current 1k smoke patches eval off.

Suggested post-implementation gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_sequence_data_single_frame.py \
  tests/test_multicam_video_data.py \
  tests/test_source_relative_cameras.py \
  tests/test_camera_swap_sampling.py \
  tests/test_temporal_sampling.py \
  tests/test_pipeline_helpers.py -q
```

Then run runtime smokes from the dynaworld root, not just `py_compile`:

- Same-view F=3 one-step smoke.
- Same-view F=32 one-step smoke.
- Multicam F=32 one-step smoke with validation/media.
- Relpose goodset one-step smoke if that family is being cited.

## Decision Implications

The repo is not blocked on inventing a new manifest format. It is blocked on
making the existing two loader families impossible to confuse, then building a
typed mixed scheduler/trainer on top.

Same-view scale data is ready enough for continued broad pretraining, with the
caveat that eval is currently disabled/empty and source-view videos must not be
reported as heldout validation.

Multicam data is ready enough for small-N novel-view smokes and diagnostics, but
not ready for broad benchmark claims without fixing sample-count docs, split
semantics, smoke gates, and W&B-backed `BASELINES.md` rows.

The shortest safe next implementation slice is:

1. Patch `load_manifest_sequence()` so `explicit_video_window` rows only require
   the fields the contract says they require.
2. Patch `load_multicam_video_bundle()` to honor record-level train/heldout
   camera split fields.
3. Add tests for those two contracts.
4. Patch `BASELINES.md` sample-count wording.
5. Add a multicam one-step smoke mode.

Only after that should the mixed scheduler be built, because otherwise the mixed
trainer will inherit ambiguous split/camera semantics and make future baseline
rows harder to trust.
