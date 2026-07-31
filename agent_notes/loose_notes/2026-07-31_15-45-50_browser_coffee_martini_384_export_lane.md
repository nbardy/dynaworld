# Coffee Martini Real-Resolution Browser Export Lane

Date: 2026-07-31

## Context

The browser SPA remains a calibrated demo over the canonical multicamera data
contract. This work prepares a real 384x288 target bundle without adding a
WebGPU SfM implementation, a second camera convention, or another train/eval
split.

Canonical inputs:

- manifest:
  `src/dataset_configs/neural3d_coffee_martini_train17_holdout1_full_300f_manifest.jsonl`
- sample:
  `neural3d_coffee_martini_train17_holdout_cam06_full_300f`
- split: `train17_holdout1`
- train cameras: all 17 manifest train cameras
- heldout camera: `cam06`, validation-only
- anchor camera: `cam04`
- synchronized frame indices:
  `[0,20,40,60,80,100,120,140,159,179,199,219,239,259,279,299]`
- verified seed:
  `research_experiments/dynamic_foam/artifacts/browser_coffee_martini_train17_known_pose_frame0_1024px.ply`
- seed construction: existing offline known-pose pycolmap adapter, 17 train
  cameras only, 815 bounded points, 768 selected for this export

The seed PLY and report remain ignored local artifacts. The checked-in recipe
pins the PLY SHA-256 and validates the report's method, input cameras,
train-only claim, and `model`/anchor-relative coordinate frame.

## Defect Found

`export_browser_multicam_dataset_bundle(...)` previously called
`load_multicam_video_bundle(...)` with sparse `frame_indices`. The canonical
loader correctly preserved the requested semantic indices, but its eager
implementation first decoded every frame through `max(frame_indices)`, stacked
all cameras, and only then selected the requested 16 frames.

At 96x72 this was survivable. At 384x288, the retained RGB tensor alone is:

```text
18 cameras * 300 frames * 288 * 384 pixels * 3 channels * 4 bytes
  = 7,166,361,600 bytes
  = 6.674 GiB
```

The list-to-stack copy creates a conservative transient floor near:

```text
2 * 6.674 GiB = 13.348 GiB
```

This is materially different from the browser worker's eventual 18-camera,
16-frame bank. The earlier 486 MiB estimate used RGBA32F for only selected
frames; the Python loader stores RGB32F, so the selected Python tensor is:

```text
18 * 16 * 288 * 384 * 3 * 4
  = 382,205,952 bytes
  = 364.5 MiB
```

## Fix

`src/train/export_dynaworld_browser_bundle.py` now has an opt-in
`sparse_frame_decode` path.

For each requested frame index it:

1. selects the original canonical manifest row;
2. derives a temporary one-frame row by adding `frame_index / fps` to the
   canonical source and target start times;
3. invokes `load_multicam_video_bundle(...)` unchanged;
4. asserts train order, heldout order, intrinsics, poses, anchor pose, and pose
   source are invariant across timestamps;
5. concatenates only the requested one-frame tensors.

The temporary row changes only the decode interval. Camera roles, calibration,
LLFF-to-OpenCV conversion, anchor-relative transforms, and frame ownership
remain in `multicam_video_data.py`.

This path is deliberately verified only for synchronized
`neural_3d_video` records. Other datasets fail closed rather than inheriting an
untested timestamp interpretation.

The ordinary eager path remains available for older/small exports. The direct
exporter CLI exposes `--dataset-sparse-frame-decode`.

## Intrinsics

The canonical Neural 3D camera adapter scales native 2704x2028 intrinsics to
the requested `[height,width]`.

Observed for `cam04`:

```text
K(96x72):
  fx=51.861103, fy=51.861103, cx=48, cy=36

K(384x288):
  fx=207.444412, fy=207.444412, cx=192, cy=144
```

`K(384x288)[:2] == 4 * K(96x72)[:2]` exactly in the checked local diagnostic;
the pose difference is zero. The serialized normalized intrinsics are
therefore unchanged. Post-export verification recomputes every camera's
canonical 384x288 intrinsics and compares them to the JSON payload.

## Reproducible Lane

Checked-in recipe:

```text
src/train_configs/browser_coffee_martini_train17_holdout1_384x288_export.jsonc
```

Fail-closed runner:

```text
src/train_scripts/export_browser_coffee_martini_384x288.py
```

The recipe pins:

- canonical manifest SHA-256;
- exact sample/split/anchor/camera-count expectations;
- exact 16-frame schedule;
- target size `[288,384]`;
- verified seed PLY SHA-256 and provenance report;
- 768 seed count;
- generated output name, separate from the legacy 96x72 bundle;
- host memory, swap, load, and disk thresholds.

The runner stages the whole bundle in a sibling temporary directory, verifies
JSON roles/provenance/intrinsics and all 6144x288 atlases, writes an
`export_report.json`, and only then atomically promotes the directory.

Default output:

```text
outputs/browser_bundles/coffee_martini_train17_holdout1_384x288_verified_sparse/
```

This does not replace or mutate the checked-in 96x72 SPA default.

## Resource Model

The runner records both the superseded eager cost and selected-frame cost:

```text
selected RGB32F bank                 364.5 MiB
raw RGB8 atlas payload                91.1 MiB
estimated sparse-export peak       1,423.3 MiB
required available memory          4,096.0 MiB
estimated output upper bound         129.9 MiB
required free disk                 2,048.0 MiB
legacy eager RGB32F bank           6,834.4 MiB
legacy eager copy peak floor      13,668.8 MiB
```

The 4 GiB memory requirement intentionally exceeds the modeled 1.39 GiB peak.
The disk requirement intentionally exceeds the compressed-output estimate.

## Host Preflight And Generation Decision

The final pre-commit preflight at `2026-07-31T06:51:11Z` was blocked. The
workstation reported:

```text
physical memory                    24 GiB
free-memory fraction               42%
swap occupied                      91.71%
free disk                          17.33 GiB
five-minute load / logical CPU     0.562
```

The checked-in maximum swap fraction is 75%. Earlier samples also showed
five-minute load above the 0.75-per-logical-CPU ceiling, although load was back
below that ceiling in the final sample. Per the task contract, the full 384x288
export was not launched and no real-resolution bundle is claimed.

The latest ignored preflight report is:

```text
outputs/browser_bundles/preflight/coffee_martini_train17_holdout1_384x288_verified_sparse.json
```

## Pixel Equivalence

A low-memory real-data check decoded the exact 16 timestamps at 96x72 with the
new path and compared all 18 generated in-memory atlases against the existing
checked-in eager atlases.

Observed:

```text
cameras: 18
frames per camera: 16
train tensor shape: [17,16,3,72,96]
maximum uint8 pixel error: 0
```

This supports the intended claim: the change removes eager intermediate state
without changing sampled pixels.

## Tests

Focused tests cover:

- one canonical loader call per requested timestamp;
- exact timestamp offsets;
- strict frame-index validation;
- camera/intrinsics/pose drift rejection;
- canonical recipe fields and separate output naming;
- manifest drift rejection;
- memory, swap, load, disk, and output-collision gates;
- malformed host telemetry rejection;
- post-export camera roles, seed provenance, atlas dimensions, and canonical
  intrinsics.

Command:

```bash
PYTHONPATH=src/train:src/train_scripts uv run --with pytest python -m pytest \
  tests/test_browser_multicam_export_adapter.py \
  tests/test_browser_multicam_384_export_lane.py \
  tests/test_multicam_video_data.py -q
```

Current result before commit: `30 passed`.

## Exact Commands

Check whether the host is safe:

```bash
PYTHONPATH=src/train .venv/bin/python \
  src/train_scripts/export_browser_coffee_martini_384x288.py \
  --preflight-only
```

Generate only after that command returns zero:

```bash
PYTHONPATH=src/train .venv/bin/python \
  src/train_scripts/export_browser_coffee_martini_384x288.py
```

The generation command repeats preflight; passing a prior preflight does not
bypass current host checks.

## Current Belief

Confidence: high.

The old blocker was not a 486 MiB final browser bundle. It was a Python eager
decode intermediate more than an order of magnitude larger than the selected
frame bank. Sparse canonical replay removes that failure mode while preserving
the existing split and camera implementation.

Could be wrong if:

- random MP4 seeks differ at 384x288 despite exact equality at 96x72;
- PNG compression temporarily exceeds the modeled headroom;
- the local ignored seed PLY is regenerated with different geometry;
- the host's `memory_pressure` free percentage is not predictive of the
  Python allocation envelope.

Each case is fail-closed: post-export pixel/geometry verification, hash checks,
staging, and resource gates prevent silent promotion.

## Remaining Action

Wait for swap occupancy below 75% and five-minute load below 0.75 per logical
CPU, rerun preflight, then run the generation command. After generation,
inspect `export_report.json`; do not wire the SPA to this bundle until the
separate browser worker/storage lane can load it without cloning the full frame
bank.
