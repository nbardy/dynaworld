# PowerFoam COLMAP CLI ALIKED Probes

Date: 2026-05-06 05:03:54 Asia/Ho_Chi_Minh

## Goal

Continue the remaining full-PowerFoam acceptance work by testing whether an
ONNX-enabled ALIKED/LightGlue host can produce a stronger clean DeepView point
cloud than the selected OPENCV_FISHEYE SIFT artifact.

## Background

The ordinary pycolmap wheels expose ALIKED/LightGlue options but abort at
runtime because they lack ONNX support. The working host route found in this
session is the official `colmap/colmap:latest` image plus
`nvidia-cudnn-cu12==9.10.2.21`, which lets the COLMAP CLI run
`ALIKED_N16ROT`, `ALIKED_BRUTEFORCE`, and `ALIKED_LIGHTGLUE`.

## Code Changes

- `research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py`
  now exposes cheap probe knobs for camera set, matcher, target size, feature
  cap, and frame indices. It can test a richer probe before spending on the
  full 1024px / 32-image artifact.
- `research_experiments/dynamic_foam/run_powerfoam_external_blockers.py`
  now prints the ALIKED builder command with `--feature-backend colmap_cli`
  instead of the stale `pycolmap-cuda12` path.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  research_experiments/dynamic_foam/run_powerfoam_external_blockers.py
```

## Probes

Wide two-camera 128px brute-force probe:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with modal \
  modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  --execute --check-onnx --probe \
  --run-id colmap_cli_probe_20260506_0458
```

Result:

- ONNX check: ok
- features: `31` and `30`
- verified pairs: `1`
- point count: `0`
- artifact: `outputs/powerfoam_aliked_geometry/colmap_cli_probe_20260506_0458/probe.json`

Near four-camera 512px brute-force probe:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with modal \
  modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  --execute --check-onnx --probe \
  --run-id colmap_cli_probe_near4_512_20260506 \
  --probe-camera-set near4 \
  --probe-target-size 512 \
  --probe-max-features 4000 \
  --probe-frame-indices 0 \
  --probe-matcher-type aliked_bruteforce
```

Result:

- ONNX check: ok
- keypoints: `4513`
- verified pairs: `6`
- point count: `9`
- track mean/p90: `2.0 / 2.0`
- unique-camera p90: `2.0`
- unique-frame p90: `1.0`
- reproj median/p90: `5.9975 / 6.4856px`
- artifact: `outputs/powerfoam_aliked_geometry/colmap_cli_probe_near4_512_20260506/probe.json`

Near four-camera 512px LightGlue probe:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with modal \
  modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  --execute --check-onnx --probe \
  --run-id colmap_cli_probe_near4_512_lightglue_20260506 \
  --probe-camera-set near4 \
  --probe-target-size 512 \
  --probe-max-features 4000 \
  --probe-frame-indices 0 \
  --probe-matcher-type aliked_lightglue
```

Result:

- ONNX check: ok
- keypoints: `4513`
- verified pairs: `6`
- point count: `27`
- track mean/p90: `2.0 / 2.0`
- unique-camera p90: `2.0`
- unique-frame p90: `1.0`
- reproj median/p90: `6.2509 / 6.9330px`
- artifact: `outputs/powerfoam_aliked_geometry/colmap_cli_probe_near4_512_lightglue_20260506/probe.json`

Near four-camera 512px LightGlue probe with known-pose guided verification:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with modal \
  modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  --execute --check-onnx --probe \
  --run-id colmap_cli_probe_near4_512_lightglue_guided_fixed_20260506 \
  --probe-camera-set near4 \
  --probe-target-size 512 \
  --probe-max-features 4000 \
  --probe-frame-indices 0 \
  --probe-matcher-type aliked_lightglue \
  --known-pose-guided-verification true
```

Result:

- ONNX check: ok
- keypoints: `4513`
- verified pairs: `6`
- known-pose guided verification: applied
- point count: `0`
- artifact:
  `outputs/powerfoam_aliked_geometry/colmap_cli_probe_near4_512_lightglue_guided_fixed_20260506/probe.json`

## Interpretation

The ONNX/CUDA host blocker is solved for the COLMAP CLI route, but ALIKED did
not produce enough useful DeepView geometry in cheap probes. The full artifact
gate wants at least `2000` points, track mean/p90 `2.5/3.0`, reproj median
`<=4.0px`, and at least `28` verified pairs before the matched Metal training
row is worth running. The best cheap ALIKED probe here has only `27` points,
two-view-only tracks, and reproj median above `6px`; known-pose guided
verification made that probe stricter and pruned it to `0` points.

Decision: do not spend the full 1024px / 32-image ALIKED run yet. The next
useful geometry work should change the verification/track-building mechanism,
not just rerun the same COLMAP-CLI ALIKED branch at full size.
