# Gate4 coeff16 fused-MSE reflection

## Context

We forked the Gate4 affine-candidate fused-MSE path to test whether the
STAR-like flat-cardinality idea could be made more competitive by packing the
candidate depth coefficients into fp16. The previous Gate4 affine candidate
path already had near-flat selected segment count across 2/4/8/16 frames, but
its num32/den16 tape used about 1.04 MB for this small high-cap probe and the
fused shader still did not show a clean speed win.

## Implementation

- Added coeff16 fused-MSE direct-atomic RGB-only and track-MSE Metal kernels in
  the world-foam lane2 fused slab variant.
- Registered C++/Objective-C++ host launchers and pybind ops.
- Added Python wrappers and train/eval modes:
  `gate4-affine-candidate-coeff16-fused-mse` and
  `gate4-affine-candidate-coeff16-trackmse-fused-mse`.
- Routed render replay through coeff16 affine candidate depth replay when the
  selected tape carries `affine_candidate_depth_coeff_f16`.
- Widened coeff16 real-ray replay to the fused-MSE boundary cap so the high-cap
  fixture is not limited by the older smaller real-ray buffer.

## Verification

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_train_eval_fused_slab_mixed_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
```

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_run_tape -v
```

Functional 2-frame smoke passed and produced nonzero gradients:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode gate4-affine-candidate-coeff16-trackmse-fused-mse \
  --frame-counts 2 --render-size 16 --site-count 8 \
  --optimizer-mode manual-vjp --steps 1 --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_trackmse_smoke_2f_render16_site8.json
```

## Scale result

Compared on the 24-site, render16, 2/4/8/16-frame ladder.

| mode | frame | total ms | backward ms | storage bytes | segments | train PSNR | heldout PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coeff16 | 2 | 7.817 | 7.024 | 708604 | 84930 | 14.204 | 15.126 |
| coeff16 | 4 | 4.466 | 4.106 | 706044 | 84609 | 14.267 | 15.138 |
| coeff16 | 8 | 13.618 | 13.239 | 702756 | 84196 | 14.414 | 15.200 |
| coeff16 | 16 | 15.504 | 14.900 | 703020 | 84225 | 14.540 | 15.324 |
| num32/den16 | 2 | 7.627 | 6.953 | 1048324 | 84930 | 14.204 | 15.126 |
| num32/den16 | 4 | 8.457 | 7.880 | 1044480 | 84609 | 14.267 | 15.138 |
| num32/den16 | 8 | 14.877 | 14.096 | 1039540 | 84196 | 14.414 | 15.200 |
| num32/den16 | 16 | 11.946 | 11.423 | 1039920 | 84225 | 14.540 | 15.323 |

Coeff16 scale summary:

- Total step 2f->16f scale: 1.98x.
- Backward 2f->16f scale: 2.12x.
- Selected storage 2f->16f scale: 0.99x.
- Selected segment 2f->16f scale: 0.99x.

## Reflection

This is a keeper for tape representation: the selected tape is now effectively
flat over frame count and about one-third smaller than the previous Gate4
affine candidate fused-MSE path, with no PSNR regression in this tiny ladder.

It is not yet the STAR UVT breakthrough. The cardinality and storage behavior
are now STAR-like on this probe, but practical time is still dominated by
per-sample replay/search/update work inside the WorldFoam fused shader. The
2f->16f timing is sublinear relative to frame count, but not flat, and the
per-row timing is noisy enough that coeff16 cannot be claimed as a speed win
over num32/den16 without a cleaner repeat gate.

The important lesson is that compressing the affine candidate representation
solves the tape-size symptom, not the execution model. STAR UVT remains cleaner
because time tubes/bins are the primary raster units; WorldFoam is still
evaluating candidate depth events per rendered sample. To become competitive,
the next fork needs to reduce candidate replay work or pre-bucket ownership in a
way the shader can consume cheaply, not just pack the same candidate stream
tighter.
