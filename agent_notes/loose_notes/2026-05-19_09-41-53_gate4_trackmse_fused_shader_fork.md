# Gate4 Track-MSE Fused Shader Fork

## Context

The 64px Gate4 affine candidate CSR capacity probe showed flat topology/storage
across `2/4/8/16` real frames, but replay iterations still scaled with frame
count. I forked the fused-MSE shader to test whether the remaining practical
bottleneck was atomic pressure rather than candidate replay.

## Code changed

- Added a new Metal kernel:
  `wf2_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only_tensor`.
- Added host launcher, Torch schema/impl, Python wrapper, and package export:
  `fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only`.
- Added train/eval tape mode:
  `gate4-affine-candidate-num32-den16-trackmse-fused-mse`.
- Preserved the existing verifier semantics:
  `gate4_affine_candidate_csr_fused_mse=true` still marks the family, with a
  new diagnostic flag `gate4_affine_candidate_csr_trackmse_fused_mse=true`.
- Added focused tests so the new mode is classified as Gate4 candidate CSR and
  the native extension verifier requires the new op.
- Extended the high-cap MPS regression to compare old sample-MSE loss/grad
  against the new track-MSE loss/grad on a row with more than 128 candidates.

The new kernel keeps the existing RGB-MSE math but dispatches one thread per
track, loops frames locally, accumulates a local site gradient array, then emits
one loss atomic per track and one gradient atomic per touched site per track.

## Verification

Build:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  python setup.py build_ext --inplace )
```

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py \
  research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
```

Passed:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_train_eval_fused_slab_mixed_mps.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py -q
```

`10 passed in 3.73s`.

Passed broader focused suite:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_probe_gate4_affine_candidate_csr_capacity.py \
  research_experiments/world_foam_lane2/test_run_gate4_affine_candidate_csr_promotion_gate.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py \
  research_experiments/world_foam_lane2/test_train_eval_fused_slab_mixed_mps.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py -q
```

`22 passed in 2.46s`.

## Runtime smoke

Single-row smoke:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode gate4-affine-candidate-num32-den16-trackmse-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 24 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_trackmse_fused_mse_smoke_2f_render16_site24.json
```

Result: `status=ok`, nonzero gradients, parameters updated, finite outputs.
This run was contended and included Metal first-dispatch cost.

Warm 2/4/8/16 ladder:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode gate4-affine-candidate-num32-den16-trackmse-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 24 \
  --optimizer-mode manual-vjp \
  --steps 3 \
  --warmup-steps 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_trackmse_fused_mse_scale_2_4_8_16_render16_site24_warm3.json
```

Artifact:

- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_trackmse_fused_mse_scale_2_4_8_16_render16_site24_warm3.json`

Rows:

- `2f`: total `7.64ms`, backward `6.96ms`, train PSNR `14.20`, heldout PSNR `15.13`
- `4f`: total `8.55ms`, backward `7.96ms`, train PSNR `14.27`, heldout PSNR `15.14`
- `8f`: total `15.17ms`, backward `14.32ms`, train PSNR `14.41`, heldout PSNR `15.20`
- `16f`: total `12.00ms`, backward `11.51ms`, train PSNR `14.54`, heldout PSNR `15.32`

The artifact reports `status=ok`, frame scale `8.0x`, total scale `1.57x`,
and backward scale `1.64x`, but `benchmark_environment.status=contended`.

## Interpretation

This fork is functionally correct but not the keeper. The warmed track-MSE path
is slower than the existing sample-thread fused-MSE path at the same 16px/24-site
scale; an earlier existing sample-MSE artifact showed roughly `3-5ms` backward
rows, while this track-MSE fork is `7-14ms`.

The negative result is useful: atomics were probably not the main practical
bottleneck. Cutting thread parallelism and carrying a per-track local
`grad_accum[64]` plus high-cap segment arrays hurts enough to erase the atomic
savings. Next fork should target candidate replay itself: owner-run/site-pair or
boundary-pair records that avoid rescanning and re-ownering the candidate list
per frame, not another atomic-reduction-only fork.
