# 2026-05-19 15:36:48 - coeff16 ownerkeep-i16 shader fork reject

## Context

We had a CPU owner-transition preflight showing a strong exact grouped-ownerkeep
opportunity: the ownergroup replay can reduce owner scans by roughly 22.6x to
23.0x versus the baseline transition scan. A smaller Metal fork was attempted
first: keep ownerkeep semantics, but store candidate boundary ids and boundary
site pairs as int16 to reduce side-stream bandwidth.

The new mode is:

```text
gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse
```

It is intentionally not the full ownergroup algorithm. It only changes the
side-stream representation and keeps the existing per-candidate boundary replay
shape.

## Implementation

Added:

- Metal kernel `wf2_fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only_tensor`
- C++ launcher `metal_fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only`
- torch binding `fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only`
- Python wrapper/export in `torch_world_foam_lane2_fused_slab`
- train/eval tape mode wiring and artifact flags
- verifier mode support
- MPS mixed correctness coverage against sample-parallel fused MSE

## Validation

Build:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
```

Focused tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

Both passed. The MPS mixed test exercises ownerkeep-i16 loss and site-gradient
matching against sample-parallel fused MSE.

Artifacts:

- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerkeep_i16_scale_2_4_8_16_render16_site24_warm3.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_for_ownerkeep_i16_pair.json`

Verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerkeep_i16_scale_2_4_8_16_render16_site24_warm3.json \
  --tape-mode gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse --allow-contended
```

passed with no failures.

## Result

Ownerkeep-i16 is correct and sublinear across the 2/4/8/16 frame scale, but it
does not beat the sample-parallel coeff16 keeper.

Paired mean timings:

| frames | ownerkeep-i16 total ms | sample total ms | total ratio | ownerkeep-i16 backward ms | sample backward ms | backward ratio | storage ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.625 | 5.786 | 0.972 | 4.975 | 5.198 | 0.957 | 1.241 |
| 4 | 5.285 | 3.537 | 1.494 | 4.876 | 3.175 | 1.536 | 1.241 |
| 8 | 4.602 | 3.236 | 1.422 | 4.225 | 2.871 | 1.471 | 1.241 |
| 16 | 4.800 | 3.817 | 1.257 | 4.407 | 3.365 | 1.310 | 1.241 |

Quality was effectively unchanged, as expected for a loss/gradient-equivalent
replay fork:

- 2f train/heldout PSNR: 14.2038 / 15.1260
- 16f train/heldout PSNR: 14.5395 / 15.3234

## Decision

Reject ownerkeep-i16 as a performance fork. It adds the boundary id and
boundary-pair streams, increasing resident tape storage by 24.1%, but does not
reduce enough work inside the hot kernel. The 2f row is slightly faster than the
paired sample control, but the useful 4/8/16 rows are 1.26x to 1.49x slower.

This supports the earlier diagnosis: the promising math is not "int16
side-streams plus ownerkeep semantics." The next serious fork needs to implement
the grouped-ownerkeep idea from the preflight, so the shader stops scanning
unrelated boundary transitions instead of merely storing them more compactly.
