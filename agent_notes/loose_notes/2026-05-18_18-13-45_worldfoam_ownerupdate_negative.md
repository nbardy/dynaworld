# 2026-05-18 18:13:45 - WorldFoam fused-MSE owner-update fork negative

## Context

The current keeper before this pass was the high-cap fused RGB-MSE inline
owner-run reverse-tape merge:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerrun_repeat20_render64_site24_2_4_8_16.json`
- total medians for `2/4/8/16f`: `2.724/3.233/6.032/6.610 ms`
- backward medians for `2/4/8/16f`: `2.396/2.915/5.627/6.205 ms`
- verifier still fails narrowly: `2.427x` total scale, `2.590x` backward scale

The next hypothesis was to port a STAR-like ownership idea into the warm
fused-MSE kernel: use candidate boundary ids plus boundary site-pairs to avoid
calling `wf2_realray_owner_at` for every segment. The implemented fork called
`wf2_realray_owner_at` for the first live segment, then toggled the current
owner across each crossed boundary pair. It was added as a separate
`fused_mse_rgb_only_ownerupdate` mode, measured, then reverted because it was
slower.

## What Passed

The small MPS boundary regression passed before the scale run. It compared the
owner-update fused-MSE path against the standard fused-MSE path on a one-boundary
two-site ray and matched loss/gradient within `1e-6`.

The full 64px/24-site train/eval artifact also preserved quality and optimizer
behavior:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerupdate_repeat20_render64_site24_2_4_8_16.json`
- train PSNR: `13.73199 / 13.75300 / 13.66118 / 13.73486`
- heldout PSNR: `14.17008 / 13.99247 / 14.22048 / 14.23198`
- artifact status: `ok`

## What Failed

The timing was worse than the current owner-run keeper, especially at 16 frames:

```text
frame  owner-update total  owner-update backward
2      3.798 ms            3.447 ms
4      5.100 ms            4.793 ms
8      7.883 ms            7.535 ms
16     16.696 ms           15.868 ms
```

Verifier artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerupdate_repeat20_render64_site24_2_4_8_16_verifier.json
```

Verifier failures:

- total median scale: `4.396x` vs target `<= 2.000x`
- backward median scale: `4.603x` vs target `<= 2.500x`
- mixed tape storage stayed near flat: `0.9958x`
- explicit ray storage still scaled `8.0x`

## Decision

Reverted the owner-update code path. The live shader remains the reverse-only
owner-run keeper.

Interpretation: avoiding per-segment `owner_at` inside the same high-cap kernel
is not enough if it requires carrying boundary ids and owner-update state through
the already register-heavy fused-MSE loop. The next serious fork should be a
larger precomputed owner-run/site-pair record path that removes both candidate
depth replay and owner scans from the warm fused-MSE kernel, rather than another
in-kernel ownership-toggle variant.

## Validation After Revert

Commands:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace --force )

PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/probe_fused_slab_affine_mse_vjp_mps.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_after_ownerupdate_revert_parity_mps.json

PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps \
  research_experiments.world_foam_lane2.test_compare_fused_slab_vjp_modes_mps \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale -v
```

Results:

- rebuild succeeded
- focused fused-slab mixed tests: `5` tests OK
- post-revert parity probe: `status ok`, max loss diff `3.725e-09`, max grad
  diff `0.0`
- focused 39-test suite: OK
