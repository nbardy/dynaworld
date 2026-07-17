# Gate4 coeff16 ownerupdate-i16 reject

## Context

After rejecting `gate4-affine-candidate-coeff16-ownerupdate-fused-mse`
and `gate4-affine-candidate-coeff16-sortnet-fused-mse`, we tested whether
the ownerupdate idea was losing mainly because its candidate-boundary side
streams were stored as int32. The fork added a packed int16 variant:

```text
gate4-affine-candidate-coeff16-ownerupdate-i16-fused-mse
```

The math stayed the same as ownerupdate: use boundary ids/site-pairs to update
the current owner across candidate transitions instead of scanning owner state
from scratch. Only the boundary id/pair tape dtype changed to `int16`.

## Implementation

Touched the fused slab variant:

- Metal kernel:
  `wf2_fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal launcher and op registration:
  `fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only`
- Python wrapper:
  `fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only`
- Train/eval mode:
  `gate4-affine-candidate-coeff16-ownerupdate-i16-fused-mse`
- Verifier and MPS mixed-test coverage for the new mode.

## Correctness gates

All local correctness gates passed:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

The verifier accepted both speed artifacts with `--allow-contended`; both had
`benchmark_environment.status = contended`, so this is a paired diagnostic,
not a clean promotion gate.

## Paired speed result

Artifacts:

- i16 ownerupdate:
  `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerupdate_i16_scale_2_4_8_16_render16_site24_warm3.json`
- same-window sample keeper control:
  `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_for_ownerupdate_i16_pair.json`

Mean timings at render16/site24/warm3:

| frames | sample total ms | i16 total ms | total ratio | sample backward ms | i16 backward ms | backward ratio | sample storage | i16 storage | storage ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4.668 | 7.447 | 1.595x | 4.074 | 6.860 | 1.684x | 708604 | 879568 | 1.241x |
| 4 | 4.638 | 6.138 | 1.324x | 4.110 | 5.604 | 1.364x | 706044 | 876366 | 1.241x |
| 8 | 4.444 | 6.071 | 1.366x | 3.892 | 5.587 | 1.436x | 702756 | 872252 | 1.241x |
| 16 | 4.116 | 7.260 | 1.764x | 3.509 | 5.816 | 1.658x | 703020 | 872574 | 1.241x |

The i16 variant is individually sublinear on this contended scale sweep:

- total step first-to-last scale: `0.9749`
- backward first-to-last scale: `0.8478`

But the same-window sample keeper is also sublinear and faster:

- total step first-to-last scale: `0.8817`
- backward first-to-last scale: `0.8613`

## Decision

Do not promote ownerupdate-i16. Packing boundary ids and boundary site-pairs to
int16 reduces the ownerupdate side stream relative to int32 ownerupdate, but it
still leaves the selected resident tape about `1.241x` larger than the sample
keeper and the branch/replay logic remains slower at every frame count in the
paired run.

The useful lesson is narrower: boundary stream dtype was not the decisive
bottleneck. At Gate4 render16/site24, added candidate-side streams and
owner-transition logic still cost more than raw sample-parallel owner scans.
The next WorldFoam fork should avoid extra per-candidate side streams unless it
removes a much larger amount of replay work.
