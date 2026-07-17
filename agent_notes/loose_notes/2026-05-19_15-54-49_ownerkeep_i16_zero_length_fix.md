# 2026-05-19 15:54:49 - ownerkeep-i16 zero-length duplicate cut fix

## Context

The first ownerkeep-i16 fork was correct on the focused MPS test but slower than
sample-parallel coeff16 in the full 2/4/8/16 timing sweep. The CPU
owner-transition preflight suggested the real issue was same-depth/tied boundary
handling: a full grouped policy is exact, while a collapsed single-boundary cut
can mismatch at 8/16 frames.

## Failed grouped attempt

I first patched the ownerkeep-i16 shader to consume all same-depth boundary ids
as one group. That compiled, but it failed the focused MPS ambiguous-cut test:

```text
ownerkeep_i16_loss vs sample_loss absolute diff: 0.012279541
```

The failed grouped loop was backed out before timing. It was too broad a change:
it changed behavior even on a simple non-duplicate boundary fixture.

## Keeper patch

The successful patch is smaller. In
`world_foam_lane2_shared_replay_tensor.metal`, the ownerkeep-i16 kernel now
keeps `current_owner` through zero-length cuts instead of invalidating it when
`length <= 1e-8`. Boundary updates still run one by one, so normal-cut behavior
stays aligned with the existing ownerkeep semantics.

This targets duplicate/tied cut depths without introducing a new tape format or
a grouped side table.

## Validation

Focused MPS test after the patch:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps.TrainEvalFusedSlabMixedMpsTests.test_coeff16_ownerupdate_mse_matches_sample_parallel_with_ambiguous_cut -v
```

passed.

Full MPS mixed suite:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

passed 8/8.

## Timing

Diagnostic artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerkeep_i16_zero_length_keep_scale_2_4_8_16_render16_site24_warm3_clean.json
```

Verifier command:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerkeep_i16_zero_length_keep_scale_2_4_8_16_render16_site24_warm3_clean.json \
  --tape-mode gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse --allow-contended
```

passed with no failures. The artifact is marked `benchmark_environment.status =
contended` because an unrelated `ai_trader` verifier started during the run, so
this is not promotion-grade timing yet.

Mean timings versus the same-session sample control from the previous
ownerkeep-i16 pair:

| frames | ownerkeep-i16 zero-length total ms | sample total ms | total ratio | ownerkeep-i16 zero-length backward ms | sample backward ms | backward ratio | storage ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.672 | 5.786 | 0.462 | 2.356 | 5.198 | 0.453 | 1.241 |
| 4 | 2.582 | 3.537 | 0.730 | 2.229 | 3.175 | 0.702 | 1.241 |
| 8 | 2.780 | 3.236 | 0.859 | 2.408 | 2.871 | 0.839 | 1.241 |
| 16 | 2.411 | 3.817 | 0.632 | 2.101 | 3.365 | 0.624 | 1.241 |

Scale:

- total step first-to-last: `0.902x`
- backward first-to-last: `0.892x`
- resident storage first-to-last: `0.992x`

Quality remains effectively identical to sample-parallel:

- 2f train/heldout PSNR: `14.2038 / 15.1260`
- 16f train/heldout PSNR: `14.5395 / 15.3234`

## Decision

Promote the zero-length ownerkeep-i16 patch as a diagnostic keeper, but do not
call the timing promotion-grade until a clean `benchmark_environment.status !=
contended` pair is captured. This is the first fork in this sequence that
materially beats the sample-parallel coeff16 keeper in the small MPS scale
sweep, and it points to duplicate/tied cut handling as the actual owner-cache
blocker.

The remaining tradeoff is storage: the extra int16 candidate boundary stream and
boundary-pair table still make resident tape storage `1.241x` sample-parallel.
