# 2026-05-19 16:14:59 - ownerkeep-i16 clean promotion reject

## Context

The zero-length ownerkeep-i16 patch produced a strong-looking diagnostic timing
artifact, but that run ended with `benchmark_environment.status=contended`
because an unrelated `ai_trader` monitor launched a high-CPU verifier during the
sweep. I tried to capture a promotion-grade clean ownerkeep/sample pair with a
bounded retry loop.

## Commands

The retry loop checked:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only
```

Then, only when clear at start, it ran:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse \
  --frame-counts 2,4,8,16 --render-size 16 --site-count 24 \
  --optimizer-mode manual-vjp --steps 3 --warmup-steps 1 \
  --require-benchmark-environment-ok
```

and the same command with:

```text
--tape-mode gate4-affine-candidate-coeff16-fused-mse
```

## Artifacts

The loop failed to capture a same-attempt clean pair because the `ai_trader`
monitor repeatedly started CPU work before the sample-control run. It did
produce clean individual artifacts:

- ownerkeep-i16: `research_experiments/world_foam_lane2/results/2026-05-19_gate4_ownerkeep_i16_zero_length_promotion_pair_owner_attempt2.json`
- sample control: `research_experiments/world_foam_lane2/results/2026-05-19_gate4_ownerkeep_i16_zero_length_promotion_pair_sample_attempt8.json`

Both verified without `--allow-contended`:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_ownerkeep_i16_zero_length_promotion_pair_owner_attempt2.json \
  --tape-mode gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_ownerkeep_i16_zero_length_promotion_pair_sample_attempt8.json \
  --tape-mode gate4-affine-candidate-coeff16-fused-mse
```

## Clean Result

Clean non-contended artifacts reject ownerkeep-i16 as a promotion candidate:

| frames | ownerkeep-i16 total ms | sample total ms | total ratio | ownerkeep-i16 backward ms | sample backward ms | backward ratio | storage ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.668 | 2.565 | 1.040 | 2.332 | 1.936 | 1.204 | 1.241 |
| 4 | 2.991 | 2.551 | 1.172 | 2.666 | 2.106 | 1.265 | 1.241 |
| 8 | 3.100 | 2.179 | 1.423 | 2.721 | 1.759 | 1.547 | 1.241 |
| 16 | 2.980 | 2.076 | 1.435 | 2.451 | 1.699 | 1.443 | 1.241 |

Ownerkeep-i16 remains correctness-green, but it loses speed and storage against
sample-parallel coeff16 in clean conditions. The earlier contended artifact was
not reliable promotion evidence.

## Decision

Reject ownerkeep-i16 zero-length as a performance fork. Keep the correctness fix
and tests only if useful for future owner-transition experiments, but do not
promote this mode as the WorldFoam keeper.

The next aligned shader work should not add per-candidate boundary side streams.
The CPU preflight still says the mathematical win is grouped owner transitions,
but the Metal path needs a representation that avoids both full owner scans and
the `1.241x` resident storage tax.
