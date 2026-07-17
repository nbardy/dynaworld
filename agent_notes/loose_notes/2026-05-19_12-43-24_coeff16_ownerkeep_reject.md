# coeff16 ownerkeep fused-MSE fork reject

## Context

We stopped iterating WorldFoam Gate4 fused shader forks and audited the latest
candidate against the current keeper, `gate4-affine-candidate-coeff16-fused-mse`.
The fork under test was:

```text
gate4-affine-candidate-coeff16-ownerkeep-fused-mse
```

Ownerkeep keeps the current owner when a candidate boundary cut is unrelated to
that owner, instead of invalidating and forcing a fresh `wf2_realray_owner_at(...)`
scan. The intent was to test whether inactive candidate cuts were the reason the
previous ownerupdate fork fell back too often.

## Correctness / build gates

Passed:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile ...
rtk zsh -lc '( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

The verifier unit suite passed `11/11`; the full MPS mixed suite passed `8/8`.

## Paired speed evidence

Clean speed gating was refused because unrelated local processes were using most
of the CPU. I ran a paired contended diagnostic instead with the same small
Gate4 shape:

```text
frame counts: 2,4,8,16
render size: 16
site count: 24
steps: 3
warmup steps: 3
benchmark_environment.status: contended
```

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerkeep_scale_2_4_8_16_render16_site24_warm3_contended.json
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_for_ownerkeep_pair.json
```

Paired ratios, ownerkeep over sample-parallel:

```text
frames  sample_total_ms  ownerkeep_total_ms  total_ratio  sample_backward_ms  ownerkeep_backward_ms  backward_ratio
2       3.922            5.165               1.317        3.389               4.583                  1.352
4       4.354            3.832               0.880        3.770               3.434                  0.911
8       4.036            4.390               1.088        3.554               4.020                  1.131
16      4.411            6.600               1.496        3.812               5.692                  1.493
```

Verifier status:

```text
sample-parallel: ok
ownerkeep: failed, total_step_scale 1.278 exceeds 1.250
```

Ownerkeep also keeps the extra boundary-id stream, so resident storage is still
near `1.04MB` at 16f versus about `0.70MB` for the sample-parallel mode.

## Decision

Reject / do not promote.

The inactive-transition hypothesis was reasonable, and the fork is
correctness-green, but the measured result says the extra candidate-id stream
plus owner-transition branch logic still costs more than raw owner scans at this
tiny Gate4 shape. It only wins the noisy 4f row and loses badly at 2f, 8f, and
16f.

The current WorldFoam keeper remains:

```text
gate4-affine-candidate-coeff16-fused-mse
```

Next useful WorldFoam work should reduce candidate replay / owner lookup cost
without adding another per-candidate side stream or serializing frames by track.
