# Owner-Run Packed-Delta Fused-MSE Parity

## Context

The packed endpoint owner-run delta storage probe showed the right topology
shape for the RGB-only `owner-run-fused-mse-nomid` path, but the harness still
needed a direct proof that the packed-delta replay is numerically the same loss
and site-gradient boundary as the lean owner-run segment-tape VJP.

## What Changed

Added a focused train/eval harness regression:

- `research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py`

The test builds the real small PowerFoam fixture with moving rays, prepares both
tapes, and compares:

- `owner-run-fused-mse-nomid`
- `owner-run-delta-packed-recompute-fused-mse-nomid`

It checks that the packed mode:

- selects the same owner-run segment count as the lean owner-run tape
- uses the owner-run-selected tape, not endpoint-run semantics
- keeps changed records below the old no-threshold endpoint-like pattern
- stores only the minimal packed-delta warm-kernel tensors on MPS
- omits `boundary_f32` and `rays_f32` from the resident hot tape
- matches loss and site gradients against the owner-run fused-MSE nomid path

## Evidence

Command:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed -v
```

Result:

```text
test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays ... ok

Ran 1 test in 9.439s
OK
```

The ad-hoc pre-test probe on the same moving-ray setup measured:

- owner/packed selected segments: `1770 / 1770`
- owner topology bytes: `18,260`
- packed topology bytes: `17,332`
- packed resident non-coeff bytes: `17,388`
- loss diff: `1.64e-7`
- max site-gradient diff: `4.10e-7`

The resident warm-kernel storage still includes a large coeff payload:

- `delta_coeff_f16`: `114,688` bytes
- selected resident bytes total: `~132k` in the parity shape

## Timing Status

Clean promotion timing was not run. The environment gate rejected the machine:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only
```

It returned `status=contended` because an `ai_trader` Python integrity verifier
was using about `90%` CPU. Do not cite a packed-delta speed ladder until
`--require-benchmark-environment-ok` passes.

## Current Decision

The packed-delta owner-run mode is now correctness-green for RGB-only fused MSE
on moving rays, but it is not promoted. The next gate is a clean
`2/4/8/16f` train/eval ladder for
`owner-run-delta-packed-recompute-fused-mse-nomid`. If timing is competitive,
the next real storage target is coeff residency/recompute, because topology is
now smaller while `delta_coeff_f16` dominates resident bytes.
