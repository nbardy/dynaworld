# coeff16 sortnet fused-MSE fork reject

## Context

We continued the WorldFoam Gate4 shader-fork lane from the current keeper:

```text
gate4-affine-candidate-coeff16-fused-mse
```

The tested fork was:

```text
gate4-affine-candidate-coeff16-sortnet-fused-mse
```

The idea was to keep the compact sample-parallel coeff16 tape and avoid a new
per-candidate side stream, but replace insertion sorting during candidate
collection with collect-valid-depths plus an in-thread bitonic sort before
segment replay.

Before the fork, a CPU ordering probe on the current Gate4 render16/site24
shape showed append-order was unsafe: all samples had adjacent inversions across
2/4/8/16 frames, so the safe no-side-stream sort alternative was collect+sort,
not append-and-replay.

## Code path

Added/wired:

- Metal helper: `wf2_realray_sort_depths_bitonic_fused_mse(...)`
- Metal kernel: `wf2_fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal launcher and bindings:
  `fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only`
- Python wrapper and train/eval mode:
  `gate4-affine-candidate-coeff16-sortnet-fused-mse`
- verifier/test flags:
  `gate4_affine_candidate_csr_sortnet_fused_mse`

## Correctness gates

Passed:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

The verifier suite passed `12/12`; the MPS mixed suite passed `8/8`. The
high-cap MPS fixture now keeps the first 128 entries unsorted but still puts the
late depths beyond the old 128 cap, so it tests both sort correctness and
candidate replay beyond 128.

## Speed evidence

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_sortnet_scale_2_4_8_16_render16_site24_warm3.json
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_for_sortnet_pair.json
```

Both runs were contended by unrelated CPU jobs, so use this as a paired
diagnostic rather than a clean promotion gate. The paired result is still clear:

```text
frames  sample_total_ms  sort_total_ms  total_ratio  sample_backward_ms  sort_backward_ms  backward_ratio
2       5.486            6.825          1.244        4.812               6.225             1.294
4       4.837            7.158          1.480        4.136               6.365             1.539
8       2.860            4.799          1.678        2.364               4.077             1.725
16      4.319            12.520         2.899        3.761               11.645            3.096
```

Verifier:

```text
sortnet: failed, total_step_scale 1.835 > 1.250, backward_scale 1.871 > 1.250
sample control: ok, total_step_scale 0.787, backward_scale 0.782
```

## Decision

Reject / do not promote.

Collect+bitonic sorting is correct but too expensive. The existing insertion
sort has the useful property that it only shifts as needed while collecting and
keeps replay hot; full in-thread sorting adds enough work that the 16f row is
almost `3x` slower on total step and `3.1x` slower on backward in the paired
window.

The keeper remains:

```text
gate4-affine-candidate-coeff16-fused-mse
```

The next fork should not spend more work on sorting. The remaining useful target
is reducing owner/candidate replay itself without adding per-candidate side
streams, full per-sample sorts, or track-serial execution.
