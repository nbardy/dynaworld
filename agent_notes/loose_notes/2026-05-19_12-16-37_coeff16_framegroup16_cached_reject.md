# Gate4 coeff16 framegroup16-cached fused-MSE reject

## Context

After the coeff16 sample-parallel fused-MSE promotion and the owner-update /
sample-reduce rejects, I tried a representation-reuse fork that keeps the same
Gate4 affine candidate CSR tape but runs one threadgroup per track/frame chunk.
The fork caches one row's coeff16 candidate depths into threadgroup memory and
reduces loss once per 16-frame chunk. Gradients still use the direct per-segment
global atomics from the promoted sample kernel; no per-candidate side stream and
no per-track frame serialization were added.

Mode:

```text
gate4-affine-candidate-coeff16-framegroup16cached-fused-mse
```

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_framegroup16cached_scale_2_4_8_16_render16_site24_warm3.json
```

Same-window sample-parallel control:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_rerun_for_framegroup16cached.json
```

## Validation

Passed:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py \
  research_experiments/world_foam_lane2/test_train_eval_fused_slab_mixed_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v

rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

The MPS suite added a multi-frame parity check against
`fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only` and passed `8/8`.

Verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_framegroup16cached_scale_2_4_8_16_render16_site24_warm3.json \
  --tape-mode gate4-affine-candidate-coeff16-framegroup16cached-fused-mse
```

Result: `status=ok`, `benchmark_environment_status=background`,
`total_step_scale=0.625`, `backward_scale=0.632`, storage scale `0.992`.

## Result

Framegroup-cached 2/4/8/16 mean timings:

```text
total ms:    5.657 / 5.427 / 6.029 / 3.536
backward ms: 5.026 / 4.803 / 5.356 / 3.176
storage:     708604 / 706044 / 702756 / 703020 bytes
PSNR train:  14.204 / 14.267 / 14.414 / 14.540
PSNR heldout:15.126 / 15.138 / 15.200 / 15.324
```

Fresh sample-parallel control in the same background environment:

```text
total ms:    5.888 / 4.249 / 4.723 / 3.719
backward ms: 5.261 / 3.638 / 4.130 / 3.174
```

Framegroup/sample ratios:

```text
frame  total  backward
2      0.961  0.955
4      1.277  1.320
8      1.277  1.297
16     0.951  1.001
```

## Decision

Reject as a default promotion. The fork is correct and storage-neutral, and it
does expose a stronger sublinear-looking scale number than the previous
contended sample artifact, but the clean same-window control shows the caching
barrier/threadgroup overhead loses at 4f and 8f and only ties at 16f backward.

The result is still informative: row-coeff global loads are not the dominant
next bottleneck at this tiny Gate4 shape. The remaining cost is still candidate
replay plus owner lookup and local sorting per active frame sample. The next
serious fork should reduce the replay/owner work itself, not just move the same
coefficients through threadgroup memory.
