# 2026-05-19 12:03:21 - Gate4 coeff16 sample-reduce fused-MSE reject

## Context

The current keeper was still `gate4-affine-candidate-coeff16-fused-mse`: sample-parallel coeff16 affine candidate replay with fused RGB MSE/VJP. The previous owner-update fork was correct but slower because it added a full boundary-id side stream and fallback-heavy control.

This fork tested a different hypothesis: maybe the backward gap was still partly per-segment atomic pressure. The new mode keeps the same sample-parallel launch and the same coeff16 CSR tape, but accumulates each sample's segment gradients into a local per-site `float4` buffer before issuing one atomic add per touched site:

```text
gate4-affine-candidate-coeff16-samplereduce-fused-mse
```

It does not add any per-candidate metadata and it does not serialize frames per track.

## Changed

- Added Metal kernel `wf2_fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only_tensor`.
- Registered the op through the Metal launcher, C++ bindings, Python wrapper, and package exports.
- Added the first-class train/eval tape mode and verifier/compare support.
- Added MPS parity coverage against the promoted sample-parallel coeff16 kernel.

## Gates

Syntax/import:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py \
  research_experiments/world_foam_lane2/test_train_eval_fused_slab_mixed_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
```

Verifier unit:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v
```

Result: `8/8` passed.

Build:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
```

Focused MPS tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

Result: `7/7` passed, including sample-reduce parity against sample-parallel and existing owner-update/high-cap coverage.

## Scale artifact

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode gate4-affine-candidate-coeff16-samplereduce-fused-mse \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 24 \
  --optimizer-mode manual-vjp \
  --steps 3 \
  --warmup-steps 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplereduce_scale_2_4_8_16_render16_site24_warm3.json
```

Verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplereduce_scale_2_4_8_16_render16_site24_warm3.json \
  --tape-mode gate4-affine-candidate-coeff16-samplereduce-fused-mse \
  --allow-contended
```

Verifier status: `ok`; contamination: `benchmark_environment status is 'contended'`.

| frames | total mean ms | backward mean ms | total median ms | backward median ms | storage bytes | train PSNR | heldout PSNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4.650 | 4.143 | 4.681 | 4.176 | 708,604 | 14.204 | 15.126 |
| 4 | 4.335 | 3.871 | 4.509 | 4.094 | 706,044 | 14.267 | 15.138 |
| 8 | 4.062 | 3.542 | 4.371 | 3.854 | 702,756 | 14.414 | 15.200 |
| 16 | 5.458 | 4.826 | 5.568 | 4.959 | 703,020 | 14.540 | 15.324 |

Scales:

- total mean: `1.174x`
- backward mean: `1.165x`
- total median: `1.189x`
- backward median: `1.187x`
- resident storage: `0.992x`
- candidate count: `0.992x`

## Matched control

To avoid judging only against an older artifact, I ran the promoted sample-parallel mode in the same window at 2f/16f:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_control_2_16_render16_site24_for_samplereduce.json
```

That control was also contended, but still beat sample-reduce clearly:

- promoted sample 2f: `4.974ms` total mean, `4.415ms` backward mean
- promoted sample 16f: `3.553ms` total mean, `2.994ms` backward mean
- sample-reduce 16f: `5.458ms` total mean, `4.826ms` backward mean

## Decision

Reject this fork for promotion.

The local per-site accumulation is correct, but it is the wrong tradeoff for this kernel. It reduces segment-level atomics, yet adds a per-thread site buffer plus initialization and final flush work. The plain sample-parallel kernel's direct segment atomics remain faster, especially at 16 frames. This makes atomic pressure a weaker next lever than candidate replay / owner-scan representation.

The next WorldFoam shader fork should target a real row/run representation change that reduces candidate replay or owner scans without adding a full per-candidate side stream. Do not retry sample-local owner reduction unless a larger fixture proves atomics dominate independently.
