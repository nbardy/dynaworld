# Gate4 coeff16 densitymask reject

## Context

After cap224 showed that local array footprint alone was not a reliable lever,
we tested an even narrower fork of the current keeper:

```text
gate4-affine-candidate-coeff16-densitymask-fused-mse
```

The idea was to keep the sample-parallel coeff16 CSR tape unchanged, but cache a
single `density_active` bit per emitted segment during the forward pass. Backward
then uses that cached bit instead of reloading `site_rgba.w` just to apply the
ReLU-density gradient gate.

## Implementation

Added/wired:

- Metal kernel:
  `wf2_fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal launcher and PyTorch op:
  `fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only`
- Python wrapper/export in the fused slab package
- Train/eval mode:
  `gate4-affine-candidate-coeff16-densitymask-fused-mse`
- Artifact/verifier flag:
  `gate4_affine_candidate_csr_densitymask_fused_mse`
- MPS parity coverage against the sample-parallel keeper, including a negative
  raw-density site so the cached density mask has to match the normal ReLU gate.

## Correctness gates

Passed:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py \
  research_experiments/world_foam_lane2/test_train_eval_fused_slab_mixed_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py

rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

The verifier suite is now `15/15`; the MPS mixed suite remains `8/8`.

## Paired speed result

Artifacts:

- densitymask:
  `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_densitymask_scale_2_4_8_16_render16_site24_warm3.json`
- same-window sample keeper:
  `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_for_densitymask_pair.json`
- verifiers:
  `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_densitymask_scale_2_4_8_16_render16_site24_warm3_verifier.json`
  `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_for_densitymask_pair_verifier.json`

Both artifacts verified with `--allow-contended`, but both were marked
`benchmark_environment.status = contended`, so this is diagnostic rather than a
promotion-quality timing gate.

Mean timings at render16/site24/warm3:

| frames | sample total ms | densitymask total ms | total ratio | sample backward ms | densitymask backward ms | backward ratio | storage ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.210 | 5.583 | 1.072x | 4.644 | 4.719 | 1.016x | 1.000x |
| 4 | 4.400 | 4.540 | 1.032x | 3.861 | 3.768 | 0.976x | 1.000x |
| 8 | 4.752 | 4.650 | 0.978x | 3.783 | 3.991 | 1.055x | 1.000x |
| 16 | 3.059 | 5.050 | 1.651x | 2.490 | 4.152 | 1.668x | 1.000x |

Verifier scales:

```text
densitymask: total 0.904, backward 0.880, total median 0.843, backward median 0.839
sample:      total 0.587, backward 0.536, total median 0.513, backward median 0.436
```

A later attempted clean rerun of densitymask was also marked contended and had
`total_step_scale_first_to_last = 1.014`, `backward_scale_first_to_last = 0.932`,
so it did not rescue the variant.

## Decision

Do not promote densitymask. It is correctness-green and storage-neutral, but it
does not beat the sample-parallel coeff16 keeper. The cached activity bit removes
one scalar load/compare in backward, but adds a local `uchar` array and another
forward write per segment. At Gate4 render16/site24 that trade is not favorable,
especially at 16f where the paired run is `1.65x` slower on total and `1.67x`
slower on backward.

The useful lesson is that density ReLU-gate loads are not the current decisive
bottleneck. The remaining target is still candidate replay / owner scan work,
not local array footprint, atomics, full sorting, boundary side streams, or
single-field reloads.
