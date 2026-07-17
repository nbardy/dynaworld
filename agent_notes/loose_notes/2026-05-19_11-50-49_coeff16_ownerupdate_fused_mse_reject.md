# 2026-05-19 11:50:49 - Gate4 coeff16 ownerupdate fused-MSE reject

## Context

The current keeper before this fork was `gate4-affine-candidate-coeff16-fused-mse`: sample-parallel fused RGB MSE/VJP with coeff16 affine boundary depths. It is frame-count sublinear on the 2/4/8/16, render16, site24 gate and beats matched STAR UVT on total step while still losing backward.

The next hypothesis was that the backward gap comes from per-segment `wf2_realray_owner_at(...)` scans. I added a first-class owner-update variant:

```text
gate4-affine-candidate-coeff16-ownerupdate-fused-mse
```

It carries `candidate_ids` and `boundary_site_pairs` to MPS, sorts boundary ids alongside depths, toggles the current owner across a boundary when the active owner is one of the boundary endpoints, and falls back to a full owner scan when the boundary sequence is ambiguous.

## Changed

- Added Metal helper `wf2_realray_insert_depth_with_boundary_fused_mse`.
- Added Metal kernel `wf2_fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only_tensor`.
- Registered the op through `world_foam_lane2_metal.mm`, `bindings.cpp`, `ops.py`, and package exports.
- Added the train/eval tape mode and MPS device tensors:
  - `affine_candidate_boundary_ids_i32`
  - `affine_boundary_site_pairs_i32`
- Added verifier/compare mode support.
- Added an MPS parity test for the ownerupdate coeff16 MSE path against the promoted sample-parallel kernel, including an ambiguous non-active cut that forces fallback.

## Gates

Build:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
```

Focused tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_verify_gate4_affine_candidate_csr_train_eval -v
```

Both passed.

Smoke:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --tape-mode gate4-affine-candidate-coeff16-ownerupdate-fused-mse --frame-counts 2 --render-size 16 --site-count 8 --optimizer-mode manual-vjp --steps 3 --warmup-steps 1 --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerupdate_smoke_2f_render16_site8.json
```

It passed, but was slower than the promoted sample path on the matched site8 smoke:

- ownerupdate: `3.581ms` total mean, `3.072ms` backward mean
- sample-parallel coeff16: `2.873ms` total mean, `2.321ms` backward mean

Full site24 ladder:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --tape-mode gate4-affine-candidate-coeff16-ownerupdate-fused-mse --frame-counts 2,4,8,16 --render-size 16 --site-count 24 --optimizer-mode manual-vjp --steps 3 --warmup-steps 1 --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerupdate_samplemse_scale_2_4_8_16_render16_site24_warm3.json
```

Verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_ownerupdate_samplemse_scale_2_4_8_16_render16_site24_warm3.json --tape-mode gate4-affine-candidate-coeff16-ownerupdate-fused-mse --allow-contended
```

It passed with `benchmark_environment.status=contended`.

## Result

The ownerupdate fork is correct and trainable but rejected for speed/storage.

Site24 2/4/8/16 ownerupdate:

| frames | total mean ms | backward mean ms | resident storage | train PSNR | heldout PSNR |
|---:|---:|---:|---:|---:|---:|
| 2 | 4.824 | 4.277 | 1,050,532 | 14.204 | 15.126 |
| 4 | 6.221 | 5.754 | 1,046,688 | 14.267 | 15.138 |
| 8 | 6.607 | 6.181 | 1,041,748 | 14.414 | 15.200 |
| 16 | 5.243 | 4.733 | 1,042,128 | 14.540 | 15.323 |

Scales:

- total mean scale: `1.087x`
- backward mean scale: `1.107x`
- total median scale: `0.994x`
- backward median scale: `0.998x`
- resident storage scale: `0.992x`

The promoted sample-parallel coeff16 path remains the keeper: its same site24 artifact is faster and smaller (`~0.70MB` resident storage versus ownerupdate `~1.04MB`) and has better mean scaling (`0.792x` total, `0.783x` backward).

## Interpretation

The simple boundary-pair owner-toggle idea is not enough. It adds a per-candidate boundary-id stream and extra branch/control work, while real candidate rows include many pair boundaries that are not necessarily active owner transitions. The conservative fallbacks keep parity but eat the intended win. Do not re-try this shape as a promotion path unless the candidate stream is first reduced to active transitions or the owner state can be represented without adding another full per-candidate memory stream.
