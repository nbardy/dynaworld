# Gate4 affine bridge scope-fix and render32 verifier

## Context

After extracting the Gate4 affine slab tape object, the next gap was stronger evidence than the render8 tiny smoke. I ran the moving-camera affine tape path at render32/site12 over 2/4/8/16 frames with VJP enabled. The run passed, but it exposed a reporting bug: owner-update acceptance keys were being marked true even when `--include-ownerupdate` was not run.

## What changed

- Updated `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py` so owner-update acceptance keys are only emitted when the owner-update path actually runs.
- Added explicit `checked` flags to:
  - `ownerupdate_diagnostics`
  - `mixed_vjp_direct_grad_only_ownerupdate_diagnostics`
- Added `research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py`.
- Added `research_experiments/world_foam_lane2/test_verify_gate4_affine_tape_bridge.py`.
- Updated `research_experiments/world_foam_lane2/README.md` with the Gate4 render32/site12 bridge artifact and verifier command.

## Validation

Compile:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_tape_bridge.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py
```

Verifier unit tests:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge -v
```

Result: `Ran 6 tests`, `OK`.

Focused Gate4 plus verifier tests:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge -v
```

Result: `Ran 10 tests`, `OK`.

Render32/site12 MPS artifact:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --include-vjp \
  --vjp-seed-mode rgba-depth \
  --timing-iters 5 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix.json
```

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix_verifier.json
```

Verifier result: `status=ok`.

Important verified numbers:

- mixed num32/den16 max error: `0.00016689300537109375`
- pure coeff16 max error: `0.02310839295387268`, rejected
- missing sample events: `0` for 2/4/8/16 frames
- compiled boundary-test ratios: `0.5 / 0.25 / 0.125 / 0.0625`
- explicit ray storage scale 2->16: `8.0x`
- mixed tape storage scale 2->16: `0.9667x`
- owner-update scope: `include_ownerupdate=false`, `ownerupdate_checked=false`, `ownerupdate_vjp_checked=false`

## Interpretation

This is a stronger Gate4 moving-camera bridge result than the render8 smoke. The affine tape representation is behaving like the STAR-style structural idea at the representation level: boundary tests amortize across frames, and the mixed tape storage stays essentially flat while explicit rays scale with frame count.

It is still not a runtime competitiveness result versus STAR UVT. The per-op timings are noisy and the VJP variants do not form a clean speed ladder at render32/site12. The practical claim is now: moving-camera affine tape + MPS bridge + mixed precision VJP correctness is verified at a less toy size, with owner-update scope correctly reported.

## Next useful step

The next shader step should move from representation correctness to a matched speed/quality comparison: same frame counts, same render size, same site/tube scale where possible, and enough warm measured iterations to compare Gate4/WorldFoam against the current STAR UVT direct-atomic reference without confusing storage scaling for wall-clock speed.
