# Gate4 affine slab tape object checkpoint

## Context

We paused the fused world-foam shader iteration after turning the Gate4 moving-ray slab compiler output into a reusable Python tape object. The intent was to stop duplicating CSR/tape construction logic inside the MPS smoke harness and give future shader or trainer bridges one canonical materialization path.

## What changed

- Added `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`.
- Introduced `Gate4AffineSlabTape`, which owns the metal-ready CSR fields:
  - per-row slab/tile indexing
  - `int32` row offsets and candidate ids
  - frame-time and affine ray coefficients
  - explicit rays for parity checks
  - candidate depth numerator/denominator views
- Kept `to_legacy_bundle()` so existing smoke tooling can consume the new object without changing the shader ABI yet.
- Rewired `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py` to build through `build_gate4_affine_slab_tape(...)`.
- Added a focused unit test in `research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py` that checks the tape materializes a metal-ready CSR layout and that replayed candidates render the same ray sequence as the original CPU oracle.

## Validation

Py-compile passed for the new tape module, the Gate4 test module, and the MPS smoke tool.

Focused unittest:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `Ran 4 tests in 0.022s`, `OK`.

MPS tiny bridge smoke:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8 \
  --render-size 8 \
  --site-count 6 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --include-vjp \
  --vjp-seed-mode rgba-depth \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_object_mps_vjp_render8_2_4_8f_rerun.json
```

Result artifact status: `ok`.

Important numbers:

- `missing_sample_events = 0` for 2f, 4f, and 8f.
- Compiled boundary-test ratio scales as expected: `0.5`, `0.25`, `0.125`.
- Mixed num32/den16 max render error: `0.0002263784408569336`.
- Mixed direct VJP relative gradient delta versus reduce path: `9.589243936860902e-07`.
- Mixed direct grad-only relative gradient delta versus reduce path: `7.991036614050751e-07`.

## Interpretation

This is a correctness and structure checkpoint, not a speed win claim. The useful result is that the STAR-like affine slab tape now has a reusable representation and the MPS bridge can consume it while preserving forward and VJP parity on the tiny moving-camera ray gate.

The pure coeff16 path is still not acceptable: max error was `0.0019558370113372803`, outside the approximate tolerance. The viable path remains mixed precision: keep depth numerator in fp32 and denominator/compact storage in fp16.

The scaling signal is theoretical/structural here, not yet practical speed proof. Boundary tests per frame drop with frame count because the compiled slab events are amortized, but the tiny cold smoke is too small and too noisy to decide competitiveness against STAR UVT.

## Next useful gate

Run a larger, less-cold benchmark through the tape object and compare against the STAR UVT reference on the same frame counts. The next question is no longer "can we represent moving first-person camera rays as a slab tape?" but "does the fused tape reduce real forward/backward wall time at useful image/site sizes without losing PSNR?"
