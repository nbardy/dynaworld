# Gate 4 Affine CSR Bridge Oracle

Continued the WorldFoam shader/test lane by targeting the bridge identified in
the side-investigation note: Gate 4 moving-ray slab candidates feeding the
real-ray CSR/fused-slab compositor/VJP path.

What changed:

- Added a CPU render oracle to
  `research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py`.
- The new test renders a toy moving affine ray from the Gate 4 compiled
  candidate set and compares RGB/alpha/depth against the per-frame real-ray
  reference over several times in the slab.
- This covers the important semantic seam: compiled candidates may include
  extras, but after depth filtering/sorting they must produce the same
  compositing result as per-frame boundary scans.

Validation:

```text
PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

Ran 3 tests ... OK
```

MPS bridge smoke:

```text
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
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_csr_bridge_mps_tiny_vjp_render8_2_4_8f.json
```

Result artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_csr_bridge_mps_tiny_vjp_render8_2_4_8f.json
```

Key result:

- `status=ok`
- zero missing sample events at 2/4/8 frames
- explicit real-ray forward parity max errors stayed at fp32 noise
- mixed num32/den16 forward max error: `0.0002264`
- direct mixed VJP max grad relative delta vs reduce: `9.59e-7`
- direct grad-only max grad relative delta vs reduce: `8.39e-7`
- compiled boundary-test ratio fell as expected: `0.5 -> 0.25 -> 0.125`
- mixed fused storage moved from worse than explicit rays at 2/4f to slightly
  below explicit ray storage at 8f: `3.976x -> 2.000x -> 0.981x`

Caveat:

- The standalone all-fp16 coefficient path is not clean across this 2/4/8f
  sweep: `coeff16_diagnostics.within_approx_tolerance=false` with max error
  `0.001956` at the aggregate level.
- The promoted bridge for this step is therefore the mixed num32/den16 path,
  not pure coeff16.

Interpretation:

This is progress toward the requested shader fork/test fix, but not completion.
The Gate 4-to-Metal seam now has a cheap CPU semantic oracle plus a tiny MPS VJP
artifact. The remaining real work is to promote the prototype tape builder into
a first-class reusable Gate 4 affine slab tape object and run a larger,
less-cold MPS/train-eval gate before claiming the WorldFoam shader path is
fixed.
