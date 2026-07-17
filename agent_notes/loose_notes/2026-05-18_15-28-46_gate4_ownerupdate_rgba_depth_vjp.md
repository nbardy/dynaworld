# Gate4 owner-update RGBA/depth VJP gate

After the midpoint-owner fix, the owner-update path had been verified with the
default RGB-only VJP seed. That was enough for the older RGB probe, but it left
a narrow escape hatch: a future artifact could satisfy the owner-update verifier
without proving that alpha/depth adjoints are routed through the owner-update
VJP kernel.

Ran a quick 2-frame probe with nonzero RGBA/depth VJP seed:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --tile-h 1 \
  --tile-w 1 \
  --include-vjp \
  --include-ownerupdate \
  --vjp-seed-mode rgba-depth \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_rgba_depth_render32_site12_2f_probe.json
```

Then ran the full 2/4/8/16 artifact:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --tile-h 1 \
  --tile-w 1 \
  --include-vjp \
  --include-ownerupdate \
  --vjp-seed-mode rgba-depth \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_rgba_depth_render32_site12_2_4_8_16.json
```

Result:

- status: `ok`
- `vjp_seed_mode`: `rgba-depth`
- `gradient_scope`: `mixed_num32_den16_site_rgba_vjp_rgba-depth_seed`
- mixed max error: `0.00016689300537109375`
- owner-update forward max error: `0.00016689300537109375`
- owner-update VJP relative delta versus reduce:
  `6.516429842513915e-6`
- RGB-only direct sidecar correctly diverged from reduce under nonzero
  alpha/depth adjoints and recorded `has_expected_seed_behavior=true`
- pure coeff16 remained rejected with max error `0.02310839295387268`

Added `--require-vjp-seed-mode {rgb,rgba-depth}` to
`verify_gate4_affine_tape_bridge.py` so strict owner-update acceptance can bind
to the actual VJP seed mode. Verified the full artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_rgba_depth_render32_site12_2_4_8_16.json \
  --require-ownerupdate \
  --require-vjp-seed-mode rgba-depth \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_rgba_depth_render32_site12_2_4_8_16_verifier.json
```

Verifier result:

- status: `ok`
- failures: `[]`
- mixed tape storage scale 2->16: `0.9667229515527287`
- explicit ray storage scale 2->16: `8.0`
- boundary ratios 2/4/8/16: `0.5 / 0.25 / 0.125 / 0.0625`
- owner-update forward and VJP checks both explicitly checked

Regression gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_tape_bridge.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

The combined Gate4 suite ran 23 tests and passed.

Interpretation: the current Gate4 owner-update shader gate now covers nonzero
alpha/depth adjoints as well as RGB. This strengthens the shader-correctness
story but does not widen the claim: still render32/site12, affine moving rays,
frozen geometry for train/eval, and not a full trainer or STAR-UVT
competitiveness result.
