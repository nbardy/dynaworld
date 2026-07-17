# Owner-Run Fused MSE No-Mid Fork

## Context

The previous owner-run fused MSE fork proved the compact owner-run tape can drive a
correct Metal RGB-MSE loss+VJP, but its selected resident storage still included
`mids_f32`. The fused RGB-only kernel does not read mids; mids are only needed for
depth/replay metrics outside the timed fused step.

This pass added `owner-run-fused-mse-nomid`, a narrow fork that keeps the same
Metal kernel math but exposes a no-mid Python wrapper and trainer mode:

- `segment_tape_nomids_mse_vjp_direct_atomic_rgb_only(...)`
- `owner-run-fused-mse-nomid`
- selected MPS tape contains only `offsets_i32`, `owners_i32`, and `lengths_f32`
- final train/heldout metric rendering lazily rebuilds a render-only segment device with mids after timing

## Verification

Syntax/import gate:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_probe_endpoint_run_tape.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
```

Focused unit/parity gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_endpoint_run_tape -v
```

Result: 5 tests OK, including no-mid fused path exactly matching the mid-carrying
fused wrapper on loss and site RGBA gradient.

Trainer smoke:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode owner-run-fused-mse-nomid --frame-counts 2 --render-size 16 --site-count 8 \
  --optimizer-mode manual-vjp --steps 1 --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_nomid_smoke_2f_render16_site8.json
```

Result: status OK, gradients nonzero, parameters updated, loss decreased. Selected
resident storage was 14,772 B instead of the owner-run tape's 20,108 B because
`mids_f32` was not moved for the fused step.

Scale ladder:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode owner-run-fused-mse-nomid --frame-counts 2,4,8,16 --render-size 16 --site-count 24 \
  --optimizer-mode manual-vjp --steps 3 --warmup-steps 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_nomid_scale_2_4_8_16_render16_site24_warm3.json
```

Result status was OK, but environment was contended. Treat timings as directional.

| frames | total ms | backward ms | selected storage | resident keys |
| --- | --- | --- | --- | --- |
| 2 | 2.900 | 2.302 | 19,028 B | offsets, owners, lengths |
| 4 | 2.852 | 2.306 | 44,012 B | offsets, owners, lengths |
| 8 | 2.914 | 2.307 | 90,684 B | offsets, owners, lengths |
| 16 | 4.416 | 3.292 | 183,668 B | offsets, owners, lengths |

Compared with `owner-run-fused-mse`, no-mid storage dropped:

- 2f: 26,492 B -> 19,028 B
- 4f: 61,920 B -> 44,012 B
- 8f: 127,832 B -> 90,684 B
- 16f: 259,116 B -> 183,668 B

## Interpretation

This fixed a real accounting/residency issue and slightly improves the owner-run
fused hot path. It does not fix the core scaling problem.

The selected owner-run segment count is unchanged and still scales 10.11x from
2f to 16f. Selected storage now scales 9.65x over the same 8x frame-count increase.
That is a little better than the mid-carrying mode's 9.78x storage scale, but still
not STAR-like temporal compression.

Current state:

- keep no-mid mode as the correct RGB-only owner-run fused path
- keep endpoint-run fused as the current fastest small hot-kernel path
- do not promote WorldFoam as competitive with STAR UVT yet
- next fork must change representation cardinality, not just remove unused fields

The next aligned fork is a persistent tube/bin or framegroup owner-run representation
whose selected records are not one owner-run per frame-local ray sample.
