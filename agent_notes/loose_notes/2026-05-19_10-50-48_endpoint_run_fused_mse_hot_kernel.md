# Endpoint-Run Fused-MSE Hot Kernel

## Context

After the direct-CSR native delta builder improved Gate4 compiler/setup cost but
did not change the warm replay kernel, I forked the already-existing compact
endpoint-run representation into a fused RGB-MSE + direct-atomic VJP shader.
The goal was to test the stronger representation change: run/site records that
avoid replaying Gate4 candidate ownership inside the hot loss/gradient kernel.

## Change

Added `endpoint_run_mse_vjp_direct_atomic_rgb_only` to the Lane2 fused-slab
variant:

- Metal kernel: `wf2_endpoint_run_mse_vjp_direct_atomic_rgb_only_tensor`
- C++ launcher/schema/dispatch
- Python wrapper/export
- train/eval mode: `endpoint-run-fused-mse`
- parity regression comparing fused loss/grad to existing endpoint-run
  `rgba_depth_replay` plus `endpoint_run_vjp_direct_atomic_grad_only` with an
  RGB-MSE seed

The train loop now routes `endpoint-run-fused-mse` directly to the endpoint-run
fused op with precomputed track-major targets. An early smoke caught that the
mode was accepted but missing from the selected-tape dictionary; that selector
now aliases the mode to the compact endpoint-run tape and marks
`selected_device["endpoint_run_fused_mse"] = True`.

## Verification

Build:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
```

Syntax:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_probe_endpoint_run_tape.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
```

Focused unit:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_run_tape -v
```

Result: `3 tests OK`.

Functional smoke:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-run-fused-mse \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 8 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_endpoint_run_fused_mse_smoke_2f_render16_site8.json
```

Result: `status=ok`, nonzero gradients, finite outputs, parameters updated.

## Scale Results

16px/24-site warm ladder:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_endpoint_run_fused_mse_scale_2_4_8_16_render16_site24_warm3.json`
- environment: `contended`
- frame scale: `8.0x`
- total step scale: `0.912x`
- backward scale: `0.831x`
- selected endpoint-run storage scale: `8.56x`
- selected endpoint-run segment scale: `8.60x`
- rows:
  - `2f`: total `2.859ms`, backward `2.229ms`, train/heldout PSNR `14.204/15.126`
  - `4f`: total `3.423ms`, backward `2.596ms`, train/heldout PSNR `14.267/15.138`
  - `8f`: total `2.878ms`, backward `2.384ms`, train/heldout PSNR `14.414/15.200`
  - `16f`: total `2.606ms`, backward `1.851ms`, train/heldout PSNR `14.539/15.324`

32px/24-site warm ladder:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_endpoint_run_fused_mse_scale_2_4_8_16_render32_site24_warm2.json`
- environment: `contended`
- frame scale: `8.0x`
- total step scale: `0.644x`
- backward scale: `0.583x`
- selected endpoint-run storage scale: `8.75x`
- selected endpoint-run segment scale: `8.80x`
- rows:
  - `2f`: total `7.555ms`, backward `6.758ms`, train/heldout PSNR `11.341/14.164`
  - `4f`: total `3.188ms`, backward `2.552ms`, train/heldout PSNR `11.406/14.250`
  - `8f`: total `3.456ms`, backward `2.838ms`, train/heldout PSNR `11.946/14.191`
  - `16f`: total `4.863ms`, backward `3.943ms`, train/heldout PSNR `12.083/14.147`

Comparison anchor from the previous track-MSE Gate4 fork at 16px/24-site:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_trackmse_fused_mse_scale_2_4_8_16_render16_site24_warm3.json`
- `16f`: total `11.946ms`, backward `11.423ms`, selected storage scale `0.992x`

## Interpretation

This is a real hot-kernel win. The endpoint-run fused shader is much faster than
the previous Gate4 replay fused-MSE variants at the same small scale, and the
functional route is now proved by parity/unit/smoke gates.

It is not a full WorldFoam completion claim. The compact endpoint-run tape used
here still has selected storage/segment counts scaling roughly with frame count
(`8.6-8.8x` from `2f -> 16f`). The hot kernel is fast because each row has a
small run count and the fused shader avoids candidate ownership replay, but the
current harness still pays linear-ish tape construction and resident endpoint
storage. The benchmark environment was also heavily contended, so promote this
as a keeper direction, not as a clean final baseline.

Next useful fork: make the endpoint/run or site-pair representation itself
sublinear for moving first-person cameras, or add a clean reference verifier for
this endpoint-run fused path and rerun in an uncontended window. More Gate4
candidate-replay micro-layout variants are lower priority now.
