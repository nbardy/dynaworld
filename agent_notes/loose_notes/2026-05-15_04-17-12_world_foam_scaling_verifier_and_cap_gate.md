# World Foam fused mixed scaling verifier and cap gate

## Context

After the `direct_atomic_grad_only` VJP path became the measured winner, the
experiment lane still had two practical gaps:

- `tools/train_eval_fused_slab_mixed_mps.py` defaulted to the slow reducer.
- The shader has a fixed local boundary array cap, but the harness only reported
  `max_candidates_per_row`; it did not fail if a CSR row would exceed the Metal
  local replay capacity.

## Changes

- Changed `tools/train_eval_fused_slab_mixed_mps.py` defaults:
  - `--vjp-mode direct_atomic_grad_only`
  - `--vjp-reduce-chunk-size 16`
- Added `MAX_REALRAY_BOUNDARIES = 128` to the Python wrapper surface and
  exported it from `torch_world_foam_lane2_fused_slab`.
- Added host-side CSR row-length validation in `ops.py`, so oversized per-row
  candidate lists fail before the Metal kernels silently truncate local depth
  insertion.
- Added the cap acceptance key to the smoke and train/eval harnesses:
  `candidate_rows_under_metal_cap`.
- Added:

```text
research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py
```

The verifier reads the saved 2/4/8/16 train/eval artifacts and the VJP smoke
artifact, then checks:

- all required modes are present;
- all saved train/eval rows are `ok`;
- `direct_atomic_grad_only` is fastest at 16f and by total-step geometric mean;
- total/render/backward scaling for the selected mode stays under configured
  limits;
- PSNR spread across VJP modes is tiny;
- smoke VJP gradients match the reducer within relative tolerance;
- smoke `max_candidates_per_row` stays below the Metal cap.

## Evidence

Verifier command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_scaling_verifier.json
```

Status:

```text
ok
```

Selected-mode scaling from the verifier:

```text
direct_atomic_grad_only total scale 2f->16f:    1.299x
direct_atomic_grad_only render scale 2f->16f:   1.055x
direct_atomic_grad_only backward scale 2f->16f: 1.742x
```

Runtime smoke for the new cap acceptance:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 8 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --time-slabs 1 \
  --timing-iters 1 \
  --include-vjp \
  --vjp-reduce-chunk-size 16 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_capcheck_smoke_2f_render16_pertrack.json
```

Status:

```text
ok, candidate_rows_under_metal_cap=true, max_candidates_per_row=27
```

Default train/eval smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke.jsonc \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 8 \
  --steps 1 \
  --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_default_gradonly_train_eval_smoke_2f_render16.json
```

Status:

```text
ok, vjp_mode=direct_atomic_grad_only, candidate_rows_under_metal_cap=true
```

Static checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py

git -C third_party/fast-mac-gsplat diff --check -- variants/world_foam_lane2_fused_slab_v0
git diff --check -- research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py agent_notes/loose_notes
```

All passed.

## Takeaway

This does not make World Foam STAR-flat. It hardens the current fused mixed lane
so the measured winner is the default path and the saved scaling claim is now
one-command verifiable instead of a manual JSON comparison.
