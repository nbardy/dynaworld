# World Foam owner-update VJP negative probe

## Context

After the scaling verifier, I tried the next non-redundant shader idea:
port the forward owner-update topology trick into a grad-only VJP. The target
was the per-segment owner scan in `direct_atomic_grad_only`; instead of calling
`wf2_realray_owner_at(...)` for every segment, the new diagnostic path seeds the
first owner once and toggles ownership through `boundary_site_pairs_i32` as it
crosses sorted candidate boundaries.

New diagnostic op:

```text
fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate
```

It is intentionally not wired into the default train/eval path.

## What changed

- Added Metal kernel
  `wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate_tensor`.
- Added host/C++/Python binding and wrapper.
- Extended the fused slab smoke so `--include-ownerupdate --include-vjp` also
  measures and checks the owner-update grad-only VJP against the reducer.
- Fixed the existing owner-update forward smoke call site by removing an
  accidental `reduce_chunk_size` keyword that the forward wrapper does not
  accept.

## Evidence

Build passed:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Tiny render16 / 2f owner-update smoke passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 8 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --time-slabs 1 \
  --timing-iters 2 \
  --include-vjp \
  --include-ownerupdate \
  --vjp-reduce-chunk-size 16 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_ownerupdate_gradonly_smoke_2f_render16_pertrack.json
```

Status:

```text
ok
owner-update VJP max rel delta vs reducer: 7.90e-7
owner-update VJP timing: 2.47 ms
current grad-only VJP timing: 2.03 ms
```

So even in the case where the topology shortcut was correct, it was slower than
the current winner.

The render32 / 2,4,8,16 timing smoke failed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --time-slabs 1 \
  --timing-iters 3 \
  --include-vjp \
  --include-ownerupdate \
  --vjp-reduce-chunk-size 16 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_ownerupdate_gradonly_vjp_render32_pertrack_2_4_8_16.json
```

Failure reason:

```text
ownerupdate forward max RGB error vs explicit: 0.424
owner-update VJP max rel delta vs reducer: 2.44e-4
```

The owner-toggle assumption does not hold robustly for this larger generated
scene. It is not safe to promote.

Control checks still pass:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_scaling_verifier_rerun_after_ownerupdate_probe.json
```

Status:

```text
ok, best_mode=direct_atomic_grad_only
```

The standard non-ownerupdate 2f render16 VJP smoke also passed after the new
diagnostic op was added:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_standard_smoke_after_ownerupdate_probe_2f_render16_pertrack.json
```

## Takeaway

Do not spend more time on boundary-pair owner toggles as a shortcut unless the
owner topology contract is made stricter. The current CSR candidate set can
include crossings where "toggle between left/right" is not equivalent to a full
power-cell owner query. The measured winner remains:

```text
vjp_mode=direct_atomic_grad_only
```
