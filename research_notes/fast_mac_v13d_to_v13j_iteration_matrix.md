# Fast-Mac v13d-v13j Iteration Matrix

Date: 2026-05-11

## Scope

This note continues the v13 multiframe raster exploration after v13a/v13b/v13c.
The goal was to iterate through several more concrete variants without copying
stable shader baselines blindly.

New benchmark runner:

```text
src/benchmarks/fast_mac_v13_iteration_matrix.py
```

The runner names seven follow-up iterations:

| Name | Mechanism | Artifact |
|---|---|---|
| `v13d_v11_serial_batch` | existing v11 serial batch strategy | timing row |
| `v13e_v11_active_auto` | existing v11 active-tile auto policy | timing row |
| `v13f_v11_active_on` | force active-tile scheduling | timing row |
| `v13g_v11_frozen_features` | freeze feature/color gradients | timing row |
| `v13h_v13a_recompute_state` | exact v13a backward-state recompute | timing row |
| `v13i_v13b_renamed_baseline` | runnable v13b renamed v11 path | timing row |
| `v13j_rgb_grad_accounting` | future RGB-gradient handoff memory accounting | accounting row |

These are benchmark versions, not all separate Metal shader forks. The point is
to prune obvious negative directions before spending another kernel fork.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  src/benchmarks/fast_mac_v13_iteration_matrix.py \
  --height 128 \
  --width 128 \
  --gaussians 8192 \
  --batch-size 2 \
  --feature-dim 32 \
  --warmup 2 \
  --iters 5
```

The runner sets:

```text
GSP_FAST_CAP=4096
GSP_FEATURE_CAP=64
```

## Results

Target row: `128x128`, `B=2`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, backward + alpha loss.

| Iteration | Median ms | Forward ms | Backward ms | Read |
|---|---:|---:|---:|---|
| `v13d_v11_serial_batch` | `30.968` | `12.351` | `19.154` | Bad for speed; serial launch overhead dominates. |
| `v13e_v11_active_auto` | `20.249` | `7.471` | `12.909` | Near baseline; auto did not unlock a new class of win here. |
| `v13f_v11_active_on` | `21.992` | `9.065` | `13.462` | Worse than auto; forced active scheduling is not a dense-row win. |
| `v13g_v11_frozen_features` | `27.869` | `13.769` | `13.958` | Not useful on this row; frozen-feature direct timing is noisy and not a standalone speed path. |
| `v13h_v13a_recompute_state` | `23.630` | `7.088` | `16.412` | Exact memory valve, still slower than save-state paths. |
| `v13i_v13b_renamed_baseline` | `19.316` | `7.209` | `12.555` | Runnable renamed baseline; timing alone is not a new kernel claim. |
| `v13j_rgb_grad_accounting` | n/a | n/a | n/a | Dense backward input would drop `528MiB -> 48MiB` at `B=16,512px,F32`. |

Accounting row:

```text
B=16 H=W=512 F=32 float32
current dense backward input: 528 MiB
handoff RGB input:             48 MiB
avoided:                      480 MiB / 90.9%
```

## Interpretation

The cheap config-level ideas mostly do not close the batch/F32 gap:

- Serializing batch is a memory fallback, not a speed path.
- Forced active-tile scheduling loses when the tile grid is dense.
- Active auto is acceptable but not enough.
- v13a recompute remains exact but slower because it pays an extra bin and
  forward-state render in backward.

The only large structural win left in this family is still v13b:

```text
image-space loss/color gradient
-> compact RGB grad
-> fused colorizer VJP at raster-backward entry
-> no dense grad_features[B,H,W,F] allocation
```

The v13d-v13j matrix strengthens the earlier conclusion: do not spend a large
kernel fork on batch scheduling alone. Spend it on the v13b streaming
RGB-gradient handoff or on real pair-level temporal active-set diagnostics.

## Next Iterations

Use the matrix to decide what deserves actual kernel work:

- Promote `v13b` from scaffold to kernel implementation.
- Add real trainer-projected input mode to `temporal_raster_overlap_profile.py`.
- Run this matrix at `256px` and `512px` before making claims about high-res
  behavior.
- If memory, not speed, blocks a run, test `v13a` recompute with a sampled MPS
  peak-memory trace in the trainer graph.
