# Fast-Mac v13 Full-Train Probe

Date: 2026-05-11

## Question

Do the v13 variants actually reduce feature-gradient memory or improve the
full multicam feature-splatting trainer path?

Short answer: not yet in the structural way we want. `v13a` is a real
recompute-state memory valve, but it trades saved autograd state for extra
backward work. `v13b` has the right RGB-gradient handoff boundary and accounting,
but the Metal handoff kernel is still scaffold-only, so the current runnable
`feature_variant="v13b_rgb_grad_handoff"` path is mostly a renamed v11-compatible
rasterizer.

## Config

Temp configs were cloned from:

```text
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_pairdelta012.jsonc
```

Only `render.fast_mac.feature_variant`, optional
`render.fast_mac.backward_state_strategy`, W&B, checkpointing, and
`train.steps=2` were patched under:

```text
/tmp/dynaworld_v13_fulltrain_configs/
```

Trainer benchmark command shape:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train WANDB_MODE=disabled \
  .venv/bin/python src/benchmarks/trainer_phase_benchmark.py \
  /tmp/dynaworld_v13_fulltrain_configs/<variant>.json \
  --warmup 1 --iters <2_or_3> --seed 123 \
  --memory-sample-interval-ms 1.0 \
  --json-output /tmp/dynaworld_v13_fulltrain_bench/<variant>.json
```

Fixed-render runs added `--fixed-render-graph`.

## Results

All rows are median milliseconds from saved JSON. Full-train rows include
sample, feature encode, projection, raster, loss, autograd backward, and
optimizer. Fixed-render rows exclude sample/encode/model backward/regularizers.

| Run | Total ms | Backward ms | Raster fwd ms | Loss ms | Encode ms | Peak current MiB | Peak driver MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v5_features` fixed | `1182.8` | `818.1` | `156.7` | `116.3` | n/a | `1560.4` | `2412.1` |
| `v11_features_gradcache_zero_bg_hostmeta_fixedbin` fixed | `861.6` | `671.6` | `89.8` | `77.3` | n/a | `1622.2` | `2420.1` |
| `v13a_temporal_recompute_state` fixed | `708.5` | `554.4` | `64.9` | `56.6` | n/a | `1574.1` | `2412.1` |
| `v13b_rgb_grad_handoff` fixed | `839.2` | `611.2` | `104.4` | `73.8` | n/a | `1622.2` | `2420.1` |
| `v11_features_gradcache_zero_bg_hostmeta_fixedbin` train | `1953.6` | `818.1` | `122.2` | `150.0` | `595.2` | `1524.6` | `2428.2` |
| `v13a_temporal_recompute_state` train | `1743.7` | `924.0` | `113.1` | `135.7` | `487.9` | `1417.7` | `2428.2` |
| `v13b_rgb_grad_handoff` train | `1721.3` | `929.1` | `102.2` | `159.2` | `426.2` | `1525.1` | `2428.2` |

The full-train rows are noisy. An earlier `v11` 2-iter run landed at
`1557.8ms` median total with `805.0ms` backward, while the first `v13b` 2-iter
run had a bad encode/project outlier at `2459.6ms`. The stable read is therefore
not "v13b is faster"; it is "v13b is in the same full-train band until the
actual RGB-gradient handoff kernel exists."

## Interpretation

- The big `528MiB -> 48MiB` claim is accounting for the intended v13b handoff at
  `B=16,H=W=512,F=32`: replace dense `grad_features[B,H,W,F]` plus
  `grad_alpha[B,H,W]` with dense `grad_rgb[B,H,W,3]`, then do the colorizer VJP
  at the raster-backward boundary.
- That drop is not yet observed in runtime memory because the handoff kernel is
  not implemented. The runnable v13b normal raster path still returns and
  backpropagates through F32 feature images.
- `v13a` really changes saved autograd state: it drops long-lived tile metadata
  and recomputes `bin(...)` plus forward-state tile stop data in backward. In
  this full-train probe it lowered sampled current allocator peak by about
  `1524.6 -> 1417.7 MiB`, but driver peak stayed flat at about `2428 MiB`, and
  backward got slower.
- The fixed-render numbers are useful for shader pressure, but full-train is
  dominated by a mix of encode/model timing plus autograd backward. Small
  fixed-render wins do not automatically promote a variant.

## Readiness

- Use `v11_features_gradcache_zero_bg_hostmeta_fixedbin` as the current stable
  F32 feature-splatting baseline.
- Use `v13a_temporal_recompute_state` only as an opt-in memory valve when a run
  is blocked by saved tile state. Expect slower backward and no guaranteed
  driver-memory reduction.
- Do not promote `v13b_rgb_grad_handoff` as a speed/memory win yet. It is ready
  as an importable scaffold and renamed v11-compatible rasterizer, but the
  fused handoff op still intentionally raises until the Metal kernel is written.

## Next

The useful next implementation is the actual v13b kernel:

```text
grad_rgb[B,H,W,3]
+ out_features/out_alpha/colorizer weights/background
-> per-pixel g_feature/g_alpha stream
-> inherited reverse raster contributor loop
```

Promotion gate for that kernel should be:

1. parity against v11 gradients on small MPS fixtures,
2. fixed-render `256/512px` phase benchmark,
3. full-train `256px` probe with sampled current and driver memory,
4. one short W&B-enabled trainer run only after the above are favorable.
