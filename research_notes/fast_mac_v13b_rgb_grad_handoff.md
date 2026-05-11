# fast-mac v13b RGB-gradient handoff

Date: 2026-05-10

## Scope

Worker 2 created an isolated fast-mac variant:

```text
third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff/
```

The fork was copied from:

```text
third_party/fast-mac-gsplat/variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin/
```

The worker did not edit shared trainer files, v11 files, or v12 files. A
main-thread follow-up later wired the renamed v11-compatible v13b raster API
into shared fast-mac dispatch as an opt-in `feature_variant`.

## What changed

- Renamed the Python package to `torch_gsplat_bridge_v13b_rgb_grad_handoff`.
- Renamed the custom op namespace to `torch.ops.gsplat_metal_v13b_rgb_grad_handoff`.
- Renamed the Metal kernel source to
  `csrc/metal/gsplat_v13b_rgb_grad_handoff_kernels.metal`.
- Kept the inherited v11 rasterizer runnable and behavior-compatible.
- Added `rgb_grad_handoff_backward(...)` in
  `torch_gsplat_bridge_v13b_rgb_grad_handoff/rgb_grad_handoff.py`.
- Registered a scaffold C++ op:
  `render_fast_backward_rgb_grad_handoff(...)`.
- Added `estimate_rgb_grad_handoff_memory(...)` and
  `benchmarks/rgb_grad_handoff_accounting.py`.

The handoff op is only a boundary today. It intentionally raises with a clear
message because the fused Metal kernel has not been written.

## Intended kernel boundary

Current v11-compatible backward consumes dense:

```text
grad_features[B,H,W,F]
grad_alpha[B,H,W]
```

The v13b target consumes:

```text
out_features[B,H,W,F]
out_alpha[B,H,W]
grad_composed_rgb[B,H,W,3]
background_rgb[B,H,W,3]
color_weight[3,F]
color_bias[3]
```

Then the missing Metal kernel should do, per pixel:

```text
logits[c] = color_bias[c] + sum_f color_weight[c,f] * out_features[f]
rgb[c] = sigmoid(logits[c])
g_alpha = sum_c grad_composed_rgb[c] * (rgb[c] - background_rgb[c])
g_feature[f] = sum_c grad_composed_rgb[c] * out_alpha * rgb[c] * (1 - rgb[c]) * color_weight[c,f]
optional g_weight[c,f] += grad_composed_rgb[c] * out_alpha * rgb[c] * (1 - rgb[c]) * out_features[f]
optional g_bias[c] += grad_composed_rgb[c] * out_alpha * rgb[c] * (1 - rgb[c])
stream g_feature/g_alpha into the inherited reverse raster contributor loop
```

This is the useful split between v12a and v12c:

- v12a proves image-space colorize/loss gradient production can be fast, but it
  still returns dense `grad_features`.
- v12c proves the full raster/color/loss fusion shape, but it scalarizes too
  much work inside raster backward and is not a good promotion candidate.
- v13b keeps the raster loop close to v11 and only moves RGB-to-feature VJP
  into the raster backward entrypoint.

## Bandwidth accounting

For `B=16,H=256,W=256,F=32,float32`:

```text
current grad_features: 128 MiB
current grad_alpha: 4 MiB
current dense backward input total: 132 MiB
handoff grad_rgb: 12 MiB
avoided dense input: 120 MiB
avoided fraction: 90.9%
```

This does not count allocator pressure, cache behavior, or the fact that
`out_features[B,H,W,F]` still exists from forward-state rasterization. The
prototype specifically targets dense feature-gradient materialization.

## Commands and results

Build:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Result: success. Built
`torch_gsplat_bridge_v13b_rgb_grad_handoff/_C.cpython-311-darwin.so`.

Python compile:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  python -m py_compile \
  third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff/torch_gsplat_bridge_v13b_rgb_grad_handoff/__init__.py \
  third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff/torch_gsplat_bridge_v13b_rgb_grad_handoff/rasterize.py \
  third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff/torch_gsplat_bridge_v13b_rgb_grad_handoff/rgb_grad_handoff.py \
  third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff/benchmarks/rgb_grad_handoff_accounting.py
```

Result: success.

Accounting:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  python third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff/benchmarks/rgb_grad_handoff_accounting.py \
  --batch 16 --height 256 --width 256 --feature-dim 32
```

Result:

```json
{
  "current_dense_backward_input_mib": 132.0,
  "handoff_dense_backward_input_mib": 12.0,
  "avoided_mib": 120.0,
  "avoided_fraction": 0.9090909090909091
}
```

Tiny MPS raster smoke:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python - <<'PY'
import torch
from torch_gsplat_bridge_v13b_rgb_grad_handoff import RasterConfig, rasterize_projected_gaussians

device = torch.device("mps")
B, G, H, W, F = 1, 16, 16, 16, 4
torch.manual_seed(123)
means2d = torch.rand(B, G, 2, device=device)
means2d[..., 0] *= W
means2d[..., 1] *= H
sig = torch.rand(B, G, 2, device=device) * 2.0 + 1.0
conics = torch.stack((1.0 / (sig[..., 0] ** 2), torch.zeros(B, G, device=device), 1.0 / (sig[..., 1] ** 2)), dim=-1)
colors = torch.rand(B, G, F, device=device, requires_grad=True)
opacities = (torch.rand(B, G, device=device) * 0.5 + 0.2).requires_grad_(True)
depths = torch.rand(B, G, device=device)
cfg = RasterConfig(height=H, width=W, tile_size=16, max_fast_pairs=2048, enable_overflow_fallback=False)
out, alpha = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
(out.square().mean() + alpha.square().mean()).backward()
print(tuple(out.shape), tuple(alpha.shape), colors.grad is not None, opacities.grad is not None)
PY
```

Result: success. Output was `(1, 16, 16, 4) (1, 16, 16) True True`.

Handoff scaffold smoke:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python - <<'PY'
import torch
from torch_gsplat_bridge_v13b_rgb_grad_handoff import rgb_grad_handoff_backward

# Minimal CPU tensors only to confirm the op is registered and fails at the
# intended boundary.
try:
    rgb_grad_handoff_backward(
        torch.zeros(1,2,2,4), torch.zeros(1,2,2), torch.zeros(1,2,2,3),
        torch.zeros(1,2,2,3), torch.zeros(3,4), torch.zeros(3),
        torch.zeros(1,2), torch.zeros(1,3), torch.zeros(1,4), torch.zeros(1),
        torch.zeros(15, dtype=torch.int32), torch.zeros(4),
        torch.zeros(15, dtype=torch.int32), torch.zeros(4),
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32), torch.zeros(1,2,2, dtype=torch.int32),
    )
except RuntimeError as exc:
    print(str(exc).split("\n")[0])
PY
```

Result: success. It printed the intended scaffold error:

```text
gsplat_metal_v13b_rgb_grad_handoff.render_fast_backward_rgb_grad_handoff is an API scaffold only; the Metal kernel that streams RGB-gradient colorizer VJP into raster backward is not implemented yet.
```

## Status

`v13b_rgb_grad_handoff` is runnable as a renamed v11-compatible variant.

It is not a true fused RGB-gradient handoff implementation yet. The exact
missing work is the Metal implementation behind
`render_fast_backward_rgb_grad_handoff(...)`, plus parity tests against the
unfused PyTorch colorizer/autograd path and timing against v11/v12a/v12c on the
same target shapes.

## Shared Renderer Integration

Main-thread follow-up wired the v11-compatible v13b raster API into
`src/train/renderers/fast_mac.py` as:

```json
"fast_mac": {
  "feature_variant": "v13b_rgb_grad_handoff"
}
```

This does not enable the missing fused RGB-gradient handoff kernel. It only
makes the isolated renamed variant selectable for trainer-path smoke/parity.

Shared dispatch smoke:

```bash
GSP_FAST_CAP=4096 GSP_FEATURE_CAP=64 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train \
  .venv/bin/python <inline shared-dispatch smoke>
```

Result:

```text
v13b_rgb_grad_handoff save (1, 32, 32, 32) (1, 32, 32) 0.7124469876289368
```
