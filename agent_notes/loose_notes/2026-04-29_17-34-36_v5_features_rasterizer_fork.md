# v5_features rasterizer fork

Date: 2026-04-29 17:34

Task: fork the active fast-mac v5 renderer into a feature-channel rasterizer
without touching the original v5 baseline path.

What changed:

- Added `third_party/fast-mac-gsplat/variants/v5_features/`.
- Renamed Python package to `torch_gsplat_bridge_v5_features`.
- Renamed custom op namespace to `torch.ops.gsplat_metal_v5_features`.
- Renamed Metal entry symbols with `v5_features_...` prefixes.
- Generalized `colors` from `[*,3]` to `[*,F]`, with `F` inferred at runtime.
- Added `GSP_FEATURE_CAP` runtime config, default `64`.
- Changed `RasterConfig.background` to accept one broadcast scalar or exactly
  `F` values.
- Added `tests/feature_contract_check.py`.
- Added `--feature-dim` to the fork's `benchmarks/benchmark_mps.py`.

Important implementation note:

- The fork uses a single generic feature-channel path. It does not keep a
  separate RGB branch. Feature channels stream from device memory during
  compositing and feature-gradient accumulation instead of being staged into
  threadgroup memory. This avoids scaling threadgroup memory by `F`.
- Original `variants/v5` remains untouched. `git -C third_party/fast-mac-gsplat
  diff --name-only -- variants/v5` was empty after implementation.

Validation:

```text
python3 setup.py build_ext --inplace
python3 tests/feature_contract_check.py
python3 tests/reference_check.py
python3 -m py_compile tests/feature_contract_check.py benchmarks/benchmark_mps.py torch_gsplat_bridge_v5_features/rasterize.py
```

Feature contract output:

```text
shape contract: ok
F=3 v5 parity max_abs=0
F=3 feature grad max_abs=2.7939677e-09
F=8 feature grad max_abs=1.8626451e-09
F=32 feature grad max_abs=4.6566129e-10
F=32 no-NaN smoke: ok
```

Small throughput sweep:

```text
height=256 width=256 gaussians=2048 case=medium_sigma_3_8 warmup=3 iters=10
F=3  forward 5.223 ms, fwd+bwd 17.067 ms
F=8  forward 5.233 ms, fwd+bwd 16.322 ms
F=32 forward 8.992 ms, fwd+bwd 36.047 ms
```

How to use:

```python
from torch_gsplat_bridge_v5_features import RasterConfig, rasterize_projected_gaussians

features = rasterize_projected_gaussians(
    means2d,
    conics,
    splat_features,  # [B, G, F] or [G, F]
    opacities,
    depths,
    RasterConfig(height=H, width=W, background=(0.0,)),
)
```

Use original v5 for RGB-only baseline training. Use v5_features when the model
needs an F-channel raster map before a colorizer.
