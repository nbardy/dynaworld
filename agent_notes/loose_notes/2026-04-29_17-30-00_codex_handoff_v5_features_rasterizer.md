# Codex Handoff: v5 → v5_features (arbitrary feature-channel rasterizer)

Date: 2026-04-29
Audience: Codex (or another agent better suited to Metal shader work)

## TL;DR

Fork the **v5** fast-mac-gsplat rasterizer to a new variant **v5_features** that
supports rasterizing splats with **arbitrary feature channel count `F`** (not
hardcoded to 3-channel RGB). `F` should be a runtime config value, not a
compile-time constant. The dynaworld trainer will then drive raster with
`F=32` (default) and use a 1×1 MLP to colorize the rasterized feature map back
to RGB.

The companion plan doc is
`agent_notes/loose_notes/2026-04-29_17-30-00_feature_splatting_plan.md` — read
that first if you want context on why.

## Currently active rasterizer

Dynaworld's training path uses **v5**:

- Variant directory: `third_party/fast-mac-gsplat/variants/v5/`
- Python bridge: `third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5/rasterize.py`
- Metal kernels: `third_party/fast-mac-gsplat/variants/v5/csrc/metal/gsplat_v5_kernels.metal`
- Bindings: `third_party/fast-mac-gsplat/variants/v5/csrc/bindings.cpp`
- Custom-op namespace: `torch.ops.gsplat_metal_v5.*`
- Wrapper in dynaworld: `src/train/renderers/fast_mac.py`
  (`_ensure_fast_mac_v5_on_path`, `_make_v5_config`, `render_fast_mac_3dgs`,
  `render_fast_mac_3dgs_batch`)

This is confirmed in the README of v5
(`third_party/fast-mac-gsplat/variants/v5/README.md`) and in
`agent_notes/loose_notes/2026-04-22_14-48-34_fast_mac_65k_batch_size_probe.md`.

**Do not** touch v6 / v7 / v8 / v9 forks. They are not on the active training
path, and changes there will not flow through.

## What "fork" means here

Create a new sibling directory:

```
third_party/fast-mac-gsplat/variants/v5_features/
```

Copy the contents of `v5/` into it, rename:

- The Python package: `torch_gsplat_bridge_v5` → `torch_gsplat_bridge_v5_features`
- The custom-op namespace: `torch.ops.gsplat_metal_v5` → `torch.ops.gsplat_metal_v5_features`
- The Metal kernel function names: keep them under a clearly distinct prefix to
  avoid Metal symbol collisions when both libraries are imported in the same
  process during testing.

The point of forking instead of editing v5 in place: v5 is what the current
baseline (`s6xnvoch`) was trained on, and we want it left untouched so we can
do bit-for-bit parity comparisons.

## What needs to change

### 1. Replace 3-channel hardcoding with runtime `F`

The 3-channel assumption is spread across both Python and Metal layers.
Concrete sites in v5 to generalize (line numbers from current HEAD; verify
when forking):

**Python bridge** (`torch_gsplat_bridge_v5/rasterize.py`):

- `:179` — explicit shape check `colors must have last dim = 3`. Generalize to
  `colors.shape[-1] == F` where `F` is read from the input tensor.
- `:359` — `colors_s.reshape(B * G, 3)`. Replace `3` with the inferred `F`.
- `:494` and `:499` — `g_colors_b = g_colors_flat.view(B, G, 3)`. Same.
- All gather/scatter/permute helpers that touch the color tensor: must work for
  arbitrary trailing dim.

**Metal kernels** (`csrc/metal/gsplat_v5_kernels.metal`):

- `:48` `inline float3 load3_sh(...)` — replace with a generalized
  `load_features_sh(base, idx, F)` that loads `F` floats. Or template/specialize
  on `F` if we want to keep the fast path for small `F`.
- `:53` `inline void atomic_add3(...)` — generalize to atomic-add over `F`
  channels.
- `:185–187` `sh_colors[b3 + 0u..2u] = colors[g3 + 0u..2u]` — loop over `F`.
- All threadgroup memory allocations that size the per-splat color buffer
  by 3: size by `F` instead.
- Atomic gradient accumulation in backward: also `F`-channel.

**Bindings** (`csrc/bindings.cpp`):

- The op signature accepts `colors` as a generic tensor. Don't hardcode the
  channel count here. Pass `F` (extracted from `colors.size(-1)`) to the
  kernel launch as part of the metadata buffer.

### 2. Threadgroup memory budget

The v5 hot path stages splats into threadgroup memory. With `F=32` instead of
`F=3`, per-splat threadgroup memory rises ~10x. You may need to:

- Reduce the number of splats staged per threadgroup (smaller `chunk` size), or
- Add a "small `F` fast path" branch that keeps the v5 layout for `F <= 4`, and
  a "large `F` slow path" that streams features from device memory for larger
  `F`.

Either is acceptable. Document the trade-off in a `v5_features/README.md`.

### 3. Output tensor shape

- Single-image path: returns `[H, W, F]` (was `[H, W, 3]`).
- Batched path: returns `[B, H, W, F]` (was `[B, H, W, 3]`).
- The dynaworld wrapper currently does `.permute(2, 0, 1)` and `.permute(0, 3, 1, 2)`
  to land in `[*, C, H, W]` for downstream PyTorch convs. That permute still
  works for arbitrary `C`, so no change needed in the wrapper for shape — only
  the variable name should move from `rgbs` to `features` for clarity.
- Also drop the `.clamp(0.0, 1.0)` in the dynaworld wrapper for the
  feature-channel path (features are unbounded reals, not RGB). Only clamp
  when the caller specifically asks for an RGB output. Plan-doc step #2
  covers this.

### 4. Backward must produce `dL/dfeatures` of shape `[B, G, F]`

- Same shape contract on input and gradient. The atomic-accumulation pattern
  in the backward kernel needs to generalize to `F` channels.
- Spot check the gradient on a tiny (`H=8, W=8, G=16, F=8`) random fixture
  against a CPU-Torch reference (compute the alpha-blended sum in pure
  PyTorch and `torch.autograd.grad` against it).

## API surface (target shape)

Match v5 exactly so the dynaworld wrapper change is one path swap:

```python
from torch_gsplat_bridge_v5_features import (
    RasterConfig,
    rasterize_projected_gaussians,
)
```

`rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, config)`:

- `colors` is `[B, G, F]` (batched) or `[G, F]` (single) for **any positive `F`**.
- All other inputs are unchanged from v5.
- Returns `[B, H, W, F]` or `[H, W, F]`. **No clamping inside the kernel.**
- `RasterConfig.background` length must match `F`. (Currently hardcoded to a
  3-tuple; generalize to a tuple/list of `F` floats. Default to zeros for
  the feature path; the dynaworld wrapper will pass `[1.0]*F` if it wants
  white-background semantics.)

## Required tests before handoff back

Land at least these in `v5_features/tests/`:

1. **Shape contract**: `F ∈ {1, 3, 4, 8, 16, 32, 64}` — forward produces the
   expected output shape for both single and batched paths.
2. **F=3 parity vs v5**: with identical inputs and an identical `RasterConfig`,
   `v5_features` at `F=3` produces output that matches `v5` to **`max abs ≤ 1e-6`**.
   This is the most important test. If it fails, the kernel generalization is
   broken in ways that will silently corrupt training.
3. **Gradient correctness**: forward + backward against a tiny pure-PyTorch
   reference (alpha-blend in fp64 on CPU). Per-element max abs ≤ 1e-4 on
   `dL/dfeatures` for `F ∈ {3, 8, 32}`.
4. **No-NaN smoke**: 100 random `F=32` forward+backward iterations in fp32 on
   MPS. No NaN anywhere in output or grads.
5. **Throughput note**: log forward and forward+backward ms for the
   benchmark sweep already in `v5/benchmarks/benchmark_mps.py`, at `F ∈ {3, 8,
   32}`. We don't need a regression target here; we just want to know how the
   curve shapes.

## Build / install

The v5 build is `python setup.py build_ext --inplace` from
`third_party/fast-mac-gsplat/variants/v5/`. Mirror that for `v5_features`.
The dynaworld project uses `uv`; the build is invoked manually (no
`pyproject.toml` integration). Document the build command in
`v5_features/README.md`.

## Out-of-scope (explicitly do not do)

- Don't change v5. Leave it alone for parity comparisons.
- Don't introduce new projection math. The 3D-to-2D projection (in the
  `renderers/projection.py` and the v8/v9 forks) is independent of channel
  count and not part of this work.
- Don't change the `depths` / sort contract. v5's
  `inputs_sorted_by_depth=True` semantics carry over unchanged.
- Don't add a separate "alpha" channel concept. Opacity stays a scalar per
  splat. Only `colors` becomes `features` of arbitrary `F`.

## Hand back when

- All five tests above pass on Apple Silicon MPS.
- A short note appended to this doc with: `F=3` parity result, `F=32`
  throughput numbers, and any threadgroup-memory or kernel-launch changes you
  had to make. We need that detail to know whether to expect the dynaworld
  trainer's per-step time to grow proportionally with `F`.

## Codex handback: 2026-04-29 17:34

Implemented the isolated fork at:

```text
third_party/fast-mac-gsplat/variants/v5_features/
```

Scope check:

- `variants/v5` was not edited.
- `git -C third_party/fast-mac-gsplat status --short -- variants/v5 variants/v5_features`
  reports only `?? variants/v5_features/`.
- `git -C third_party/fast-mac-gsplat diff --name-only -- variants/v5` is empty.

Package / op namespace:

- Python package: `torch_gsplat_bridge_v5_features`
- Custom op namespace: `torch.ops.gsplat_metal_v5_features`
- Metal symbols are prefixed with `v5_features_...`

Feature-channel behavior:

- `colors` is now `[G,F]` or `[B,G,F]`.
- Output is `[H,W,F]` or `[B,H,W,F]`.
- `F` is inferred at runtime from `colors.shape[-1]`.
- `RasterConfig.background` accepts one scalar to broadcast across channels, or
  exactly `F` values.
- Default `GSP_FEATURE_CAP=64`; set it before import for larger feature dims.

Kernel implementation note:

- The fork uses one generic feature-channel path, not an RGB-special branch.
- Feature channels are streamed from device memory during compositing and
  feature-gradient accumulation. They are not staged in threadgroup memory,
  avoiding the `GSP_CHUNK * F` memory expansion at `F=32`.
- Geometry/conic/opacities still use the v5 tile/depth/binning contract.

Verification run on Apple Silicon MPS:

```text
python3 setup.py build_ext --inplace
python3 tests/feature_contract_check.py
python3 tests/reference_check.py
python3 -m py_compile tests/feature_contract_check.py benchmarks/benchmark_mps.py torch_gsplat_bridge_v5_features/rasterize.py
```

Key results:

```text
shape contract: ok
F=3 v5 parity max_abs=0
F=3 feature grad max_abs=2.7939677e-09
F=8 feature grad max_abs=1.8626451e-09
F=32 feature grad max_abs=4.6566129e-10
F=32 no-NaN smoke: ok
```

Copied v5 reference check also passed. Largest reported max error was
`2.086162567138672e-07` on the saturated image case.

Throughput note, small fixed sweep:

Command shape:

```text
python3 benchmarks/benchmark_mps.py --height 256 --width 256 --gaussians 2048 --case medium_sigma_3_8 --feature-dim F --warmup 3 --iters 10 [--backward] --json
```

Results:

| F | forward mean ms | backward mean ms | total mean ms |
|---|----------------:|-----------------:|--------------:|
| 3 | 5.223 | 0.000 | 5.223 |
| 8 | 5.233 | 0.000 | 5.233 |
| 32 | 8.992 | 0.000 | 8.992 |
| 3 | 10.113 | 6.954 | 17.067 |
| 8 | 7.386 | 8.936 | 16.322 |
| 32 | 12.025 | 24.022 | 36.047 |

Use v5 for RGB baseline runs. Use v5_features for feature raster maps before
the model-side colorizer.
