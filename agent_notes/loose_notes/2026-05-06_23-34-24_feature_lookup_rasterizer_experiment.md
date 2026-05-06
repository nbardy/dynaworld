# Feature Lookup Rasterizer Experiment

Created an isolated prototype fork:

```text
third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/
```

The starting point was `v6_refined_features`, because that fork already has the
F-channel Metal path, accumulated alpha, active-tile scheduling, and backward
contract needed to evaluate whether the F32 path is memory-bound.

Feasibility read:

- A compact-basis path is feasible without changing compositing math: splat K
  compact coefficients with the existing dense-channel kernel, then reconstruct
  full F features with `compact @ lookup`.
- This should reduce rasterizer output/intermediate/backward-channel pressure
  when K is much smaller than F. It still materializes the final `[H,W,F]`
  tensor after lookup, so the trainer path must measure where peak memory lands.
- A true sparse ID/weight Metal path is deeper than a small fork. The existing
  kernel assumes dense `colors[BG,F]` in forward and dense `g_colors[BG,F]` in
  backward. Sparse IDs require new op signatures, new Metal loops over
  `feature_ids[BG,L]`, and an explicit lookup-gradient contract.

Prototype status:

- New namespace: `torch_gsplat_bridge_v6_feature_lookup_experiment`.
- New custom-op namespace: `torch.ops.gsplat_metal_v6_feature_lookup_experiment`.
- Added `rasterize_projected_gaussians_feature_lookup(...)`, which renders K
  compact channels with zero compact background and reconstructs F features with
  an output-space background tail.
- Added `rasterize_projected_gaussians_feature_ids(...)`, but it currently
  densifies IDs to `[G,K]` / `[B,G,K]` before rasterization. This is an API
  skeleton, not the true sparse-kernel memory win.

Validation update:

The fork now builds and has a tiny MPS parity check:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/tests/feature_lookup_parity_check.py
```

Observed:

```text
features max_abs=8.9406967e-08
alpha max_abs=0
loss max_abs=0
grad_means max_abs=5.8207661e-11
grad_conics max_abs=1.4901161e-08
grad_weights max_abs=6.0535967e-09
grad_lookup max_abs=2.5611371e-09
grad_opacities max_abs=1.8626451e-09
feature lookup direct parity: ok
id_skeleton feature max_abs=0
id_skeleton alpha max_abs=0
feature id skeleton parity: ok
```

Interpretation:

- The compact-basis math is correct for the low-rank case where direct full
  features equal `feature_weights @ lookup`.
- Gradients through the compact rasterizer plus post-raster lookup match the
  direct full-feature rasterizer with a Torch matmul in front.
- This is still not the true sparse-ID memory win, because the ID helper
  densifies to `[G,K]` before rasterization and the final `[H,W,F]` feature
  image still exists after reconstruction.

Next decisive test:

Profile peak memory and time for `K in {4,8,16}` versus direct `F=32` on a
bounded shape before doing any trainer integration. If K does not lower peak
memory or wall time, stop this branch and keep it as a math/API note only.
