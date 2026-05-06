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

Next decisive test:

Build the variant, run a tiny MPS parity/smoke against direct F32 where
`feature_weights @ lookup` reconstructs the original features, then profile
peak memory for `K in {4,8,16}` versus direct `F=32`.
