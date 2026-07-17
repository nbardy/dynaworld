# 2026-05-19 12:29:28 - coeff16 site-cache reflection

## What changed

Stopped the current Gate4 shader fork loop after two cache-oriented variants:

- `gate4-affine-candidate-coeff16-framegroup16cached-fused-mse`: threadgroup-cached 16-frame coefficient replay.
- `gate4-affine-candidate-coeff16-sitecache-fused-mse`: sample-parallel coeff16 fused-MSE with per-threadgroup cached site positions and RGBA.

The site-cache fork added a cached owner-scan helper and host dispatch padding so the full threadgroup can fill cache slots before sample-id early return. It preserves the sample-parallel launch shape instead of serializing frames per track.

## Validation

Mechanical gates passed after the site-cache fork:

- Python `py_compile` over the verifier, train/eval harness, comparison script, MPS tests, and Python op wrappers.
- Verifier unit suite: `10/10`.
- Native rebuild of `world_foam_lane2_fused_slab_v0`: passed.
- MPS mixed fused-slab suite: `8/8`; parity now also covers the site-cache loss/gradient path.

## Speed evidence

Framegroup16 cached was clean/background and clearly rejected against the paired sample-parallel control:

```text
frames  sample_total  framegroup_total  ratio   sample_back  framegroup_back  ratio
2       5.888 ms      5.657 ms          0.961   5.261 ms     5.026 ms         0.955
4       4.249 ms      5.427 ms          1.277   3.638 ms     4.803 ms         1.320
8       4.723 ms      6.029 ms          1.277   4.130 ms     5.356 ms         1.297
16      3.719 ms      3.536 ms          0.951   3.174 ms     3.176 ms         1.001
```

Site-cache passed correctness but only has contended timing, so it cannot be promoted. The same-window contended pair was mixed:

```text
frames  sample_total  sitecache_total  ratio   sample_back  sitecache_back  ratio
2       4.412 ms      5.333 ms         1.209   3.704 ms     4.725 ms        1.276
4       5.159 ms      4.460 ms         0.864   4.274 ms     3.842 ms        0.899
8       4.498 ms      4.395 ms         0.977   3.792 ms     3.816 ms        1.006
16      4.562 ms      3.928 ms         0.861   3.920 ms     3.209 ms        0.819
```

Artifacts:

- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_framegroup16cached_scale_2_4_8_16_render16_site24_warm3.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_rerun_for_framegroup16cached.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_sitecache_scale_2_4_8_16_render16_site24_warm3_clean_rerun.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3_for_sitecache_pair.json`

## Reflection

This fork family is not the STAR-style math port we want. It made the existing WorldFoam sample-parallel coeff16 path more elaborate, but it did not remove the thing that scales with frame count.

Framegroup coefficient caching loses because the cache/barrier cost is real and the tiny Gate4 coefficient payload is not expensive enough to justify it. Site-cache may help the larger-frame rows under a noisy run, but the 2f regression and 8f backward tie mean raw site/RGBA table loads are not the decisive bottleneck either.

The practical conclusion is to stop cache-only variants here. The next competitive path should change the representation of owner transitions or candidate replay, not add another threadgroup cache around the same owner scan. STAR UVT is cleaner because its hot path is time-tubed and tile/bin structured; WorldFoam is still doing frame/sample owner work unless we compress the event structure more aggressively.
