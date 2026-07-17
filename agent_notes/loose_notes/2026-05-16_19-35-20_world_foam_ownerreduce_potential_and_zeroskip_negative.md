# WorldFoam Owner-Reduce Potential And Zero-Skip Negative

## Context

We were trying to make the current WorldFoam framegroup fused-MSE lane more competitive with STAR UVT by finding another sublinear/reduced-work shader fork. The current selected lane is still:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse
```

The prior best saved full repeat-loaded render32/site12 run remained:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json
```

with 16/32/64/128 total mean ms `3.046 / 3.701 / 3.590 / 4.459`, backward mean ms `2.510 / 3.269 / 3.032 / 3.857`, and selected tape storage scaling only `1.026x` from 16f to 128f.

## Owner-Reduce Topology Probe

Added:

```text
research_experiments/world_foam_lane2/probe_delta_framegroup_owner_reduce_potential.py
```

The probe estimates current framegroup gradient atomics versus an owner-list reduction that would reduce each track/chunk to unique owner ids capped at 16, with fallback to current direct atomics when a chunk exceeds the cap. It handles synthetic delta tapes and sampled real moving-camera train/heldout topology.

Syntax check passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/probe_delta_framegroup_owner_reduce_potential.py
```

Synthetic site64 / tracks128 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_ownerreduce_potential_synthetic_site64_tracks128_16_32_64_128.json
```

Key synthetic result: fallback fraction stayed `0.0`; owner-reduce/current atomic ratio improved from `0.1321` at 16f to `0.0366` at 128f.

Sampled real render32/site12 stride4 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_ownerreduce_potential_real_stride4_render32_site12_16_32_64_128.json
```

Site12 sampled real result:

```text
frame split   ratio   factor  fallback  unique_mean unique_max
16    train   0.4427  2.26x   0.0       5.31        8
16    heldout 0.3490  2.87x   0.0       4.19        8
32    train   0.4434  2.26x   0.0       5.32        8
32    heldout 0.3516  2.84x   0.0       4.22        8
64    train   0.4007  2.50x   0.0       4.81        8
64    heldout 0.3184  3.14x   0.0       3.82        7
128   train   0.3727  2.68x   0.0       4.47        7
128   heldout 0.3057  3.27x   0.0       3.67        7
```

Sampled real render32/site64 stride8 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_ownerreduce_potential_real_stride8_render32_site64_16_32_64_128.json
```

Site64 sampled real result:

```text
frame split   ratio   factor   fallback  unique_mean unique_max
16    train   0.0876  11.41x   0.0       10.69       14
16    heldout 0.1015  9.85x    0.0       9.00        12
32    train   0.0437  22.88x   0.0       10.66       13
32    heldout 0.0503  19.89x   0.0       8.94        12
64    train   0.0377  26.49x   0.0       9.23        12
64    heldout 0.0426  23.45x   0.0       7.59        10
128   train   0.0345  28.98x   0.0       8.44        12
128   heldout 0.0367  27.25x   0.0       6.53        10
```

The full real render32 site12/site64 unsampled probes were killed because Python topology enumeration was too slow. The script now supports `--sequence-spatial-stride` and `--progress`; future full probes should either use sampling first or write partial rows incrementally.

## Zero-Skip Shader Shortcut Negative

Tried a low-risk shortcut in the live i16x3 framegroup Metal shader: after threadgroup reduction for small `site_count <= 16`, skip the final `wf2_atomic_add4` when the reduced `float4 grad_sum` is exactly zero.

Focused parity passed after the edit:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk -v
```

But the train/eval timing gate was a clear negative. Partial artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_zeroskip_fused_mse_short_warm3_steps5_render32_site12_16_32_64_128.partial.json
```

Partial rows:

```text
16f total/backward mean ms: 4.713 / 3.965
32f total/backward mean ms: 389.371 / 353.084
```

The sweep was killed before 64/128 because 32f was catastrophically slower. The shader edit was reverted, the extension was rebuilt, and focused parity passed again. A search confirmed the zero-skip condition and the earlier compact-state `trans_before_run` experiment are absent from the live Metal source.

## Next Best Shader Fork

Do not retry the zero-skip shortcut. The measured useful direction is a real owner-list fork:

- producer builds `track_chunk_owner_offsets_i32` and flattened `track_chunk_owner_i16`
- Metal kernel loads up to 16 unique owner ids for each track/chunk
- reverse pass accumulates into `tg_owner_grad[frame, owner_slot]`
- final flush emits one global `wf2_atomic_add4` per unique owner slot
- if owner count exceeds 16, fallback to the current direct-atomic path

This should target the site64 path especially strongly, and should also reduce site12 final atomics without the branch pathology of the zero-skip shortcut.
