# World Foam Packed Delta Framegroup Probe

Context: after the i16x4 framegroup fork stayed cadence-sensitive and was not
promotable, I tried a smaller structural fork against the selected
delta-replace coeff16 framegroup path: pack each `(owner,left,right)` endpoint
record into one `int32` instead of the current i16x3 row. The goal was to keep
the same fused-MSE replay/VJP math while making the record stream smaller and
more aligned for Metal.

Files touched:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_metal.mm
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py
research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py
research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
```

The new op is:

```text
endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only
```

It reuses the existing packed record bit layout from the block-coeff packed
fork: owner in 8 bits, left/right cut codes in 12 bits each. That limits this
fork to `site_count <= 256` and `boundary_count <= 4093`, which is fine for the
current small-site framegroup lane but not a general full-trainer claim.

Build and parity:

```text
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay
```

Result: 17 tests OK. The focused delta-framegroup tests now compare scalar
i16x3, selected i16x3 framegroup, packed framegroup, materialized framegroup,
and i16x4 framegroup. Packed matches loss/grad with max grad diff at zero or
float noise in the saved timing probes.

Timing artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_packed_variant_timing_probe_tracks128_prewarm_warm3_steps8_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_packed_variant_timing_probe_tracks128_prewarm_warm8_steps20_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_packed_variant_interleaved_timing_probe_tracks128_prewarm_warm4_steps12_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_packed_variant_interleaved_timing_probe_tracks1024_prewarm_warm3_steps8_64_128.json
```

The timing probe now has `--interleave-variants` so every variant is measured
under the same process cadence instead of timing one whole variant block at a
time. This matters because the earlier i16x4 results and these packed results
both show large order/cadence effects.

Best larger-track read from the 1024-repeat interleaved artifact:

```text
64f i16x3 mean/median:   175.35 / 176.72 ms, storage 184332
64f packed mean/median:  189.86 / 195.42 ms, storage 143372

128f i16x3 mean/median:  190.15 / 206.48 ms, storage 192524
128f packed mean/median: 180.90 / 208.23 ms, storage 151564

packed storage ratio: 0.778x at 64f, 0.787x at 128f
packed 64f->128f mean scale: 0.953x, median scale: 1.066x
```

Interpretation: packed records are correctness-green and a real persistent
storage win, but not a clean speed promotion. At larger track count, packed is
slightly faster by mean at 128f but slower at 64f and essentially tied by
median at 128f. The result is useful as a storage fork and as another signal
that the timing harness/cadence must be controlled, but it does not replace
the selected i16x3 framegroup loss-reduce path.

Train/eval harness smoke:

```text
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 16,32 \
  --render-size 32 \
  --site-count 12 \
  --steps 5 \
  --warmup-steps 3 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --repeat-loaded-frames \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_packed_framegroup16_fused_mse_train_eval_repeat32_warm3_steps5_render32_site12_16_32.json
```

Result: status OK. PSNR is stable across 16f and 32f under the repeated-frame
speed-scaling smoke:

```text
16f final train/heldout PSNR: 12.859 / 13.642
32f final train/heldout PSNR: 12.849 / 13.665

16f total/backward mean: 159.70 / 143.24 ms
32f total/backward mean:  84.32 /  74.24 ms

selected tape storage: 1.263 MB at 16f, 1.273 MB at 32f
selected tape storage vs full: 8.02% at 16f, 4.04% at 32f
```

The train/eval smoke is technically sublinear by the script's acceptance
flags, and storage is almost flat while the full tape doubles. But it is not a
speed promotion over the selected i16x3 path: the prior paired i16x3 artifact
had about 104.21 / 92.40 ms total/backward at 16f and 93.72 / 83.15 ms at 32f.
Packed is slower at 16f and only comparable around 32f on this noisy smoke.

Follow-up paired compare:

```text
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_delta_framegroup_i16x3_packed_train_eval.py \
  --frame-counts 16,32 \
  --render-size 32 \
  --site-count 12 \
  --steps 5 \
  --warmup-steps 3 \
  --repeat-loaded-frames \
  --prewarm-sweep \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_i16x3_packed_train_eval_compare_repeat32_prewarm_warm3_steps5_render32_site12_16_32.json
```

That paired same-process prewarmed artifact changes the read from
"storage-only fork" to "narrow speed candidate":

```text
16f i16x3 total/backward: 109.33 / 96.51 ms
16f packed total/backward: 78.17 / 70.14 ms

32f i16x3 total/backward: 104.90 / 85.54 ms
32f packed total/backward: 75.16 / 68.62 ms

max packed/i16x3 total mean ratio: 0.716
max packed/i16x3 backward mean ratio: 0.802
max packed/i16x3 selected-storage ratio: 0.955
max heldout PSNR delta: 0.0
```

I added a first-class status-summary guardrail for this artifact:

```text
research_experiments/world_foam_lane2/compare_delta_framegroup_i16x3_packed_train_eval.py
research_experiments/world_foam_lane2/test_compare_delta_framegroup_i16x3_packed_train_eval.py
research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py
research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py
research_experiments/world_foam_lane2/test_verify_fused_slab_status_summary.py
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
```

The summary records packed as `packed_speed_promotion_candidate=true` but keeps
`completion_claim=false`, `full_trainer_claim=false`, and
`star_uvt_competitive_claim=false`. This is intentionally not a final default
promotion: the earlier standalone/interleaved probes were cadence-sensitive and
the packed candidate still needs a broader 64/128 or real-loaded guard before
replacing selected i16x3 as the default speed path.

Broader interleaved guard:

```text
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_delta_framegroup_i16x3_packed_train_eval.py \
  --frame-counts 64,128 \
  --render-size 32 \
  --site-count 12 \
  --steps 3 \
  --warmup-steps 1 \
  --repeat-loaded-frames \
  --prewarm-sweep \
  --interleave-modes \
  --out-json research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_i16x3_packed_train_eval_compare_repeat64_128_interleaved_prewarm_warm1_steps3_render32_site12_64_128.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_i16x3_packed_train_eval_compare_repeat64_128_interleaved_prewarm_warm1_steps3_render32_site12_64_128.partial.json
```

I added `--interleave-modes` and `--partial-out-json` to the compare harness so
each frame count can rotate mode order and leave a recoverable partial artifact.

The broader guard rejects packed as a default speed promotion:

```text
64f i16x3 total/backward: 331.62 / 292.68 ms
64f packed total/backward: 203.82 / 165.17 ms

128f i16x3 total/backward: 217.55 / 185.53 ms
128f packed total/backward: 278.33 / 260.32 ms

64f packed/i16x3 total/backward ratio: 0.615 / 0.564
128f packed/i16x3 total/backward ratio: 1.279 / 1.403
max packed/i16x3 selected-storage ratio: 0.952
max heldout PSNR delta: 0.0
```

The status summary now records both facts at once:

```text
framegroup16_packed_prewarm_candidate_recorded=true
framegroup16_packed_broad_nonpromotion_recorded=true
```

Interpretation: packed is a real storage win and a narrow 16/32 speed
candidate, but the 64/128 interleaved guard says it is not fixed as a broad
default. The next useful fork is probably either a 128f-specific packed-kernel
fix or a different broad-frame layout, not simply flipping the default to
packed.

Next decision: keep selected i16x3 as the speed path, keep packed as a
current narrow speed candidate and broad non-promotion artifact. Do not make it
the default until it fixes the 128f regression under a cadence-controlled guard.
The next shader work should target the 128f packed regression or the remaining
full-trainer gap rather than more record packing alone.
