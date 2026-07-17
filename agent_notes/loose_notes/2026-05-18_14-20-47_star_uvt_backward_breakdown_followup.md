# STAR UVT Backward Breakdown Follow-Up

Date: 2026-05-18
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

## Question

The user asked whether we ever broke STAR UVT backward down, because the thesis
was that UVT STAR shares work across frames and should be sublinear/fast.

## Answer

We had aggregate `loss.backward()` timings for the promoted direct-atomic branch
and design notes for the deterministic compact branch, but the May 17 audit did
not save a full direct-atomic microbreakdown separating:

- direct Metal backward kernel
- reducer cost
- projection/model VJP back into trainable tube parameters

The saved May 17 shader audit only split STAR UVT train steps into forward and
total backward. It showed that `512px / 64f / 32768 tubes` direct atomic was
still backward-bound:

```text
total:   1288.7ms
forward: 139.9ms
backward:1096.2ms
```

## Fresh Direct-Atomic Probe

Ran the existing direct backward probe:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_backward_breakdown_probe.py \
  data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4 \
  --target-size 512 \
  --max-frames 64 \
  --tube-count 32768 \
  --seed 5 \
  --spatial-precision 0.125 \
  --temporal-precision 2.0 \
  --opacity 0.7 \
  --uvt-tile-t 1 \
  --uvt-tile-capacity 256 \
  --uvt-sample-emission-mode direct_atomic \
  --uvt-reduction-mode index_add \
  --warmup-iterations 1 \
  --iterations 3 \
  --out-json outputs/benchmarks/2026-05-18_backward_breakdown/star_uvt_512_64f_32768_directatomic_breakdown_warm3.json
```

Result:

```text
sample_unit: direct_tube_grad
sample_count: 32768
allocated_sample_slot_count: 32768
compact_sample_fraction: 1.0
sample_backward_ms median: 485.5ms
reduce_bundle_ms median: 0.005ms
sample_plus_reduce_median_ms: 485.5ms
unstable_tile_fraction: 0.0
```

Cold one-iteration comparison points:

```text
256px / 16f / 32768: direct kernel 139.2ms
256px / 64f / 32768: direct kernel 196.8ms
512px / 16f / 32768: direct kernel 206.9ms
512px / 64f / 32768: direct kernel 947.1ms cold, 485.5ms warmed median
```

## Interpretation

The direct-atomic branch is compact in gradient output: it returns one gradient
record per tube, not a giant per-pixel sample table, so the reducer is gone.
That is the direct-atomic speed valve.

But compact output is not the same as frame-independent work. The direct kernel
still has to traverse the covered pixels/tiles/frames to accumulate each tube's
gradient. At `512px / 64f`, that direct Metal backward kernel is hundreds of
milliseconds by itself, and the full trainer `loss.backward()` is slower because
it also includes the projection/model VJP from projected screen-time tube
parameters back into trainable model parameters.

So the correct statement is:

```text
STAR UVT forward has sparse/sublinear evidence.
Direct-atomic STAR UVT backward is compact and much faster than the exact
per-pixel sample/reduce branches, but it is still resolution/frame-covering
work, not solved O(tube_count) training backward.
The promoted direct branch is fast enough for source-view overfit, but high-res
64f training remains backward-bound.
```

The deterministic compact branch is even less solved: earlier notes found that
changing `tile_t` reduced forward tile/tube pairs without proportionally reducing
the current backward sample table. The desired next kernel is still a compact
backward whose work unit matches active UVT tile/tube pairs without exploding
into per-pixel rows and without nondeterministic float atomics.
