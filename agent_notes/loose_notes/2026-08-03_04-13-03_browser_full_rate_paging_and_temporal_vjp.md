# Browser Full-Rate Paging, Temporal VJP, And Floater Guard

Date: 2026-08-03

## Trigger

The user noticed that Coffee Martini appeared temporally short and that novel
orbit views still had floaty translucent clouds. They asked for the complete
frame rate and duration, bounded nonblocking paging, a paper audit, practical
novel-view improvements, and preserved training throughput.

## What The Data Actually Contains

- Probed source: 300 frames, 30/1 fps, 10.0 seconds for every used camera.
- The prior browser bundle sampled 16 synchronized frames over the complete
  interval.
- Therefore the clip is genuinely only 10 seconds, but the old optimizer saw
  16 times rather than all 300.

## Implementation

### Bundle adapter

Extended `src/train/export_dynaworld_browser_bundle.py` with a versioned
`temporal_stream` description and per-camera video URL/start fields. Kept the
canonical manifest, calibration, train/heldout split, and fallback atlas.

Generated 18 checked-in 384x288 H.264 streams with 300 frames at 30 fps. Exact
logical total is 17,936,497 bytes. Each stream was probed after encode.

### Paging

Added `temporalPagingPlanner.js`. It creates 19 interleaved pages: 18 pages of
16 frames and a final page of 12. The interleaving makes each resident cycle
span the whole timeline while one complete rotation visits every exact index.

Added async page decode in `dataset.js`. The current and prefetched page are
bounded RGBA8 banks. The training worker owns optimization, and page decode
never becomes an awaited operation in its pump. Page replacement validates
camera identity, pose, dimensions, and split, then uses queue ordering rather
than a completion barrier. The validation worker switches to the same page.

The progressive 96 to 384 handoff loads the current temporal page at the new
resolution before restoring trainer state.

### Memory

For 384x288, 18 cameras, 300 frames:

```text
eager RGBA8 target corpus = 2,388,787,200 bytes = 2.225 GiB
two 16-frame RGBA8 pages  =   254,803,968 bytes = 243.000 MiB
FP32 camera means         =    31,850,496 bytes = 30.375 MiB
bounded page + mean total =   286,654,464 bytes = 273.375 MiB
```

The tiled trainer still puts one selected frame on the GPU.

### Temporal VJP bug

During parity review, found that staged projection backward reconstructed the
temporal gate as the dynamic component only. Forward uses
`mix(dynamicGate, 1, staticMix)`. This was a serious mismatch at the default
static-heavy initialization. Rebuilt the exact forward gate and direct dynamic
core in the staged VJP. Added production static-mix fixtures to parity.

Also moved scale LR from color's 10x decay to geometry's 100x decay while
preserving the former initial scale LR.

### Density score and papers

Code inspection showed that the browser already sums `length(barMu)` per
covered pixel before reduction. This is non-cancelling and AbsGS-like, not the
ordinary norm after all pixel gradients cancel. It is not exact AbsGS because
AbsGS sums absolute x/y components separately.

Implemented Pixel-GS near-camera depth scaling only on the density statistic:

```text
clip((cameraDepth / (0.37 * cameraSceneRadius))^2, 0, 1)
```

The optimizer gradient and renderer are unchanged. Added a default-on
`Near-Camera Floater Guard` reset-time toggle so the paper-backed intervention
can be ablated honestly.

## Browser Checks

Live full-rate page:

- observed native source frame 202 and later frame 72;
- crossed progressive resolution at step 10,688;
- continued past 12,480 at 384x288;
- no browser warning/error;
- no tile overflow;
- result remained coherent across the three calibrated cameras.

Live parity:

```text
maximum RGB error        1.1920928955078125e-7
objective absolute error 2.258587120662625e-7
gradient families        9 / 9 active
tile overflow            0
```

## Performance Caveat

The host was not valid for a promoted benchmark:

- 1-minute load per logical CPU: 1.098;
- competing CPU fraction: 0.559;
- Apple GPU: 37%;
- swap occupancy: 0.838 (6.295 / 7.516 GB).

The live SPA showed roughly 433 completed steps/s during one diagnostic window,
but it included contention, paging, progressive preload, UI, and validation.
Do not append it to a baseline table. Retain the existing isolated artifacts:
roughly 1,243-1,260 steps/s at 8K/96, roughly 410 at 30K/96, and the saved
30K/384 artifact around 239.3 steps/s with zero-overflow validity requirements.

## Failed Or Abandoned Work

An attempted full atlas re-export redundantly re-decoded fallback frames and
was taking minutes. It was stopped before writing output. The existing checked
atlases were kept and their structured temporal metadata was updated instead.
This avoided turning the new stream contract into a second expensive asset
pipeline.

The first post-Pixel-GS JS suite had one failure: resource-plan expected totals
were exactly 16 bytes low after the tiled uniform grew from 176 to 192 bytes.
The implementation tests passed. Updated only those two exact accounting
values, then reran the suite.

## Paper Conclusions

Highest-value order:

1. matched Pixel-GS guard on/off;
2. exact AbsGS statistic only if parent selection differs materially;
3. residual plus coarse-depth, multi-view-supported fixed-budget relocation,
   combining the useful allocation ideas from SpacetimeGS and 3DGS-MCMC;
4. complete Mip-Splatting forward and backward for zoom/scale instability;
5. dropout or rigidity only when diagnostics establish sparse support or
   inconsistent local motion.

Softmax splatting remains the wrong compositing model for this renderer.
Native 4DGS and World Tubes remain separate model lanes, not labels for the
current trajectory 3DGS shader.

## Files And Durable Followup

The detailed math, memory derivation, failure branches, paper matrix, and A0-A5
ablation protocol live in:

`research_notes/browser_full_rate_paging_and_novel_view_roadmap_2026-08-03.md`

Next evidence should be the guarded versus unguarded matched quality run on a
quiet host. After that, instrument residual/contribution/depth support before
implementing fixed-budget relocation.
