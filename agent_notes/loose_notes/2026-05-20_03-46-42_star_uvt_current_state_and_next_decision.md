# STAR UVT Current State And Next Decision

Date: 2026-05-20 03:46 +07

## Original Goal

The active goal was:

> Repeat and harden the STAR UVT fast feature-shader plan docs, fill any missing
> implementation details, then execute the plan gate by gate with benchmarks and
> progress logs recorded in markdown.

The broader thread also asked whether STAR UVT has the fastest practical UV
STAR/UVT route with cached V-JEPA targets, how it compares against dynamic
gsplat at matched `64f/512px/8192` scale, whether backward/data-loading were
real bottlenecks, what should be ported from feature splatting, whether the
result is overfitting, and whether to continue, reset the goal, or wrap up.

## Recorded State

The key learnings are recorded. `agent_notes/key_learnings.md` is still at the
199-line cap and now compresses the important lessons:

- direct atomic/index-add remains the practical STAR UVT path;
- cached V-JEPA target-grid and frozen-probe routing are real and fast;
- feature-tube backward bottlenecks moved from generic rasterizer guesses into
  specific renderer/probe/colorizer/support slices depending on the route;
- compact target-area / native hidden / vec4 `W^T` work produced speed wins but
  not a dense visual-quality promotion;
- same-support alpha/grid/opacity pressure does not solve dense visibility;
- sparse-F1 alpha rendering is the keeper implementation trick for alpha-only
  diagnostics, not a quality fix;
- matched fixed-512 dynamic gsplat is slower and lower quality than the current
  STAR UVT diagnostic route, so it is not the escape hatch.

The detailed chronology is also recorded in dated loose notes and benchmark
reports. Current routing pointers live in `TODO/README.md`, `PROJECT_INDEX.md`,
`EXPERIMENTS.md`, `BASELINES.md`, and
`research_experiments/star_uvt_feature_tubes/README.md`.

## What Was Actually Accomplished

Real progress was made on measurement and implementation:

- STAR UVT now has a first-class cached V-JEPA target-grid/frozen-probe route,
  not just an RGB-only STAR route.
- The backward question was decomposed. In the earlier RGB-target feature path,
  `FeatureToColor`/loss VJP dominated; in the later frozen-probe/target-grid
  path the renderer side became dominant; in dense visual support gates the
  hidden64 `W^T` feature-gradient work and colorizer atomics were isolated.
- `direct_atomic + index_add` remains the practical STAR route. Deterministic
  compact backward is still not the promoted path.
- Sparse-F1 all-pixel alpha rendering reuses the sparse-pixel Metal path with
  cached bins, keeps alpha/gradient parity, and cuts dense-alpha trainer
  step/backward from `2558.6/1114.2ms` to `873.3/370.0ms`.
- Dynamic gsplat at the same fixed `64f/512px/8192` scale got a real local
  media comparator: mean timed step/backward `2940.1/1926.1ms`, final eval
  PSNR/SSIM/L1 `5.587/0.165/0.469`, and smeared media. It is not currently a
  better quality or speed path.
- WorldFoam has a clean separate micro-gate: native-cutwalk render64/site24
  fused-MSE beats matched STAR timing at `2/4/8/16f` under strict background
  checks, but it is only a fused-MSE speed row, not broad RGB-quality parity.

This is meaningful progress, but it is mostly a negative/diagnostic progress
cycle. We learned what not to scale and which implementation tricks are real.

## Current State

The STAR UVT shader diagnostic phase is closeable. The docs, loose notes,
benchmark reports, config rows, and `BASELINES.md` entries are current enough
for another agent to resume without rediscovery.

The full active goal is not fully complete if interpreted as the entire broad
thread plan. These remain open:

- no 300-video STAR UVT scale run was launched;
- no longer/final dynamic-gsplat ranking run was launched beyond the fixed-512
  smoke and 20-step media comparator;
- feature-splatting lessons were ported into sparse-F1 alpha and target-area
  support diagnostics, but a true world-space UVT feature-tube lane was not
  implemented;
- WorldFoam was kept separate and has a clean current micro-gate, but no new
  no-GPU side investigation was launched in this continuation.

The important distinction: current STAR UVT feature tubes are screen/projected
UVT feature tubes with cached V-JEPA targets and visual probes. They are not
yet the full "world feature tube" compiler imagined in the older STAR notes.

Update after the next continuation: a CPU-first visibility bridge mechanism is
now started. `visibility_support_bridge_prototype.py` proves that a soft
projected-tube coverage proxy can send center/velocity gradients from target
pixels that begin with zero alpha support, raising target alpha `>0.10`
coverage to `0.324` in the toy gate while same-support dense alpha stays at
`0.0`. That does not close visual quality; it narrows the next implementation
step to trainer integration plus dense media validation.

## Future Experiments Worth Running

Do not scale current STAR UVT to the 300-video set yet. The single-video dense
media gate is still sparse/streaked or blurry, and scaling that failure would
only produce a larger expensive failure.

Worth doing next:

1. Implement a support-changing visibility/model bridge for STAR UVT and rerun
   the selected dense media gate. Success means dense full RGB leaves the
   current `5.6-6.0` PSNR band without giving up feature/probe quality. The
   CPU proxy mechanism now passes; the missing part is first-class trainer
   integration and real single-video media evidence.
2. Turn the world-feature-tube idea into a CPU/PyTorch parity gate before any
   new Metal work: compile world tube parameters to projected UVT parameters,
   backprop projected UVT gradients to world parameters, and prove parity on a
   tiny same-view clip.
3. Only after a dense visual gate clears, run the prepared 300-video STAR UVT
   scale path with cached V-JEPA targets.
4. Run a longer dynamic-gsplat fixed-512 baseline only if final dynamic-gsplat
   ranking matters. The current comparator already says it is not the immediate
   fast-route escape.
5. Keep WorldFoam as a separate lane. It has a clean current fused-MSE speed
   result, so the next WorldFoam step should be CPU/docs/shader design unless a
   clean benchmark window is explicitly available.

## Recommendation

Wrap this STAR UVT shader-diagnostic phase, but do not mark the broader active
goal complete unless the goal is narrowed to "record and close the diagnostic
slice."

The next useful goal should be one of these:

- **STAR UVT visibility bridge:** make a support-changing visual-quality route
  beat the current dense RGB failure band on one clip.
- **World feature tube parity:** build the CPU/PyTorch world-to-projected-UVT
  compiler and gradient parity gate without GPU contention.
- **WorldFoam side lane:** keep going on the clean fused-MSE Gate4 line, but do
  not mix it with STAR UVT scale benchmarking.

The most useful next goal is the first one. It attacks the actual blocker:
visibility/support quality, not more same-support alpha pressure, data loader
work, or another dynamic-gsplat escape check.
