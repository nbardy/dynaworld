# STAR UVT Current Goal And Remaining Work

Date: 2026-05-19

## Original Goal

The original handoff request was not just "make one shader faster." It was a
full STAR UVT / dynamic-gsplat workday plan:

1. Re-run and audit the renderer benchmarks for Gaussian splats and STAR UVT
   across different frame counts, resolutions, splat/tube counts, and both
   forward and backward timing.
2. Decide what is actually missing, what needs fixing, and what shader path is
   good enough to use.
3. Build a nice fast single-video overfit path for the selected UVT STAR route
   and the dynamic Gaussian-splat route.
4. Break down the real training bottlenecks, especially whether backward time is
   rasterizer, feed-forward/colorizer/V-JEPA loss, data loader, or optimizer.
5. Scale the selected route to the prepared 300-video dataset.
6. Keep a separate feature-splatting / UVT STAR feature-world-tube lane alive.
7. Keep the WorldFoam investigation separate, with notes and shader work, and
   avoid letting that lane fight the main GPU benchmarks.

The active narrowed goal became: repeat the core plan in docs, record missed
details, then execute the STAR UVT fast-feature-shader plan gate by gate with
benchmarks and progress logs.

## What Is Recorded

Completed work is now heavily recorded in markdown:

- Thread closeout:
  `agent_notes/loose_notes/2026-05-17_17-01-49_star_uvt_thread_closeout.md`.
- Main routing map:
  `TODO/README.md`, `PROJECT_INDEX.md`, `README.md`, `EXPERIMENTS.md`,
  `BASELINES.md`, `research_experiments/star_uvt_feature_tubes/README.md`,
  and
  `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`.
- Dense lessons:
  `agent_notes/key_learnings.md` stays under the 200-line cap and carries the
  compressed direct-atomic / sparse-forward / hidden-native lesson.
- Benchmark reports and JSON outputs are under `outputs/benchmarks/`.
- Each meaningful STAR UVT gate has a loose note in
  `agent_notes/loose_notes/`.

The latest fully completed and documented native shader gate is:

- Loose note:
  `agent_notes/loose_notes/2026-05-19_22-10-40_star_uvt_hidden_sigmoid_mse_native_gate.md`
- Report:
  `outputs/benchmarks/2026-05-19_star_uvt_hidden_sigmoid_mse_native_gate.md`

That gate proved the hidden sigmoid-MSE native path is correct but not the
speed keeper: H32 scalar totals are `317.54/610.90/2549.39ms` at
`128/256/512px`, H64 at 256px is `817.27ms`, and vec4 reduce is slower than
scalar. The lesson is that fusing dense hidden MLP work into the traversal is
not enough; the next speed gate needs sparse support, compact visual gradients,
or a visibility/prefix tape.

## Current State

The best measured STAR UVT cached-V-JEPA training helper is the sparse-forward
batched-VJP route. In the routing docs it is recorded around the
64f/512px/8192t continuation chain:

- target-grid sparse-forward batched-VJP rows are around `0.25-0.54s/step`
  depending on schedule and timing window;
- rendered sparse visual probe rows are around `0.24-0.34s/step`;
- dense hidden full-cell visual routes are much slower, with full-cell8 hidden
  rows in the `4-7s/step` range;
- dense native hidden sigmoid-MSE correctness works, but 512px H32 is
  `2.55s` synthetic total and still relies on dense support.

The selected practical path is therefore still:

- use sparse-forward / batched target/probe VJP for fast 512px cached-V-JEPA
  helper training;
- keep direct atomic and index-add semantics as the correctness base;
- only promote a new native shader if it beats this path with parity and real
  trainer relevance.

The current in-progress implementation is a benchmark-only sparse hidden
sigmoid-MSE native gate. It has partial code in:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`

It is not complete, not built, and not benchmarked yet. It should not be
described as done.

## What Is Left

Immediate next work:

1. Finish binding the sparse hidden sigmoid-MSE cached-bin native op through
   `star_uvt_metal.mm`, `common.h`, and `bindings.cpp`.
2. Add a focused benchmark harness for sparse hidden native loss/backward:
   tiny parity first, then 64f/128px, 64f/256px, and 64f/512px timing.
3. Compare it against the current sparse-render plus Python hidden-VJP plus
   sparse backward path.
4. Record the result in a benchmark markdown report, a loose note, and routing
   docs. If it loses, call it a negative gate and stop sinking time into dense
   hidden fusion.

Remaining larger plan after that:

1. Convert the selected fast 512px STAR UVT route into the cleanest single-video
   overfit script and config set.
2. Re-check dynamic Gaussian-splat baseline timing at the same resolution,
   frame count, and effective primitive count before making any "faster than"
   claim.
3. Decide whether the chosen STAR UVT route is an acceptable source-overfit /
   V-JEPA helper despite current feature-quality weakness.
4. If acceptable, scale to the prepared 300-video manifest with explicit W&B
   logging and checkpoint/media artifacts.
5. Keep WorldFoam and feature-splatting investigations documented but separate
   from the main GPU benchmark lane.

## Short Answer For Future Agents

Yes, completed work is recorded. No, the active sparse hidden native gate is not
complete or recorded as a result yet. The original goal was a full renderer,
training, bottleneck, and scale-up audit; the current blocker is proving or
rejecting a native sparse hidden/visual backward gate against the already-fast
sparse-forward batched-VJP route.
