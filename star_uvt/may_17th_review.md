# STAR UVT May 17 Review

Date: 2026-05-17
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`
Branch: `codex/satar-prt-compact-backward`

## User Goal

Get STAR UVT running as the fast 64-frame single-video overfit lane. The target is
not heldout quality today. The target is:

- one source video window;
- 64 frames;
- source-view overfit;
- fast training wall clock;
- loss down;
- PSNR up;
- SSIM up across all frames;
- enough artifacts and notes that the next agent can see what actually ran.

The user also asked whether the blocker is nondeterministic backward. The short
answer is:

```text
STAR UVT forward: yes, sparse/sublinear evidence exists.
STAR UVT training backward: the fast branch exists but is nondeterministic.
Promotable deterministic compact backward: still the main blocker.
64-frame overfit today: use direct_atomic first as an exploration/overfit lane,
then compare deterministic branches only if they are close enough to matter.
```

## Immediate Correction

The prior 300-clip training lane used the main Dynaworld token/gaussian trainer
with `fast_mac`; it was not the STAR UVT harness. That explains the mismatch
between the user's expectation ("STAR UVT should be fast") and the observed
36-second step totals in the V-JEPA-loss runs.

Those 36-second steps were not STAR UVT and were not raster-forward dominated.
The recorded timing showed:

- render/rasterize under about 1% of wall time;
- V-JEPA feature loss plus its backward path dominating the long step;
- recon-only removing most of that cost, but still using the token/gaussian
  trainer rather than STAR UVT.

For this review, STAR UVT means the code under:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/
third_party/fast-mac-gsplat/variants/star_uvt_prt_v0/
```

The current main trainer renderer modes do not expose STAR UVT as a first-class
`renderer_mode`. The existing STAR work is a research harness and benchmark
surface, not the production pretrain path.

Latest integration bridge:

```text
STAR UVT itself remains a research harness.
The production precomputed-feature trainer now has an opt-in
train.render_size_schedule so it can copy the useful STAR finding:
cheap 256px optimization first, then 512px finishing.
```

New configs:

```text
src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256to512.jsonc
src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc
```

The 400-step overfit bridge config completed locally:

```text
wall: 16:36
final eval PSNR/SSIM: 24.766 / 0.5316
256px median step_total: 1.676s
512px median step_total: 3.187s
```

That is a speed/throughput win versus the V-JEPA-loss lane, not a quality win.

## STAR UVT v0 Code Inventory

Root:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/
```

Core compiled files:

```text
csrc/bindings.cpp
csrc/metal/star_uvt_kernels.metal
csrc/metal/star_uvt_metal.mm
torch_gsplat_bridge_star_uvt/rasterize.py
```

Important Python surface:

```text
torch_gsplat_bridge_star_uvt/rasterize.py
research_project/trainer_harness/model.py
research_project/trainer_harness/tile_metal_autograd.py
research_project/trainer_harness/data.py
research_project/benchmarks/video_fit_comparison.py
research_project/benchmarks/uvt_forward_speed_probe.py
research_project/benchmarks/uvt_backward_breakdown_probe.py
research_project/benchmarks/uvt_train_step_timing_probe.py
research_project/benchmarks/multicam_heldout_compare.py
research_project/benchmarks/multicam_train_step_timing_probe.py
research_project/benchmarks/multicam_star_repeatability_probe.py
research_project/benchmarks/deterministic_compact_promotion_gate.py
```

The narrow primitive is a projected screen-time tube:

```text
ma:         [N,3] center in u,v,t
q_uvt:      [N,6] symmetric quadratic support in u,v,t
depth0:     [N]
depth_beta: [N,3]
opacity:    [N]
color:      [N,3]
```

`UVTRenderConfig` is process-static with respect to Metal tile constants:

```text
height
width
frames
tile_x in {8,16}
tile_y in {8,16}
tile_t in {1,2,4}
tile_capacity in {32,64,128,256}
```

The benchmark harness sets:

```text
STAR_UVT_TILE_X
STAR_UVT_TILE_Y
STAR_UVT_TILE_T
STAR_UVT_TILE_CAPACITY
```

before the first Metal call. This matters because a single Python process cannot
freely switch tile constants after the Metal extension is initialized.

## STAR UVT v0 Backward Modes

`video_fit_comparison.py` exposes these reducer modes:

```text
index_add
sorted_cpu
scan_metal
compensated_scan_metal
sort_scan_metal
sort_compensated_scan_metal
key_sort_scan_metal
key_sort_compensated_scan_metal
key_sort_segmented_metal
```

and these sample/backward emission modes:

```text
atomic_append
with_keys
tile_pair
tile_pair_compensated
tile_pair_grouped
tile_pair_parallel
tile_pair_scanline
tile_pair_sharedsort
tile_pair_target_bounds
tile_pair_suffix
direct_atomic
direct_fixedpoint
direct_split_fixedpoint
direct_serial
tile_pair_atomic
tile_pair_fixedpoint
tile_pair_reduced
tile_pair_reduced_parallel
tile_pair_suffix_reduced
```

Practical interpretation:

- `direct_atomic + index_add`: fastest useful exploration branch. It bypasses the
  reducer and accumulates gradients with float atomics. It is the branch most
  likely to make the user happy for a local 64-frame overfit.
- `tile_pair + key_sort_scan_metal`: deterministic quality/reporting branch.
  It is much slower and is the reference family for exact repeatability.
- `tile_pair_suffix` / `tile_pair_grouped`: intermediate speed/quality probes.
  Some are useful at same-time overfit, but they are not currently the promoted
  reporting branch.
- `tile_pair_reduced` and relatives: tried as compact backward probes, but the
  full trainer timing did not become the desired solution.
- fixed-point branches: diagnostic attempts for determinism; not currently the
  selected fast path.

## What Deterministic Compact Backward Means

The desired backward is not simply "make gradients deterministic." It means:

```text
forward work unit: active UVT tile/tube pair
desired backward work unit: same compact tile/tube pair, or similarly compact
bad current exact path: emit/sort/reduce a large per-pixel sample table
bad current fast path: use unordered float atomics, fast but nondeterministic
```

The current state review says the forward path has real sparse evidence, but the
training path still does too much per-pixel backward work or uses nondeterministic
atomics. The strongest diagnostic line is:

```text
tile_t=1, tile_t=2, and tile_t=4 all emitted 5,342,341 compact backward samples
on the same 256px/16-frame initialized scene.
```

That means changing the forward temporal tile shape can reduce forward tile/tube
pairs without proportionally reducing the current backward sample table. The
future solved kernel should avoid this per-pixel row explosion.

## Prior Result Read

### Single-video 256px, 16 frames

Strong prior rows from the STAR README:

```text
50 steps, tile_t=1, unregularized:
  PSNR 22.3639, wall 25.24s, render median 9.43ms

50 steps, tile_load_reg=0.003,target=60:
  PSNR 21.9738, wall 18.75s, render median 5.82ms

50 steps, tile_load_reg=0.01,target=60:
  PSNR 21.3108, wall 10.92s, render median 5.03ms

200 steps, tile_load_reg=0.003,target=60:
  PSNR 23.9762, wall 42.52s, render median 5.74ms

200 steps, unregularized:
  PSNR 24.4697, wall 70.66s, render median 18.97ms
```

Interpretation:

- STAR can overfit source video.
- Tile-load regularization buys a real speedup.
- It costs some quality.
- The old rows are 16-frame, not 64-frame.
- They report MSE/PSNR, not SSIM.

### 512px, 16 frames, multicam goodset

The state review records the strongest 512px fast branch:

```text
direct_atomic + index_add, 600 steps, 320 tubes:
seed 0: train loop 30.20s, heldout 13.6691
seed 1: train loop 23.73s, heldout 13.9040
seed 2: train loop 21.19s, heldout 13.8199
```

That cleared the V-JEPA F32 heldout reference on all three seeds in that narrow
goodset comparison, but it was not promotable because same-process repeatability
was not exact:

```text
final state max abs delta: 1.4310
final state mean abs delta: 0.1183
heldout PSNR span: 0.0330
```

The deterministic exact branch is much slower:

```text
keyed per-pixel: exact repeatable, train loops about 206-221s
zero-pruned tile-pair: exact repeatable, train loop about 113s
```

This is the core answer to "is that nondeterministic backward?":

- yes for the fastest branch;
- no for the exact branch;
- the exact branch exists but is too slow for the goal.

## Projective STAR UVT / PRT v0 Inventory

Root:

```text
third_party/fast-mac-gsplat/variants/star_uvt_prt_v0/
```

Core compiled files:

```text
csrc/bindings.cpp
csrc/metal/star_uvt_kernels.metal
csrc/metal/star_uvt_metal.mm
torch_gsplat_bridge_star_uvt_prt/rasterize.py
torch_gsplat_bridge_star_uvt_prt/tile_config.py
```

Important trainer/benchmark files:

```text
research_project/trainer_harness/projective_rational.py
research_project/trainer_harness/projective_rational_metal_autograd.py
research_project/benchmarks/projective_rational_video_overfit_compare.py
research_project/benchmarks/projective_rational_multicam_splat_compare.py
research_project/benchmarks/projective_rational_multicam_train_breakdown.py
research_project/benchmarks/projective_rational_train_step_timing_probe.py
research_project/benchmarks/projective_rational_train_step_repeatability_probe.py
research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py
research_project/benchmarks/projective_rational_multicam_learnable_camera_train.py
```

PRT is not a replacement for the static 64-frame overfit gate. It is the
moving-camera completion path: fit or compile camera motion into a rational UVT
primitive so one world tube can cover many frames under camera motion. It has
lots of gates done, including fused MSE timing and real-clip learnable-camera
rows, but for today's source-view single-video overfit, the simpler screen-space
STAR UVT v0 harness is the right first target.

## Data For Today's 64-frame Gate

The exact first overfit manifest row is:

```text
data/single_video_pretrain/dynaworld_single_video_pretrain_300_youtube_64f_512_v0/train_manifest_overfit_first.jsonl
```

It points at:

```text
video_path: data/youtube_scene_distinct/raw/KUDJ8HDFVQo.mp4
start_seconds: 2.0
fps: 23.976023976023978
frame_count: 64
duration_seconds: 2.6693333333333333
target_size: 512
image_crop_mode: center_square
```

The video exists locally. `ffprobe` reports:

```text
width: 640
height: 360
fps: 24000/1001
duration: 8.008s
frames: 192
```

Current STAR `video_fit_comparison.py` only accepts a video path and loads from
the start. That is wrong for parity with this manifest. It needs:

- `--start-seconds`;
- `--fps`;
- `--image-crop-mode center_square`;
- optional manifest-record helper later.

## Current Harness Gaps

The harness can run a STAR UVT overfit, but for today's target it lacks:

- all-frame SSIM metric;
- all-frame SSIM min/mean/per-frame output;
- manifest-window loading;
- explicit source FPS sampling;
- explicit center-square crop;
- 64-frame result rows;
- full-frame media/contact-sheet beyond the first four frames.

The first code change should be measurement/parity, not a new kernel.

## Expected Scaling Pressure

The natural cheap capacity heuristic is:

```text
tubes_per_frame ~= tube_count / frames
pixels = frames * width * height
pixels_per_tube = pixels / tube_count
```

Examples:

```text
512 tubes over 64f:   8 tubes/frame, 32768 pixels/tube at 512px
2048 tubes over 64f: 32 tubes/frame, 8192 pixels/tube at 512px
8192 tubes over 64f: 128 tubes/frame, 2048 pixels/tube at 512px
16384 tubes over 64f:256 tubes/frame, 1024 pixels/tube at 512px
```

The user is right that "256 splats/tubes per frame" is a cheap mental scale-up
target. For STAR UVT that means roughly:

```text
64 frames * 256 tubes/frame = 16384 tubes
```

But this is only cheap if the active support/tile load stays bounded. If support
expands during optimization, backward rows can explode. The tile-load proxy is
there because prior runs showed exactly this.

## Fast Lane Versus Promotion Lane

For the user goal, split the work:

```text
Fast overfit lane:
  direct_atomic + index_add
  tile_t=1
  tile_capacity=128 or 256
  tile_load_reg bracket
  accepts nondeterminism for exploration

Promotion lane:
  tile_pair + key_sort_scan_metal or selected deterministic branch
  exact repeatability
  slower
  only worth running after fast lane proves quality and target scale
```

This is not lowering the bar. It prevents the deterministic promotion blocker
from stopping the practical "can STAR overfit 64 frames fast?" gate.

## Working Belief

Current belief:

```text
The fastest path to a useful result is not to solve deterministic compact
backward immediately. It is to wire the 64-frame source-window overfit correctly,
measure all-frame SSIM/PSNR, and run direct_atomic with tile-load controls.
```

Confidence: medium-high.

Could be wrong if:

- direct atomic becomes unstable at 64f/512px for this video;
- the video-sample init undercovers high-frequency detail too badly;
- tile-load regularization prevents accuracy from rising;
- Metal process-static tile constants make matrix exploration too clumsy;
- the extension is stale or broken on the current branch.

Cheap falsification:

1. Build `star_uvt_v0`.
2. Run a tiny 64f/128px direct-atomic smoke from the manifest window.
3. Confirm loss decreases and SSIM fields appear.
4. Run 64f/256px direct-atomic 50/200 steps.
5. Only then consider 512px/64f and 8192/16384 tubes.
