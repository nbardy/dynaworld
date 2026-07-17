# May 17 Workstreams

Date: 2026-05-17
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

This note reorganizes the current STAR UVT / Dynaworld concerns into explicit
lines of work. It separates what is already running from what actually proves
the STAR UVT thesis.

## Current State

The active local run is still the modular Dynaworld Gaussian-sequence trainer,
not STAR UVT:

```text
screen:
dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143

log:
outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log

latest checked status:
about 2190/3000
still before the step-2400 512px render-size switch
feature-cache hits only
W&B offline because launch env had WANDB_MODE=offline
```

The active trainer path is:

```text
src/train/train.py
arch=precomputed_feature_implicit_camera
-> src/train/train_precomputed_feature_implicit_dynamic.py
-> PrecomputedFeatureImplicitTrainer
-> VideoTokenImplicitTrainer in src/train/train_video_token_implicit_dynamic.py
```

It emits a standard per-frame `GaussianSequence` and renders through:

```text
render.renderer=fast_mac
fast_mac.rgb_variant=v6_refined
fast_mac.feature_variant=v5_features
```

That run is useful for broad dataset/convergence signal, but it does not prove
the STAR UVT claim that multi-frame training should approach single-frame cost.

Update: the run reached the step-2400 256px -> 512px switch, slowed into the
`4-8s/step` range, then produced NaN total/camera losses around step `2429`.
It was stopped. Treat this as a Gaussian-trainer stability warning, not a STAR
UVT result.

## Existing STAR UVT Trainers

Yes, STAR UVT trainers already exist, but they are currently research harnesses
and benchmark scripts rather than first-class Dynaworld trainer-router entries.

Relevant STAR UVT v0 harness:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/train.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/train_video.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/train_synthetic.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/video_fit_comparison.py
```

Relevant projective STAR UVT / PRT harness:

```text
third_party/fast-mac-gsplat/variants/star_uvt_prt_v0/research_project/benchmarks/projective_rational_video_overfit_compare.py
third_party/fast-mac-gsplat/variants/star_uvt_prt_v0/research_project/benchmarks/projective_rational_train_step_breakdown_probe.py
```

Why this distinction matters:

```text
current Dynaworld trainer output:
  GaussianSequence [T, G, C]
  xyz / scales / quats / opacities / rgbs

STAR UVT output:
  screen-time tubes
  center_uv / center_t / velocity_uv / q_uvt / opacity / color / depth

STAR UVT PRT output:
  projected-rational or world-tube parameters
  different camera/projection contract
```

A pure renderer swap would not test the STAR UVT thesis because the model would
still be producing per-frame Gaussian sequences instead of compact time-tube
support.

## Workstream 1: Finish And Interpret The Active Gaussian Trainer Run

Question:

```text
Does the current precomputed-feature static/dynamic/register Gaussian trainer
learn anything useful across 300 natural 64-frame YouTube clips?
```

Why it exists:

This run answers a data/training-signal question. It is not the STAR UVT speed
proof.

Immediate next steps:

```text
record the stopped-at-512 NaN verdict
compare final visual quality against the 400-step single-window overfit and the
STAR UVT high-motion overfit rows
```

Done criteria:

```text
final loss/PSNR/SSIM/media recorded
quality verdict: useful dataset signal or not
explicit note that this is not STAR UVT speed evidence
```

Current verdict:

```text
stopped near step 2436/3000 after NaNs began at step 2429
the 512px promotion is not a stable baseline yet
do not spend more local time on this exact lane without a NaN/stability fix
```

## Workstream 2: STAR UVT First-Class Trainer Fork

Question:

```text
Can we keep the modern Dynaworld data/config/W&B/media shell but train UVT tubes
directly?
```

Why it matters:

This is the real path to the thesis: 64-frame training should not cost 64x a
single-frame train step.

Proposed implementation:

```text
add a new trainer arch, for example:
  star_uvt_video_overfit

reuse:
  manifest/window loading
  center-square crop contract
  online W&B logging
  image/video media logging
  PSNR/SSIM/temporal metrics

replace:
  GaussianSequence output contract
  fast_mac Gaussian renderer
  static/dynamic/register token layout assumptions

with:
  ScreenTimeTubeModel-style UVT tube parameters
  STAR UVT metal_tile renderer
  direct_atomic/index_add fast branch first
```

Done criteria:

```text
one config launches through src/train/train.py or a clear sibling launcher
one high-motion 64f 256px overfit runs with online W&B
logs sample/data, forward, render, backward split, optimizer
exports side-by-side MP4 and PSNR/SSIM
```

Current result:

```text
done:
  arch=star_uvt_video_overfit routes through src/train/train.py
  src/train/train_star_uvt_video_overfit.py wraps the STAR UVT video harness
  online W&B works
  JSON/contact-sheet/side-by-side MP4 export works

direct_atomic row:
  config: src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc
  W&B: https://wandb.ai/nbardy/dynaworld/runs/jba7kztn
  PSNR: 29.823
  SSIM mean/min: 0.8572 / 0.7788
  final loss: 0.0010415

compact row:
  config: src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc
  W&B: https://wandb.ai/nbardy/dynaworld/runs/641gxm9l
  PSNR: 17.599
  SSIM mean/min: 0.6148 / 0.5446
  final loss: 0.017382
```

Still open: the wrapper logs final metrics/media only. If we want full timing
breakdown parity with the main trainer, the harness needs step-level timing
hooks or a smaller STAR-native trainer loop instead of only wrapping the
existing comparison function.

## Workstream 3: Representation Redesign For STAR UVT

Question:

```text
What replaces the current static/dynamic split under UVT?
```

Current answer:

The existing static/dynamic split is a Gaussian-token prior. It does not map
cleanly onto STAR UVT. UVT already represents time support and motion through
tube centers, temporal precision, velocity, and the `q_uvt` shape.

Better STAR UVT representation knobs:

```text
static-ish tubes:
  low or zero velocity
  broad temporal support
  stable appearance

dynamic-ish tubes:
  nonzero velocity
  tighter temporal support
  optional temporal split/promotion

detail tubes:
  higher spatial precision
  possibly shorter temporal support
  more aggressive sampling from high-error regions
```

Avoid carrying over the current `24 static + 8 dynamic + register/detail`
layout blindly. For STAR UVT, the better split is likely a tube-family or
initialization policy, not the current decoded-token taxonomy.

Done criteria:

```text
documented STAR UVT tube-family config
one ablation with all tubes free
one ablation with static-ish/dynamic-ish initialization or constraints
compare convergence, SSIM, tile load, and temporal artifacts
```

## Workstream 4: Data Loading And Prefetch

Question:

```text
Can sample/data time disappear from the critical path?
```

Current answer:

`sample_clip` is dataloader-ish cost. It includes record selection, cached RGB
window load, cached V-JEPA conditioning feature load, device transfer, and
clip/times prep. It should be overlapped under GPU work.

Already staged:

```text
data.train_manifest_prefetch
single-worker bounded CPU sequence prefetch
main-thread MPS transfer
```

Still open:

```text
benchmark frame-cache reads vs correct compressed-video decode
consider a real Dataset/DataLoader path if it improves overlap and ergonomics
avoid blocking the GPU on disk or CPU crop/resize
avoid W&B offline for future runs
```

Done criteria:

```text
profile run shows sample/data time near zero on the critical path
no cache misses
online W&B run records timing payloads
```

## Workstream 5: Backward Attribution

Question:

```text
How much of step time is model backward versus raster/loss backward?
```

Already staged:

```text
train.profile_backward_split=true
```

Diagnostic sections:

```text
backward/raster_loss_to_boundary
backward/model_from_boundary
backward/regularizers
```

This should be run as a short profile probe only. The active long run has
profiling disabled by design.

Done criteria:

```text
one 20-50 step probe on the current Gaussian trainer
one 20-50 step probe on STAR UVT direct_atomic
one 20-50 step probe on deterministic compact candidate
clear bottleneck table
```

## Workstream 6: STAR UVT Kernel Promotion

Question:

```text
Which STAR UVT kernel path is fast enough, accurate enough, and promotable?
```

Known branches:

```text
direct_atomic + index_add:
  fastest useful overfit/exploration branch
  nondeterministic

tile_pair + key_sort_scan_metal:
  deterministic promotion/reference branch
  currently too slow

tile_pair_suffix + key_sort_segmented_metal:
  better deterministic probe at short rows
  load growth is still a blocker
```

Current policy:

Use `direct_atomic/index_add` to prove trainability and speed first. Keep
deterministic compact backward as a separate promotion line, not a blocker for
the first STAR UVT trainer fork.

Done criteria:

```text
direct_atomic trainer row: fast 64f overfit with good SSIM
deterministic compact row: close enough to compare without load blowup
documented decision on what is research-only versus promotable
```

## Workstream 7: Experiment Matrix And Gates

Minimum useful runs:

```text
A. STAR UVT high-motion single-window overfit
   64 frames, 256px, direct_atomic, online W&B

B. STAR UVT multires promote
   256px coarse -> 512px fine

C. Gaussian trainer profile probe
   same clip/sample, profile_backward_split on

D. STAR UVT deterministic compact probe
   20-50 steps, same clip/sample

E. Current 300-clip Gaussian run final readout
   dataset-scale signal only
```

Primary metrics:

```text
wall seconds
samples/s
frames/s
PSNR
SSIM mean/min
render median
raster/loss backward
model backward
sample/data critical-path time
tile load proxy
visual side-by-side MP4
```

## Workstream 8: Checkpointing, Resume, And Run Hygiene

Question:

```text
Why are restarts expensive and why are offline runs hard to reason about?
```

Current problem:

The active trainer has no useful checkpoint/resume artifact in this lane, and
the current run is offline because of launch env.

Next fixes:

```text
online W&B by default
short-run checkpoint for overfit probes
simple resume for long local runs
run manifest embedded in notes
clear final sync command for offline W&B when needed
```

Done criteria:

```text
restart does not lose a long run
W&B links are available for important training lanes
notes name exact config, screen, log, media, and verdict
```

## Priority Order

1. Let the current active Gaussian trainer reach a useful gate or finish.
2. Run a short backward-split profile probe on the Gaussian trainer.
3. Fork/register a STAR UVT video-overfit trainer that reuses the modern
   data/W&B/media shell.
4. Prove fast 64f 256px high-motion overfit with direct_atomic/index_add.
5. Add multires 256->512 STAR UVT promotion.
6. Revisit deterministic compact backward after the fast branch is clearly
   useful.
