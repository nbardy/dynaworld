# Thread Recap: Softmax-GS, STAR UVT Support, Prefix-Alpha, And WorldFoam

Date:
    2026-06-09

Scope:
    Durable recap of the thread that began with arXiv `2604.27437`
    Softmax-GS and then narrowed into STAR UVT support/composition work.
    This note records the goals, math, challenge taxonomy, attempted
    solutions, measured targets, and current decision.

Primary local artifacts:

- Paper PDF:
  `research_notes/gaussian_splatting_papers/pdfs/2604_27437_softmax_gs.pdf`
- Converted paper Markdown:
  `research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_converted.md`
- Integration note:
  `research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_dynaworld_integration.md`
- Short-term plan:
  `research_notes/gaussian_splatting_papers/2026-05-25_short_term_softmax_gs_plan.md`
- Long-term plan:
  `research_notes/gaussian_splatting_papers/2026-05-25_long_term_splats_vs_worldfoam_plan.md`
- STAR prefix tape diagnostic:
  `research_experiments/star_uvt_feature_tubes/visibility_prefix_tape_diagnostic.py`
- STAR prefix-alpha trainer loss:
  `src/train/star_uvt_feature_overfit_trainer.py`
- Experiment registry:
  `EXPERIMENTS.md`

## User Questions That Drove The Thread

Initial user asks, paraphrased:

1. Download arXiv `2604.27437`, sort it into research notes, convert it to
   Markdown.
2. Understand whether Softmax-GS can integrate into dynamic GS, STAR UVT, and
   WorldFoam.
3. Clarify when Softmax-GS is applied:
   during rasterization, after 3D projection, or final compositing?
4. Clarify whether the non-softmax path needs order and whether Softmax-GS
   needs different order/tape state.
5. Decide how this interacts with STAR UVT and WorldFoam.
6. Decide whether Softmax-GS fixes issues WorldFoam already fixes, whether
   "moving to foam" is the future, or whether we should build better splats.
7. Write short-term and long-term plan docs.
8. Follow those plans through the shader/support work, run some trains, and
   record the results.

Current answer in one sentence:
    We did not promote Softmax-GS or WorldFoam as the main path. We kept
    Softmax-GS as an opt-in dynamic-GS renderer probe, kept STAR UVT support
    as the near-term mainline, fixed a real STAR sparse-binner bug, measured
    prefix/order contribution, implemented a prefix-alpha loss, ran 20-step
    and 50-step STAR trains, and found that prefix-alpha is locally positive
    but not a dense-support fix.

## Paper Intake And Sorting

The Softmax-GS paper was archived locally:

```text
research_notes/gaussian_splatting_papers/pdfs/2604_27437_softmax_gs.pdf
research_notes/gaussian_splatting_papers/sources/2604_27437_softmax_gs/
research_notes/gaussian_splatting_papers/text/2604_27437_softmax_gs.txt
research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_converted.md
research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_dynaworld_integration.md
```

The converted note records the paper's own core claim:

- Vanilla 3DGS assumes trimmed Gaussian supports do not overlap along a ray.
- That assumption gives efficient alpha-over compositing but creates blurry
  boundaries and view/order inconsistency when splats overlap.
- Softmax-GS introduces softmax competition between overlapping Gaussians,
  with learnable/controllable parameters for boundary sharpness and overlap
  competition.

## Vanilla Alpha-Over Math

For one pixel/ray, contributors are sorted front-to-back by per-pixel depth.
Let:

```text
a_i  = absorbance / opacity contribution of contributor i at this pixel
c_i  = feature/color of contributor i
T_i  = prefix transmittance before contributor i
w_i  = final visible contribution weight of contributor i
```

Then vanilla alpha-over is:

```text
T_i = product_{j < i} (1 - a_j)
w_i = T_i * a_i
C   = sum_i w_i * c_i
A   = sum_i w_i
    = 1 - product_i (1 - a_i)
```

Important interpretation:

- `a_i` asks "how opaque is this primitive if it were considered at this
  pixel?"
- `w_i` asks "how much did it actually contribute after everything in front
  of it?"
- Vanilla alpha-over is already order-sensitive.
- Vanilla backward can often reconstruct prefix state from output
  transmittance because:

```text
T_in = T_out / (1 - a_cur)
```

That simple reverse recurrence is what Softmax-GS breaks.

## Softmax-GS Math

Softmax-GS changes the per-ray compositing model for overlapping contributors.
It does not solve projection, data, world-token identifiability, or STAR support
coverage by itself.

For two close splats with absorbances `a_j, a_k`, exponents `p_j, p_k`, depths
`d_j, d_k`, competition strength `beta`, and depth-decay `gamma`, the simplified
paper mechanism is:

```text
softmax_k = exp(beta * p_k) / (exp(beta * p_j) + exp(beta * p_k))
ahat_k   = softmax_k * a_k
s        = exp(-gamma * abs(d_k - d_j))
abar_k   = s * ahat_k + (1 - s) * a_k
```

Meaning:

- If two splats are near the same surface/depth, let them compete.
- If they are separated in depth, decay back toward vanilla alpha-over.
- `beta` controls winner-take-all versus smooth blend.
- `gamma` controls how quickly competition disappears with depth separation.
- The paper also has a generalized exponential footprint parameter,
  effectively a boundary-sharpness parameter. To avoid name collision with
  alpha compositing, this note calls it `alpha_shape`.

The full algorithm collapses all prior contributors into a single "past"
entity, competes the current splat against that past, then solves small
corrections to keep:

```text
same-depth two-way contribution ratio follows softmax weights
final transmittance matches original product transmittance
```

The backward challenge:

- Softmax-GS forward uses intermediate state such as past absorbance, past
  depth, past exponent, pairwise corrections, and rescale factors.
- That state cannot be inferred from final transmittance alone.
- Therefore the efficient backward needs a forward tape. The paper uses a
  K-limited tape; their real-scene setting uses `K=128`.

## Answer To "When Is It Applied?"

Softmax-GS is not a high-level neural postprocess after an image exists.

It is applied in the renderer's per-pixel accumulation/compositing stage:

```text
project 3D primitives -> bin/rasterize candidate splats per tile/pixel
evaluate per-pixel absorbance/exponent/depth/color
sort or process front-to-back
apply Softmax-GS overlap competition while accumulating color/alpha
write final image
```

So it is after projection/evaluation of candidate splats but inside the
rasterization/compositing loop, not after final RGB output.

Does non-softmax need order?
    Yes. Vanilla alpha-over needs front-to-back order because `T_i` depends on
    every previous `a_j`.

Does Softmax-GS need different order?
    It still processes an ordered ray/tile list, but tries to be more stable
    for near-same-depth overlaps by making same-surface competitors share a
    softmax rule rather than relying on arbitrary front/back ordering.

Does Softmax-GS need more backward state?
    Yes. The final alpha/transmittance is no longer enough to reconstruct all
    needed intermediate values, so tape or recomputation is needed.

## Dynamic-GS Work Done

Implemented/recorded in the dynamic-GS Softmax lane:

- Torch/reference implementation:
  `research_experiments/softmax_gs/reference.py`
- Reference report:
  `research_experiments/softmax_gs/REFERENCE_REPORT.md`
- Tests:
  `tests/test_softmax_gs_reference.py`
  `tests/test_softmax_gs_metal_forward.py`
  `tests/test_softmax_gs_tape_coverage_diagnostic.py`
- Fast-mac variant:
  `third_party/fast-mac-gsplat/variants/v5_softmax_gs/`
- Config-selected RGB variant path.
- Center-camera-z depth mode for Softmax-GS, because rank-depth is a vanilla
  sorting artifact, not a meaningful softmax depth signal.
- Native Metal forward and backward for fast and overflow tiles.
- Bounded contribution tape ABI/kernel:

```text
rasterize_softmax_gs_bounded_tape(...)
    -> selected_ids
    -> selected_weights
    -> residual_weight
    -> final_alpha
```

- Tape-backed color gradients.
- Tape-backed selected scalar geometry/opacity/depth VJP when
  `softmax_gs_tape_k > 0`.

Key reference invariants:

```text
vanilla parity when disabled
same-depth two-splat order swap invariance when enabled
separated-depth fallback toward vanilla
final transmittance preservation
finite gradients through absorbance/exponent/beta/gamma
weights @ features reconstructs output feature/color
bounded tape residual mass bounds omitted feature error for features in [0,1]
```

## Dynamic-GS Evidence

Tiny 64px/4f/128-splat heldout diagnostic:

```text
No-op:
    heldout PSNR/SSIM = 4.7369 / 0.0503
Enabled K=16:
    heldout PSNR/SSIM = 11.7255 / 0.0794
Train loss:
    tied, about 0.226
```

This was the first positive sign stronger than source-view-only evidence.

Primitive-count repeat, 64px/4f/512 splats:

```text
No-op:
    heldout PSNR/SSIM = 12.5002 / 0.0817
Enabled K=16:
    heldout PSNR/SSIM = 11.8847 / 0.0950
```

Interpretation:

- Enabled fit train/source better.
- Heldout PSNR got worse by about `0.6155dB`.
- Heldout SSIM got slightly better.
- Not a clean promotion.

Tape residual diagnostic at 512 splats:

```text
K=16 heldout residual/alpha mean = 0.001930
K=16 heldout residual/alpha p99  = 0.012332
K=8  heldout residual/alpha mean = 0.040167
K=8  heldout residual/alpha p99  = 0.112505
```

Interpretation:

- K=16 tape truncation tail is small.
- The 512-splat heldout PSNR miss is not explained only by bounded tape
  approximation.

Practical 128px/16f stride16 scale check:

```text
No-op:
    heldout PSNR/SSIM = 12.1234 / 0.1244
Enabled K=16:
    heldout PSNR/SSIM = 12.2092 / 0.1088
```

Interpretation:

- Tiny heldout PSNR nudge.
- Worse heldout SSIM and train-view metrics.
- Still mixed.

Decision:
    Softmax-GS stays as an active dynamic-GS renderer probe. Do not port it to
    STAR UVT or WorldFoam without a repeated heldout/rate win.

## STAR UVT Context

STAR UVT feature tubes are time-tubed splat-like primitives. The relevant
per-tube state is:

```text
ma          = center in screen/time coordinates (u, v, t)
q_uvt       = 3D quadratic precision over (u, v, t)
depth0      = base depth
depth_beta  = local depth slope with respect to (u, v, t)
opacity     = tube opacity
feature     = learned F32 feature
```

For a target point `p = (u, v, t)`:

```text
delta_i = p - ma_i
qv_i    = delta_i^T Q_i delta_i
a_i     = opacity_i * exp(-0.5 * qv_i)
a_i     = clamp(a_i, 0, max_alpha)
a_i     = 0 if a_i < alpha_threshold
d_i     = depth0_i + dot(delta_i, depth_beta_i)
```

Then contributors are sorted by `d_i` and alpha-over applies.

The STAR problem in this thread was not primarily speed anymore. It was
support/coverage/composition:

- Feature/probe losses could improve.
- Local patch objectives could learn.
- Dense RGB stayed weak or blurry.
- Forced-alpha and target-background oracle diagnostics were much better than
  normal black-background compositing.

That means a lot of the color/content is available if alpha/support were
stronger, but ordinary render output remains dark or uncovered.

## STAR Support-Birth Target

The support-birth idea:

```text
find target points where current model is under-covering important pixels
select low-opacity/dead tubes
reallocate those tubes around the target points
give them support radius / temporal radius / initial feature values
train with local target pressure
```

Target point sources tried along the prior ladder included:

- brightness / uncovered brightness
- cap slack / low tile-load points
- residual scoring
- footprint-aware residual scoring
- target-grid feature initialization
- support-target alpha
- support-target-area 2x2 patches

The specific strong row under discussion used:

```text
target_point_source = cap_slack_footprint_residual_uncovered_brightness
center_count        = 16
reallocate_tubes    = 16
support_radius_px   = 40
opacity             = 0.4
target_area_loss    = 2x2 black-background composition loss
tile_capacity       = 128
tile_overflow_repair_guard_refs = 2
```

## STAR Binner Bug And Repair

A selected-patch diagnostic found a real renderer bug:

Observed contradiction:

```text
analytic selected-tube alpha: nonzero
sparse rendered selected-only alpha: zero
```

Root cause:

- Chunk-shifted moving tubes had valid analytic target alpha.
- Sparse binning dropped them because `tube_bounds` rejected small valid
  `3x3` determinants.
- Fallback temporal bounds used local chunk size as half-extent and did not
  cover shifted local centers.

Repair:

```text
determinant tolerance: max(eps^2, 1e-20)
fallback bounds: cover abs(m) + local domain, not only local chunk frames
```

Focused regression:

```text
tests/test_star_uvt_feature_binning.py
```

## Binner Repair Evidence

Before/after selected target patches:

Repaired pre-train selected-patch diagnostic:

```text
targetinit / targetalpha / targetarea2 normal patch PSNR:
    4.606 / 4.686 / 4.684
forced patch PSNR:
    14.529 / 14.694 / 14.677
selected-only alpha:
    about 0.30 instead of 0.0
```

First repaired targetarea2 50-step train:

```text
row:
outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_50step_media.json

pass: true
tile overflow: 0
max tile count: 110 / 128
loss: 0.889263 -> 0.863064
feature loss: 0.612217 -> 0.610967
rgb-probe loss: 0.003756 -> 0.003587
support-target-area loss: 0.253626 -> 0.217254
```

Post-train selected-patch diagnostic:

```text
normal / forced / oracle patch PSNR:
    6.644 / 19.452 / 26.994
patch alpha mean:
    0.481
selected-only alpha:
    0.444
```

Interpretation:
    The support exists locally after the binner repair. The bug fix was real.

## Dense Support Diagnostic Math

The dense support diagnostic asks why dense RGB is bad.

Definitions:

```text
normal PSNR:
    PSNR of actual black-background composite alpha * RGB

forced-alpha PSNR:
    ignore alpha; compare RGB/colorizer output directly to target

target-background oracle PSNR:
    composite prediction over target background
    output = alpha * predicted_rgb + (1 - alpha) * target_rgb

posthoc alpha gain:
    multiply rendered alpha by a scalar after the render

alpha floor:
    clamp rendered alpha to a minimum floor after the render

raw-opacity bias:
    rerender after adding a logit-space bias to tube opacity
```

What each diagnostic falsifies:

- If forced-alpha is much better than normal, color/content is less bad than
  alpha/coverage.
- If target-background oracle is high, black holes dominate error.
- If raw-opacity bias works, the same support shape could be fixed by opacity.
- If raw-opacity bias fails but forced-alpha works, support geometry/ownership
  is still wrong, not just scalar opacity.

Binfix dense diagnostic:

```text
normal / forced / oracle PSNR:
    7.269 / 14.736 / 21.439
alpha mean:
    0.3456
alpha > 0.1:
    75.4%
alpha > 0.5:
    29.1%
best posthoc gain:
    12.399 PSNR @ 16x
best alpha floor:
    14.736 PSNR @ 1.0
best raw-opacity bias:
    8.039 PSNR @ +4
```

Comparison to pre-binfix targetarea2 repair:

```text
pre-binfix dense normal/forced/oracle:
    6.507 / 14.085 / 21.627
pre-binfix alpha > 0.1:
    65.7%
```

Interpretation:

- Binner fix gave a real dense-support gain.
- But the large normal-vs-forced/oracle gap remains.
- Raw opacity bias barely helps.
- Therefore the remaining issue is not simply "turn opacities up."

## Prefix Tape Diagnostic Math

After binfix, the next question was:

```text
Are selected born tubes present but hidden behind older/front tubes?
```

The prefix tape diagnostic reconstructs, for selected support-target rays:

```text
order_i              = contributor ids sorted by depth
ordered_alpha_i      = alpha in that order
prefix_i             = product_{j < i} (1 - ordered_alpha_j)
weight_i             = prefix_i * ordered_alpha_i
selected_weight_sum  = sum_{i selected} weight_i
final_alpha          = sum_i weight_i
selected_share       = selected_weight_sum / final_alpha
selected_prefix      = prefix at the selected tube's max-alpha contribution
```

The important distinction:

```text
selected_alpha_i high
```

does not mean selected tubes are visible. They may be behind already-opaque
front mass. The visible contribution is:

```text
selected_weight_i = prefix_i * selected_alpha_i
```

Binfix prefix tape result:

```text
sampled target rays: 256
normal / forced / oracle sampled PSNR:
    6.522 / 19.129 / 26.831
final alpha mean:
    0.4755
selected alpha max mean:
    0.2670
selected weight sum mean:
    0.4363
selected weight share mean:
    0.9308
selected prefix at alpha max mean:
    0.8734
selected absent fraction:
    0.0%
selected prefix-hidden fraction:
    1.6%
top contributor selected fraction:
    95.7%
```

Interpretation:

- Selected born tubes are present.
- They are not meaningfully hidden by older/front tubes on the sampled target
  rays.
- They already own most of the local prefix tape.
- The problem is weak final alpha and insufficient spread to dense pixels, not
  "old tubes hide born tubes."

This directly shaped the prefix-alpha experiment.

## Prefix-Alpha Loss Math

The prefix-alpha loss was implemented in:

```text
src/train/star_uvt_feature_overfit_trainer.py
```

Config keys:

```text
support_birth_split.prefix_alpha_loss_weight
support_birth_split.prefix_alpha_target
support_birth_split.prefix_alpha_max_points
```

For each selected support-birth target point:

```text
a_i = opacity_i * exp(-0.5 * qv_i)
d_i = depth_i
order = argsort(d_i)
T_i = product_{j < i} (1 - a_j)
w_i = T_i * a_i
selected_weight = sum_{i in selected_tubes} w_i
final_alpha = sum_i w_i
loss = (selected_weight - alpha_target)^2
```

In the actual 50-step row:

```text
prefix_alpha_loss_weight = 2.0
prefix_alpha_target      = 0.85
prefix_alpha_max_points  = 512
```

Implementation detail:

- Depth ordering is computed under `torch.no_grad()`.
- Gradients flow through alpha/opacity/shape terms after the gather.
- This is not a native Metal rasterizer replacement. It is a compact autograd
  auxiliary loss on selected support points.

What it addresses:
    It addresses the specific possibility that selected support tubes are
    locally present but not contributing enough visible alpha-weighted mass
    under alpha-over compositing.

What it does not address:

- It does not create new target ownership elsewhere in the frame.
- It does not change which pixels are sampled unless the target set changes.
- It does not replace alpha-over with Softmax-GS.
- It does not fix dense coverage if the sampled selected rays already have high
  selected share and the rest of the image lacks ownership.

## Prefix-Alpha Train Evidence

20-step probe:

```text
config:
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_prefixalpha_from1500_lr001_20step_media.jsonc

row:
outputs/benchmarks/2026-05-28_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_prefixalpha085w2_20step_media.json

pass: true
loss: 1.285825 -> 1.253901
support-target-area loss: 0.253626 -> 0.239840
prefix-alpha loss: 0.198281 -> 0.188115
selected weight: 0.4114 -> 0.4234
final alpha: 0.4456 -> 0.4571
tile overflow: 0
```

Fair 50-step comparison:

```text
config:
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_prefixalpha_from1500_lr001_50step_media.jsonc

row:
outputs/benchmarks/2026-05-28_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_prefixalpha085w2_50step_media.json

pass: true
loss: 1.285825 -> 1.210325
feature loss: 0.612217 -> 0.611005
rgb-probe loss: 0.003756 -> 0.003590
support-target-area loss: 0.253626 -> 0.219786
prefix-alpha loss: 0.198281 -> 0.172906
selected weight: 0.4114 -> 0.4419
selected share: 0.9333 -> 0.9382
final alpha: 0.4456 -> 0.4751
tile overflow: 0
```

Prefix-alpha dense support diagnostic:

```text
normal / forced / oracle PSNR:
    7.262 / 14.732 / 21.438
alpha mean:
    0.3452
alpha > 0.1:
    75.4%
alpha > 0.5:
    29.0%
best posthoc gain:
    12.394 PSNR @ 16x
best alpha floor:
    14.732 PSNR @ 1.0
best raw-opacity bias:
    8.037 PSNR @ +4
```

Prefix-alpha prefix tape:

```text
sampled target rays: 256
normal / forced / oracle sampled PSNR:
    6.465 / 19.051 / 26.734
final alpha mean:
    0.4719
selected weight sum mean:
    0.4374
selected weight share mean:
    0.9381
selected absent fraction:
    0.0%
selected prefix-hidden fraction:
    0.8%
top contributor selected fraction:
    96.9%
```

Comparison to no-prefix binfix:

```text
no-prefix binfix dense:
    7.269 / 14.736 / 21.439
prefix-alpha50 dense:
    7.262 / 14.732 / 21.438
```

Conclusion:
    Prefix-alpha works locally and is a useful measurement/control surface. It
    is not a dense-support fix at this target distribution and 50-step budget.

## WorldFoam Discussion And Decision

The user asked whether Softmax-GS fixes issues WorldFoam already fixes, and
whether the future should be foam rather than better splats.

Working distinction:

```text
Softmax-GS:
    renderer/compositing fix for overlapping fuzzy Gaussian contributors

STAR UVT:
    time-tubed splat/world-token bridge with projective interval trace atlas
    and support-changing feature tubes

WorldFoam:
    deeper primitive-family challenger: bounded cells / ray-cell intervals /
    surface-detail sites
```

Softmax-GS and WorldFoam overlap conceptually because both respond to fuzzy
overlap/ownership problems, but they operate at different levels:

- Softmax-GS changes how existing splats divide contribution.
- WorldFoam changes the geometry/support primitive itself.
- STAR UVT changes temporal/world-token structure and has its own
  projective/trace atlas math.

Thread decision:

```text
short term:
    continue STAR support/composition diagnostics

medium term:
    keep Softmax-GS as dynamic-GS renderer probe
    keep WorldFoam as challenger

long term:
    run matched representation tournament
```

Promotion rule:

```text
heldout predictive quality
rate / primitive efficiency
trainability
clean export
wall-clock and backward cost
```

Not sufficient for promotion:

- source-view media alone
- one tiny heldout row
- shader speed without RGB quality
- elegant math without matched evidence
- local sparse loss movement without dense support movement

## Short-Term Plan Written

The short-term plan now says:

```text
Do not start with a STAR/WorldFoam Softmax-GS port.
Treat STAR support-birth/binfix as active.
Treat dense support and prefix tape as measured.
Treat prefix-alpha as measured and locally positive but not dense-positive.
Next STAR move: broaden support ownership/coverage or change target sampling.
```

Stop criteria recorded:

```text
local pointwise / patch / prefix-alpha loss learns but dense support stays flat
support-selection scorer improves scalar loss but not forced/oracle gap
Softmax-GS repeats source/train wins without heldout repeat
WorldFoam has no matched heldout-quality row
```

## Long-Term Plan Written

The long-term plan now says:

```text
not "move to foam now"
not "port every Softmax-GS idea into STAR"

short term: STAR visibility/support/composition gate
medium term: small representation tournament
long term: promote the family that wins heldout predictive quality per cost
```

Current strategic stance:

- Better splats / STAR UVT remain mainline.
- Softmax-GS is a renderer probe with mixed repeat evidence.
- WorldFoam is serious, but must win a matched tournament.

## Main Challenges Exposed

### Challenge 1: Alpha/support coverage, not only color

Evidence:

```text
normal dense PSNR around 7
forced-alpha dense PSNR around 14.7
target-background oracle around 21.4
```

Meaning:
    Color/features are not hopeless. Holes/weak alpha/support dominate.

### Challenge 2: Raw opacity is not enough

Evidence:

```text
raw opacity logit bias +4 only reaches about 8.04 PSNR
```

Meaning:
    Same support shapes cannot simply be made opaque. Need different ownership,
    support geometry, sampling, or dense visibility pressure.

### Challenge 3: Selected support exists locally after binner repair

Evidence:

```text
selected-only alpha nonzero
hide-selected hurts local patch PSNR
selected patch forced/oracle high
```

Meaning:
    The binner bug was real and fixed, but local selected support is not enough.

### Challenge 4: Selected support is not hidden by old tubes

Evidence:

```text
selected absent 0%
selected hidden 1.6% -> 0.8%
selected share 93%+
top selected 95.7% -> 96.9%
```

Meaning:
    Occlusion-order debugging is not the next main lever.

### Challenge 5: Local objectives can learn without moving dense RGB

Examples:

```text
support-target alpha learns
support-target-area learns
prefix-alpha learns
dense support barely moves or stays flat
```

Meaning:
    The loss target distribution is too narrow or the support distribution does
    not spread the learned ownership to dense pixels.

### Challenge 6: Softmax-GS helps some tiny dynamic-GS cases but is mixed at repeat

Evidence:

```text
128-splat tiny heldout jump: positive
512-splat repeat: heldout PSNR negative
128px/16f stride16: tiny PSNR positive, SSIM/train negative
```

Meaning:
    Do not spend STAR/WorldFoam engineering on Softmax-GS yet.

### Challenge 7: WorldFoam speed evidence is not the same as quality promotion

WorldFoam has important shader/micro-gate work elsewhere in the repo, but the
thread conclusion for this question is:

```text
WorldFoam must beat STAR/dynamic splats on matched quality/trainability/export
before it becomes the main representation.
```

## Attempted Solutions And Outcomes

| Attempt | Goal | Outcome |
| --- | --- | --- |
| Download/convert Softmax-GS | Understand paper and local applicability | Done; PDF/source/text/Markdown/integration notes exist |
| Dynamic-GS Softmax reference | Make math executable before Metal | Done; reference report and tests |
| `v5_softmax_gs` Metal path | Test renderer-local Softmax-GS on MPS | Done; no-op parity, forward, backward, bounded tape |
| Tiny dynamic-GS heldout | See if Softmax-GS helps | Positive tiny row |
| Primitive-count repeat | Check robustness | Mixed/negative PSNR |
| Tape coverage diagnostic | Check if K=16 truncation caused failure | K=16 tail small; not the whole cause |
| 128px/16f stride16 repeat | Practical scale probe | Mixed; not promotion |
| STAR support-birth/binner diagnostic | Explain selected support contradiction | Found real sparse-binner bug |
| STAR binner repair | Make selected moving support render | Done; focused test and local patch evidence |
| Dense support diagnostic | Separate alpha holes from color/content | Showed coverage/composition bottleneck |
| Visibility prefix tape | Check if selected tubes are hidden | They are present and dominant locally |
| Prefix-alpha loss | Push visible selected contribution | Learns locally, fixed-bin pass, dense flat |
| Plan docs | Decide what to do now/long-term | Updated short/long plans, registry, TODO, README |

## Current Working Model

Current belief:
    The active STAR UVT feature-tube failure is not "selected support missing"
    anymore and not "old support hides selected support" on the sampled target
    rays. It is broader dense ownership/coverage: local selected support can
    learn, but the learned mass does not cover enough of the actual rendered
    image under black-background alpha-over.

Confidence:
    Medium-high for the local selected-ray claim, because prefix-tape metrics
    directly measure contribution order. Medium for the dense-ownership claim,
    because dense diagnostics strongly support it but we still need the next
    sampling/coverage experiment to localize exactly which unsampled pixels or
    tubes fail.

Could be wrong if:

- The dense diagnostic has a mismatch between target points and rendered
  full-frame pixels.
- The prefix-alpha loss samples too few points (`512`) and misses important
  target rays.
- The target point distribution repeats the same top strip/bright structures
  and ignores other dense holes.
- A longer prefix-alpha train or different target would move dense support, but
  the current evidence says not to assume that without changing sampling.

## What We Should Do Next

Immediate next STAR direction:

```text
broaden support ownership/coverage
change target sampling distribution
make diagnostics report which dense pixels lack selected ownership
avoid repeating local alpha pressure unless point/tube ownership changes
```

Concrete next tests:

1. Dense ownership map:
   For full-frame sampled pixels, compute `selected_weight_share`,
   `final_alpha`, residual, and target-point-nearest distance. Ask whether
   dense holes are near or far from the selected target set.

2. Broadened target sampler:
   Add target diversity constraints by frame/region/residual bucket, not just
   top score. Keep fixed-bin cap by reducing radius/tube allocation per region.

3. Support distribution A/B:
   Same total selected tubes, different spatial/temporal spread:

   ```text
   current multicenter K16/r40
   broader region-stratified K16/r32
   per-frame quota K16
   residual quantile quota K16
   ```

4. Prefix-alpha with changed ownership:
   Only rerun prefix-alpha if the selected target set changes. The old target
   set already had selected share above `93%`.

5. Softmax-GS:
   Only continue if the dynamic-GS probe gets a repeated heldout/rate win or if
   STAR support becomes dense enough that overlap/order is visibly the remaining
   artifact.

6. WorldFoam:
   Use a matched representation tournament, not intuition:

   ```text
   dynamic GS baseline
   dynamic GS + Softmax-GS if repeated positive
   STAR UVT support/composition route
   WorldFoam selected Metal route
   ```

## Final Thread Decision

The thread started with "Should we integrate Softmax-GS, and does it change
STAR/WorldFoam strategy?"

The measured answer is:

```text
Softmax-GS:
    useful renderer probe, not promoted

STAR UVT:
    remains active mainline; binner repaired; local support works; dense
    ownership still insufficient

prefix-alpha:
    useful diagnostic/control loss; locally positive; not dense-positive

WorldFoam:
    serious challenger; not an automatic switch; needs matched tournament

next work:
    broader STAR ownership/coverage sampling, then tournament only after a
    clean dense-support gate
```
