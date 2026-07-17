# Softmax-GS Integration Notes for DynaWorld

Date:
    2026-05-24

Paper:
    Softmax-GS: Generalized Gaussians Learning When to Blend or Bound

Local artifacts:
    `research_notes/gaussian_splatting_papers/pdfs/2604_27437_softmax_gs.pdf`
    `research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_converted.md`
    `research_notes/gaussian_splatting_papers/sources/2604_27437_softmax_gs/`

## Context

Softmax-GS attacks two standard 3DGS renderer failures:

- diffuse individual Gaussian boundaries;
- view-dependent popping/order artifacts when overlapping splats change order
  under camera motion.

The mechanism is renderer-local, not an encoder/world-token mechanism. It adds
per-Gaussian shape/competition parameters:

- `alpha`: generalized exponential boundary sharpness for the individual
  footprint.
- `beta`: strength of softmax competition between overlapping splats.
- `gamma`: decay of same-surface competition as depth separation grows.

The paper keeps the "splat first, compose second" structure by replacing the
per-splat absorbance used in alpha compositing with a softmax-adjusted
absorbance, then correcting the two-way competition so same-depth pairs are
approximately order invariant and final transmittance matches the original
product transmittance.

## Key Mechanism, Compressed

For two close splats with absorbances `a_j, a_k`, exponents `p_j, p_k`, and
depths `d_j, d_k`:

```text
w_k = exp(beta * p_k) / (exp(beta * p_j) + exp(beta * p_k))
ahat_k = w_k * a_k
s = exp(-gamma * abs(d_k - d_j))
abar_k = s * ahat_k + (1 - s) * a_k
```

The full algorithm folds the previously accumulated prefix into one
"past" entity and competes that against the current splat. It then solves a
small correction so:

```text
same-depth two-way contribution ratio follows softmax weights
final transmittance equals the original product transmittance
```

Backward cannot infer all needed prefix state from final transmittance, so the
paper caches a K-limited forward tape. Their real-scene setting uses `K=128`.

## What It Is Not

Softmax-GS does not solve DynaWorld's main world-token contract by itself.
It does not add held-out camera supervision, prevent frame-local leaks, or make
same-view training identify a predictive world quotient. Under
`training_contract_v1.md`, it is an allowed fixed-rasterizer variant and should
be judged only by supported-query prediction and rate/quality tradeoffs.

It is also not a STAR UVT coverage fix by itself. The current STAR feature
failure is mostly that alpha support covers too little of the target image.
Softmax-GS changes how overlapping contributors divide existing support; it
does not create contributors where there are none.

## Integration Verdict

Dynamic GS:
    High-value, bounded experiment. The first RGB/F3 fast-mac fork now exists
    as `v5_softmax_gs`, with no-op parity and Softmax-GS forward behavior
    verified on MPS. It now trains through native fast/overflow backward, and
    `softmax_gs_tape_k > 0` routes color plus selected scalar
    geometry/opacity/depth gradients through the bounded tape. Full-tape tests
    are exact; bounded K is an approximation. K=16 is the first useful
    post-shader source-view diagnostic, K=8 is too lossy, and K=32 does not
    improve the tiny endpoint. The first tiny RGB-pyramid multicam diagnostic
    is positive on heldout (`4.7369 -> 11.7255` PSNR at matched final train
    loss), but the first larger-primitive repeat does not preserve heldout
    PSNR: at 64px/4f/512 splats, no-op reaches `12.5002/0.0817` heldout
    PSNR/SSIM while enabled K=16 reaches `11.8847/0.0950`. The enabled row fits
    source/train loss better, so this is not a clean heldout promotion. The
    old 128px/8f MPS model-forward blocker is now localized to large-memory
    `nn.MultiheadAttention` and fixed with a manual MPS cross-attention
    fallback. Full-memory 128px/16f training remains locally too slow, so the
    practical stride16 128px/16f row is the scale check: no-op reaches heldout
    `12.1234/0.1244`, enabled K=16 reaches `12.2092/0.1088`. That is a tiny
    heldout-PSNR nudge with worse SSIM/train-view metrics, not promotion. The
    512-splat K=16 tape tail is small after a local 20-step diagnostic (heldout
    residual/alpha mean/p99 `0.001930/0.012332`), so the heldout PSNR miss is
    unlikely to be explained only by bounded-tape truncation.

STAR UVT:
    Useful but not first-line. There are two plausible routes:

    1. Use the Softmax-GS ideas inside the current direct feature Metal path as
       a local overlap/order/composition variant.
    2. Use the same-surface softmax competition as a visibility-strata rule in
       the projective interval atlas.

    Route 1 is simpler but may not address the known STAR failure. Route 2 is
    philosophically aligned with the trace-atlas work but requires careful
    event/cell semantics and a K-limited tape per tile/cell. It should follow,
    not precede, the existing support-changing birth/split work.

## Dynamic GS Implementation Sketch

Current dynamic/token-GS rendering enters through:

- `src/train/renderers/fast_mac.py`
- `src/train/pipeline/render.py`
- `src/train/objective/objective.py`

Important local detail:
    `project_for_fast_mac(...)` used to pass artificial rank-depths into the
    fast-mac rasterizer. It now supports `depth_mode="center_camera_z"` while
    preserving `rank_depth` as the default no-op behavior. Softmax-GS must use
    the center-depth path until a pixel-affine depth signal exists.

Smallest useful dynamic-GS experiment:

1. Add `fast_mac.feature_variant` / `rgb_variant` entries for a new
   Softmax-GS shader fork.
2. Add per-primitive `smgs_alpha`, `smgs_beta`, and `smgs_gamma` parameters only
   to the free/dynamic-gsplat model variant under test.
3. Initialize to near-vanilla behavior:
   `alpha = 1`, `beta = 0` or small, `gamma` large enough that separated
   depths mostly fall back to vanilla alpha compositing.
4. Run tiny parity gates:
   `beta=0` and `alpha=1` should match vanilla feature/RGB renders within
   tolerance.
5. Run a sparse dynamic-gsplat matched smoke:
   same primitive count, same config, compare eval PSNR/SSIM/L1, alpha coverage,
   temporal render flicker, and primitive count/step time.

Promotion criterion:
    Either heldout quality improves at matched primitive count, or matched
    quality holds with fewer primitives. Source-view-only improvement is useful
    but not a world-token claim.

## Implementation Status: 2026-05-25

Implemented:

- CPU/Torch ray reference and report:
  `research_experiments/softmax_gs/reference.py` and
  `research_experiments/softmax_gs/REFERENCE_REPORT.md`.
- Focused unit/reference tests:
  `tests/test_softmax_gs_reference.py`.
- Fast-mac depth signal plumbing in `src/train/renderers/fast_mac.py`.
- Config-selected RGB variant `rgb_variant="v5_softmax_gs"`.
- Metal Softmax-GS forward path, including overflow fallback.
- Native Metal recompute backward for `softmax_gs_enabled=true`, covering
  fast and overflow tiles.
- Bounded top-K contribution tape ABI/kernel lowering for fast and overflow
  tiles.
- Tape-backed color-gradient and selected scalar-gradient consumer when
  `softmax_gs_tape_k > 0`.
- Tape-backed selected scalar VJP for geometry/opacity/depth when
  `softmax_gs_tape_k > 0`.
- MPS regression:
  `tests/test_softmax_gs_metal_forward.py` proves same-depth two-splat swapped
  colors are order-invariant to `1e-5`, while no-op vanilla remains order
  sensitive. It also checks native fast/overflow backward behavior, bounded
  tape outputs, and full-tape color/scalar-gradient parity.
- Mechanical train smoke:
  `src/train_configs/local_mac_softmax_gs_noop_smoke_64_4f_128splats.jsonc`
  completes one MPS optimizer step with Softmax-GS disabled.
- Matched tiny trainability smokes:
  `src/train_configs/local_mac_softmax_gs_enabled_smoke_32_2f_64splats.jsonc`
  and
  `src/train_configs/local_mac_softmax_gs_noop_smoke_32_2f_64splats.jsonc`
  both complete one MPS optimizer step.
- MPS large-memory cross-attention fallback:
  `tests/test_mps_safe_cross_attention.py` covers parity against
  `nn.MultiheadAttention` and an MPS smoke at 40,960 memory tokens. A
  128px/16f forward/tape smoke now completes where the old path aborted.
- Practical 128px/16f stride16 comparison configs:
  `local_mac_multicam_softmax_gs_noop_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc`
  and
  `local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc`.

Not implemented:

- Learned per-Gaussian `beta/gamma` or generalized boundary-shape parameters.
- F32 feature-splat Softmax-GS.
- A repeated heldout-PSNR win beyond the tiny 128-splat row.

Current decision:
    Keep this as an active dynamic-GS renderer probe, but do not promote or
    benchmark it as a quality variant until a stronger heldout/source quality
    row repeats the win. The current repeat/scale rows are mixed or negative.
    Do not port to STAR or WorldFoam from the current evidence.

## STAR UVT Integration Sketch

Current STAR feature route:

- `src/train/star_uvt_feature_tube_model.py` defines screen-time Gaussian-like
  tubes with `ma`, `q_uvt`, `depth0`, `depth_beta`, `opacity`, and feature.
- `src/train/star_uvt_feature_overfit_trainer.py` routes direct feature Metal,
  sparse visual VJP, and projective interval paths.
- `src/train/star_uvt_projective_interval_backend.py` owns atlas producer
  settings, support guards, tail-alpha certificates, and visibility/order
  refresh logic.

The clean STAR version is not "drop Softmax-GS into the trainer." It is:

```text
for each pixel/tile/time cell:
    preserve current support and visibility certificates
    compute current contributor prefix as usual
    when contributors are near-coplanar/same-surface:
        compete feature/color absorbance by beta * exponent
        decay competition by depth separation / visibility stratum
        preserve output transmittance
```

Questions that must be answered before coding:

- What is the STAR equivalent of Softmax-GS `p`? Candidate: the negative UVT
  quadratic `-0.5 * qv`, not the final alpha after opacity.
- Does competition happen in feature space before `FeatureToColor`, or in RGB
  after colorization? Feature-space competition is cheaper and closer to the
  renderer, but RGB-space competition may better express boundary color
  assignment.
- Does `beta/gamma` live per tube, per trace-cell, or global? Per-tube is the
  direct paper analog; per-cell could be more stable for projective atlases but
  changes export semantics.
- How does the K-limited tape interact with cap128/cap256 tile capacity and the
  existing gradcache/fixedbin plans?

STAR falsification gate:

1. CPU dense reference on `dense_render_feature_tubes(...)` with `alpha=1`,
   `beta=0` parity to vanilla.
2. Synthetic two-overlapping-tube same-depth case: swapping input order should
   keep RGB/features stable.
3. Existing sparse-1500 checkpoint diagnostic: compare dense RGB, forced-alpha
   RGB, alpha `>0.1`, oracle target-background PSNR, and step time.

If dense RGB stays around the current `5.7-6.1` PSNR and alpha coverage stays
    near `0.43`, do not continue this as the visibility bridge.

## Why This Might Help

Dynamic GS:
    Sparse free/dynamic splats often need many small primitives to make sharp
    color changes. Softmax-GS may let the same primitive budget represent
    sharper boundaries and reduce popping under camera deltas.

STAR UVT:
    Current projective interval work already cares about depth-order roots,
    same-tile support, and omitted-alpha certificates. Softmax-GS gives a
    concrete local law for "same-surface overlap should not be arbitrary
    prefix-order alpha compositing." This may reduce streaks or order noise
    once support coverage is adequate.

## Why This Might Fail

- STAR's current hard blocker is missing support, not how existing support is
  blended.
- Softmax-GS has an explicit limitation for three or more overlapping distinct
  colors; STAR target-area cells often contain many contributors.
- The paper's K-limited forward tape is natural for CUDA/desktop 3DGS, but
  DynaWorld's Metal feature path already fights memory and atomic pressure.
- Current fast-mac projection plumbing uses rank-depths, so a naive port would
  produce a fake `gamma` decay signal.
- For F32 feature splats, softmax competition may choose feature vectors in a
  way that helps RGB but harms V-JEPA target-grid loss, or vice versa.

## Proposed Work Order

1. Keep ingestion complete: PDF, source, text, converted Markdown, and this
   note are enough for future implementation.
2. Dynamic-GS CPU/Torch reference prototype:
   implement a slow per-ray Softmax-GS composition for tiny tensors and write
   order-invariance/transmittance tests.
3. Dynamic-GS Metal fork:
   add one `fast_mac` variant behind config, with parity fallback to vanilla.
4. Matched dynamic smoke:
   compare the current fixed-512 dynamic-gsplat media comparator against the
   Softmax-GS variant.
5. STAR CPU reference only after dynamic smoke:
   try same-depth two-tube and sparse-1500 checkpoint diagnostics before any
   Metal STAR port.

## Bottom Line

Softmax-GS is worth integrating as a renderer experiment, especially for
dynamic GS. For STAR UVT, it is a promising compositing law but probably not
the next scale bridge until coverage/support is improved. Treat it as a
post-support quality/stability lever or as a projective-visibility research
branch, not as a replacement for birth/split or mixed same-view plus heldout
training.
