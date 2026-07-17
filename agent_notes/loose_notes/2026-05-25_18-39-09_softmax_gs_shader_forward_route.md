# Softmax-GS Shader Forward Route

## Context

The Softmax-GS paper is now ingested under
`research_notes/gaussian_splatting_papers/`, with short-term and long-term
plan docs. This chunk moved from planning into the first dynamic-GS renderer
probe.

Relevant plan docs:

- `research_notes/gaussian_splatting_papers/2026-05-25_short_term_softmax_gs_plan.md`
- `research_notes/gaussian_splatting_papers/2026-05-25_long_term_splats_vs_worldfoam_plan.md`
- `research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_dynaworld_integration.md`

## Current Model

Softmax-GS is best treated as a dynamic-GS renderer/compositing ablation first.
It is not yet a STAR UVT support fix and not a reason to move the project to
WorldFoam. The useful question is narrow:

```text
Does same-surface overlap-aware splat compositing improve quality or stability
enough to justify a trainable fast-mac variant?
```

The forward shader now answers the smallest mechanics question positively:
same-depth two-splat ordering can be made invariant on MPS while preserving a
vanilla-equivalent no-op route.

## What Changed

Implemented pieces:

- Tiny Torch reference:
  `research_experiments/softmax_gs/reference.py`
- Reference report:
  `research_experiments/softmax_gs/REFERENCE_REPORT.md`
- Reference tests:
  `tests/test_softmax_gs_reference.py`
- Fast-mac depth plumbing:
  `src/train/renderers/fast_mac.py`
- Metal shader fork:
  `third_party/fast-mac-gsplat/variants/v5_softmax_gs/`
- MPS forward regression:
  `tests/test_softmax_gs_metal_forward.py`
- No-op smoke config:
  `src/train_configs/local_mac_softmax_gs_noop_smoke_64_4f_128splats.jsonc`

The `v5_softmax_gs` fork now accepts sorted center-camera depths and carries
them through eval, train-state forward, and overflow forward kernels. The
forward transform is enabled by `softmax_gs_enabled`; backward remains disabled
for that enabled mode.

## Evidence

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q
```

Result:

```text
15 passed
```

No-op MPS parity against vanilla `v5`:

```text
forward_max_abs = 0.0
grad_max_abs = 2.9802322387695312e-08
```

Forward-only Softmax-GS same-depth swapped-order check:

```text
vanilla_swap_max_abs = 0.47309088706970215
softmax_swap_max_abs = 2.384185791015625e-07
vanilla_center_ab = [0.7852748632, 0.0974588320, 0.3121839762]
vanilla_center_ba = [0.3121839762, 0.0974588320, 0.7852748632]
softmax_center_ab = [0.5487294197, 0.0974588692, 0.5487294197]
softmax_center_ba = [0.5487294197, 0.0974588692, 0.5487294197]
```

No-op trainer smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_noop_smoke_64_4f_128splats.jsonc
```

Result:

```text
DynamicVideoTokenGSImplicitCamera training complete (W&B disabled).
```

## Assumptions

- First dynamic-GS Softmax-GS row can keep `beta/gamma` fixed scalar knobs.
- `p = -0.5 * Mahalanobis2D` is the right first exponent signal.
- `center_camera_z` is adequate for the first dynamic-GS gamma decay signal.
  Pixel-affine depth can wait.
- STAR UVT should not receive a Metal port until dynamic-GS has trainable
  evidence or STAR support coverage improves.

## Math / Implementation Invariant

For a two-splat same-depth case, the shader must preserve the ordinary
alpha-over final transmittance while making contribution weights symmetric when
the exponent signals are equal.

The current forward update uses:

```text
T_orig = T * (1 - alpha_cur)
a_past = 1 - T
w_cur = 1 / (1 + exp(beta * (p_past - p_cur)))
soft_cur = w_cur * alpha_cur
soft_past = (1 - w_cur) * a_past
```

It then rescales the pair so the post-correction transmittance matches
`T_orig`. This is why the red/blue equal-depth case converges to equal red and
blue center contributions instead of prefix-order alpha-over.

## Branches

Hypothesis:
    Softmax-GS will improve dynamic-GS quality/stability cheaply.

Why it might be true:
    The synthetic shader test hits exactly the paper's order artifact. Dynamic
    free splats likely contain many near-coplanar, overlapping color-boundary
    regions.

What would make it false:
    Real training might learn around the artifact already, or the backward tape
    might add too much cost for the quality gain.

Cheap test:
    Implement backward for fixed scalar `beta/gamma`, then run the matched
    1-clip dynamic smoke against vanilla.

If supported:
    Promote a small dynamic-GS Softmax-GS benchmark row and consider F32 later.

If invalidated:
    Park Softmax-GS as a renderer note and return attention to STAR support and
    WorldFoam geometry.

Hypothesis:
    This should not be moved directly into STAR UVT.

Why it might be true:
    STAR's current failure is missing support/coverage. Softmax-GS changes the
    blend among existing contributors.

What would make it false:
    A CPU STAR diagnostic shows dense RGB improves despite unchanged alpha
    coverage, or support is already adequate on a new STAR route.

Cheap test:
    After dynamic-GS result, add a CPU dense STAR two-tube diagnostic and a
    sparse checkpoint diagnostic.

## Backtracks

Earlier concern:
    This might be "just final compositing" and therefore simple to add.

Status:
    Weakened. Forward is local to the rasterization/compositing loop, but
    trainability needs prefix state and a backward replay/tape. It is not a
    pure postprocess after the 3D rasterizer has emitted a finished image.

Earlier concern:
    Softmax-GS might not need order.

Status:
    Partially true only for the same-depth two-splat invariant. The algorithm
    still walks the sorted per-pixel contributor order and treats the prefix as
    a moving "past" entity. It reduces order artifacts; it does not remove the
    need for a contributor stream.

## Falsification Tests

1. Backward parity microtest:
   Compare finite differences or Torch reference gradients on tiny two/three
   splat cases against the Metal backward for fixed scalar knobs.
2. No-op preservation:
   `softmax_gs_enabled=false` must keep matching vanilla forward and gradients.
3. Trainable smoke:
   One-step dynamic-GS smoke with `softmax_gs_enabled=true` must complete and
   produce finite gradients.
4. Matched quality row:
   The Softmax-GS trainable row must improve heldout/source metrics or reduce
   primitive count at matched quality. Source-only prettiness is not enough.

## Decision Implications

Immediate next work is the backward/tape design, not STAR or WorldFoam. The
short-term plan remains:

```text
dynamic-GS Softmax-GS backward -> trainable smoke -> matched dynamic row ->
only then STAR CPU diagnostic
```

Do not update `BASELINES.md` from the current results. They are mechanics and
shader-contract evidence, not model quality baselines.

## Open Questions

- Should backward cache a compact per-pixel prefix tape, or recompute the
  forward prefix per splat during backward to save memory?
- Do we need gradients for `beta/gamma` in the first row, or are fixed scalar
  knobs enough to answer the renderer question?
- How expensive is Softmax-GS in overflow-heavy tiles on real dynamic-GS
  scenes?
- Does the synthetic order-invariance win translate to heldout camera stability
  or only source-view boundary appearance?
