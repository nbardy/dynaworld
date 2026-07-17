# Softmax-GS Reference Report

Date:
    2026-05-25

Purpose:
    Establish a tiny, auditable Torch implementation of the Softmax-GS
    per-ray forward pass before changing fast-mac/Metal shaders.

Scope:
    The reference consumes already-sorted, already-evaluated per-pixel values:

    ```text
    absorbance a[k]
    exponent p[k]
    depth d[k]
    feature/color c[k]
    beta, gamma
    ```

    It deliberately does not implement projection, tile binning, GEF boundary
    sharpness, backward kernels, or per-Gaussian learned parameters. The
    fast-mac fork now has a separate Metal ABI that lowers the bounded tape
    contract for projected splats.

Covered cases:

1. Vanilla alpha-over parity when Softmax-GS is disabled.
2. Same-depth two-splat order swap invariance.
3. Separated-depth fallback toward vanilla alpha-over when gamma decay turns
   softmax competition off.
4. Final alpha/transmittance preservation relative to original alpha-over.
5. Finite gradients through absorbance, exponent, beta, and gamma.
6. Contribution-tape reconstruction: `weights @ features` matches the
   Softmax-GS color for arbitrary feature vectors.
7. Contribution-tape color gradients: `dL/dfeatures[k]` equals
   `weights[k] * dL/dcolor`.
8. Bounded contribution tape selects the exact top-K final color weights while
   returning selected rows in front-to-back/ray order.
9. Bounded contribution tape exposes a residual weight that bounds each
   output-channel error when omitted feature values are in `[0, 1]`.

Implementation note:
    The code follows the supplemental Algorithm 1 sequential approximation:
    collapse prior splats into a moving "past" entity, compete past against the
    current splat with a softmax over exponents, decay the effect by depth
    separation, then rescale past/current absorbance so final transmittance
    matches ordinary alpha-over.

    The two-absorbance correction is implemented from the algebraic
    order-invariance equations in the main paper. This makes the two-splat,
    same-depth case invariant under input-order swap; it is the behavior the
    shader path must preserve.

Status update:
    The reference has now been carried into a `v5_softmax_gs` Metal forward
    path and native fast-tile plus overflow recompute backward. The MPS shader
    matches vanilla in no-op mode and makes the same-depth two-splat
    swapped-order case invariant within float tolerance. The first
    contribution-tape contract is executable in tests, and the native recompute
    backward matches the Torch reference on tiny projected MPS scenes for
    means, conics, colors, opacities, and depths with nonzero `gamma`.
    The bounded reference tape is now executable too: it selects top-K final
    contribution weights and returns the residual mass needed for an explicit
    approximation/error contract. The `v5_softmax_gs` Metal fork now exposes
    `rasterize_softmax_gs_bounded_tape(...)` and passes fast-tile plus
    forced-overflow MPS tests against that reference. The bounded tape now has
    backward consumers: when `softmax_gs_tape_k > 0`, color gradients and
    selected geometry/opacity/depth scalar gradients are accumulated from the
    tape. Full-tape fast and forced-overflow tests match the Torch reference
    for means, conics, colors, opacities, and depths. Bounded-K scalar
    gradients are explicitly approximate.

Numerical note:
    The absorbance-pair rescale now uses the rationalized quadratic form:

    ```text
    scale = 2 * target_absorbance / (pair_sum + sqrt(discriminant))
    ```

    This avoids float32 cancellation in small-current-absorbance cases and
    keeps the Torch fallback aligned with the paper invariant.

Next gate:
    Measure the residual/quality tradeoff for bounded scalar tape sizes before
    larger quality rows. K=8 moves the route off full recompute but is a poor
    50-step source-view diagnostic; K=16 recovers the old seeded no-op/enabled
    source-view bracket while running through the selected-row tape path. K=32
    trains but does not improve that endpoint, so K=16 is the current setting
    for the next matched dynamic-GS quality row.
