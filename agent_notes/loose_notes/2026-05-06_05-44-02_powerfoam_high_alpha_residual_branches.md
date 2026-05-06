# PowerFoam High-Alpha Heldout Residual Branches

Date: 2026-05-06 05:44:02 Asia/Ho_Chi_Minh

Scope: thinking note only. No code, TODO, baseline, key-learning, or existing
note edits. This records the current PowerFoam heldout finding and turns the
plausible mechanisms into falsifiable tests.

## Finding

Selected row:

```text
config:
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc

diagnostic:
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/heldout_error_diagnostics.json
```

The heldout failure is not mainly blank coverage:

- heldout PSNR / SSIM / L1: `12.5099 / 0.1169 / 0.1794`
- heldout alpha mean: `0.9776`
- alpha `> 0.9` fraction: `0.9708`
- alpha `< 0.05` fraction: `0.0174`
- selected sample: heldout `camera_0040`, frame `0`
- sample support-hit fraction: `0.99994`
- high-residual support-hit fraction: `0.99969`
- alpha `>= 0.5` pixels contain `95.67%` of total residual
- top-20%-residual, alpha `>= 0.5` pixels contribute `43.61%` of total residual
- that high-alpha/high-residual bucket has mean L1 `0.4301`

Current belief:

```text
The model confidently renders an opaque, supported heldout surface, but the
opaque explanation is wrong. The next lever is not "make alpha nonblank"; it is
"why is a supported opaque explanation wrong?"
```

## Caveats

- The support panel is a decoded center/radius ray-sphere proxy, not true
  COLMAP per-track unique-camera support.
- The finding is for this selected regular-triangulation row at checkpoint step
  `0`; do not generalize it to every PowerFoam path.
- "Wrong residual" is broader than "wrong depth." High-alpha wrong pixels can
  come from depth/order error, normal/material mismatch, SV texel drift,
  view-conditioned color drift, or a diagnostic artifact.
- Color/exposure is a partial lever, but prior affine/oracle color bounds did
  not reach the SSIM gate. Tiny support/material thaw improved source metrics
  without improving heldout. Both weaken cheap coverage/color/schedule-only
  explanations.

## Branch 1: Depth Or Order Alignment

Hypothesis:
    Heldout rays hit opaque support, but the selected layer is at the wrong
    depth, traversed in the wrong order, or composited with a wrong local
    normal. Alpha is high because something is there, not because the right
    surface is there.

Why plausible:
    The residual concentrates in high-alpha pixels. The high-residual bucket
    has higher mean normal-distance than the all-high-alpha bucket (`0.0967`
    vs `0.0500`). Earlier graph evidence also showed Cech/AABB is not a
    superset of regular-triangulation edges on the frozen real checkpoint.

Falsifiers:
    Residual does not correlate with rendered depth, normal-distance,
    traversal count, nearest power margin, or train-view reprojection error.
    A depth/order perturbation leaves heldout L1/SSIM unchanged while material
    or color corrections explain the residual.

Cheap tests:
    1. Extend the diagnostic JSON/panel, not training, with residual stratified
       by rendered depth, normal-distance, traversal count, and nearest power
       margin.
    2. Run a tiny two-layer synthetic where both layers are high-alpha but only
       one is heldout-correct; verify the diagnostic marks wrong-layer pixels
       as high-alpha/high-residual.
    3. Reproject rendered-depth pixels into nearest train cameras. If train
       reprojected color agrees with the render but not heldout GT, the
       heldout structure/order is likely wrong.

If supported:
    Prioritize traversal/order/topology or depth/normal objectives before
    appearance expressivity.

If invalidated:
    Treat opaque support as geometrically plausible and shift pressure to
    material coordinates, SV texels, or color calibration.

## Branch 2: Material Transport Under SV Texels

Hypothesis:
    Geometry support is present, but the `quaternion_height_sv_texel_surface`
    payload is attached to train-view-friendly texel coordinates instead of a
    heldout-stable material surface. The heldout ray sees supported geometry
    carrying the wrong local texture/feature state.

Why plausible:
    The selected row uses an SV texel surface. Source-only support/material thaw
    improved source metrics but did not improve heldout, consistent with local
    appearance absorbing train error without becoming cross-view material.

Falsifiers:
    Residual has no relationship to texel-site gradients, tangent-frame
    discontinuities, per-cell appearance variance across train views, or
    material-coordinate extrapolation distance. A simpler per-cell or low-rank
    material model is no worse on heldout with matched geometry.

Cheap tests:
    1. Freeze geometry and compare train-fit heldout residual for:

       ```text
       A. per-cell constant color/features
       B. current SV texel surface
       C. low-rank or smoothed texel field
       ```

    2. Add residual buckets by nearest-cell texel-gradient magnitude and
       tangent-frame discontinuity.
    3. For high-residual pixels, report train-view color disagreement for the
       nearest contributing cell.
    4. Preserve alpha/geometry but shuffle local texel payloads among nearby
       cells. If residual barely changes, this branch is weak.

If supported:
    Add material-coordinate or texel-smoothness diagnostics before changing the
    renderer. Any loss should be train-only and gauge-aware.

If invalidated:
    Stop spending complexity on SV transport for this row; return to
    depth/order or color.

## Branch 3: View-Conditioned Color Drift

Hypothesis:
    Geometry/material support is acceptable, but the heldout camera's color or
    exposure transform is outside the train-fitted appearance distribution.

Why plausible:
    Train-fit affine postprocessing materially improved heldout PSNR, so color
    drift is real.

Falsifiers:
    After train-only affine/background correction, the same high-alpha
    structural residual remains and SSIM stays below gate. Prior oracle
    color/background bounds already suggest this is secondary, not primary.

Cheap tests:
    1. Save the same residual panel after train-fit affine and oracle affine.
    2. Compare residual edge maps before/after affine. If low-frequency color
       improves but edge/shape residual remains, color drift is not the main
       blocker.
    3. Use leave-one-train-camera-out affine calibration to test whether a
       train-only color head predicts camera-heldout improvement.

If supported:
    Add explicit no-heldout-RGB camera/color calibration.

If invalidated:
    Keep affine as a reporting bound and focus on structure.

## Branch 4: Gauge, Dynamic, And Fluid Analogies As Tests Only

The useful analogy is not "make PowerFoam a fluid simulator." It is: an opaque
supported surface can still be wrong if internal coordinates, tangent frames,
or material payloads are gauge-inconsistent.

Tangent-frame gauge drift:
    Test whether high-residual projected cells have high tangent-frame
    discontinuity or graph-cycle holonomy:

    ```text
    holonomy(i -> j -> k -> i) =
        composed relative tangent-frame rotation around a local cycle
    ```

    If high-alpha residual does not concentrate on high gauge-defect cells, do
    not add gauge machinery.

Material conservation / transport:
    Across frames `0/4/8/12`, compute source-rate diagnostics without changing
    the renderer:

    ```text
    source_rate_f = ||f_i(t + dt) - transported_f_i(t)|| / dt
    source_rate_rho = |rho_i(t + dt) - transported_rho_i(t)| / dt
    ```

    This branch matters only if source-rate cells predict heldout residual. If
    there is no correlation, fluid/transport language is decorative here.

Dynamic repaint vs geometry motion:
    Compare payload change to cell-center motion:

    ```text
    geometry_motion_i = ||x_i(t + dt) - x_i(t)||
    payload_motion_i = ||f_i(t + dt) - f_i(t)||
    repaint_ratio_i = payload_motion_i / (geometry_motion_i + eps)
    ```

    Pursue motion/transport constraints only if high residual tracks high
    repaint ratio or a moving-sheet synthetic separates repaint from transport.

## Decision Rule

1. Do one diagnostic pass first: no training, no coverage-only patch.
2. If residual follows depth/normal/order, work on traversal/order/topology or
   depth/normal objectives.
3. If residual follows texel/gauge/material variables, freeze geometry and test
   simpler material decoders before adding renderer complexity.
4. If train-only affine predicts heldout-train splits, add no-heldout-RGB color
   calibration.
5. If none explain it, reopen the support proxy and map true per-track
   unique-camera support from PLY points to cells.

Current next action:

```text
Do not run another coverage-only or source-only thaw schedule until a diagnostic
shows high-residual pixels are actually missing support. Current evidence says
they are high-alpha and high-support wrong residuals.
```
