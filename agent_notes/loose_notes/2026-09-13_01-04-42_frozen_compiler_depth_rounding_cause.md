# The frozen F4 image/VJP failure is an ambiguous-depth rounding discontinuity

The previous goal turn answered the user's source-overfit question but did not
advance experiment state. This turn reproduced the frozen failure, localized
it, checked a causal intervention, and ran the existing full image/world-VJP
contract on that intervention. No training or production renderer change was
needed. This is new diagnostic evidence, not accepted compiler scaling.

## Frozen input and unchanged contract

The input is the September 11 two-update, 256-tube Coffee Martini world with
four frames at 96x128, evaluated at heldout cam06. Checkpoint file SHA-256 is
`8ca44d076cf17e6aed274f2986a5cb033a7220fa2a1c14a41ca1e589345949f9`;
logical world SHA-256 is
`ccf3d00cfe5463a8b14b2dccfb36ee6e1b2695fdb392cb29e08a7e779e50c64a`.
The strict loader checked both identities. Rendering uses 8x8x2 tiles,
capacity 128, peak-splat alpha, threshold 1/255, max alpha .99, black
background, and transmittance threshold 1e-4. Static dataset-lens projection
is used by both routes. Current STAR HEAD is a20fc97 plus existing WIP; no
production source was edited in this turn.

## Measurements

| Quantity | Original saved failure | Centered-depth intervention |
| --- | ---: | ---: |
| Maximum RGB error | 4.6917796e-4 | 6.8545341e-7 |
| Global normalized world VJP error | 5.2766229e-4 | 7.0533735e-7 |
| Maximum parameter-group normalized VJP error | 3.8248538e-3 | 6.7818462e-7 |
| Color parameter normalized VJP error | 3.8248538e-3 | 2.3153064e-7 |
| Fallback fraction | 0.3888889 | 0.3888889 |

The fresh unmodified image replay exactly reproduces the old maximum error.
All 349 RGB components above 1e-5 occur in fallback regions. The native fast
regions have maximum error 6.8545341e-7 and zero components above 1e-5. Their
mean absolute error is 1.8278135e-8. This localization rejects a general native
interval-forward failure as the explanation of this particular artifact.

With the intervention, the existing report passes image, loss, global VJP,
every parameter-group VJP, gradient coverage/nonzero, and checkpoint identity.
All seven world parameter tensors are covered. Before/after route world hashes
are unchanged. The report remains `accepted:false` because fallback exceeds
the unchanged 20% budget: 280/720 active tile samples, 241/1783 cells. Paper
counts and BASELINES standings are unchanged.

## Causal counterexample

At frame 2 (centered time .5), pixel x=3,y=49, primitive ids 178 and 242 are
adjacent in the replay depth order. Their common anchor depth is
2.180581569671631. Slopes are -0.006274801678955555 and
0.006274800281971693; time anchors are -.48883867263793945 and
1.4888386726379395.

Replay evaluates d0 + b*(t-t0). Both float32 depths are 2.1743767261505127,
so the primitive-id tie break puts 178 first. The compiler stores d0-b*t0
and evaluates that intercept plus b*t. The first depth becomes
2.174376964569092 while the second stays 2.1743767261505127. The live-depth
fallback sort reverses the two primitives. These expressions agree in exact
arithmetic but cross a discrete ordering boundary after float32 rounding.

For two adjacent alpha-composited primitives, swapping a,b to b,a changes RGB
by T*alpha_a*alpha_b*(color_b-color_a), where T is transmittance before the
pair. At this pixel T=.7153020295, both alphas are about .28963113, and their
blue colors differ by .0078191161. The predicted blue change is
.00046917797149393617; the measured change is .0004691779613494873. Residual
is 1.0144e-11 in blue and 7.1531e-9 across RGB. The large output discrepancy
therefore follows directly from one rounding-induced depth swap, without a
large footprint or opacity approximation error.

The independent standard-library float32 reproduction is committed at
`research_experiments/paper_runner_suite/reproduce_frozen_depth_rounding.py`.
It preserves measured primitive values and checks the adjacent-swap identity.
It is an executable counterexample, not a regression claiming the bug fixed.

## Intervention and limits

The first intervention renders one sample at a time, replacing only the
atlas depth polynomial at that sample with depth computed in the original
centered form. It leaves footprints, colors, opacity and cell metadata intact.
All four images then fall below the existing image threshold.

The full VJP intervention changes only
`ProjectiveCellIntervalTrainerState.render_reference_with_fallback` during
the diagnostic process. It supplies centered source depth for each fallback
reference sample, preserving the normal differentiable atlas graph for
footprint, opacity and color. The order values are detached because the
current derivative treats sorting topology as fixed. The original method is
restored in a finally block. This fixture has zero spatial depth coefficients;
the probe explicitly asserts that restriction rather than claiming arbitrary
pixel-varying depth coverage.

This is not a production fix. The diagnostic adds per-sample work and carries
source depth data outside the atlas, so its existing timing/storage fields
cannot support a performance claim. Its report explicitly marks the
intervention and `performance_claim_eligible:false`. No threshold was loosened.
The next source change should preserve an agreed depth evaluation and stable
source-id tie policy through lowering, slicing, fallback, and updates, with
all additional state charged. It must exercise this saved world and a small
colored near-tie fixture, including world VJPs. Simply quantizing depth or
raising tolerances is not a demonstrated fix. Keep the production atlas
self-contained; do not ship this per-sample diagnostic as a faster compiler.
The separate excessive-fallback problem remains after numerical parity.

## Retained execution

Artifacts are under `outputs/benchmarks/2026-09-13_frozen_parity_diagnostic/`:

- `comparison.json`, `route_tensors.pt`: fresh failure localization and inputs.
- `swap_identity.json`, `minimal_reproduction.json`: independent swap check.
- `centered_depth_probe.json`, `centered_depth_images.pt`: image intervention.
- `centered_depth_vjp/report.json`: existing full contract with intervention.
- `retained_sha256.json`: hashes of the saved measurements and tensors.
- Three resource receipts plus launch/source/preflight files and stdout logs.

`launch.py` runs the three retained scripts sequentially under the existing
600-second, 3-GiB RSS, 2-GiB MPS, host, swap and disk guards. Peak sampled
tree/launcher RSS is 813,694,976 / 846,757,888 / 1,770,831,872 bytes. Every
job ends successfully, with no guard trip and zero new swap. These are
mechanical correctness probes, so no W&B run was created; no training,
network upload, native build, or application termination occurred. All process
handles are terminal. The broader overnight goal remains active.
