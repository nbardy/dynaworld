# Native SPD(4) parallel training and bounded Metal smoke

- **Time:** 2026-07-27 16:19:33 +0900
- **Lane:** World Tubes / STAR UVT representation and engineering
- **Author/role:** Codex coordinator, implementation integrator, and hostile
  scope reviewer
- **Objective:** Turn the strict `mu4 + SPD(4)` reference object into a
  separately selectable trainable source beside the original legacy tube,
  preserve a controlled common raster back end, and establish the smallest
  safe CPU and Metal gates needed before longer comparison runs.
- **Why attempted:** The mathematical source/compiler existed, but the user
  correctly asked whether it had actually landed in parallel with the
  original implementation. A reference oracle alone could not answer that or
  support a fair empirical comparison.

## Inputs and predecessors

- `2026-07-27_04-56-24_spd4_worldfoam_four_pass_implementation_plan.md`
- `research_experiments/spd4_world_tubes/{model.py,compiler.py,retained_fiber.py}`
- The original restricted `WorldTubeModel` and six-field STAR UVT/Metal ABI.
- `multicam_heldout_compare.py` and the unified paper runner.
- Coffee Martini four-frame, two-step smoke protocol and existing local safety
  guidance after the unified-memory/kernel-task incident.

No publication-scale MPS run was authorized or attempted. The 300-frame
progressive protocol remained closed. Metal jobs were deliberately executed
one at a time in fresh processes.

## Four-pass review outcome

1. **Representation pass:** A native strict-SPD(4) source is useful only if
   its full spatial conditional covariance and spacetime tilt survive
   compilation. A cosmetic wrapper around the legacy tuple would not test the
   hypothesis.
2. **Fairness pass:** Same atom count is scientifically useful but not
   parameter matched: legacy has 14 total trainable scalars/atom and full
   SPD(4) has 18. Therefore both 256-vs-256 and 256 legacy vs 199 SPD(4)
   comparisons are required. Initialization also needed an explicit depth
   precision so full SPD(4) did not silently begin with a much thicker
   projected object.
3. **Shader pass:** Full covariance does not require a second geometry
   rasterizer. It can be compiled to the existing `ma`, packed `q_uvt`,
   `depth0`, `depth_beta`, `opacity`, and `color` ABI. Reusing this back end
   controls the comparison and reduces engineering risk.
4. **Safety/claim pass:** A tiny Metal VJP and a four-frame/two-step
   end-to-end smoke can establish dispatch, finite gradients, accounting,
   overflow, and memory behavior. It cannot establish convergence, visual
   quality, or speed.

## Implemented source model and math

The production chart stores a conditional spatial covariance
\(C=L_xL_x^\top\), spacetime tilt \(v\), and temporal variance \(c>0\):

\[
\Sigma_4=
\begin{bmatrix}
C+cvv^\top & cv\\
cv^\top & c
\end{bmatrix}.
\]

This is strict SPD whenever \(C\) is strict SPD and \(c>0\). Conditioning on
time gives

\[
\mathbb E[x\mid t]=x_0+v(t-t_0),\qquad
\operatorname{Cov}[x\mid t]=C.
\]

Thus affine motion is derived from one native 4D covariance rather than being
added to the renderer as an independent trajectory law. The trainable source
has 14 geometry DOFs:

- four spacetime-center coordinates;
- six entries of a 3x3 conditional spatial Cholesky chart;
- three spacetime-tilt coordinates;
- one temporal log-precision/scale coordinate.

RGB and opacity bring the total to 18 scalars/atom. The original model remains
14 total scalars/atom.

Added production implementation:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/spd4_world_atom.py`

Added explicit selector and metadata:

```text
--uvt-world-representation {legacy_tube,full_spd4}
```

`legacy_tube` remains the default. Added independent initialization controls:

```text
--uvt-spd4-init-precision-z <positive scalar>
--uvt-spd4-min-spatial-scale <positive scalar>
```

The direct multicamera benchmark and unified paper runner propagate and
validate the representation and initialization identity. Old reports missing
the representation field normalize only to the legacy default. Unsupported
moving-camera/D-NeRF use of full SPD(4) fails early instead of silently
lowering through the static-camera approximation.

## Capacity certificate

`research_experiments/spd4_world_tubes/run_capacity_gate.py` fits one legacy
atom and one native SPD(4) atom to the projected precision observations from
one target native atom under three static camera charts.

The camera observation design has rank six for the six independent entries of
the symmetric conditional spatial covariance. The target contains nonzero
depth variance and cross-covariances outside the legacy axis-aligned XY
subclass. Exact linear analysis gives:

- full-SPD observation residual: `1.26e-15`;
- best legacy observation RMSE: `4.2026`;
- initial SPD4/legacy loss ratio: `0.998517`;
- legacy final MSE: `2.066366e-4`;
- full-SPD(4) final MSE: `1.157288e-13`;
- final MSE ratio: approximately `1.79e9`;
- capped PSNR gap: `83.15 dB`.

Artifact:
`artifacts/foundation_gates/spd4_native_multiview_capacity_cpu.json`.

**Claim status:** proved algebraic identifiability of all six symmetric
conditional spatial-covariance DOFs for this camera design, plus
computational evidence that the implemented optimizer reaches the full-model
solution. This is a controlled representation-capacity certificate, not a
public quality result and not parameter matched.

## Tiny Metal forward and VJP

The bounded gate used 8x8 pixels, two time samples, and two SPD(4) atoms.
Compiled records were sent through the unchanged STAR Metal forward/backward.

- Maximum forward absolute error versus the thresholded official
  `brute_force_render_uvt_tubes` reference: `2.235174e-8`.
- All SPD source parameter groups had finite, nonzero gradient norm:
  `x0=.04398`, `t0=.003668`, raw spatial scales `.01195`, spatial
  off-diagonals `.004183`, tilt `.009491`, temporal precision `.0004868`,
  opacity `.004204`, and color `.001746`.
- MPS current allocation was 10,752 bytes; sampled driver allocation was about
  29.2 MB.
- Conditional depth variance was finite and positive.

The existing hard-order backward intentionally returns zero cotangents for
`depth0` and `depth_beta`; the source still receives geometry gradients
through the projected mean and UVT quadratic. This limitation is now tested
and must not be mistaken for a differentiable visibility/order method.

**Claim status:** numerical Metal interface and source-VJP evidence at tiny
scale. It is not a throughput result.

## Sequential end-to-end Coffee Martini smokes

All rows used static multicamera `cam04` and `cam09` for training, `cam06` for
heldout evaluation, 128px images, four frames, two optimizer steps, 30,720
rasterized pixels, and fresh sequential processes.

| Row | Atoms | Parameters | Train wall | Steady forward | Backward | Heldout PSNR | Driver |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 256 | 3,584 | 0.631 s | 0.159 s | 0.165 s | 6.240 dB | ~37.6 MB |
| SPD4 near-planar | 256 | 4,608 | 1.052 s | 0.182 s | 0.370 s | 6.224 dB | ~37.6 MB |
| SPD4 isotropic | 256 | 4,608 | 0.805 s | 0.183 s | 0.290 s | 6.387 dB | ~37.6 MB |
| SPD4 parameter-matched | 199 | 3,582 | 0.819 s | 0.237 s | 0.156 s | 6.181 dB | ~37.6 MB |

Every row had zero tile overflow. Same-count isotropic SPD4 generated more
projected tile/tube pairs than legacy, so support/tile pressure, rather than
the roughly 12 KB of extra parameter/Adam storage, is the larger memory risk
for scaled experiments.

Artifact: `artifacts/spd4_parallel_smoke/summary.json`.

**Claim status:** end-to-end mechanical evidence for independent selection,
shared raster dispatch, memory plateau, same-atom accounting, and
matched-parameter accounting. The metrics after two optimizer steps are
reported to preserve the run, but support no quality, convergence, or speed
ranking.

## Failures and corrections preserved

1. The first attempted initialization comparison gave SPD4 an unmatched depth
   scale. The model and runners were changed to carry an explicit independent
   `init_precision_z`; the near-planar legacy-lift control uses a large value,
   while the isotropic exploration uses the XY precision.
2. The first dense CPU/Metal comparison appeared to differ by about `0.0032`.
   The dense calculation retained Gaussian tails below the renderer's alpha
   threshold. Comparing against the renderer's official thresholded
   brute-force reference reduced the maximum error to `2.235174e-8`. The
   original discrepancy was a reference-semantics error, not silently erased
   as a shader success.
3. The first real legacy smoke completed training but failed during LPIPS
   evaluation because the locked experiment dependencies were absent from the
   local environment. The experiment dependency group was installed from the
   lockfile and the run was repeated in a fresh process.
4. Python 3.14 treated a literal `%` in one argparse help string as
   interpolation syntax. It was escaped as `%%`; parser tests now cover the
   CLI.

## What landed and what did not

Landed locally in parallel:

- default original legacy source;
- opt-in trainable native SPD(4) source;
- common static-camera STAR Metal forward/backward;
- direct and unified-runner representation axes;
- same-count and matched-parameter protocols;
- CPU capacity, finite-gradient, and tiny-Metal gates;
- bounded sequential real-data smoke artifacts.

Not landed:

- simultaneous dual-job execution (intentionally avoided for host safety);
- moving-camera full-SPD(4) compilation;
- production use of conditional depth variance;
- differentiable depth-order changes;
- physical Beer--Lambert primitive alpha/VJP;
- retained-fiber colored-overlap production fallback;
- longer public-scene convergence, multi-seed quality, or speed evidence.

The working tree changes are not yet a durable remote landing: the new
production source is untracked inside the nested `fast-mac-gsplat` repository,
and neither nested nor parent changes have been committed/pushed.

## Precise next actions

1. Preserve the current legacy-default behavior and run both lanes only into
   distinct output roots.
2. On an adequate host, run a bounded convergence ladder before any
   progressive-512/full-300-frame matrix: same atoms first, then matched
   parameters, with tile-pair and driver-memory plateaus recorded.
3. Add a non-piecewise camera-gauge/atlas compiler for moving cameras and test
   it against the affine reference before enabling D-NeRF.
4. Add variance-aware order certificates, then a separately named physical
   Beer--Lambert mode and retained-fiber fallback rather than conflating them
   with peak-splat opacity.
5. Commit the nested implementation and parent integration intentionally once
   the user chooses the desired durability boundary.
