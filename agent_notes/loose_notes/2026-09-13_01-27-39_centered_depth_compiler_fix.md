# Centered source depth now survives production compilation and fallback

The preceding turn established a causal diagnosis using a per-sample
intervention. This turn integrates the correction into the actual compiler,
checks cached training updates, and reruns the saved world without monkeypatches.
STAR commit `048f797` contains the compiler/fallback change. This closes the
demonstrated numerical parity defect; excessive fallback and scaling remain
open. No new training run or paper-acceptance row is claimed.

## Implementation

UVT lowering retains `depth_reference_uvt`, a float32 [N,7] tensor containing
the original center (u,v,t), anchor depth, and three depth slopes. Ambiguous
fallback ordering evaluates the source-centered expression. Stable tie keys
use source primitive ids rather than renumbered trace-table positions.
The native interval kernel and its ABI are unchanged.

Atlas time slicing, topology rebinning/stratification, quadrature remapping,
CPU fallback copies, and native-autograd state reconstruction preserve this
metadata. Topology-only reconstructions now use dataclass replacement to
avoid dropping evaluator fields. Direct edits to polynomial center/depth or
spatial depth coefficients reject stale source-depth metadata; a caller must
regenerate the UVT projection, or explicitly choose polynomial depth by
removing the UVT reference. World optimization already regenerates the
projection; it does not require differentiating discrete sort choices.

The root trainer cache had a second live-UVT producer that would have omitted
the new field. It now generates current source-depth data on each tensor
update and preserves it through cached topology changes, including empty
atlases. The actual live-update/refresh/render regression changes source
depth and compares the resulting image with direct UVT compositing.

Both logical tensor accounting and serialized retained state include the
new tensor. At N=256 it adds 7,168 payload bytes: retained tensor payload grows
from 25,600 to 32,768 bytes. F4 total serialized atlas size is 981,498 bytes,
including topology/container overhead. Older six-tensor artifacts still
verify: the actual historical F4 file hash
`6be5a28b07a7e7281821b0de83a5a0648d30582ebb2e5997993d9a9586ab3b82`
was rechecked through the updated existing verifier.

## Saved-world runtime results

All rows use the unchanged strict-loaded September 11 world:

- File SHA: `8ca44d076cf17e6aed274f2986a5cb033a7220fa2a1c14a41ca1e589345949f9`.
- Logical world SHA: `ccf3d00cfe5463a8b14b2dccfb36ee6e1b2695fdb392cb29e08a7e779e50c64a`.
- 256 tubes, full interval four frames, cam06, 96x128, peak-splat alpha,
  8x8x2 tiles/capacity128, alpha threshold1/255, transmittance1e-4,
  max alpha.99, black background, static dataset-lens camera.

| Final-code row | Max RGB error | Global normalized world VJP | Max parameter-group VJP | Fallback |
| --- | ---: | ---: | ---: | ---: |
| F4, indices [0,1,2,3] | 6.8545341e-7 | 7.3183136e-7 | 6.8855594e-7 | 38.8889% |
| F3, indices [0,2,3] | 6.8545341e-7 | 9.3419876e-7 | 9.9598826e-7 | 38.1481% |

Every image, loss, gradient-coverage/nonzero, global VJP, per-parameter VJP,
and world-state identity check passes without intervention. All seven world
parameter groups are covered. Both rows remain `accepted:false` solely
because fallback exceeds the unchanged 20% budget. The F3 selected-time
full-atlas versus chunk-sliced test also passes: images are identical,
global normalized world-VJP difference4.0148675e-7, maximum parameter-group
difference4.3295829e-7, and the world state is unchanged.

An earlier production F4 check, before strengthening the stale-metadata
guard, also passed (RGB6.85e-7, globalVJP7.20e-7). The final-code rows above
supersede it for the committed core change. Their tiny VJP variation is
consistent with direct atomic accumulation; no timing comparison is claimed.
The later root cache fix is exercised separately by its actual runtime test;
the frozen report does not use that cache helper.

## Validation and resources

The new `tests/test_world_tube_depth_order.py` checks a colored near-tie
fixture against uncapped UVT images and cotangents, whole/sliced frames,
stable source-id ties after trace-table reordering, stale coefficient edits,
source-depth updates, and retained byte accounting. Two actual Metal cases
check mixed-fallback images and gradients with and without pixel-varying
depth. These fail on the old depth-order behavior; they are not tests of a
preferred helper shape.

- Core atlas/reference/visibility/storage CPU gate: 81 passed, 20 explicitly
  skipped while MPS availability was masked. No skipped case counts as Metal.
- Guarded focused test run on real Metal: 7 passed, including both native
  mixed-fallback gradient cases (before the later cache regression was added).
- Producer/cache follow-up: 30 passed, 12 skipped with MPS masked, including
  the live cached-update regression. Its first attempt exposed missing alpha
  fields in the new test config; adding the required fields fixed the fixture.
- Final F4 and non-unit F3 existing-contract runs: completed on Metal as above.

All native jobs were sequential under the unchanged 600-second wall,
3-GiB tree/launcher RSS, 2-GiB MPS, host/swap/disk gates. Maximum sampled
tree/launcher RSS was 2,024,767,488 bytes for the final two-row job; each
job reports zero new swap and no guard trip. No native build, application
termination, training, or upload was performed. These mechanical correctness
probes do not create W&B runs. All process handles are terminal.

## Artifacts, ownership, and next step

Retained root: `outputs/benchmarks/2026-09-13_frozen_depth_fix/`.
Final reports: `frames_4/report.json` and `frames_3/report.json`; each directory
contains the real serialized atlas. Launch/source/preflight/resource receipts,
`windows.log`, `metal_tests.log`, and the first production `report.json` remain.
Focused test logs are also copied into this directory, along with artifact and
source hashes. Scripts use the retained strict checkpoint/bundle setup and
the existing guarded launcher; they do not patch runtime methods.

Unrelated existing root/STAR WIP is preserved. The STAR commit includes its
HEAD-existing logical payload accounting; an additional one-line accounting
update inside the pre-existing uncommitted interaction-byte helper remains
within that WIP. The reports bind the working runtime sources, so these are
diagnostics rather than clean-source publication evidence.

Next diagnose the remaining fallback distribution: distinguish unavoidable
ambiguous order in this barely trained world from overly conservative
visibility bounds before changing policy or increasing the scaling sweep.
Do not lower the budget to accept these rows or treat numerical parity as a
sublinear-speed result. Existing successful quality controls remain intact.
Paper counts and BASELINES standings are unchanged; the overnight goal stays
active.
