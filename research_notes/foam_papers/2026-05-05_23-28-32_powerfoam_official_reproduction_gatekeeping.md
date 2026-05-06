# PowerFoam Official Reproduction Gatekeeping

Date: 2026-05-05 23:28:32 Asia/Ho_Chi_Minh

Purpose: keep future PowerFoam claims honest. This is not a status recap. It is
a gatekeeping note for separating:

1. official PowerFoam reproduction gaps, which require upstream CUDA/Warp
   evidence and paper-scale quality gates, from
2. local Metal implementation gaps, which can be closed on this Mac with
   Torch/Metal/raytrace/4K/trainability tests.

Local Metal can be strong and still not be an official reproduction. Official
reproduction requires an upstream-generated fixture and paper-clean heldout
quality. A local approximation, even if fast and trainable, must not be allowed
to silently satisfy those gates.

## Claim Boundary

Allowed claim today:

```text
PowerFoam Metal has a local trainable bounded-cell raster/raytrace/backward
core with quaternion height+SV primitive coverage, cech_aabb topology input,
synthetic 4K forward/backward evidence, and real-scene training probes.
```

Forbidden claim until the gates below pass:

```text
PowerFoam proper / official PowerFoam is reproduced on Metal.
```

The forbidden claim remains forbidden even if:

- local Torch and Metal match each other,
- `cech_aabb` local adjacency passes local tests,
- synthetic 4K benchmarks and optimizer-step trainability pass,
- a real-scene probe trains without crashing,
- a local clean row improves after initialization.

Those are necessary local implementation gates. They are not upstream parity
and not paper acceptance.

## Gate A: Official CUDA/Warp Fixture

The missing official artifact is:

```text
research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json
```

It must be generated on a host with:

```text
linux
cuda
torch.cuda.is_available() == True
warp-lang
pinned upstream PowerFoam checkout at 96392252ebd0059fe6ca98881b62e12295d9242f
```

The canonical command is:

```bash
PYTHONPATH=src/train python research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py \
  --backend official \
  --upstream-root /tmp/powerfoam_official \
  --fixture research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json \
  --output research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json
```

The fixture must identify itself as official:

```text
metadata.backend == "official"
metadata.upstream_powerfoam_commit == "96392252ebd0059fe6ca98881b62e12295d9242f"
```

It must contain the same input scene contract as the local fixture:

```text
points
radii
densities
quaternions
normals
local_texel_sites
texel_sites
texel_height
texel_sv_axis
texel_sv_rgb
adjacency
rays
render_options
official_camera
```

It must contain official expected outputs for forward and backward:

```text
rendered
alpha
normal_distance
normal
contrib
visible_mask
loss
grad_points
grad_radii
grad_density
grad_normals
grad_texel_sites
grad_texel_height
grad_texel_sv_axis
grad_texel_sv_rgb
```

No Mac-local regenerated fixture can close this gate unless
`metadata.backend == "official"` and the expected tensors came from upstream
PowerFoam on CUDA/Warp. The local backend fixture is only a dry run for the
same camera and payload shape.

## Gate B: Official Parity Tests

After copying the official JSON back to this repo, run:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present \
  -q -rs
```

The first test proves the local Torch direct reference matches upstream
PowerFoam's official fixture on the shared primitive/camera scene:

```text
rendered
alpha
normal_distance
contrib
visible_mask
loss
direct shared gradients
```

The second test proves the Metal height+SV path matches the same official
fixture for the channels with shared parameterization:

```text
rendered
alpha
normal_distance
loss
grad_density
grad_texel_height
grad_texel_sv_axis
grad_texel_sv_rgb
```

This is intentionally narrower than "the whole paper system." Points, radii,
local sites, and quaternions do not have a one-to-one gradient comparison in
the Metal fixture because Metal derives world texel sites and frames from
local-site/quaternion parameters while the official/direct fixture stores
world-texel-site and normal gradients. Do not inflate this parity test beyond
what its shared parameterization proves.

Passing Gate B means:

```text
the local direct and Metal primitive paths agree with upstream CUDA/Warp on a
tiny official-compatible fixture.
```

It does not mean:

```text
paper-scale training, densification, pruning, resampling, losses, schedules,
or heldout reconstruction quality are reproduced.
```

## Gate C: Paper-Clean Heldout Quality

The paper-scale acceptance gate is separate from official primitive parity.
Current local evidence is below the acceptance thresholds:

```text
selected clean DeepView heldout PSNR: 10.8536  threshold: 13.0
selected clean DeepView heldout SSIM: 0.0766  threshold: 0.15
```

The current selected clean row is useful because it is distortion-consistent,
W&B-offline backed, selected after optimization, and has nonblank heldout
support. It is not good enough to call paper acceptance.

The acceptance verifier remains:

```bash
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py --allow-incomplete
```

The completion audit remains:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py --run-local-tests
```

If these report `ok: false`, do not summarize the lane as complete. The right
summary is "local Metal gates pass; official fixture and/or paper quality still
block full reproduction."

## Why Cech/AABB Topology Is A Gate

PowerFoam is not just a bag of independently rendered spheres. The cell
adjacency graph defines which support overlaps the renderer considers, which
neighbors can bound ray traversal, and which cell interfaces are available to
the optimizer.

The local correctness mode is `cech_aabb`:

```text
candidate edge: AABB_i intersects AABB_j
cech edge: ||p_i - p_j|| <= r_i + r_j
```

This matters for three reasons.

First, KNN is not a conservative replacement. A KNN graph can miss a true
overlapping neighbor, which changes the topology and can remove real cell
interfaces. Missing a true overlap is a correctness failure, not just a speed
tradeoff.

Second, a Cech/AABB graph can include false-positive edges. False positives
are usually safer than missed overlaps because the traversal can still see the
needed neighbor set, but they are not free: they can increase traversal work,
create noisy competition, and let local quality regressions hide behind "the
graph is conservative."

Third, regular triangulation remains a teacher/verifier, not necessarily the
first production path. The small SciPy-backed regular-triangulation parity path
can check whether Cech/AABB is conservative and whether false edges matter.
The current fast 4K path uses Cech/AABB because it is the selected practical
Metal path. If later work swaps topology, it needs frozen-state forward,
alpha, and gradient parity before training claims are allowed.

Gate condition for topology changes:

```text
reg_missing_edges == 0, or every missing edge has a documented geometric reason
forward/alpha deltas are within tolerance on frozen states
shared gradient parity remains within tolerance
4K traversal benefit is measured without heldout-quality regression
```

## Anti-Masquerade Rules

These rules prevent local approximations from being rebranded as official
reproduction.

1. Name the backend. Use `official CUDA/Warp`, `local Torch direct`, `local
   Metal`, `local regular-triangulation verifier`, or `local cech_aabb`.
   Never write "PowerFoam parity" without the backend pair.
2. Do not let skipped official tests count as passing evidence. A skip caused
   by the missing official fixture is a blocker.
3. Do not use a local backend fixture to satisfy an official fixture gate.
   `metadata.backend == "local"` is dry-run evidence only.
4. Do not let synthetic 4K trainability stand in for paper-scene quality. It
   proves an optimizer step at UHD, not clean heldout reconstruction.
5. Do not let EX4DGS, pretrained reconstruction, heldout RGB, heldout
   residuals, heldout masks, or heldout-derived depth/normal priors satisfy a
   paper-clean gate. They can be diagnostics only unless the result is labeled
   non-clean.
6. Do not tune mechanisms on the official heldout camera. Use source
   leave-one-camera-out or train-only diagnostics for selection, then evaluate
   the real heldout once per selected candidate.
7. Do not compress "Metal primitive parity" into "full paper system parity."
   Densification, pruning, resampling, official losses, schedules, and quality
   thresholds are separate gates.
8. Do not claim RadFoam/Radiant Foam status from PowerFoam work. It is a
   different upstream system and is not locally ported.

## External Artifact Handoff

The deterministic handoff manifest is:

```text
research_experiments/dynamic_foam/external_powerfoam_artifact_handoff.json
```

It is the source for the official-host command bundle and copy-back list. The
expected copy-back artifacts are:

```text
research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_aliked_n16rot_aliked_lightglue_minucam2.ply
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_aliked_n16rot_aliked_lightglue_minucam2.json
```

The geometry artifact is separate from the official fixture. The fixture closes
primitive parity. The ALIKED/LightGlue artifact tests whether a stronger clean
multi-view reconstruction can move heldout quality toward paper acceptance.

The artifact quality floor is:

```text
point_count >= 2000
track_mean >= 2.5
track_p90 >= 3.0
reproj_median <= 4.0 px
reproj_p90 <= 8.0 px
verified_pairs >= 28
```

Meeting those floors admits the artifact to the matched Metal training row. It
does not itself prove heldout acceptance.

## Next-Experiment Ladder After Heldout Diagnostic

Start by running the heldout-view diagnostic on the current best selected
checkpoint:

```bash
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py \
  <selected_clean_config.jsonc> \
  --checkpoint <selected_checkpoint.pt>
```

Then follow this ladder. Stop at the first failed gate; do not skip forward to
a larger sweep.

1. Classify the heldout failure from the diagnostic.
   - Low alpha / missing coverage: geometry support is the first suspect.
   - High alpha with high L1: appearance/material or camera/ray convention is
     the first suspect.
   - Train and heldout both weak: primitive/training capacity is not isolated;
     do not make a heldout-only claim.

2. If coverage is weak, prioritize clean geometry construction.
   - Run the external ALIKED/LightGlue artifact path on an ONNX-capable host.
   - Verify the artifact floors above before writing a training config.
   - Train the matched Metal row and validate heldout PSNR/SSIM through the
     handoff runner.

3. If coverage is present but colors are wrong, run source-camera internal
   holdout before touching the official heldout.
   - Rotate one source camera into an internal query role.
   - Select mechanisms using internal-query PSNR/SSIM/L1 only.
   - Evaluate official heldout only after a candidate is selected.

4. If topology appears implicated, run frozen topology verifiers before
   training changes.
   - Compare Cech/AABB against regular-triangulation edges on frozen states.
   - Record missing regular edges, extra Cech edges, forward/alpha deltas, and
     shared gradient parity.
   - Only train a topology regularizer if frozen evidence says topology
     predicts internal-query error.

5. If primitive parity is in question, do not run more scene sweeps.
   - Generate the official CUDA/Warp fixture.
   - Run the two official fixture tests.
   - Fix primitive mismatch before interpreting heldout PSNR.

6. If the ALIKED/LightGlue row passes artifact floors but still misses heldout
   quality, run the no-leakage local ablation matrix.
   - baseline clean init
   - uncertainty-weighted point selection
   - train-only surfel/normal lift
   - witnessed topology ledger or regularizer
   - selection + surfel + witnessed topology only if the individual reports
     point in the same direction

7. If two local mechanism probes fail to improve internal holdout, kill the
   local approximation lane and return to external clean geometry or official
   schedule fidelity.

The ladder is deliberately narrow. The project does not need another broad
hyperparameter sweep until the diagnostic says the failure is an optimization
or schedule problem rather than an official-parity or clean-geometry problem.

## Completion Wording Template

Use this wording until all gates pass:

```text
Local PowerFoam Metal implementation gates pass for the checked primitive,
raytrace/backward, synthetic 4K, and optimizer-step trainability paths. Full
official PowerFoam reproduction remains blocked until the upstream CUDA/Warp
fixture is generated and the official Direct/Metal parity nodes pass, and until
the clean heldout row satisfies paper acceptance thresholds.
```

Only after Gate A, Gate B, and Gate C pass may future notes use:

```text
PowerFoam proper is reproduced on Metal for the audited scope.
```
