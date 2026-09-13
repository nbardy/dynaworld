# Cam09 gradient removal with cam04 exposure and loss weight fixed

The preceding user-facing review recovered the old successful source fits but
added no new runtime evidence. This continuation completed the in-flight masked
photometric experiment, re-evaluated both frozen checkpoints on Metal, and
verified the retained measurements. Training session 51155 and evaluation
58982 are terminal exit 0. One lead, no subagents; sequential accelerator jobs,
offline W&B, no app termination or remote compute. The broad goal stays active.

## Controlled intervention

The reference is the robust-L1 run under
`outputs/benchmarks/2026-09-13_world_tube_loss_control/robust_l1/`.
Both runs use Coffee Martini, seed 17, cam04/cam09 initialization, 32 frames,
1024 tubes at 48x64 for 20 updates and 2048 at 96x128 for 780 updates, the same
depth-9 matched footprint, and the same learning rates and regularization.
All 800 actual batch camera/frame IDs match an independent sampler replay.
Each frame of each camera is sampled 25 times, for 1600 rendered target images
and 19,292,160 target RGB pixels. Only cam04 supplies photometric gradients in
the intervention: 800 image contributions, unchanged from the reference.

For residuals r and camera mask m, the optimized photometric term is
`sum(m * sqrt(r*r + 1e-6)) / full_batch_RGB_element_count`.
It is implemented by selecting cam04, calling the original mean robust loss,
then multiplying by selected_images / batch_images. Its cam04 residual
derivative is r / (BHW3 * sqrt(r*r + 1e-6)); cam09's is zero. The scalar is the
actual reduced objective, not a detached nonzero cam09 term. Geometry
regularization is unchanged. Both cameras are still initialized and rendered;
cam09 is not a clean unseen view. Cam06 is development validation.

The initial world SHA is bit-identical to the reference:
`2a0b428bba306747f292157d1546f136f857ad344f57e869e389b3ea5bde562a`.
The initial residual is also exact. First-batch views are [1,0], mask [false,true].
The retained cam04 residual derivative is bit-identical to the reference and
cam09's derivative is exactly zero. Independent NumPy float64 checking gives
loss absolute error 3.9781e-9 and derivative relative L2 error 4.6202e-8 under
the unchanged 1e-7/1e-6 gates. The observer records all subsequent batch masks
and shapes. These are actual first-step derivatives, not a claim that gradients
stay equal after the optimizer trajectories diverge.

## Accepted measurements

| Photometric cameras | cam04 PSNR | cam04 SSIM | cam09 PSNR | cam06 PSNR | cam06 SSIM | cam06 LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cam04 + cam09 | 20.62914 | 0.68586 | 21.21857 | 15.35368 | 0.34544 | 0.74335 |
| cam04 only, original exposure/weight | 21.67876 | 0.76450 | 9.51642 | 11.75457 | 0.33901 | 0.67966 |

Cam04 gains 1.04962 dB and 0.07864 SSIM; MSE falls 21.4695%. Thus the favorable
source-fit effect survives removing the earlier doubled-exposure confound.
This is a treatment effect in one seed/recipe, not proof of an intrinsic
representation ceiling or universal gradient conflict. Atomic backward is
nondeterministic. The earlier first_only run reached 23.29674 dB with twice
cam04 exposure and different batch/loss normalization; subtracting these gains
does not cleanly measure an exposure-only effect.

The source preview is recognizable but blurry. The heldout preview has clear
geometric distortion, and PSNR/SSIM fall even though LPIPS improves. Do not
declare a novel-view improvement from LPIPS alone. The generic declared-train
mean is 12.27047 dB because it includes unoptimized cam09; it is not cam04's
source-fitting score.

Training took 427.13365 seconds, excluding startup/evaluation/W&B; the enclosing
launch remained under 600 seconds. Training/evaluation peak tree+launcher RSS
is 1,224,753,152 / 1,286,307,840 bytes (maximum 1.20 GiB), with zero new swap
and zero tile overflow. The sampled training MPS allocator is 710,608,128 bytes.
All existing 3-GiB process, 2-GiB MPS, host/disk and wall guards are unchanged.
This is the actual 24-GiB host, not an actual 8-GiB-device measurement.

## Verification defect and repair

The first independent verification stopped at exact dictionary equality of
separately accumulated per-camera metrics. Only cam04 MSE and derived PSNR
differed, by two float64 ULPs (PSNR difference 7.1054e-15 dB). With exactly the
same retained float32 pixels, CPU reduction using one/two threads reproduces
the re-evaluation value; four threads reproduces the training value.
`scalar_reduction_roundoff.json` retains this CPU-only reproduction.

The existing view verifier now permits at most four ULPs for these per-camera
scalar re-evaluations, checking camera/metric key sets and recording every ULP
difference, including heldout metrics. This replaces its former bit-equality
condition; it is an explicit gate correction, not an assertion that all gates
were identical. Independent raw-pixel PSNR/MSE/L1, aggregate re-evaluation,
world/source/target identity, initial derivative, sampler, resource, offline
W&B and zero-overflow checks are unchanged. The failed verifier source/log are
preserved before repair. No fitting or rendering was repeated for this fix.
The older single-camera control also passes the updated verifier; its prior
comparison is preserved. Historical source checking prefers retained execution
archives and reports live differences, rather than pretending old runs used
today's diagnostic producer. The old loss pair still passes the mask-aware
loss verifier. Raw historical reports and checkpoints are unchanged.

## Files and continuation

Config: `src/train_configs/world_tube_fixed_exposure_control_20260913.jsonc`.
Output: `outputs/benchmarks/2026-09-13_world_tube_fixed_exposure_control/`.
It retains training report/probe/batches/checkpoints/resource receipt, frozen
`evaluation/` raw cam04 pixels and PNG/MP4 previews, `comparison.json`, execution
source archives under `after/`, and verification logs. W&B offline IDs are
training `pa3f35a15ef47a` and paired evaluation `clfevmny`.

Use root `.venv` with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:.`.
The existing `run_world_tube_loss_control.py CFG robust_l1`,
`evaluate_world_tube_view_control.py CFG`, and
`verify_world_tube_view_control.py CFG` own this contract. Choose fresh output
paths for new training/evaluation; do not overwrite these attempts. No new
production renderer, loss, sampler, trainer or independent-verifier family
was introduced. The four diagnostic helpers were extended in place.

The fixed-exposure local control advances 0/1 -> 1/1; the previous single-view
control remains 1/1 and loss pair 2/2. Public counts remain 0/7 contexts and
0/21 lanes. This local measurement is not promoted into the public paper table.

Code inspection supplies a concrete next hypothesis: the active legacy
pixel-Jacobian projector uses `rotation[:, :2]` and world variances 1/p_x,1/p_y,
equivalent to fixed-time covariance diag(1/p_x,1/p_y,0), plus a screen-space
numerical floor. Centers and velocities are still fully 3D. This fixed world-XY
footprint is a candidate limitation, not the demonstrated cause of this result.
The existing full-SPD4 lane already supports a complete conditional spatial
covariance and lowers to the same Metal ABI; its earlier 40-step/256-or-199-atom
smokes are not the current 2048-tube quality contract. Next measure that existing
route under the bounded two-camera recipe, explicitly accounting for additional
parameters and initialization differences, before inventing more camera math.
