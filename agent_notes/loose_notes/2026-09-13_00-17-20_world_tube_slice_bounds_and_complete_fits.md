# Slice bounds remove the demonstrated overflow without dropping contributions

The preceding turn was progress: it recovered historical source fits, identified
tile-list truncation in the larger shared-world diagnostics, and committed the
fail-closed RGB guard. This continuation fixes a demonstrated source of excess
candidates and obtains complete training/evaluation results. The broad overnight
goal remains active; publication acceptance counts remain unchanged.

## Retain the learned world before evaluation

The prior capacity-256/tile_t=1 run reached evaluation but lost its learned state
when evaluation overflowed. The multicamera runner now saves final legacy-world
weights plus a training receipt before evaluation. It uses the existing strict
checkpoint format, including logical state and file hashes. The receipt is a
snapshot taken before evaluation; it does not claim a completed evaluation. The
compiler route keeps a separate pre-evaluation checkpoint so its later snapshot
cannot overwrite the training receipt's file identity. Optimizer state is not
saved, so exact optimizer continuation is not claimed.

The same 80-step cap256/tile_t1 recipe was rerun with this retention fix. It
completed all 80 optimizer updates in 41.85134 seconds and then rejected
overflow during evaluation. Its 118,652-byte checkpoint survived and passes
the existing strict CPU loader. File SHA-256 is
`dcdb7bbea0be1be9a8e1efe7bcaeeef9d7bc1374e6b9a773d1969808617f1b93`;
logical world SHA-256 is
`b15090677912e9388ca834e3578a9642da9ebf6825d23e2bd9f321c22f4caa49`.
Artifacts live under the sampling-ablation root's
`random_2048_balanced_cap256_t1_checkpoint/world_tubes/` directory. This failed
overall run has no finalized W&B or outer resource receipt; its training log and
checkpoint are retained evidence, not an accepted quality row.

## Mathematical diagnosis and native fix

The old native binner used the axis-aligned box of the entire spacetime
ellipsoid at every temporal tile. Thus a tube could consume a tile slot even
when its footprint at that tile's rendered times was far away.

For packed precision Q partitioned as [A b; b^T c], completing the square at
dt=t-m_t gives slice center m_uv-A^-1 b dt and residual budget
tau-(c-b^T A^-1 b)dt^2. The coordinate radii are the square roots of this
budget times diag(A^-1). The new binner unions these slice boxes over the
discrete frame samples in each temporal tile. It retains the broad bound for
ill-conditioned or non-finite calculations and uses outward numerical slack.
This is standard Gaussian algebra applied to discrete raster binning; it is
not new camera gauge theory or a continuous-time compiler certificate. The
floating-point slack has regression evidence, not a formal interval proof.

CPU float64 support estimates on the saved learned world gave:

| Camera | Lifetime-box maximum | Slice-box maximum |
| --- | ---: | ---: |
| train cam04 | 206 | 77 |
| train cam09 | 241 | 64 |
| heldout cam06 | 289 | 122 |

The true-overlap guard remains: 129 coincident tubes still overflow capacity
128 and are rejected. A second fixture has 129 moving tubes with disjoint
time-local footprints. It failed under the old bounds and now matches complete
CPU per-pixel depth-ordered RGB and parameter VJPs for both peak-splat and
Beer–Lambert alpha, including a gated-forward check. The focused suite passes
29 cases, and the existing shifted-time sparse feature-binning regression also
passes because feature rendering shares this native binner. Logs are
`/tmp/world_tube_sliced_before.txt`, `/tmp/world_tube_sliced_after.txt`, and
`/tmp/world_tube_sliced_feature.txt`.

## Complete real-data evaluations and fresh training

The unchanged saved 80-step world evaluates after the bin fix at train/heldout
PSNR **14.512481/8.005188 dB**, SSIM **0.375881/0.033221**. Native tile maxima
are exactly 77/64/122, with zero overflow in all three cameras. Peak measured
tree/launcher RSS is 1,303,265,280 bytes, with zero new swap. Offline W&B is
`wkqf9tm6`. This is evaluation of an existing world, not another optimizer run.

A fresh 80-update run using the corrected bins reaches **14.466347/8.005325
dB**, SSIM **0.374730/0.032954**, with tile maxima 78/65/118 and zero overflow.
It consumed the same 160 target images and 1,597,440 target pixels. Training
took 40.042493 seconds. Peak tree/launcher RSS is 1,416,806,400 bytes; new swap
is zero. Offline W&B is `pa4de6ce29d421`. Its final checkpoint passes strict
reconstruction and finite-parameter checks, with file SHA-256
`dea1c6f86cb6cab58e7d530c2662a6adba583976d1cfad963525ba606cb316c3`.
The roughly 0.046-dB difference between the two fits is not a quality win.
No controlled timing or sublinear-scaling claim follows from these runs.

An independent check of 81 actual scene pixels (three cameras, three times,
nine spatial positions) compares native rendering with full uncapped CPU
per-pixel compositing. Maximum RGB error is **2.98023224e-7**. This is sampled
real-world forward evidence, not exhaustive image/VJP certification. The
mechanical pixel check did not create a W&B run; both quality evaluations did.

Artifacts and replay scripts:

- `outputs/benchmarks/2026-09-13_world_tube_support_diagnostic/`: support-count
  estimates, checkpoint evaluation, PNG/MP4 media, offline W&B, resource
  receipts, and sampled pixel parity. Its scripts use the strict checkpoint
  loader and the existing evaluator/resource monitor.
- `outputs/benchmarks/2026-09-12_world_tubes_sampling_ablation/random_2048_balanced_sliced_cap256_t1/`:
  full optimizer/evaluation report, source hashes (including the dynamically
  loaded Metal shader), checkpoint, media, resources, and offline W&B.

The native fix and portable checkpoint-retention change are STAR commit
`ed6f393`. Unrelated multicamera streaming/compiler WIP remains unstaged.
No concurrent accelerator jobs, resource-limit changes, online upload, or
C++ rebuild occurred; the existing extension compiles the changed Metal source
when loaded by a new process.

## What the results now mean

The overflow blocker in this 80-step world is resolved through tighter valid
support, not increased capacity or ignored overflow. Training-view fitting is
moderate and visibly blurry; heldout quality is still poor. The old successful
screen-space source fits remain distinct controls. The next experiment is a
longer training-duration control under these complete bounds, with checkpoints
retained during training so a later guard failure cannot erase the trajectory.
The previous 400-step truncated result remains a confounded negative.
