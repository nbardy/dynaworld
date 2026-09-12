# Longer training and camera-informed initialization separate two quality gaps

The preceding turn was progress: slice-local bounds enabled complete 80-step
Metal fits. This continuation retained intermediate optimizer states, ran two
duration controls and two initialization controls, and checked their saved
identities. All measurements below are Coffee Martini, seed 17, 2048 legacy
world tubes, 32 frames, train cam04/cam09, heldout cam06, corrected LLFF v2,
and the same 48x64 -> 96x128 schedule. These are exploratory local diagnostics,
not newly accepted paper or BASELINES rows.

## Completed measurements

| Initial depth / spatial precision | Updates | Train PSNR | Heldout PSNR | Train SSIM | Heldout SSIM | Heldout LPIPS | Offline W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 / 30, preceding control | 80 | 14.46635 | 8.00532 | 0.37473 | 0.03295 | 0.88426 | pa4de6ce29d421 |
| 2 / 30 | 400 | 16.38455 | 9.19926 | 0.41633 | 0.06311 | 0.89691 | paa186a146d011 |
| 9 / 30 | 80 | 7.09543 | 6.90907 | 0.06923 | 0.03195 | 0.79954 | pa640cec6acea7 |
| 9 / 1.48148148, source footprint matched | 80 | 14.39679 | 14.41132 | 0.38697 | 0.26744 | 0.90575 | pa307c8447ebb4 |
| 9 / 1.48148148, source footprint matched | 400 | 19.69050 | 14.96450 | 0.59827 | 0.30468 | 0.83233 | pa9fe1981515a7 |

Every completed training/evaluation run has zero tile overflow. Final tile
maxima for the last row are 135/119/120 across the three cameras. The 80-step
rows consume 160 target images and 1,597,440 pixels; the 400-step rows consume
800 images and 9,461,760 pixels. Both 400-step schedules retain the first 20
coarse updates and extend the existing fine stage. Training takes 184.45 s at
depth 2 and 189.73 s at depth 9 with matched footprint. These are not controlled
performance measurements.

Maximum sampled tree/launcher RSS across these new rows is 1,684,602,880 bytes;
all report zero new swap. The largest reported MPS driver peak in the last row
is 1,312,964,608 bytes. Existing host, 3-GiB RSS, 2-GiB MPS, swap, disk, and
600-second diagnostic limits were unchanged. Jobs ran sequentially; no online
W&B sync or application termination occurred.

## Why depth and footprint were tested together

The current initializer's depth 2 is small relative to the camera geometry.
Using only the two training camera poses, their centers are 6.37997 units
apart. The closest points on their forward optical axes have depths
9.15095/8.93169 and are separated by 0.04951 units. Depth 9 was chosen from
this calibration calculation before the depth-control runs. It is an
initialization heuristic: optical axes need not meet on an actual surface.
The two training rows' retained metadata bounds were also inspected, but no
heldout RGB was used to derive the depth or footprint correction.

For a point moved along its source camera ray from depth z to s*z, the local
pixel Jacobian scales by 1/s. Leaving world covariance fixed therefore shrinks
its screen footprint. To preserve that footprint, scale world covariance by
s^2, or spatial precision by 1/s^2. Here s=9/2, giving
30*(2/9)^2 = 1.48148148. The depth-only negative and the matched-footprint
positive show why moving points farther away without adjusting their extent
was insufficient at this primitive budget. This is ordinary projective
Jacobian algebra, not evidence that a new gauge formalism is needed.

On the old 80-step world, no tube center at its own anchor time is behind the
near plane in any evaluated camera. A small number of center trajectories
intersect the near plane over their temporal opacity support (25/13/31 of
2048). This does not certify full-volume visibility, but it does not support
blaming the initial heldout failure on a majority of centers being behind the
camera. The retained CPU geometry/depth diagnostics are in
`outputs/benchmarks/2026-09-13_world_tube_support_diagnostic/`.

## Checkpoint retention and verification

STAR commit `a20fc97` saves a hashed legacy-world checkpoint at each logged
update (step 1 and every 10 updates), plus an atomic latest-receipt JSON. Each
400-update run retains 41 snapshots. This does not save Adam state or claim
exact optimizer resume. It preserves learned geometry and logs if a later
renderer call fails. An actual CPU trainer test injects a later renderer error,
loads the preceding checkpoint, and verifies its tensors equal an independently
completed optimizer update exactly. The focused trainer suite passed 23 tests;
the strengthened survival test also passed independently. The real 400-step
runs exercise the new path on Metal. Initialization depth is now explicit in
the final report.

`outputs/benchmarks/2026-09-13_convergence_depth_controls/comparison.json`
binds the five rows to their raw reports, checkpoints, offline W&B ids, costs,
and resource receipts. Its replayable summarizer checks equal decoded bundle,
native library, and dynamically loaded Metal shader hashes; ordered sample
schedule hashes match within each update budget. It checks completed steps,
zero overflow, and checkpoint file hashes. W&B's report digest uses canonical
JSON, while the separately retained raw-file SHA hashes the actual bytes.
These are different valid identities, not interchangeable hash contracts.

The raw runs are the correspondingly named variant directories under
`outputs/benchmarks/2026-09-12_world_tubes_sampling_ablation/`. Configs remain
in `src/train_configs/world_tubes_local_sampling_ablation_20260912.jsonc` and
the 80/400-step capacity-2048 protocol files. The last final checkpoint file
SHA-256 is `7b65338e6f24d79e5a1d5093a49e587814a12fc63e52368ab09b91aa835d6cd0`.

## Interpretation and next work

Longer training helps when rendering is complete. At 80 updates, the
calibration-informed, footprint-matched initializer preserves roughly the
same train PSNR while raising heldout PSNR by 6.41 dB. At 400 updates it reaches
19.69/14.96 dB. Both source and heldout previews remain blurry; the 80-step
PSNR/SSIM improvement did not improve LPIPS, while the longer fit improves all
three relative to that 80-step row. This is substantial diagnostic progress,
not convergence to an exact optimum, a SOTA comparison, or an untouched test
set result. Depth and footprint were selected from training geometry, but
these heldout measurements have now been inspected during development.

Keep these complete fits as controls. Before claiming sublinear performance,
resolve the existing frozen replay/compiled image/VJP/fallback mismatch; the
new quality results do not close that independent gate. Further quality work
should distinguish remaining spatial capacity/footprint limits, optimization
under scene units, and projection/visibility approximations. Do not discard
the successful source-space lineage or attribute the former 6-dB results to a
fundamental inability to fit images. The broad overnight goal remains active.
