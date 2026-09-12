# Longer shared-world fitting improves both views, with a persistent quality gap

The preceding goal turn produced a verified profiling-resource failure and
stopped that profiling mode. This turn left it stopped and returned to the
independent convergence lane. Its ordinary Metal training and evaluation both
passed the same resource guards. No renderer/source implementation changed.

## Run and primary comparison

Coffee Martini, seed 17, 32 frames, train cam04/cam09 and heldout cam06,
corrected LLFF v2, 2048 legacy world tubes, random balanced initialization,
depth 9 and spatial precision 30*(2/9)^2. The checked-in 800-update protocol
preserves the first 20 coarse updates at 48x64/1024 tubes, then uses
96x128/2048 tubes, two target images per update and learning rate 0.015.
Only the name, duration and final stage endpoint differ from the 400-update
protocol. The 600-second wall guard and all memory/disk guards are unchanged.

The fresh run completed all 800 updates in 363.62 s of training. It consumed
1600 target images / 19,292,160 target pixels and retained 81 periodic learned
states. These checkpoints omit Adam state and do not claim exact optimizer
resume. Offline W&B run: `pa2d8aafc2109d`.

To isolate duration within one actual optimizer trajectory, the saved step-400
checkpoint of this new run was separately loaded and evaluated with the same
full dataset, cameras, evaluator and current source. The primary result is:

| Metric | Step 400 of this run | Step 800 | Change |
| --- | ---: | ---: | ---: |
| Train PSNR | 19.66510 | 20.84216 | +1.17706 dB |
| Heldout PSNR | 14.93240 | 15.30286 | +0.37045 dB |
| Train SSIM | 0.59522 | 0.67731 | +0.08209 |
| Heldout SSIM | 0.30224 | 0.33887 | +0.03663 |
| Heldout LPIPS (lower is better) | 0.83127 | 0.73922 | -0.09206 |

Train MSE falls 23.74%, heldout MSE 8.18%. No target or renderer threshold
changed between these evaluations. Both use all declared train and heldout
frames rather than media samples. The step-400 evaluation logs images, videos,
metrics and checkpoint identity offline as `gyfkjlve`.

The preview inspection still shows broad blurred shapes, missing fine detail
and substantially worse novel-view reconstruction. This supports continued
learning at this budget, not exact convergence, broad generalization, or a
matched SOTA claim. The heldout view has already been used in development and
is an exploratory validation view, not an untouched final test set.

## Why the same-trajectory comparison matters

The previous independent 400-update run measured 19.69050/14.96450 dB.
Its source-file hash does not equal the new trainer-file hash. Commit 048f797
accounts for one known change in frozen-compiler byte accounting, but reversing
that change alone did not reconstruct the full old hash. The whole source
difference is therefore not certified as irrelevant to training. The aggregate
records this mismatch explicitly and does not call the separate runs an
exact-source duration control.

Independent sampler reconstruction reproduces each actual 400/800 schedule
digest and proves their first 400 ordered target updates and stage settings
match. The digest also encodes stage endpoints, so the full hashes differ even
for a shared prefix. Native library, shader and decoded dataset identities
match across the historical controls. The same-trajectory step400/800 evaluator
comparison additionally matches the full decoded-data/evaluator identities and
common bound source hashes directly.

The separate runs' largest common logged-loss difference is only 0.001325,
but their step-400 velocity tensors differ by normalized L2 0.146 (positions
0.00719, raw colors 0.0210). Their training PSNRs nevertheless differ by only
0.0254 dB. Do not infer identical learned worlds from close losses. Atomic
backward is nondeterministic, and the source hash also differs; this comparison
does not isolate either as the cause of coefficient drift.

Code inspection confirms another historical experiment distinction:
video_fit_comparison.fit_uvt optimizes full-clip MSE, whereas the shared-world
paper_batch branch optimizes robust L1. Current PSNR is computed separately
from full-image MSE. Do not convert sampled robust-L1 logs into PSNR or treat
the old source fitter and new world fitter as the same objective. This turn
kept the existing robust-L1 loss fixed.

## Gates, resources and artifacts

The training run has zero final tile overflow, with maxima 139/115/116 across
the three cameras. Maximum sampled process-tree/launcher RSS is 1,530,445,824
bytes; new swap is zero. The step400 evaluation also has zero overflow and
new swap, with peak tree/launcher RSS 1,330,102,272 bytes. No guard was loosened,
and accelerator jobs were sequential. The shape-recording profiler stayed
stopped. These measurements are on the 24-GiB host, not physical 8-GiB hardware.

The evaluation script initially passed a tuple to an API requiring ImageSize;
it exited before rendering. That local argument error was corrected using
normalize_image_size. The failed script, log, preflight and hashes remain
beside the successful run with `.input_type_failure` suffixes. The summary's
attempted source-equality assertion also failed and is reported as a mismatch,
not weakened into a passing source-identity claim. Its schedule serialization
now uses ImageSize.as_list. No production code was changed for either issue.

The existing convergence summarizer now includes the 800-update row and the
within-trajectory control. It rechecks actual schedule digests, decoded data,
evaluator/source hashes, checkpoint hashes, zero overflow, resources, and
offline W&B report backing. Raw results remain under
`outputs/benchmarks/2026-09-12_world_tubes_sampling_ablation/random_2048_balanced_depth9_footprint_matched_800/`.
The separate step400 evaluation is its `step400_evaluation/` subdirectory.
The aggregate and replayable summarizer are in
`outputs/benchmarks/2026-09-13_convergence_depth_controls/`.

Final checkpoint SHA:
`da6eac63e2497a47fe7bcfa5be645b7b665199c318e17384b31cbf493f4e0c77`.
Logical world SHA:
`2ac9be4f45e1304d9ed2e8208303d3ed2ccf3a1713d3cd1ab89be6b46233434e`.

## Next evidence

Use this better-fitted 2048-tube world in a bounded four-selected-time
replay/compiled check before expanding a scaling sweep. The existing compiler
checks use a much smaller 256-tube world trained for only two updates; their
passing tolerances and fixed-time cost are not automatically representative
of this learned scene. Retain the same checkpoint/camera/target/alpha law on
both routes and all existing resource and numerical gates. This is a next
experiment, not completed evidence or a sublinear claim.

Keep the 800-update fit as a local quality control. It shows that the new
shared-world route fits its training views increasingly well and still has a
large heldout gap; it does not invalidate the older successful source-space
fits. Public paper counts remain 0/7 contexts and 0/21 lanes. BASELINES is
unchanged, and the broad overnight goal remains active.
