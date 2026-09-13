# Single-camera shared-world fitting improves the optimized view

The preceding turn was progress: root commit 4d5ef88 completed a matched MSE
versus robust-L1 pair. MSE did not close the quality gap. This turn kept
robust L1 and measured a cam04-only optimizer run using the same shared-world
representation and the existing two-camera initializer. One new 800-update
fit and two frozen checkpoint re-evaluations now pass independent checks.
No production trainer, renderer, native binary or dataset was changed.
One lead, no subagents; all accelerator jobs are terminal. No application
termination, remote compute or online W&B upload occurred.

## Experiment and measurements

Coffee Martini, seed 17, 32 frames, 2048 final legacy world tubes, corrected
LLFF v2 cameras, random balanced initialization, depth 9 with precision 30*(2/9)^2.
The first 20 updates use 48x64/1024 tubes and LR 0.03; the remaining 780 use
96x128/2048 and LR 0.015. Each run consumes 1600 target images and 19,292,160
target pixels. Only the optimizer view selector changes from all to first_only.
Both initial worlds are bit-identical with logical SHA:
2a0b428bba306747f292157d1546f136f857ad344f57e869e389b3ea5bde562a.

The new fit finishes 800 updates in 276.17 seconds of training. Its generic
train/eval mean includes cam09 even though cam09 is not optimized, and reads
12.49 dB. That mean is not the cam04 fitting result. The canonical evaluator
already returns per-camera metrics; the diagnostic runner now retains them.

Both final worlds were loaded from their exact saved checkpoints and rendered
again on all declared images with the same evaluator and actual Metal camera
arithmetic. The results are:

| Optimized cameras | cam04 PSNR | cam04 SSIM | cam09 PSNR | cam06 PSNR | cam06 LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| two_camera | 20.62914 | 0.68586 | 21.21857 | 15.35368 | 0.74335 |
| cam04_only | 23.29674 | 0.83284 | 9.66767 | 11.92116 | 0.55803 |

Cam04 gains 2.66760 dB and 0.14698 SSIM; its MSE drops 45.89%. The preview shows
clearer person/window structure but still softened detail. Cam09, no longer
optimized, collapses from 21.22 to 9.67 dB. Cam06 PSNR drops 15.35->11.92 dB,
despite LPIPS improving 0.743->0.558 and SSIM rising 0.345->0.359. The heldout
preview is visibly distorted. Do not turn one perceptual metric into a broad
novel-view win. Cam06 remains exploratory validation, already used in development.

This supports source-view fitting in the WorldTubeModel implementation itself,
in addition to the historical ScreenTimeTubeModel positives. It does not prove
pixel-perfect convergence or that the representation has reached its ceiling.
The difference from the old single-video evidence was never just an added
heldout evaluation switch.

## Exposure and initialization confounds

The initializer still reads cam04 AND cam09 in both runs. Cam09 is therefore
unoptimized but initialization-exposed, not a clean heldout camera. The initial
state equality rules out an initialization change as the difference here.

Independent sampler replay reproduces both actual 800-update digests. In the
two-camera fit, each of the 32 frames from EACH camera contributes 25 times:
800 image contributions per camera. In the cam04-only fit, each cam04 frame
contributes 50 times, with zero cam09 optimizer targets. Thus cam04 exposure
doubles even though total target images, pixels and updates remain fixed.
The 2.67 dB gain cannot all be attributed to removing cross-camera constraints.
One seed and nondeterministic atomic backward also limit the inference.

Next isolate that confound: preserve the two-camera sampling schedule,
initialization, 800 updates, regularization and original per-image loss
normalization, but detach only cam09's photometric contribution. Verify the
actual saved batch camera IDs and per-residual derivatives: cam04 must retain
its original weight and cam09 must have zero photometric derivative. This
keeps cam04 target exposure fixed while removing cam09's photometric gradient.
Continue to report geometry regularization and initialized camera exposure.
That next experiment is not implemented or run in this chunk.

## Verification and implementation

run_world_tube_loss_control.py gains a boundary-normalized optimizer view
selector (old configs default all) and an observational evaluator wrapper that
retains named per-camera metrics. Its real 800-update run exercises training,
checkpoint saving, evaluation and logging. The underlying trainer is unchanged.

evaluate_world_tube_view_control.py loads each fixed world, checks data,
evaluator and native identities, re-evaluates all cameras, and saves 32 raw
cam04 RGB predictions and targets per world plus PNG/MP4 previews. It uses
the same process/MPS/host/disk guards and 600-second enclosing wall limit.

verify_world_tube_view_control.py reuses the loss verifier's world hashing,
first-loss NumPy derivative check and offline W&B parser. It checks matching
initial states; exact command equality apart from output/optimizer view; all
common execution-source hashes except the documented diagnostic wrapper edit;
sampler digests and exact per-frame exposure counts; checkpoint identity;
re-evaluation agreement; zero overflow; resources; and actual offline metrics.
It independently recomputes cam04 MSE/L1/PSNR from saved full-frame float32
pixels using NumPy float64 arithmetic, and proves both raw target videos exact.
SSIM/LPIPS are retained evaluator values, not independent NumPy recomputations.
The accepted comparison and execution-bound source archives live under
outputs/benchmarks/2026-09-13_world_tube_single_view_control/comparison.json and after/.

Training peak process-tree/launcher RSS is 1,222,115,328 bytes; successful
evaluation peaks 1,306,165,248 bytes (1.22 GiB). Both have zero new swap and
zero tile overflow. Training maximum tile counts are 139/129/134 for the
three cameras, below capacity 256. Sampled training allocator use is 711,205,888
bytes. All old 3-GiB process/2-GiB MPS/host/disk guards remain unchanged.
These measurements use the physical 24-GiB Mac. The training wall above excludes
startup/evaluation/W&B; total launch wall was bounded but not retained exactly.

The first frozen evaluation completed rendering and offline W&B but failed
the provenance helper because run.dir was relative. The helper's error said
the directory was missing even though its actual issue was the absolute-path
requirement. That failed attempt, log, report, raw tensors and exact producer
source remain under evaluation/ (offline fm9xc6a8). It has no successful final
resource receipt and is not accepted. The corrected evaluator normalizes its
output directory and supports an explicit fresh --output-dir. Its successful
attempt is evaluation_absolute_paths/; per-camera metrics are EXACT against
the failed attempt. No training was repeated for this path fix.
The successful evaluation's existing-output guard also rejects a repeated
launch without changing its report hash; evaluation_preservation_check.json
retains that check.

The historical loss verifier now checks archived execution source when the
live producer changes, and explicitly lists live-source differences. It still
requires the pair's bound sources to match. The old verifier source and old
comparison are preserved under the loss-control output; the updated verifier
passes both historical runs without numerical/metric changes. Only the
diagnostic runner differs from their archived live sources. This does not
claim that historical runs used the new producer.

## Artifacts and next continuation

Config: src/train_configs/world_tube_single_view_control_20260913.jsonc.
Training offline W&B: pac4b80cd5b0c3; successful paired
evaluation offline W&B: 9bj7qq7w.
Launch from repo root with PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:.
and .venv/bin/python. Pass the config and robust_l1 to the existing runner.
Pass the config and --output-dir to the new evaluator; use a fresh directory.
Pass the config and that same evaluation directory positionally to the view
verifier. A fresh training rerun also requires a fresh config output_dir.
Keep the reference_run pinned to the retained two-camera control.

Local single-camera controls advance 0/1->1/1, with two frozen worlds evaluated.
The loss pair remains 2/2. Public counts stay 0/7 contexts and 0/21 lanes.
The broad overnight goal remains active; next is the fixed-exposure gradient
ablation described above, not another loss-function sweep.
