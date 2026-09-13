# Matched loss substitution: MSE does not fix the shared-world quality gap

The previous turn was a verified wait on live session 85763 while answering
the user's question about historical overfits. Those successful source fits
remain valid: the old 128px recipe reproduced at 21.76853 dB. This turn
finished the live robust-L1 control, ran the MSE counterpart sequentially,
and accepted both measurements with an independent retained-artifact verifier.
All accelerator jobs started for this pair are terminal (85763 and 65181).
One lead, no subagents, no application termination, remote compute or upload.

## Fixed experiment and results

Coffee Martini, seed 17, 32 frames, train cam04/cam09, exploratory heldout
cam06, 800 updates, 2048 final legacy world tubes, random balanced initial
samples, depth 9 and spatial precision 30*(2/9)^2. The first 20 updates use
48x64/1024 tubes and LR 0.03; the remaining 780 use 96x128/2048 and LR 0.015.
Both consume 1600 target images / 19,292,160 target pixels, two images per
update. Dataset, native library, shader, calibrated cameras, optimizer,
regularization, sample schedule, evaluator and all 18 bound source files match.

| Loss | Train PSNR | Heldout PSNR | Train SSIM | Heldout SSIM | Heldout LPIPS | Train loop s | Offline W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| robust_l1 | 20.91387 | 15.35368 | 0.68091 | 0.34544 | 0.74335 | 361.34 | pa370e31d5ffa7 |
| mse | 20.93021 | 14.68559 | 0.65377 | 0.32552 | 0.74651 | 369.31 | paf91acfd3f77e |

MSE changes train PSNR by only +0.01635 dB, train SSIM by -0.02714, heldout
PSNR by -0.66809 dB, heldout SSIM by -0.01992 and heldout LPIPS by +0.00316.
These are one run per loss, not a seed/stability study. Atomic backward is
nondeterministic. The near-zero train-PSNR difference is not a demonstrated
improvement. Both previews remain blurred; MSE does not visibly recover the
missing fine structure. Keep robust L1 as the practical current recipe.

This is a negative result for the hypothesis that simply replacing the
world-space robust-L1 objective with the historical fitter's MSE closes its
quality gap. It does not establish that tuned MSE can never help, or isolate
the loss's shape from its scale relative to unchanged regularization weights.
The historical source fitter also has a different representation, dataset,
capacity, initialization and full-clip target exposure; those remain confounds.

## What was implemented and independently checked

run_world_tube_loss_control.py invokes the existing benchmark through the
existing resource guard, with a process-local substitution of its photometric
function. It does not edit the production trainer or common robust_l1.
The three auxiliary photometric weights are zero and exactly 800 calls occur.
An observational constructor hook saves the pre-optimizer world; the first
loss call saves its residual, actual scalar and actual derivative with
respect to that residual. Both hooks execute symmetrically in both runs.

For n residual elements, the two objectives and derivatives are:

    L_robust = mean(sqrt(r^2 + 1e-6)),   dL/dr = r / (n*sqrt(r^2 + 1e-6))
    L_mse    = mean(r^2),               dL/dr = 2*r/n.

Independent CPU NumPy float64 calculations give loss errors 9.41e-9/8.03e-9
and derivative relative-L2 errors 4.58e-8/2.63e-8 (robust/MSE), below the
preselected 1e-7/1e-6 tolerances. The initial world tensors and residual are
bit-identical; logical initial-world SHA is
2a0b428bba306747f292157d1546f136f857ad344f57e869e389b3ea5bde562a.
First losses are 0.2618719935 and 0.1068380177. Raw loss values are not
comparable quality metrics. All 800 sampler updates reproduce the retained
schedule SHA 6c45d9448449dc3e4dc578cf2177ccaa7522ab8bcc4410ef90be148ff0bda195.

verify_world_tube_loss_control.py also checks final world/checkpoint hashes,
81 periodic checkpoints per run, complete declared train/heldout evaluation,
PSNR arithmetic from global MSE, zero tile overflow for all three cameras,
unchanged resources, saved media and exact offline W&B report/source digests,
metrics and media records. It parses the retained .wandb files directly.
It does not independently re-render final checkpoints; those quality values
are the existing evaluator's retained output. No public evidence count changes.
The 18 run-bound files are archived under outputs/benchmarks/2026-09-13_world_tube_loss_control/after/ and rehashed.
The existing-attempt guard was exercised for both losses; both calls reject
before execution and leave the retained report hashes unchanged. The result
is saved in artifact_preservation_check.json beside the comparison.

The W&B reader initially looked only at config-update records, whereas the
initial config lives in the run record. The verifier also initially read
frames_per_step from the protocol rather than its stage; the retained
verification_attribute_error.log records that failed check. Both reader/API
mistakes were corrected and the full verifier passes. Neither changed training.
No new renderer tests were added or broad suite repeated for this runner-only
experiment: the two real optimizer paths and independent artifact checks are
the runtime evidence.

Peak tree/launcher RSS is 1,331,740,672 bytes robust and 1,398,751,232 bytes MSE;
both have zero new swap and zero final overflow. Existing 3-GiB process,
2-GiB MPS, host reserve, disk and 600-second wall limits are unchanged.
The outer wall guard includes evaluation and offline W&B finalization. The
table records only training-loop time; exact total launch wall was not retained.
These are measurements on the physical 24-GiB Mac, not physical 8-GiB hardware.

## Reproduction and next action

Config: src/train_configs/world_tube_loss_control_20260913.jsonc.
Run each loss sequentially from repo root with PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=src/train:. .venv/bin/python and the canonical runner, passing
that config and robust_l1 or mse. Choose a fresh output_dir for reruns;
the runner refuses existing attempt directories. Verify with the same
environment and verify_world_tube_loss_control.py plus the config path.
Raw reports, initial/final/periodic states, media, logs, W&B and receipts:
outputs/benchmarks/2026-09-13_world_tube_loss_control/. The pair's comparison.json is the accepted local summary.

Next measure a single-training-camera WorldTubeModel overfit using the
existing first_only optimizer view selector, preserving the renderer and
resource guards. Evaluate cam04 separately in both the two-view and one-view
worlds; do not compare a one-camera metric to the old two-camera average.
Keep initializer exposure explicit (the existing initial world includes both
training cameras), and account for different per-camera exposure under a
fixed total target budget. Cam09 would be unoptimized but initialization-
exposed; it is not a fresh heldout view. The next experiment is not yet run.

Accepted local loss controls advance 0/2->2/2. Public 0/7 contexts and 0/21 lanes
remain unchanged. The broad overnight goal remains active.
