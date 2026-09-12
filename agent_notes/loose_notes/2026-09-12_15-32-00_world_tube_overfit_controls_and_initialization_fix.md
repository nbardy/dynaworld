# Overnight iteration: source fit reproduced, world initialization isolated

## Goal and starting evidence

The user renewed the broad overnight iteration goal. The preceding review was
progress because it recovered retained successful fits and changed the next
experiment: reproduce the source fitter before diagnosing the much smaller
world-space multicamera run. No new token budget was supplied. One lead, no
subagents, sequential accelerator jobs, unchanged small-playground resource
guards, and a 600-second wall timeout per diagnostic were used.

The initial live snapshot passed: available memory 7,630,012,416 bytes, swap
427,556,864 bytes, disk free 43,721,244,672 bytes, and load/core 0.2795.
Existing Python processes were market recorders, not accelerator jobs. Chrome
was open and GPU utilization sampled at 14%; no application was terminated.
These runs support quality comparisons, not a controlled performance claim.

## Historical control: reproduced

The July 128px/16-frame/2048-tube/60-step STAR UVT recipe was copied with only
fresh output paths and offline W&B logging enabled. Scientific config sections
are identical to the old recipe. Current native PSNR is 21.76852703 versus
historical 21.76852942; SSIM is 0.519189298 versus 0.519189238. Initial MSE is
identical and final MSE differs by 2.79e-9. The existing video-metric evaluator
independently reads 21.8093 dB from the encoded side-by-side video. The preview
shows the recognizable running dog and landscape with soft detail.

Artifacts: `outputs/benchmarks/2026-09-12_source_overfit_control/` contains
result JSON, PNG, MP4, resource receipt, metric/config verification, run log,
and offline W&B `l61kgq5d`. Peak sampled tree plus launcher RSS was 587,497,472
bytes and new swap was zero. `launch.py` retains the guarded launch command.
The source fitting implementation has not suffered the suggested regression.

## World-space controlled measurements

All four rows below use Coffee Martini, corrected LLFF v2, train cam04/cam09,
heldout cam06, 32 frames, seed 17, 80 updates, two target images per update,
and the same 48x64 -> 96x128 resolution schedule. All use MPS peak-splat legacy
world tubes and the same native binary and evaluator.

| Initialization / order | Final tubes | Train PSNR | Heldout PSNR | Train SSIM | Heldout SSIM | W&B offline |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Boundary grid, grouped | 256 | 6.486528 | 6.504323 | 0.04152 | 0.01484 | pa1021ab3b4c17 |
| Random, grouped | 256 | 8.686841 | 6.111591 | 0.13358 | 0.01490 | pac56587c10e6b |
| Random, grouped | 2048 | 13.061873 | 7.873699 | 0.30688 | 0.02946 | pac3ca9f0a60ef |
| Random, balanced groups | 2048 | 13.923467 | 8.027669 | 0.34637 | 0.03417 | pa5a0f2fc3af0f |

Each row consumed 160 target images and 1,597,440 target pixels. The full
schedule hash includes primitive counts, so the 256- and 2048-tube hashes
appropriately differ. Independently replaying the sampler reproduces each
retained hash and confirms identical ordered target-only sequences across all
four rows. Data, evaluator, and native-binary identities also match.

Peak sampled process-tree RSS across these runs was 1,562,279,936 bytes; all
reported zero new swap. The largest reported sampled driver allocation was
1,304,576,000 bytes in the pre-fix 2048 run. These remain within the existing
3-GiB process / 2-GiB MPS caps on the 24-GiB physical host.

Root: `outputs/benchmarks/2026-09-12_world_tubes_sampling_ablation/`.
`comparison.json` binds the summaries to raw reports; each variant directory
contains argv, full report, train/heldout PNGs and MP4s, resource receipts, and
offline W&B. `source_snapshots/` preserves the exact pre/post initializer
source; both hashes were checked against their run-bound identities.

## Demonstrated defects and interpretation

At N=256, V=2, F=32, initialization assigns N/(VF)=4 tubes to each source
camera/time group. The boundary-grid sampler repeats the same four corner
pixels in every group: four unique source coordinates, zero interior samples.
The random control has 253 unique coordinates, 96.48% interior. Randomization
alone helps training but hurts heldout quality; it is not a generalization fix.

The initializer also concatenated complete camera/time groups. Progressive
`model.batch()` takes a prefix, so its first half contained only camera zero.
The fix interleaves groups by their within-group sample index, preserving the
full initialized point/color/time multiset while balancing smaller prefixes.
It handles unequal group sizes. Static/dynamic partition placement is retained.
The STAR submodule commit is `7f2482c` (9 insertions, 1 deletion); unrelated
streaming/frozen-world WIP was deliberately left unstaged.

New behavioral tests run the actual initializer and WorldTubeModel.batch and
verify every source camera/time group remains represented at reduced budgets.
They failed on the demonstrated empty-group coverage before the change and
pass afterward. The broader focused suite passed 27 tests, including existing
SPD4 training/projection/Metal checks and streamed metrics. An existing eager
test fixture lacked the current `deferred_target_frames=False` field; that
fixture was updated so its trainer checks reach the actual call graph. The
80-step balanced Metal row above is the real optimizer-path smoke after all
edits, and improves both train and heldout metrics relative to the matched
2048-tube pre-fix run. This is one seed, not a universal ordering win.

Training-time support still shrinks physically if frame count is increased
while temporal precision in frame-index units stays fixed. Together with
N/(VF) initialization density, that is a confound for quality-vs-duration
experiments. The frozen-world compiler sweep must retain its fixed-world,
fixed-physical-interval contract; these fits establish no sublinear exponent.

## Reproduction, limits, and next work

Configs are `star_uvt_source_control_128_16f_2048t_60step_20260912.jsonc`,
`world_tubes_local_sampling_ablation_20260912.jsonc`, and
`paper_protocols/coffee_martini_local_capacity2048_20260912.jsonc` under
`src/train_configs/`. The sampling output root retains the original `launch.py`
and `launch_scoped_provenance.py` for subsequent guarded diagnostics. Invoke
the latter from repo root with `PYTHONPATH=src/train:. .venv/bin/python` and
one configured variant name. Earlier runs used excessive whole-tree hashing;
the follow-up launcher records Git state and relevant source-file hashes.
Pre-fix reproduction requires its retained source snapshot; current code
intentionally changes coarse-stage coverage.

These are diagnostic controls in a dirty worktree, not accepted publication
rows. BASELINES standings and paper acceptance counts are unchanged. No online
W&B upload or remote compute occurred. No larger compiler sweep was attempted.

The next useful experiment is a training-duration control at the same 2048
tubes and balanced random initialization, preserving data and renderer. The
current fit remains blurry at 13.92 dB and heldout quality is still weak.
Longer training must be measured before attributing the remaining gap to
convergence or geometry. Keep the source-space 21.77-dB control separate from
this more constrained world-space fit. The renewed overnight goal remains active.
