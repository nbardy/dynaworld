# Tile overflow qualifies the new world-space convergence results

This follows the September 12 source-overfit correction and initialization
diagnostics. The user again asked whether the remembered positive fits were
real and whether only heldout views fail. The retained May source fits are
29.823/29.138 dB at 256/512px; the July source recipe reproduces at 21.768527
dB. Those successful screen-time image fits must not be conflated with the
much smaller shared-world multicamera experiment. The original September
256-tube grid and random controls have poor training-view PSNR too
(6.486528 and 8.686841 dB). Their final evaluations report zero overflow.

## Duration control and discovered implementation limit

The balanced random 2048-tube run was extended from 80 to 400 updates, with
the first 20 coarse updates and subsequent fine-stage settings retained.
It completed 400 updates, 800 sampled target images, and 9,461,760 target
pixels. Train/heldout PSNR deteriorated to 9.131199/7.752918 dB, with SSIM
0.108824/0.046369. Training took 322.144522 seconds. Peak sampled process
tree plus launcher RSS was 1,405,714,432 bytes; new swap was zero. The
existing resource limits were unchanged. Offline W&B is `pa00398da8947a`.

Raw artifacts are under
`outputs/benchmarks/2026-09-12_world_tubes_sampling_ablation/random_2048_balanced_400/`.
The retained report's final tile statistics expose dropped primitives:

| Run | Capacity | cam04 max / overflow tiles | cam09 max / overflow tiles | cam06 max / overflow tiles |
| --- | ---: | ---: | ---: | ---: |
| Random 2048, 80 steps | 128 | 295 / 2451 | 254 / 1938 | 297 / 1098 |
| Balanced random 2048, 80 steps | 128 | 231 / 1985 | 260 / 2144 | 304 / 929 |
| Balanced random 2048, 400 steps | 128 | 983 / 3072 | 1059 / 3072 | 963 / 3072 |

Each evaluation has 3072 counted tile instances per camera. Every evaluated
tile overflows in the final 400-step row. The native list stores only up to
capacity and the renderer drops excess entries. Consequently the previously
reported 13.06/7.87 and 13.92/8.03 dB improvements describe the truncated
implementation, not complete rendering. The loss deterioration does not
isolate generalization, optimization, or representational limits. The training
tile-load proxy also grows sharply, but no final checkpoint was saved for this
run, so a near-plane or footprint-growth explanation remains unverified.

## Demonstrated guard and failed capacity retries

`render_uvt_tubes` now raises on overflow for ordinary image rendering;
explicit `return_aux=True` remains a diagnostic path returning counts with
the image. `render_uvt_tubes_gated` also raises. A 129-overlapping-tube fixture
at capacity 128 demonstrates a pixel error from truncation and requires both
ordinary and gated calls to reject it. Both cases failed before the fix; the
focused tile-capacity, initialization, and SPD4 training suite passed all 27
tests afterward (`/tmp/world_tube_overflow_after.txt`). The guard adds a scalar
GPU synchronization; future timing must include this cost. This is not a
claim that all other renderer families now guard overflow.

A capacity-512 retry failed before training because both Python and native
runtime support only capacities 32/64/128/256. No binary rebuild or guard
relaxation was attempted. The supported capacity-256, tile_t=1 retry reached
post-training evaluation and raised on tile overflow there. It exited with
code 1 and has no final metrics, resource receipt, or finalized W&B result.
Its argv, source identity, run metadata, and traceback remain in
`random_2048_balanced_cap256_t1/`. Do not count it as a completed quality run.

An uncommitted checkpoint-save addition in the multicamera runner is still
after evaluation; this failure did not exercise it. It therefore remains
unverified and outside the overflow-guard commit. Saving learned state before
fallible evaluation is a concrete next fix, so a failed evaluation need not
discard the optimization result. Stop this capacity-sweep lane until overflow
can be handled without dropping contributions; do not launch another long fit
by disabling the guard. The existing 256-tube zero-overflow controls and old
source-fit artifacts remain separate evidence.

## Interpretation and status

Successful source-image fitting is real; it is not proof of an exact zero-error
optimum, novel-view geometry, or compiled sublinear scaling. The current shared
world route has both a training-fit gap and heldout weakness, with a proven
overflow confound in its larger runs. Preserve the successful source fitter
as a control, repair complete rendering and checkpoint retention, then measure
train and heldout separately. No accepted paper counts or BASELINES standings
change. No online upload or concurrent accelerator run occurred.
