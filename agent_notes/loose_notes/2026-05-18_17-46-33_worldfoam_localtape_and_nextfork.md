# WorldFoam Local-Tape Follow-Up And Next Fork

Continuation after the corrected high-cap insert fix. The goal was to try
small fused-MSE shader forks against the real 64px/24-site path, where max row
length is above 128 and the corrected helper actually inserts up to the 256-cap.

## Shellsort Fork

Hypothesis: appending depths then shell-sorting the row would beat sorted insert
when the high-cap rows have many inversions.

Result: negative. Parity and quality were fine, but timing worsened at every
frame count:

```text
artifact:
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_shellsort_repeat20_render64_site24_2_4_8_16.json
verifier:
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_shellsort_repeat20_render64_site24_2_4_8_16_verifier.json

frame  insert-fix total  shellsort total  insert-fix backward  shellsort backward
2      3.572 ms          4.004 ms         3.211 ms             3.649 ms
4      4.725 ms          5.380 ms         4.360 ms             5.026 ms
8      6.819 ms          8.508 ms         6.441 ms             8.134 ms
16     14.091 ms         15.310 ms        13.678 ms            14.950 ms
```

The shader change was reverted; keep the artifact only as a negative result.

## Local-Tape Fork

Hypothesis: the high-cap fused RGB-MSE VJP was spilling or losing occupancy due
to local tape footprint. Remove per-segment `segment_alpha`, `weights`, and
`segment_rgb` arrays, then recompute those values from `owner`,
`trans_before`, `segment_trans`, and `site_rgba` during the reverse pass.

Result: positive but insufficient. Parity stayed exact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_localtape_parity_mps.json
status ok
max loss diff 3.725290298461914e-09
max grad diff 0.0
```

Train/eval quality stayed effectively unchanged:

```text
train PSNR:   13.731994 / 13.753005 / 13.661237 / 13.735021
heldout PSNR: 14.170085 / 13.992464 / 14.220458 / 14.231979
```

Timing versus corrected insert-fix:

```text
artifact:
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_localtape_repeat20_render64_site24_2_4_8_16.json
verifier:
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_localtape_repeat20_render64_site24_2_4_8_16_verifier.json

frame  insert-fix total  local-tape total  speedup  insert-fix backward  local-tape backward  speedup
2      3.572 ms          3.400 ms          1.05x    3.211 ms             3.019 ms             1.06x
4      4.725 ms          4.511 ms          1.05x    4.360 ms             4.120 ms             1.06x
8      6.819 ms          6.493 ms          1.05x    6.441 ms             6.137 ms             1.05x
16     14.091 ms         11.549 ms         1.22x    13.678 ms            11.070 ms            1.24x
```

The verifier still reports `status failed` on broad scale:

```text
total mean scale 3.153 exceeds 2.000
backward mean scale 3.403 exceeds 2.500
total median scale 3.397 exceeds 2.000
backward median scale 3.667 exceeds 2.500
mixed tape storage scale: 0.9922x
explicit ray storage scale: 8.0000x
```

## STAR Comparison

Fresh local-tape STAR comparison:

```text
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_localtape_scale_64px_896t_vs_24site_2_4_8_16.json
status ok
```

```text
frame  STAR total  WF local-tape total  STAR/WF total  STAR backward  WF local-tape backward  STAR/WF backward
2      4.684 ms    3.400 ms             1.378x         2.591 ms       3.019 ms                0.858x
4      5.507 ms    4.511 ms             1.221x         3.301 ms       4.120 ms                0.801x
8      6.577 ms    6.493 ms             1.013x         4.117 ms       6.137 ms                0.671x
16     8.220 ms    11.549 ms            0.712x         5.528 ms       11.070 ms               0.499x
```

Interpretation: local-tape turns the 16f cliff from terrible to merely bad and
keeps WorldFoam total step competitive through 8f on this small gate. It does
not solve frame scaling. STAR still has the cleaner scaling curve
(`1.755x` total, `2.134x` backward) while WorldFoam local-tape is
`3.397x` total and `3.667x` backward.

## Next Fork

Do not keep raising caps or replaying shell-sort ideas. The next fork should
reduce the actual number of per-frame segments/owners visited by the fused-MSE
backward. The strongest direction is an owner-run or interval tape: during
compile/build, merge adjacent boundary intervals that resolve to the same owner
under the affine ray/time sample or store enough interval ownership to skip
same-owner repeats in the fused kernel. This is a bigger correctness surface
than local-tape, so it needs the high-cap beyond-128 regression, the parity
probe, the 64px/24-site train/eval artifact, and the verifier before citing.
