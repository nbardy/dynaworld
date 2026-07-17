# WorldFoam Owner-Run Reverse-Tape Gate

Follow-up to the local-tape pass on the corrected 64px/24-site high-cap
WorldFoam fused-MSE kernel.

## Keeper: Inline Owner-Run Reverse Tape

Patch: inside
`third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`,
the high-cap `wf2_fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only_tensor`
still evaluates each interval forward, but when adjacent intervals resolve to
the same owner it merges their reverse-tape entry by accumulating length and
multiplying segment transmittance. This preserves the RGB-only compositing math
for same-owner constant-density runs and reduces reverse-pass work and atomics.

Validation:

```text
PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

Result: 5 tests passed, including the beyond-128 Metal regression.

Parity:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_ownerrun_final_parity_mps.json
status ok
max loss diff 3.725290298461914e-09
max grad diff 0.0
```

Train/eval artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerrun_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerrun_repeat20_render64_site24_2_4_8_16_verifier.json
```

Timing against local-tape:

```text
frame  local-tape total  owner-run total  speedup  local-tape backward  owner-run backward  speedup
2      3.400 ms          2.724 ms         1.25x    3.019 ms             2.396 ms            1.26x
4      4.511 ms          3.233 ms         1.40x    4.120 ms             2.915 ms            1.41x
8      6.493 ms          6.032 ms         1.08x    6.137 ms             5.627 ms            1.09x
16     11.549 ms         6.610 ms         1.75x    11.070 ms            6.205 ms            1.78x
```

Quality remains unchanged:

```text
train PSNR:   13.732002 / 13.753016 / 13.661242 / 13.735022
heldout PSNR: 14.170086 / 13.992479 / 14.220483 / 14.231969
```

The verifier still reports `status failed`, but the failure is now close to the
formal threshold rather than a broad miss:

```text
total median scale:    2.4266x, threshold 2.000x
backward median scale: 2.5899x, threshold 2.500x
mixed tape storage scale: 0.9922x
explicit ray storage scale: 8.0000x
```

STAR comparison artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_ownerrun_scale_64px_896t_vs_24site_2_4_8_16.json
```

The STAR rows in this rerun were noisy, especially 16f. Against the stable STAR
medians from the immediately preceding local-tape comparison, owner-run
WorldFoam wins total step at all checked frame counts:

```text
frame  stable STAR total  owner-run WF total  STAR/WF total  stable STAR backward  owner-run WF backward  STAR/WF backward
2      4.684 ms           2.724 ms            1.720x         2.591 ms              2.396 ms               1.081x
4      5.507 ms           3.233 ms            1.703x         3.301 ms              2.915 ms               1.133x
8      6.577 ms           6.032 ms            1.090x         4.117 ms              5.627 ms               0.732x
16     8.220 ms           6.610 ms            1.244x         5.528 ms              6.205 ms               0.891x
```

Interpretation: this is the first 64px/24-site WorldFoam gate that is locally
competitive with STAR on total warm step time across 2/4/8/16 frames. It is not
the final fix because the WorldFoam verifier still misses the total/backward
scale thresholds and the kernel still replays candidate depths and calls
`wf2_realray_owner_at` per interval.

## Negative: Forward-Merge Variant

I also tried delaying same-owner forward accumulation so the kernel would
compute RGB/loss once per owner run instead of once per interval. It passed
the same parity probe:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_ownerrun_forwardmerge_parity_mps.json
```

But the full 64px/24-site timing was worse at 16f:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_ownerrun_forwardmerge_repeat20_render64_site24_2_4_8_16.json

frame  forward-merge total  forward-merge backward
2      2.878 ms             2.527 ms
4      3.197 ms             2.856 ms
8      4.061 ms             3.705 ms
16     10.909 ms            9.189 ms
```

That patch was reverted. Current code keeps reverse-only owner-run merge.

## Next

The next serious fork should use the existing endpoint/owner-run tape machinery
to precompute owner-run or site-pair records and feed a new fused RGB-MSE kernel
that does not rebuild depths or scan owners in the warm path. Subagents agreed
that STAR's useful portable idea is tile-pair/work-owner assignment, not another
direct-atomic port.
