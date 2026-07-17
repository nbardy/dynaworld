# WorldFoam High-Cap 24-Site Gate

Follow-up to the 64px/12-site STAR-vs-WorldFoam gate. The 64px/24-site
WorldFoam fused-MSE attempt originally failed before train/eval because a CSR
row had `222` candidates and the fused path was still guarded by the old Metal
local boundary cap `128`.

Code change:

- Added `MAX_REALRAY_FUSED_MSE_BOUNDARIES = 256`.
- Scoped the 256-boundary Python guard to
  `fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only`.
- Scoped the 256-depth Metal arrays to the fused-MSE kernel and the affine
  num32/den16 forward replay used for final train/heldout eval.
- Kept the regular real-ray and non-MSE affine VJP paths on the old 128 cap.
- Updated the train/eval harness so `candidate_rows_under_metal_cap` and
  `max_realray_boundaries` report the fused-MSE cap when `vjp_mode` is
  `fused_mse_rgb_only`.

Capacity proof:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_capcheck_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_capcheck_render64_site24_2_4_8_16_verifier.json
```

The 1-step capcheck is `status ok` with `max_realray_boundaries: 256`; max train
candidate rows were `222 / 217 / 215 / 216` at `2/4/8/16f`.

Full repeat20 gate:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_repeat20_render64_site24_2_4_8_16_verifier.json
```

Verifier result: `status ok`, `failures []`. WorldFoam total median scale is
`1.117x`, backward median scale is `1.487x`, mixed tape storage scale is
`0.992x`, and explicit ray storage scale is still `8.000x`.

STAR comparison:

```text
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_highcap_scale_64px_896t_vs_24site_2_4_8_16.json
```

The comparison is `status ok`, but high-cap WorldFoam no longer wins speed:

```text
frame  STAR total  WF total   STAR/WF total  STAR backward  WF backward  STAR/WF backward
2      5.001 ms    13.722 ms  0.364x         2.702 ms       9.294 ms     0.291x
4      5.973 ms    7.204 ms   0.829x         3.404 ms       6.231 ms     0.546x
8      7.146 ms    7.602 ms   0.940x         4.311 ms       6.605 ms     0.653x
16     9.342 ms    15.332 ms  0.609x         5.674 ms       13.819 ms    0.411x
```

Interpretation: raising the fused-MSE cap makes 24-site WorldFoam run and
preserves sublinear-ish frame scaling, but candidate cost dominates enough that
STAR direct-atomic is faster at this 64px/896-tube vs 24-site gate. The next
real shader problem is reducing high-cap candidate replay cost or tile/tape
build time, not just lifting caps.
