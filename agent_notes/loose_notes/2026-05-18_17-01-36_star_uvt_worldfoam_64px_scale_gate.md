# STAR UVT vs WorldFoam 64px Scale Gate

Follow-up to the tiny 32px matched gate. I kept the fixed Gate4
`fused_mse_rgb_only` WorldFoam path and ran a less tiny 64px comparison against
STAR UVT `direct_atomic/index_add`.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_affineclear_repeat20_render64_site12_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_affineclear_repeat20_render64_site12_2_4_8_16_verifier.json
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_scale_64px_896t_vs_12site_2_4_8_16.json
```

The WorldFoam verifier passed with `status ok` and no failures. Key verifier
numbers:

- total median scale from 2f to 16f: `0.979x`
- backward median scale from 2f to 16f: `1.107x`
- train mixed tape storage scale: `0.997x`
- train explicit ray storage scale: `8.000x`

The matched comparison passed with `status ok` and no failures. Warm-step
median ratios were:

```text
frame  STAR total  WF total  STAR/WF total  STAR backward  WF backward  STAR/WF backward
2      11.560 ms   3.608 ms  3.204x         5.131 ms       2.847 ms     1.802x
4      13.998 ms   2.226 ms  6.288x         6.687 ms       1.791 ms     3.735x
8      8.805 ms    2.630 ms  3.348x         4.983 ms       2.279 ms     2.186x
16     9.044 ms    3.534 ms  2.559x         5.953 ms       3.151 ms     1.889x
```

Interpretation: the fixed WorldFoam fused warm step stays faster than STAR
direct-atomic in this local 64px gate. The result is still not a full
WorldFoam-over-STAR claim because the comparison is not capacity matched:
WorldFoam is 12 sites while STAR is 896 tubes, and WorldFoam still carries the
explicit ray/tape build side path outside the fused kernel.

Capacity check: trying 64px/24-site WorldFoam with the same fused-MSE mode fails
before train/eval:

```text
ValueError: candidate_row_offsets_i32 row contains 222 candidates, exceeding Metal local boundary cap 128
```

So the next useful shader work is candidate-cap handling or candidate tiling,
not more proof that the already-fused 12-site warm step is fast.
