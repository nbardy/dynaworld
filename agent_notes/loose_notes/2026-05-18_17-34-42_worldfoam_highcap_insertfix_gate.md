# WorldFoam High-Cap Insert Fix Gate

The first 64px/24-site high-cap WorldFoam artifacts were too optimistic to cite.
The Python validators and local arrays allowed `256` fused-MSE boundaries, but
the shared Metal `wf2_realray_insert_depth` helper still stopped at the old
`128` boundary cap. Rows with max length `222` therefore ran with silent depth
truncation.

Fix:

- added `wf2_realray_insert_depth_capped(...)` plus
  `wf2_realray_insert_depth_fused_mse(...)` in
  `world_foam_lane2_shared_replay_tensor.metal`
- kept legacy kernels on the `128` cap
- routed only the high-cap affine forward replay and fused RGB-MSE VJP kernels
  through the `256` helper
- added a Metal regression that builds a 140-candidate row and checks that both
  affine replay RGB and fused-MSE loss match a CPU reference using candidates
  past 128

Verification:

```text
PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

Result: 5 tests passed, including
`test_fused_mse_highcap_metal_replays_candidates_beyond_128`.

The standard fused-MSE parity probe also passed:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_highcap_insert_fix_parity_mps.json
status ok
max loss diff 3.725290298461914e-09
max grad diff 0.0
```

Corrected 1-step capcheck:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_capcheck_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_capcheck_render64_site24_2_4_8_16_verifier.json
```

Verifier status: `ok` with relaxed one-step PSNR thresholds. Max train candidate
rows are still `222/217/215/216` for `2/4/8/16f`.

Corrected repeat20 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_highcap_insertfix_repeat20_render64_site24_2_4_8_16_verifier.json
```

The repeat20 artifact itself is `status ok` and quality is stable:

```text
train PSNR:   13.731995 / 13.753005 / 13.661238 / 13.735021
heldout PSNR: 14.170086 / 13.992464 / 14.220456 / 14.231981
```

The verifier intentionally reports `status failed` on timing scale:

```text
WF total medians:    3.572 / 4.725 / 6.819 / 14.091 ms
WF backward medians: 3.211 / 4.360 / 6.441 / 13.678 ms
total median scale:    3.9446x
backward median scale: 4.2604x
mixed tape storage scale: 0.9922x
explicit ray storage scale: 8.0000x
```

Corrected STAR comparison:

```text
research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_highcap_insertfix_scale_64px_896t_vs_24site_2_4_8_16.json
```

```text
frame  STAR total  WF total   STAR/WF total  STAR backward  WF backward  STAR/WF backward
2      4.686 ms    3.572 ms   1.312x         2.598 ms       3.211 ms     0.809x
4      5.479 ms    4.725 ms   1.160x         3.297 ms       4.360 ms     0.756x
8      6.504 ms    6.819 ms   0.954x         4.114 ms       6.441 ms     0.639x
16     8.268 ms    14.091 ms  0.587x         5.633 ms       13.678 ms    0.412x
```

Interpretation: the corrected path is better than the invalid old table at
small frame counts and proves the flat-tape storage thesis, but it is not
compute-sublinear enough. Once all candidates are actually inserted, per-frame
sample replay dominates and WorldFoam loses the 16f comparison. Next shader work
should reduce or amortize candidate replay/order cost; do not cite the earlier
high-cap table that reported `1.117x` total scale.
