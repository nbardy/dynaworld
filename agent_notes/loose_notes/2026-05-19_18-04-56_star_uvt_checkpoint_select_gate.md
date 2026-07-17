# STAR UVT checkpoint-selection gate

Date: 2026-05-19 18:04

## What changed

Ran a matched checkpoint-selection gate for the current STAR UVT cached-V-JEPA
feature path. The question was whether the next quality continuation should
start from the lr005-sparse 1400 checkpoint or the lr001-sparse 1400 checkpoint.

Both configs use the selected speed path:

- `analytic_sparse_grid_forward_batched`
- `feature_direct_gradcache_reduce_vec4`
- 64f/512px/8192t/F32
- target-grid V-JEPA loss plus frozen RGB-probe40
- effective `lr=0.001`
- 50 local steps from global 1400

## Result

The lr005-sparse checkpoint wins as the continuation point:

- pass: true
- loss: `0.880512 -> 0.877791`
- feature loss: `0.627021 -> 0.625976`
- probe PSNR: `21.981 -> 22.010`
- mean step/back/render: `262.7 / 106.3 / 94.8 ms`
- last-20 step/back/render: `301.8 / 122.9 / 109.1 ms`
- zero overflow, tile max/p95 `69/46`

The lr001-sparse checkpoint fails:

- pass: false
- loss: `0.880903 -> 0.893426`
- feature loss: `0.630543 -> 0.631770`
- probe PSNR: `22.035 -> 21.843`
- mean step/back/render: `325.8 / 143.0 / 107.0 ms`
- zero overflow, tile max/p95 `63/43`
- it improves until global step `1444`, then jumps at `1444 -> 1445`
  (`+0.014899` weighted loss, `+0.001797` feature loss,
  `+0.000328` probe loss)

## Interpretation

The earlier lr001 sparse run was useful because it proved the dense-lr001
quality endpoint can be reached with the fast sparse-forward path. It is not the
best checkpoint to continue from. Its 1400 state repeats the transient jump
pattern and fails the exact matched 50-step continuation.

The next quality/media run should continue from:

`outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr005sparse_1450step.pt`

Do not spend the next gate on the failed lr001-sparse 1450 state unless the goal
is specifically to diagnose the jump.

## Artifacts

- Report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_checkpoint_select_1400_to1450.md`
- lr005-sparse config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume50_from1400_lr005sparse_checkpointselect.jsonc`
- lr005-sparse result:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr005sparse_checkpointselect.json`
- lr001-sparse config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume50_from1400_lr001sparse_checkpointselect.jsonc`
- lr001-sparse result:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr001sparse_checkpointselect.json`
