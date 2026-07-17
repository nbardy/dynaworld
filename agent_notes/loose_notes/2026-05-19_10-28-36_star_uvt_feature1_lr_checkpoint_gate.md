# STAR UVT Feature1 LR Checkpoint Gate

Date: 2026-05-19

## Goal

Follow the chunk-trace result with an optimizer/LR checkpoint gate from the
1300-step feature1/probe40 STAR UVT checkpoint. The chunk trace showed the
global-step 1318 spike was distributed across chunks, not tile overflow or a
single bad frame range, so the next cheap question was whether the continuation
schedule could avoid that region before starting native VJP/scalar fixedbin
work.

## Change

Added two diagnostic configs:

- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resumeopt_chunktrace20_from1300.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resetopt_chunktrace20_from1300.jsonc`

While running the retained-optimizer row, the first result exactly matched the
old `lr=0.005` loss path. The reason was PyTorch optimizer-state loading:
`optimizer.load_state_dict` restores the checkpoint param-group LR. The trainer
was reporting config `lr=0.001` while the loaded optimizer was still stepping
at the checkpoint LR.

Trainer fix in `src/train/train_star_uvt_feature_overfit.py`:

- record `resume_optimizer_lrs_loaded`
- re-apply the configured LR after loading optimizer state
- record final effective `optimizer_lrs`

That makes resumed diagnostic artifacts auditable: the corrected retained
optimizer row records loaded/effective LRs `[0.005] -> [0.001]`.

## Runs

Baseline already existed:

- `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace20_from1300.json`
- original `lr=0.005`, `resume_optimizer=true`
- `pass=false`
- end loss `0.895442`, feature loss `0.632250`, probe PSNR `21.818`
- spike delta at 1318 vs 1317: `+0.014559` weighted loss, `27/32` chunks worse
- no-first timing `1924.9ms/step`, render `673.2ms`, backward `1037.5ms`

Corrected retained-optimizer lower-LR row:

- `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_resumeopt_chunktrace20_from1300.json`
- W&B offline run: `wandb/offline-run-20260519_102156-urx6omer`
- checkpoint optimizer LR loaded as `[0.005]`, effective optimizer LR `[0.001]`
- `pass=true`
- end loss `0.884576`, feature loss `0.631648`, probe PSNR `21.991`
- spike delta at 1318 vs 1317: `-0.000067` weighted loss, `5/27` chunks worse/better
- no-first timing `1384.4ms/step`, render `546.1ms`, backward `748.9ms`
- zero tile overflow, tile max/p95/cap `63/42/128`

Reset-optimizer lower-LR row:

- `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_resetopt_chunktrace20_from1300.json`
- W&B offline run: `wandb/offline-run-20260519_101941-r92j1i75`
- `resume_optimizer=false`
- `pass=true`
- end loss `0.884902`, feature loss `0.631614`, probe PSNR `21.984`
- spike delta at 1318 vs 1317: `-0.000059` weighted loss, `5/27` chunks worse/better
- no-first timing `1608.9ms/step`, render `603.5ms`, backward `860.0ms`
- zero tile overflow, tile max/p95/cap `63/42/128`

Report:

- `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.json`
- generator:
  `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_lr_reset_report.py`

## Read

The 1318 objective jump is schedule-state sensitive. It is not a tile-capacity
failure and not a one-chunk rendering pathology. Lowering effective LR to
`0.001` from the 1300 checkpoint removes the spike.

Retaining optimizer moments with the corrected effective LR gives the best
weighted/probe result in this 20-step diagnostic. Resetting optimizer state
gives a marginally lower feature MSE but weaker weighted objective and slower
timing in this run.

This is a quality-continuation fix, not the renderer-speed fix. The current
target-grid/frozen-probe objective is still renderer-backward dominated, so the
speed lane remains native VJP/scalar fixedbin/tile-slot feature-gradient work.

## Next

- Continue quality from the 1300-step checkpoint with effective `lr=0.001`.
- Keep the retained-optimizer path unless the next longer run shows feature MSE
  is more important than weighted/probe score.
- Do not chase tile-overflow debugging for the 1318 spike; overflow is zero.
- For speed, move to native VJP/scalar fixedbin rather than more schedule
  tracing.
