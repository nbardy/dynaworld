# STAR UVT Fast Feature-Shader Port Plan

Date: 2026-05-18
Updated: 2026-05-20

## End Goal

Build a feature-valued STAR UVT renderer that carries F32/F64 tube features
through the same time-tubed representation as the RGB STAR UVT path, while
importing the best fast-mac feature-splat shader lessons:

```text
feature tubes -> [T,H,W,F] feature image + alpha -> FeatureToColor -> RGB loss
```

The short-term target is not deterministic compact backward. The short-term
target is a fast, trainable, direct path that can overfit one 64-frame video at
256px/512px without the feature-raster backward explosion seen in the stable
F32 fast-mac path.

## Current Execution Plan

As of 2026-05-19, the core plan has three separated lanes:

1. Keep `star-feature-512-fast` as the current cached V-JEPA target-grid speed
   diagnostic: `analytic_sparse_grid_forward_batched` plus
   `feature_direct_gradcache_reduce_vec4`, 64f/512px/8192t/F32, resume from the
   1300-step checkpoint. The lr005 100-step helper/media gate passes with mean
   `399.9ms/step`, `176.9ms` backward, and last-20 `262.9ms/step`; the lr001
   rerun preserves the dense lr001 quality endpoint with mean `372.3ms/step`,
   `158.9ms` backward, feature loss `0.630549`, and probe PSNR `22.034`, but
   has noisy late timing and worse feature loss than lr005.
2. Keep the older RGB-target speed diagnostic available as
   `star-feature-512-rgbfast`: no pre-norm, `feature_direct_gradcache_reduce_vec4`,
   64f/512px/8192t/F32, `2.491s/step`, `1.184s` backward. It is not the
   cached V-JEPA target route.
   The native-prep handoff gate now proves the next benchmark-only shader step:
   `logit_handoff_reduce_vec4_native_prep` moves linear sigmoid-MSE prep to
   Metal, passes F4/F32 parity, and cuts matched `64f/512px/8192t/F32`
   prep+backward `826.35 -> 428.98ms` and total `1446.53 -> 1108.50ms`. Use
   this as the template for native hidden/frozen-probe prep or visibility/prefix
   tape, not as the final trainer route. The hidden sigmoid-MSE native follow-up
	   passes the mechanical parity gate, but rejects naive dense hidden fusion as
	   the next keeper: H32 scalar totals `317.54/610.90/2549.39ms` at
	   128/256/512px, H64 256px totals `817.27ms`, and vec4 reduce is slower than
	   scalar. The native target-area follow-up is the positive full-support port:
	   it bins once, computes native hidden64 RGB cell sums, and cuts the matched
	   full-cell8 star-only trainer row `5801.7 -> 3496.0ms/step`, while 512px
	   native-only synthetic support passes where the Torch hidden-VJP baseline
	   OOMs. It still leaves dense full RGB at `5.648`, so it is a speed/memory
	   baseline for visual-VJP work, not a visual-quality promotion. The hidden32
	   native follow-up proves smaller hidden recompute is faster (`2464.6ms/step`)
		   but rejects decoder shrinkage as the answer (`pass=false`, probe PSNR
		   `19.481`, full RGB `5.632`). The benchmark-only skip-feature-grad
		   diagnostic then isolates raw feature-gradient atomics and finds only a
		   small hidden64 native backward slice (`594.9 -> 562.2ms` at 256px,
		   `1918.6 -> 1854.3ms` at 512px), so the next factor win is not a
		   feature-atomic-only reducer.
3. Keep the STAR V-JEPA target route as the cached-feature scale gate:
   64f/512px/8192t/F32, `vjepa_tokens [1,8192,768]`, chunked logical
   `[64,32,512,512]`. The original streaming chunked row is `3.743s/step`
   with `1.734s` target chunk/loss; the new `cached_chunks` materialization
   precomputes the adapted target into 32 resident chunks (`2048MiB`) and
   cuts the same 5-step gate to `1.655s/step`, `0.770s` backward, `0.601s`
   render, and `0.202s` target/loss. The `target_grid` materialization keeps
   only the channel-adapted `[32,32,16,16]` grid resident (`1.0MiB`) and is the
   fastest V-JEPA target diagnostic at `1.351s/step`, `0.705s` backward,
   `0.548s` render, and `0.041s` target/loss. The 20-step target-grid media
   follow-up stays monotonic in feature-target loss (`0.999935 -> 0.997425`) at
   `1.451s/step`, but it is not RGB quality evidence because RGB loss is
   disabled and the colorizer is not trained. The RGB-aux1 control trains the
   colorizer and decreases both feature and RGB losses, but only moves RGB PSNR
   `4.709 -> 4.746` in 20 steps while slowing to `2.000s/step`. RGB-aux10 only
   nudges RGB PSNR to `4.750` and slightly worsens feature loss, so weight alone
   is not the visual fix. The 100-step aux10 row moves more clearly
   (`RGB PSNR 4.709 -> 5.109`, feature loss `0.999935 -> 0.964670`) at
   `1.876s/step`, so schedule length matters. A matched RGB-warm20 schedule
   (`feature=0/rgb=20` for 20 steps, then `feature=1/rgb=10`) is faster
   (`1.639s/step`) but worse on final RGB PSNR (`5.046`) and feature loss
   (`0.973557`), so feature-loss-skipping warmup is a negative visual-control
   gate. The hidden64 standalone target-grid feature-to-RGB probe now passes:
   it trains only `FeatureToColor` on the cached `[32,32,16,16]` V-JEPA target
   grid and reaches `23.401` grid PSNR plus `20.073` full-video upsampled PSNR
   at `2.427ms/step`. That proves the target features are decodable; the next
	   step was to load/freeze this decoder inside STAR training or probe logging,
	   not another RGB aux schedule. That integration gate now passes at
	   `1.220s/step` with feature loss `0.999935 -> 0.998357` and frozen-probe
	   PSNR `13.985 -> 14.060`; it proves plumbing and cheap gradient flow, but
	   the 20-step visual movement is still too small to promote quality. The
	   100-step frozen-probe follow-up is the first stronger visual diagnostic:
	   feature loss reaches `0.970035` and probe PSNR reaches `14.641` at
	   `1.268s/step`, cheaper than 100-step RGB-aux10. The 300-step extension
	   reaches feature loss `0.811652` and probe PSNR `16.560` at `1.355s/step`,
	   so the objective keeps working but still sits below the standalone
	   `20.073` full-video number. The checkpoint/no-media rerun matches that
	   curve at `1.268s/step`, and the resumed 300-step continuation reaches
	   feature loss `0.655366` and probe PSNR `19.884` at `1.440s/step`. That
	   nearly reaches the standalone full-video upsample number. The probe-emphasis
	   600->800 continuation reaches probe PSNR `21.425` at `1.512s/step`, but
	   feature loss drifts `0.655132 -> 0.703820`, so visual gain now needs to be
	   balanced against V-JEPA target alignment. The scheduled 800->1000 balance
	   row recovers feature loss `0.703862 -> 0.643852` at `1.308s/step`, but
	   gives back probe PSNR (`21.428 -> 21.382`) and is nonpassing, so simple
	   two-stage alternation is not the oracle fix. The feature0.5/probe40
	   1000->1100 Pareto row passes the combined gate at `1.461s/step`, moves
	   probe PSNR `21.384 -> 21.789`, and keeps zero overflow, but feature loss
	   drifts `0.643823 -> 0.656728`. The 1100->1200 recover schedule lowers
	   feature loss `0.656765 -> 0.635093` at `1.521s/step`, but gives back a
	   little probe PSNR (`21.792 -> 21.738`) and is nonpassing. The short
	   feature0.75/probe40 1200->1250 continuation passes and restores probe
	   PSNR `21.740 -> 21.929` at `1.523s/step`, but feature loss rises
	   `0.635066 -> 0.638799`. The feature1/probe40 1250->1300 continuation is
	   the first current both-improving objective-balance row: feature loss
	   `0.638803 -> 0.632192`, probe PSNR `21.933 -> 21.963`, zero overflow, and
	   `1.285s/step`. The 1300->1400 extension keeps both improving to feature
	   loss `0.627129` and probe PSNR `21.979`, but slows to `1.690s/step` on the
	   older dense target-grid path. The matched timing repeat is `1.711s/step`
	   with zero overflow and `68/45/128` max/p95/cap tile count, so this is not a
	   tile-overflow diagnosis. The sparse-forward batched-VJP helper/media row
	   preserves the same 100-step objective movement at mean step/backward/render
	   `399.9/176.9/125.2ms`, last-20 `262.9ms/step`, and zero overflow. The
	   effective-lr001 sparse-forward rerun keeps the dense lr001 quality endpoint
	   at mean step/backward/render `372.3/158.9/119.9ms`, feature loss
	   `0.630549`, and probe PSNR `22.034`, but gives up lr005's better feature
	   loss and has noisy late timing. The 1400 checkpoint selector rejects that
	   lr001-sparse state, and the selected lr005-sparse 1450->1500 media gate
	   passes at last-20 `254.0ms/step` while still looking blurry. The
	   whole-graph profile gate then splits the current
	   target-grid/frozen-probe objective and shows renderer backward is
	   `81.3-81.4%` of manual backward, but the isolated split does not reproduce
	   the trainer slowdown (`1565.9ms` manual total at step 1250 vs `1504.0ms`
	   at step 1300). The end-to-end trainer trace adds `step_timings_ms` to the
	   trainer JSON and does reproduce the slowdown after dropping the first
	   optimizer/warmup step (`1850.7ms` from step 1300 vs `1705.3ms` from step
	   1250), with a late loss/probe spike at global step `1318`. The chunk
	   trace around `1317-1319` shows this spike is distributed (`27/32` chunks
	   worsen, `44.5%` of weighted delta in frames `0-15`) and persists at
	   `1319`. The LR checkpoint gate confirms this is schedule-state sensitive:
	   after the trainer was fixed to re-apply config LR after optimizer-load,
	   the retained-optimizer `lr=0.001` row records loaded/effective LRs
	   `[0.005] -> [0.001]`, removes the spike, and passes with end loss
	   `0.884576`, feature loss `0.631648`, probe PSNR `21.991`, no-first
	   `1384.4ms/step`, and `748.9ms` backward. The reset-optimizer `lr=0.001`
	   control also passes (`0.884902`, `0.631614`, `21.984`) but is slower in
	   this diagnostic. Use the 1300-step checkpoint with effective `lr=0.001`
	   as the safer probe/visual continuation; this is not the renderer-speed
	   fix. The 100-step effective-lr001 continuation from 1300 passes with
	   media/checkpoint and reaches feature loss `0.630549`, probe PSNR `22.034`,
	   mean `1463.8ms/step`, and `778.4ms` backward. It avoids the early 1318
	   jump but later has a smaller transient at `1377->1378`; the older lr005
	   1300->1400 row is slower and lower on probe PSNR, but better on final
	   feature loss (`0.627129`) and slightly better weighted loss (`0.880751`).
	   The first explicit optimizer-LR schedule gate (`0.001` until global step
	   `1375`, then `0.00025`) is a negative promotion result: it removes the
	   `1377->1378` jump, but a comparable jump reappears at `1385->1386`; the
	   100-step scheduled row is worse than static lr001 on weighted loss
	   (`0.881602` vs `0.880942`), feature loss (`0.630803` vs `0.630549`),
	   probe PSNR (`22.027` vs `22.034`), and timing (`1506.9ms` / `807.2ms`
	   backward vs `1463.8ms` / `778.4ms`). The late 88-step trace is diagnostic
	   and expected to fail quality pass because it stops after the spike; it
	   confirms `26/32` chunks worsen at `1385->1386` with summed weighted-loss
	   delta `0.015248` and largest frame-0 chunk delta `0.001802`. The next
	   quality gate is checkpoint selection or a schedule keyed to measured
	   transient recovery, not lower LR forever.
	   The chain/profile/trace reports are
	   `outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`
	   and
	   `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`
	   and
	   `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`
	   and
	   `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`
	   and
	   `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`
	   and
	   `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`
	   and
	   `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md`.
	   Longer probes can now be
	   resumed:
	   `output.checkpoint`
	   writes model/colorizer/optimizer/config/row/losses, and
	   `train.resume_checkpoint` loads that state as warm-start local steps.
	   `train.global_step_offset` records explicit continuation steps for schedule
	   audits. The contract passed a real 8f/64px RGB-pyramid smoke and the real
	   64f/512px frozen-probe scale gate with optimizer resume and zero overflow.
	   The first dense-analytic target-grid render-mode trainer matrix said
	   reducer modes were not the keeper-objective speed fix: all modes pass and
	   land on the same 5-step loss/probe PSNR, but repeat-top no-first timing is
	   direct-atomic `1249.0ms`, cached-bins `1410.9ms`, vec4 `1509.6ms`, and
	   fixedbin-request `1422.6ms`; fixedbin-request is still
	   `kernel_backward_mode=direct_atomic`. The sparse-grid/sparse-forward gates
	   below supersede that dense-analytic matrix. The sparse-pixel VJP gate is
	   the first current-objective speed win that survives the trainer loop:
	   `feature_target.image_vjp_mode=analytic_sparse_pixels` reuses forward
	   bins and only visits the nonzero target-grid image-gradient pixels. The
	   repeat-3 profile keeps parity (`4.61e-08` max grad error, zero loss
	   error) and cuts dense analytic bridge total `1245.9ms -> 920.5ms`, with
	   renderer backward `557.6ms -> 46.3ms` while sparse packing still costs
	   `184.0ms`. The matched 5-step trainer smoke passes from the 1300
	   checkpoint and cuts no-first step `1318.0ms -> 973.7ms` at identical
	   loss/probe PSNR, visiting only `65,536` pixels per step (`0.390625%` of
	   dense). The direct sparse-grid VJP follow-up now supersedes it:
	   `feature_target.image_vjp_mode=analytic_sparse_grid` generates those
	   sparse source pixel ids/values directly from the trilinear target-grid/probe
	   gradient, passes profile parity (`4.60e-08` max grad error), cuts bridge
	   total to `760.6ms`, and passes the matched 5-step trainer gate at
	   `795.3ms` no-first step / `88.6ms` no-first backward. The sparse-grid
	   render-mode matrix keeps `feature_direct_gradcache_reduce_vec4` selected:
	   no-first `730.5ms`, mean backward `78.3ms`, zero overflow, ahead of
	   gradcache `759.4ms` and direct atomic `779.3ms`. The sparse-forward
	   follow-up then removes dense feature-image rendering for this objective:
	   `feature_target.image_vjp_mode=analytic_sparse_grid_forward` matches dense
	   feature/alpha values exactly, initially cuts forward render
	   `515.9ms -> 70.5ms` (`7.322x`), and passes the matched 5-step trainer gate
	   at `492.3ms` no-first / `413.7ms` last step. The follow-up 128/256/512
	   matrix passes all rows with zero overflow but reveals run-order timing
	   sensitivity: sequential no-first is `379.2ms`, `494.2ms`, and `973.0ms`,
	   while an isolated 512px repeat after scale is `598.2ms` no-first /
	   `477.6ms` last step. A dedicated 512px repeat-3 timing gate then gives
	   no-first step mean/min/max/stdev `504.9/411.0/626.4/110.3ms`, last-step
	   `468.8/409.3/549.9/72.7ms`, and no-first backward
	   `142.2/114.7/174.4/30.1ms`, with all rows passing and zero overflow. This
	   becomes the selected target-grid/frozen-probe diagnostic, not a stable
	   hard speed baseline. The batched target/probe VJP path is now the selected
	   opt-in trainer speed lever: one batched MPS pass over all 32 chunks
	   preserves loss/gradient-pack parity (`7.45e-09`, `6.55e-11`) and cuts
	   isolated loss+VJP `38.0ms -> 4.8ms`; the 5-step optimizer harness is
	   trainable at `173.1ms` no-first step with zero overflow, and the
	   integrated trainer repeat-3 gate gives no-first `179.3/159.7/215.6/31.5ms`.
	   The next speed gate should now either run a longer overfit with this mode
	   or beat it with native target-grid/probe loss+VJP or scalar fixedbin/tile-slot
	   feature-gradient accumulation. The target-cache budget gate
   shows why dense cached chunks remain bounded: the
   same float32 adapted target is `4GiB` at 128f/512px/F32 or 64f/512px/F64,
   and `8GiB` at 64f/1024px/F32.
3. Keep Gaussian/token cached conditioning as the dataset-scale route until
   its 512px promotion is guarded: recon-only cached conditioning profiles at
   `3.460s/step`, while prediction-side V-JEPA loss is still a negative
   `38.621s/step` because frozen V-JEPA dominates backward.

The next implementation gates are:

- extend or reschedule the passing frozen target-grid feature-to-RGB STAR
  objective before judging target-grid visual quality; the first 20-step
  integration gate is cheap but barely moves probe PSNR, the 100-step
  frozen-probe row moves more clearly, and the 300-step extension reaches
  `16.560` probe PSNR. The checkpointed 300+300 continuation reaches `19.884`,
  nearly matching the standalone full-video upsample number. The probe-emphasis
  600->800 row reaches `21.425` probe PSNR but drifts feature loss upward, so
  the scheduled 800->1000 balance row was run and recovers feature alignment
  while giving back a little probe quality. The feature0.5/probe40 1000->1100
  Pareto row is passing and pushes probe PSNR to `21.789`, but drifts feature
  loss back to `0.656728`. The 1100->1200 recover schedule pulls feature loss
  down to `0.635093`, but gives probe PSNR back to `21.738`. The short
  1200->1250 feature0.75/probe40 row restores probe PSNR to `21.929` but
  raises feature loss to `0.638799`. The 1250->1300 feature1/probe40 row is the
  first current both-improving row, ending at `21.963` probe PSNR and
  `0.632192` feature loss. The 1300->1400 extension keeps both improving to
  `21.979` probe PSNR and `0.627129` feature loss, but slows to `1.690s/step`
  on the older dense path. The sparse-forward batched-VJP helper keeps the same
  movement at `0.400s/step` mean and `0.263s/step` last-20, so the next gate is
  visual quality or native VJP that beats that speed surface. The LR gate says
  effective `lr=0.001` from the 1300 checkpoint is safer for
  probe/visual continuation, but the 100-step run is not a clean quality
  dominance: it reaches higher probe PSNR (`22.034`) and lower timing
  (`1.464s/step`) than lr005, while ending worse on feature loss (`0.630549`
  vs `0.627129`) and weighted loss (`0.880942` vs `0.880751`). The next quality
  gate is checkpoint selection or a schedule keyed to measured transient
  recovery because the explicit `0.001 -> 0.00025` optimizer-LR schedule only
  moves the jump from `1377->1378` to `1385->1386` and ends worse than static
  lr001 (`0.881602` weighted, `0.630803` feature, `22.027` probe PSNR).
	  Sparse forward plus sparse-grid VJP plus vec4 reduce is the selected
	  current objective diagnostic, but the 128/256/512 scale matrix and
	  repeat-3 gate show timing is repeat-sensitive (`492.3ms` best isolated
	  no-first, `598.2ms` isolated repeat, `973.0ms` sequential 512px row,
	  repeat-3 no-first mean/min/max/stdev `504.9/411.0/626.4/110.3ms`). The
	  batched target/probe VJP path is now integrated as an opt-in trainer mode:
	  isolated loss+VJP `38.0ms -> 4.8ms`, 5-step harness no-first `173.1ms`,
	  and first-class repeat-3 no-first step `179.3/159.7/215.6/31.5ms`, all
	  zero-overflow. Native GPU target-grid/probe loss+VJP or a scalar
	  fixedbin/tile-slot accumulator remains the lower-level speed gate while
	  closing the same-grid `23.401` oracle, but must beat the batched trainer
		  distribution. The native target-area full-support visual-VJP gate now passes
		  as the compact visual-gradient baseline for dense support: use it before any
		  new full-cell visual objective, but do not call it the selected overfit route
		  until either native reverse recompute drops or dense RGB quality improves. The
	  100-step aux10 control remains far below
  RGB STAR quality, and RGB-warm20 is a matched negative. Prototype native-VJP
  losses so larger
  V-JEPA target runs do not require 4-32GiB resident adapted targets;
- port the sparse-grid target/probe loss+VJP into a native GPU path or build a
  scalar fixedbin/tile-slot feature-gradient accumulator for STAR F32; avoid
  another duplicate-traversal two-pass fork, dense-VJP pack-only iteration, or
  dense-forward regression;
- fix Gaussian/token 512px promotion NaNs before using the 300-clip multires
  run as a scale baseline;
- keep feature STAR quality work focused on objective/decoder shape because
  Gate 4 still loses badly to RGB STAR source overfit.

## Current State

Recorded evidence:

```text
outputs/benchmarks/2026-05-18_renderer_scaling_report.md
outputs/benchmarks/2026-05-19_renderer_scaling_report.md
outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md
outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.md
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.json
outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.json
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.json
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.json
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.json
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.json
outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract.json
outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_tiny_parity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_128_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_directatomic_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun2_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_tiny_parity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun3_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_tiny_parity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun4_after_fused_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun2_after_fused_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_tiny_parity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_rerun2_after_skip_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_tiny_parity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_serial_rerun2_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun5_after_linear_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_tiny_parity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun6_after_logit_64f_256_32768_f32.json
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_report.md
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_matrix_256_512_64f_32768t_f32/summary.md
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_64f512_from1300.md
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_8f64_smoke.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_report.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_profile.md
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsepixvjp_64f512_from1300_5step.json
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_report.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_profile.md
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsegridvjp_64f512_from1300_5step.json
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_tiny_parity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_serial_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun7_after_vec4_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_rerun2_after_vec4_sequential_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun3_after_vec4_sequential_64f_256_32768_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_mode_matrix_128_256_64f_32768t_f32/summary.md
outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32.json
outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32_chunkparity.json
outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_video_overfit_8f_64px_512t_f32_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_8192t_f32_chunk4_3step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_cap256_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_cap256_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_cap256_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_vec4_alpha1_72_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_2048t_f32_chunk2_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_4096t_f32_chunk2_gradcache_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_2step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_2step.json
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_media.json
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_media.json
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_media.json
outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_same_session_before_cachedbins_64f_256_32768_f32.json
outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_cachedbins_64f_256_32768_f32.json
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_cachedbins_prenorm_2step.json
outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached/summary.md
outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json
outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md
outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.json
outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_bridge_smoke.md
outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_bridge_smoke.md
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_512_scale_gate.md
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_chunkedtarget_lr005_5step.json
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_cachedchunks_lr005_5step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step.json
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_side_by_side.mp4
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_vec4_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_vec4_alpha1_72_cap256_20step_side_by_side.mp4
agent_notes/loose_notes/2026-05-18_17-13-51_renderer_scaling_matrix_star_dynamic_feature.md
```

Key rows from the 64f / 32768 primitive stress:

```text
STAR UVT RGB direct_atomic:
  128px 110.4 ms
  256px 182.4 ms
  512px 521.5 ms

Dynamic RGB projected raster, best rows:
  256px v8 252.7 ms total / 172.2 ms backward
  512px v8 693.0 ms total / 541.3 ms backward

F32 feature projected raster:
  256px v11 fixedbin 1582.2 ms vs stable 3642.2 ms
  512px f32_fixedbin 5920.8 ms vs stable 41036.8 ms

STAR UVT F32 feature first-class real-video rows:
  synthetic direct_atomic serial A/B, 64f/256px/32768t/F32:
    630.2 ms total / 485.6 ms backward, zero overflow
  synthetic gradcache serial A/B, 64f/256px/32768t/F32:
    621.4 ms total / 471.3 ms backward, zero overflow
  synthetic gradcache skip-feature-gradient diagnostic:
    515.0 ms total / 327.7 ms backward, zero overflow, feature grad intentionally zero
  synthetic gradcache rerun after diagnostic:
    786.5 ms total / 592.5 ms backward, zero overflow
  synthetic gradcache reduce-feature-gradient prototype:
    709.8 ms total / 523.8 ms backward, zero overflow, full grad parity
  synthetic full-gradcache same-session comparison:
    654.0 ms total / 491.1 ms backward, zero overflow
  synthetic fused first3 sigmoid-MSE RGB handoff prototype:
    468.7 ms total / 309.3 ms backward, zero overflow, full grad parity for
    the narrow first3 objective
  synthetic generalized linear sigmoid-MSE handoff with colorizer grads:
    800.1 ms total / 618.5 ms backward, zero overflow, full tube/feature and
    colorizer weight/bias grad parity; rerun 792.9 ms / 615.5 ms backward
  synthetic generalized linear sigmoid-MSE with colorizer weight/bias grads
  skipped:
    956.0 ms total / 714.1 ms backward on first run, 801.6 ms / 598.5 ms
    backward on rerun; tube/feature parity still passes, colorizer grads are
    intentionally zero
  synthetic gradcache after generalized handoff:
    634.0 ms total / 477.5 ms backward, zero overflow
  synthetic image-space-prep logit handoff:
    835.6 ms total / 180.3 ms forward / 60.2 ms handoff prep /
    595.2 ms renderer backward, zero overflow, tube/feature grad parity
  synthetic gradcache after logit handoff:
    693.2 ms total / 529.0 ms backward, zero overflow
  synthetic gradcache_reduce_feature_grad_vec4:
    F4/F32 parity passes, 648.0 ms total / 484.7 ms backward, zero overflow
  synthetic same-session controls after vec4:
    gradcache 690.9 ms total / 528.2 ms backward; scalar reduce 721.8 ms /
    516.4 ms backward; skip-feature-gradient 553.6 ms / 326.5 ms backward
  generated sequential 128/256 mode matrix:
    outputs/benchmarks/2026-05-18_star_uvt_feature_direct_mode_matrix_128_256_64f_32768t_f32/summary.md
    128px: gradcache 332.7 ms / 285.1 ms backward, vec4 reduce 315.5 ms /
      256.2 ms backward, but all 32768t rows overflow at 128px under cap128
    256px: gradcache 962.5 ms / 731.3 ms backward, vec4 reduce 968.2 ms /
      718.8 ms backward, fused-first3 696.1 ms / 465.5 ms backward
  synthetic gradcache after fused:
    717.4 ms total / 547.6 ms backward, zero overflow
  synthetic skip-feature-gradient after fused:
    516.9 ms total / 351.6 ms backward, zero overflow, feature grad intentionally zero
  256px/64f/32768t alpha>=1/72/cap256 1320.9 ms total / 1021.2 ms backward
  256px/64f/32768t alpha>=1/80/cap256 1173.1 ms total / 931.3 ms backward but overflows late
  256px/64f/32768t feature_direct_fixedbin requested, unpruned/cap256:
    fallback required after 216 overflow tiles, 1603.1 ms total / 1146.6 ms backward
  256px/64f/32768t feature_direct_fixedbin requested, alpha>=1/72/cap256:
    fixedbin-eligible direct-atomic row, zero overflow, 1252.8 ms total /
      991.7 ms backward
  256px/64f/32768t feature_direct_gradcache, alpha>=1/72/cap256:
    zero overflow, rerun 1807.0 ms total / 1333.1 ms backward
  256px/64f/32768t feature_direct_gradcache_reduce, alpha>=1/72/cap256:
    zero overflow, rerun 1889.7 ms total / 1394.8 ms backward; trainable but
    slower
  256px/64f/32768t feature_direct_gradcache_reduce_vec4, alpha>=1/72/cap256:
    zero overflow, 2094.8 ms total / 1412.5 ms backward; trainable but slower
    than gradcache/reduce in the first-class cap256 path
  512px/64f feature_direct_gradcache chunk2 scale probes:
    4096t passes, zero overflow, max tile 18, p95 9:
      6456.4 ms total / 4208.9 ms backward / 1220.7 ms forward
    8192t passes, zero overflow, max tile 33, p95 17:
      7937.3 ms total / 4882.8 ms backward / 1385.8 ms forward /
      1223.4 ms color-loss
  first-class manual backward split:
    256px/32768t/alpha>=1/72/cap256 gradcache:
      1994.7 ms manual total, 987.8 ms colorize/loss backward,
      553.0 ms renderer backward; renderer is 35.9% of backward
    256px same split, vec4 reduce diagnostic:
      1727.7 ms manual total, 867.4 ms colorize/loss backward,
      494.7 ms renderer backward; renderer is 36.3% of backward
    512px/4096t gradcache:
      6566.8 ms manual total, 3775.0 ms colorize/loss backward,
      1071.7 ms renderer backward; renderer is 22.1% of backward
    512px/8192t gradcache:
      5372.8 ms manual total, 3430.1 ms colorize/loss backward,
      700.0 ms renderer backward; renderer is 16.9% of backward
    512px/4096t gradcache, no pre-norm diagnostic:
      2018.9 ms manual total, 317.1 ms colorize/loss backward,
      674.8 ms renderer backward; renderer is 68.0% of backward
    512px/8192t gradcache, no pre-norm diagnostic:
      2403.5 ms manual total, 400.6 ms colorize/loss backward,
      751.5 ms renderer backward; renderer is 65.2% of backward
  512px/8192t first-class no-pre-norm trainer:
      pass, zero overflow, loss 0.33817 -> 0.33764 in 2 steps,
      3715.4 ms/step, 1585.6 ms backward, 1268.3 ms forward,
      440.9 ms colorize/loss
  512px/8192t identity/no-pre-norm 20-step media diagnostic:
      pass, zero overflow, loss 0.34748 -> 0.32446,
      PSNR 4.591 -> 4.888, 2536.6 ms/step, 1173.5 ms backward,
      936.2 ms forward, 306.0 ms colorize/loss
  512px/8192t hidden64 pre-norm 20-step media diagnostic:
      pass, zero overflow, loss 0.33869 -> 0.31716,
      PSNR 4.702 -> 4.987, 19179.6 ms/step, 13769.3 ms backward,
      2229.2 ms forward, 2415.3 ms colorize/loss
  512px/8192t pre-norm gain2 20-step media diagnostic:
      pass, zero overflow, loss 0.33844 -> 0.31719,
      PSNR 4.705 -> 4.987, 14119.5 ms/step, 8913.4 ms backward,
      2567.8 ms forward, 2047.3 ms colorize/loss
  synthetic cached-bin sidecar same-session comparison:
      gradcache 1697.4 ms total / 1068.0 ms backward;
      gradcache_cached_bins 1544.0 ms total / 935.8 ms backward,
      zero overflow, parity passes
  512px/8192t first-class cached-bin sidecar 2-step diagnostic:
      pass, zero overflow, loss 0.33874 -> 0.33432,
      16196.4 ms/step, 10241.4 ms backward, no end-to-end win versus
      same-session plain gradcache 16210.3 ms/step / 9675.8 ms backward
  128/256/512 direct-mode matrix with cached-bin modes:
      all 39 rows pass at 64f/32768t/F32.
      512px fastest full-gradient total row:
        gradcache_cached_bins 1978.6 ms total / 1102.8 ms backward.
      512px fastest diagnostic row:
        gradcache_skip_feature_grad 1713.5 ms total / 803.7 ms backward.
      cached-bin deltas are mixed:
        gradcache 2020.4 -> 1978.6 ms total at 512px,
        but gradcache 1128.3 -> 1384.1 ms total at 256px.
  feature-gradient-only / two-pass split diagnostic:
      tiny F4/F32 parity passes for `gradcache_feature_grad_only` and
      `gradcache_two_pass_feature_grad`.
      refreshed 256px matrix:
        gradcache 972.2 ms total / 691.9 ms backward,
        feature-only 817.1 ms total / 529.1 ms backward,
        two-pass 1342.6 ms total / 1063.2 ms backward.
      refreshed 512px matrix:
        gradcache 2466.8 ms total / 1379.2 ms backward,
        feature-only 1873.6 ms total / 862.0 ms backward,
        two-pass 2471.4 ms total / 1613.1 ms backward.
      reverse-order 512px check:
        two-pass 3216.7 ms total / 1821.1 ms backward,
        gradcache 2066.3 ms total / 1203.8 ms backward.
  fixedbin/tile-slot accumulator budget:
      artifact:
        outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_budget_128_256_512_64f_32768t_f32/summary.md
      theoretical feature-gradient atomic-write reduction:
        128x at 128/256/512px.
      naive prefix recompute multiplier:
        64.4x at 128px, 39.8x at 256px, 10.8x at 512px.
      scalar f32 contribution-weight tape:
        0.499 GiB at 128px, 1.171 GiB at 256px, 1.195 GiB at 512px.
      wrong per-channel f32 tape:
        16.0 GiB at 128px, 37.5 GiB at 256px, 38.2 GiB at 512px.
  tile-slot reducer isolation:
      artifact:
        outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_reduce_matrix_256_512_64f_32768t_f32/summary.md
      feature-only reducer:
        gradcache_feature_grad_only 532.8 ms backward at 256px,
        gradcache_feature_grad_only_reduce_vec4 449.9 ms backward at 256px,
        gradcache_feature_grad_only 869.1 ms backward at 512px,
        gradcache_feature_grad_only_reduce_vec4 774.8 ms backward at 512px.
      full-gradient refresh:
        outputs/benchmarks/2026-05-19_star_uvt_feature_full_reduce_refresh_256_512_64f_32768t_f32/summary.md
        gradcache 1284.2 ms backward at 512px,
        gradcache_reduce_feature_grad_vec4 1108.0 ms backward at 512px.
```

Interpretation:

- STAR UVT already has the better time-scaling representation for RGB.
- fast-mac feature forks have the better high-F shader tricks.
- STAR UVT now has the feature-valued renderer, but still lacks the F32
  backward tricks that made the projected feature raster competitive.
- Gate 0 dense feature-tube contract now passes on CPU and MPS. This proves the
  shape/gradient/microbatch contract, not the Metal feature shader speed.
- `feature_uvt.render_mode=feature_direct_fixedbin` now exists as a trainer and
  reporting contract. Today it gates fixed-bin eligibility and records
  `effective_render_mode` / `mode_fallback_required` around the existing direct
  feature Metal path; it is not yet a separate optimized fixedbin feature
  backward kernel.
- `feature_uvt.render_mode=feature_direct_gradcache` is the first actual
  feature-backward fast mode. It caches each pixel's F32
  `grad_feature_image[pixel]` vector once in the direct backward kernel when
  `feature_dim <= 64`. The win is real but small, so it should be treated as a
  baseline improvement before the larger feature-gradient atomic/reduction
  port, not as the final fast path.
- `gradcache_skip_feature_grad` is benchmark-only and intentionally returns
  incorrect/zero `grad_feature`. It keeps geometry/opacity gradients intact and
  isolates feature-gradient atomic overhead. On the 64f/256px/32768t/F32
  synthetic probe it cuts backward to `327.7ms`, so the next real kernel should
  target feature-gradient reduction, not another grad-vector load cache.
- `gradcache_reduce_feature_grad` is the first trainable attempt at that target:
  it ports the v11-style per-tile/simd feature-gradient reduction into STAR UVT
  stable tiles and falls back to the direct path for unsupported/unstable cases.
  It passes F4/F32 tiny parity and real-video training, but it is slower on the
  target row (`523.8ms` synthetic backward versus `491.1ms` same-session
  gradcache; `1000.3ms` first-class backward versus `973.2ms`). Do not promote
  it as the default; keep it as evidence that barrier-heavy per-contributor
  reduction is the wrong shape for this STAR kernel.
- `gradcache_reduce_feature_grad_vec4` is the follow-up vectorized reduction
  probe. It changes only the channel reduction shape, packing feature channels
  into `float4` SIMD reductions while preserving full gradients. It passes
  tiny F4/F32 parity and improves the synthetic direct-kernel cap128 row
  against same-session gradcache (`484.7ms` vs `528.2ms` backward), but the
  first-class cap256 real-video row is slower than gradcache and scalar reduce
  (`1412.5ms` backward vs `1333.1ms` and `1394.8ms`). Keep it available as a
  diagnostic/trainer-selectable mode, not as the default. This narrows the
  next optimization target: changing scalar SIMD math is not enough; the
  feature-gradient path needs a different accumulation topology or a true
  fixedbin/sidecar strategy.
- Fresh 512px/8192t first-class gates change the practical fast-overfit
  selection: with the default pre-norm colorizer, `feature_direct_gradcache`
  and `feature_direct_gradcache_reduce_vec4` are nearly tied
  (`7.825s/step`, `5.181s` backward versus `7.690s`, `5.088s` on the 2-step
  rerun). With `colorize.pre_norm=false`, the same 20-step media gate selects
  the vec4 mode as the fastest current feature-tube diagnostic:
  `2.858s/step`, `1.327s` backward for gradcache versus `2.491s/step`,
  `1.184s` backward for reduce-vec4, identical loss/PSNR (`0.32053`,
  `4.941`) and zero overflow. This is now preserved as
  `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
  star-feature-512-rgbfast`, but it is still a speed diagnostic because Gate 4
  feature quality trails RGB STAR badly.
- The selected-shader 128/256/512 scale gate shows that this choice is specific
  to high resolution. With no-pre-norm 8192-tube first-class rows, vec4 reduce
  is a tie/slight backward loss at 128px, a small win at 256px, and a clear
  win at 512px. The same gate exposed the low-resolution support trap: 128px
  overflows at cap128/default alpha, cap256/default alpha, and
  cap256/`alpha>=1/72`; the first valid 128px row needed cap256 plus
  `alpha>=1/32`.
- `fused_first3_sigmoid_mse` is a benchmark-only RGB handoff prototype through
  the existing direct feature backward op. It interprets channels 0..2 of the
  input image as RGB targets, reconstructs the pixel feature/alpha locally,
  computes `alpha * sigmoid(feature[:3]) -> mean MSE` VJP inside Metal, and
  feeds that gradient into the STAR reverse contributor loop. It is deliberately
  not the learned F32 colorizer path, but it proves the handoff shape: full
  parity and `309.3ms` synthetic backward, faster than both full gradcache and
  skip-feature-gradient reruns in the same regime. The matched
  `64f/512px/8192t/F32` rerun also passes and records `494.09ms` backward /
  `1152.58ms` total, so the boundary remains interesting at the active V-JEPA
  scale.
- `direct_linear_sigmoid_mse_backward` is the generalized benchmark handoff:
  it supports a real `[3,F]` linear colorizer, bias, sigmoid, mean-MSE target,
  and colorizer parameter gradients. It passes F4/F32 parity down to the
  colorizer weights and biases, but it is not a speed win: `615-619ms`
  backward on two full-gradient timing runs versus `477.5ms` for a same-session
  gradcache rerun. A skip-colorizer-gradient variant was noisy and did not
  produce a convincing mean win (`598.5-714.1ms` backward), so the regression is
  not explained by colorizer parameter atomics alone. Do not promote this
  direct in-tile generalized handoff to the trainer; use it as evidence that
  the next handoff needs a different reduction shape.
- `direct_logit_handoff_backward` is the cheaper image-space-prep handoff:
  Torch/image space computes `grad_logits` and `grad_alpha`, while Metal only
  applies `W^T @ grad_logits` inside the STAR reverse traversal. It passes
  F4/F32 tube/feature parity, but it is still slower than same-session
  gradcache on the target row (`595.2ms` renderer backward plus `60.2ms` prep,
  `835.6ms` total, versus `529.0ms` / `693.2ms` for gradcache). This means
  merely replacing the dense `[T,F,H,W]` gradient input with RGB/logit inputs is
  not enough; the remaining cost is the per-pixel `W^T` work plus unchanged
  per-channel feature-gradient atomics.
- `logit_handoff_reduce` and `logit_handoff_reduce_vec4` combine the image-space
  logit/alpha prep with the stable-tile feature-gradient reducers. The 256/512
  direct matrix passes F4/F32 parity and zero overflow. Vec4 improves synthetic
  backward at 256px (`571.7 -> 510.6ms`) and narrowly at 512px
  (`654.8 -> 642.3ms`); scalar reduce regresses 512px backward to `722.5ms`.
  Because forward and handoff-prep timings also move in the 512px row, keep this
  as a diagnostic native-VJP/tile-slot bridge candidate, not a trainer-default
  claim.
- `star_uvt_logit_handoff_rgb_vjp_profile.py` is the first real-video profile
  of that bridge against a trainer-style linear RGB reconstruction loss. It
  loads the 64f/512px/8192t 1300-step checkpoint, compares standard autograd
  against manual `logit_handoff_reduce_vec4`, and matches model/colorizer
  gradients (`9.43e-09` max abs error, zero loss error) while measuring
  `1691.0 -> 1587.4ms` (`1.065x`). The 8f/64px smoke is a stronger small-scale
  speed result (`78.8 -> 34.7ms`, `2.27x`). This proves compatibility for a
  linear RGB loss only; target-grid V-JEPA MSE and the hidden64 frozen-probe
  objective still need a generic image-space VJP/native loss bridge.
- `gradcache_cached_bins` is the forward-bin sidecar diagnostic. It avoids the
  backward clear/bin stage by saving forward `tile_counts`, `tile_tube_ids`,
  `tile_depths`, and `tile_unstable`. It passes parity and wins the isolated
  64f/256px/32768t/F32 synthetic renderer comparison (`1068.0ms -> 935.8ms`
  backward), but it does not improve the first-class 512px trainer row
  (`16.20s/step`, `10.24s` backward versus same-session plain gradcache
  `16.21s/step`, `9.68s` backward). The follow-up 128/256/512 direct-mode
  matrix keeps the same decision: cached-bin rows pass and sometimes help
  renderer timing, but the effect is mixed (`gradcache` total improves
  `2020.4 -> 1978.6ms` at 512px and regresses `1128.3 -> 1384.1ms` at 256px).
  Keep it as a diagnostic; rebinning is not the main 512px blocker.
- `gradcache_feature_grad_only` and `gradcache_two_pass_feature_grad` test the
  simplest split of geometry/opacity gradients from feature gradients. The
  split is correct, but the naive two-kernel recompute is slower than full
  gradcache at the relevant backward boundary: `1.063s` versus `0.692s` at
  256px and `1.613s` versus `1.379s` at 512px in the refreshed matrix, with a
  reverse-order 512px rerun also negative. Keep the diagnostic modes for
  measurement, but do not implement first-class training by duplicating STAR
  traversal. The next viable "two-pass" design must precompute a compact
  fixedbin/tile-slot contribution structure or move more VJP work native.
- `tile_slot_accumulator_budget.py` sizes that contribution structure before a
  larger Metal fork. The good news is that tile-slot accumulation can reduce
  feature-gradient write count by `128x` by moving from per-pixel/slot/channel
  atomics to one atomic per tile slot and channel. The bad news is that naive
  per-slot prefix recompute is much too expensive (`39.8x` extra slot-pixel
  work at 256px, `10.8x` at 512px), and storing per-channel contribution
  weights is impossible at the target (`37-38GiB`). A scalar contribution
  weight/prefix tape is closer to feasible (`~1.2GiB` at 256/512px, before
  chunking or f16), but it must be scalar over channels. This turns the next
  implementation target into a compact scalar weight/prefix tape plus channel
  reduction, or a native image-space VJP that avoids the tape entirely.
- The reducer-only isolation gate clarifies that the current tile-slot reducer
  is not purely a dead end: when isolated to feature gradients only,
  `gradcache_feature_grad_only_reduce_vec4` improves backward by `15.6%` at
  256px and `10.9%` at 512px versus direct feature-only atomics. A refreshed
  full-gradient synthetic run also has `gradcache_reduce_feature_grad_vec4`
  beating plain gradcache at 512px (`1108.0ms` versus `1284.2ms` backward).
  This keeps single-pass vec4 tile-slot reduction alive as a candidate, but it
  does not rescue two-pass composition: combining geometry-only with reduced
  feature-only still duplicates traversal and remains behind or tied with
  single-pass gradcache at the trainer-relevant boundary.
- `firstclass_backward_breakdown.py` now separates the real trainer graph into
  render forward, `FeatureToColor`/loss forward, `FeatureToColor`/loss backward
  to image gradients, and Metal renderer backward. This corrects the working
  assumption that the 512px problem is mostly the rasterizer: at 512px the
  renderer is only `16.9-22.1%` of backward on the 4096/8192t gradcache rows,
  while the image-space colorizer/loss backward is `77.9-83.1%`. At the
  256px/32768t/cap256 target, renderer backward is still only about `36%` of
  backward. Pure feature-gradient shader work is necessary, but it cannot be
  the whole speed plan unless the dense `FeatureToColor`/loss VJP is also
  avoided, simplified, or made native.
- A no-pre-norm colorizer A/B makes that cost concrete. With only
  `colorize.pre_norm=false`, the 512px manual split drops colorizer/loss
  backward from `3775.0/3430.1ms` to `317.1/400.6ms` on 4096/8192 tubes. The
  actual 512px/8192t 2-step trainer also passes and drops mean step from
  `7937.3ms` to `3715.4ms`, with backward down from `4882.8ms` to `1585.6ms`.
  The 20-step media A/B keeps no-pre-norm faster (`7.37s/step` versus
  `11.10s/step`), but default pre-norm ends slightly better (`0.31742` loss
  versus `0.32053`), so no-pre-norm remains a speed candidate rather than a
  promoted quality setting. Removing sigmoid as well gives the fastest row
  (`2.54s/step`, `1.17s` backward), but worsens quality to `0.32446` loss /
  `4.888` PSNR, so the simple decoder-unclamp route is only a speed diagnostic.
  A hidden-64 pre-norm decoder barely improves PSNR (`4.984 -> 4.987`) while
  slowing to `19.18s/step` and `13.77s` backward, so naive dense decoder
  capacity is not the practical quality bridge either. Reducing pre-norm
  sigmoid init gain from `4` to `2` is similar (`4.987` PSNR) and slower than
  gain-4 linear pre-norm (`14.12s/step`, `8.91s` backward), so simple init gain
  is not the bridge either.
- The precomputed V-JEPA bridge audit originally closed a naming/target
  ambiguity by showing the old `star-feature-512-fast` was RGB-target only. That
  is now superseded: `star-feature-512-fast` launches the cached V-JEPA
  target-grid/frozen-probe sparse-forward batched route, while
  `star-feature-512-rgbfast` preserves the old RGB-target row. The bridge is no
  longer config plumbing; the next comparison is visual quality from the
  trained/frozen feature-to-RGB probe and whether native-VJP can beat the
  batched target/probe timing surface without multi-GiB resident targets.

## Precomputed V-JEPA Bridge Contract

Current answer: **selected fast helper now uses cached V-JEPA targets; old
RGB-target helper kept separately**.

The fastest STAR UVT feature diagnostic today is:

```text
src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-fast
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc
```

That route renders sparse F32 tube features only on the target-grid support
pixels, batches target-grid/frozen-probe loss+VJP across chunks, and trains
against cached V-JEPA target-grid features. The older RGB reconstruction route
is now `star-feature-512-rgbfast`.

Port the cached-feature lane in this order:

1. Add a STAR feature-target config section separate from RGB
   `colorize`/reconstruction loss.
2. Reuse `VideoFeatureCache` or a narrow adapter so STAR and the existing
   precomputed-feature Gaussian/token trainers share cache semantics.
3. Define the target-grid adapter between dense STAR feature images and cached
   V-JEPA token grids. Do not hide resize/pooling in ad hoc tensor reshapes.
4. Start with an `rgb_pyramid` cache smoke so the contract can run without
   extractor cost.
5. Run the selected no-pre-norm `feature_direct_gradcache_reduce_vec4`
   renderer under cached-feature loss.
6. Before long runs, reduce the now-visible target chunk/loss cost or move
   target evaluation closer to the renderer.
7. Scale to the 300-prepared set and compare against the existing
   Gaussian/token V-JEPA rows with explicit baseline entries.

Bridge smoke result:

```text
config:
  src/train_configs/star_uvt_feature_testvideo_8f_64_rgbpyramid_target_gradcache_reduce_vec4_10step.jsonc
target:
  VideoFeatureCache extractor=rgb_pyramid, layer=rgb_x1,
  source_shape=[1,3,8,64,64], adapted_shape=[8,32,64,64],
  channel_adapter=repeat_truncate, temporal_spatial_adapter=trilinear
loss:
  feature_target_loss=mse, feature_target_loss_weight=1.0, rgb_loss_weight=0.0
cache-hit rerun:
  pass=true, loss 0.34006 -> 0.24809, zero overflow,
  mean step 93.5ms, render forward 17.9ms, loss prep 5.7ms,
  backward 43.0ms
```

This completes the cheap cached-target contract gate. It does not complete the
real V-JEPA gate, because `rgb_pyramid` is only a deterministic local smoke
extractor. The target adapter now also handles cached token tensors when a
config provides `feature_target.token_grid_shape=[T,H,W]`, but the real V-JEPA
grid shape still has to be chosen and benchmarked.

Real V-JEPA bridge smoke result:

```text
config:
  src/train_configs/star_uvt_feature_testvideo_8f_64_vjepa_target_gradcache_reduce_vec4_10step.jsonc
target:
  VideoFeatureCache extractor=vjepa_torchhub, model=vjepa2_1_vit_base_384,
  layer=vjepa_tokens, source_shape=[1,1024,768],
  token_grid_shape=[4,16,16], adapted_shape=[8,32,64,64],
  channel_adapter=truncate_or_pad, temporal_spatial_adapter=trilinear,
  normalization=channel_standardize
loss:
  feature_target_loss=mse, feature_target_loss_weight=1.0, rgb_loss_weight=0.0
cache-hit rerun:
  pass=true, loss 1.00082 -> 0.90042, zero overflow,
  mean step 181.1ms, render forward 92.3ms, loss prep 9.5ms,
  backward 53.8ms
```

This completes the real cached V-JEPA target smoke gate.

Real V-JEPA 512px scale gate:

```text
config:
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_chunkedtarget_lr005_5step.jsonc
target:
  VideoFeatureCache extractor=vjepa_torchhub, model=vjepa2_1_vit_base_384,
  layer=vjepa_tokens, source_shape=[1,8192,768],
  channel_adapted_source_shape=[32,32,16,16],
  token_grid_shape=[32,16,16], adapted_shape=[64,32,512,512],
  channel_adapter=truncate_or_pad, temporal_spatial_adapter=trilinear,
  normalization=channel_standardize,
  channel_adapter_applied_before_grid=true, materialization=chunked
loss:
  feature_target_loss=mse, feature_target_loss_weight=1.0, rgb_loss_weight=0.0
cache-hit rerun:
  pass=true, loss 1.000014 -> 0.999545, zero overflow,
  mean step 3743.3ms, render forward 815.6ms, target chunk/loss 1734.2ms,
  backward 1077.4ms
```

Cached-chunks target-layout follow-up:

```text
config:
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_cachedchunks_lr005_5step.jsonc
target:
  same V-JEPA source and logical adapted shape as above,
  materialization=cached_chunks, cached_chunk_count=32,
  cached_target_mib=2048.0, feature_target_load_ms=2043.8
cache-hit rerun:
  pass=true, loss 1.000014 -> 0.999545, zero overflow,
  mean step 1654.7ms, render forward 600.8ms, target chunk/loss 202.5ms,
  backward 769.6ms
```

Target-grid loss follow-up:

```text
config:
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_5step.jsonc
target:
  same V-JEPA source as above,
  materialization=target_grid, target_grid_shape=[32,32,16,16],
  target_grid_mib=1.0, feature_target_load_ms=138.4
loss:
  rendered feature chunks are downsampled to the V-JEPA target grid before MSE;
  this changes the loss surface from dense render-grid MSE to coarse token-grid
  MSE
cache-hit rerun:
  pass=true, loss 0.999935 -> 0.999467, zero overflow,
  mean step 1350.8ms, render forward 547.7ms, target/loss 41.0ms,
  backward 705.1ms
20-step media rerun:
  pass=true, loss 0.999935 -> 0.997425, zero overflow,
  mean step 1451.2ms, render forward 629.7ms, target/loss 37.5ms,
  backward 722.1ms, target_grid_mib=1.0,
  media written, but RGB PSNR is not quality evidence because rgb_loss_weight=0
RGB-aux1 20-step probe:
  pass=true, total loss 1.338106 -> 1.332599,
  feature loss 0.999935 -> 0.997336,
  RGB loss 0.338171 -> 0.335263, RGB PSNR 4.709 -> 4.746,
  colorizer_grad_seen=true, zero overflow,
  mean step 1999.6ms, render forward 586.3ms, target/loss 51.8ms,
  backward 1113.6ms, target_grid_mib=1.0
RGB-aux10 20-step probe:
  pass=true, total loss 4.381647 -> 4.347160,
  feature loss 0.999935 -> 0.997547,
  RGB loss 0.338171 -> 0.334961, RGB PSNR 4.709 -> 4.750,
  colorizer_grad_seen=true, zero overflow,
  mean step 1996.9ms, render forward 605.8ms, target/loss 51.6ms,
  backward 1089.4ms, target_grid_mib=1.0
RGB-aux10 100-step probe:
  pass=true, total loss 4.381647 -> 4.048905,
  feature loss 0.999935 -> 0.964670,
  RGB loss 0.338171 -> 0.308424, RGB PSNR 4.709 -> 5.109,
  colorizer_grad_seen=true, zero overflow,
  mean step 1876.4ms, render forward 580.1ms, target/loss 43.1ms,
  backward 1032.9ms, target_grid_mib=1.0

RGB-warm20 -> aux10 100-step probe:
  pass=true, schedule=[rgb_warm20 feature0/rgb20 steps 0-19,
  target_grid_aux10 feature1/rgb10 steps 20-99],
  total loss 6.763425 -> 4.102212,
  feature loss 0.999868 -> 0.973557,
  RGB loss 0.338171 -> 0.312865, RGB PSNR 4.709 -> 5.046,
  colorizer_grad_seen=true, zero overflow,
  mean step 1639.1ms, render forward 548.3ms, target/loss 27.7ms,
  backward 872.5ms, target_grid_mib=1.0,
  negative versus constant aux10 despite cheaper steps

Frozen RGB-probe 100-step STAR gate:
  pass=true, total loss 1.399375 -> 1.313538,
  feature loss 0.999935 -> 0.970035,
  frozen-probe loss 0.039944 -> 0.034350,
  frozen-probe PSNR 13.985 -> 14.641,
  colorizer_grad_seen=false, zero overflow, fixedbin_eligible=true,
  mean step 1268.4ms, render forward 531.6ms, target/loss 17.2ms,
  probe loss 31.2ms, backward 630.4ms, target_grid_mib=1.0

Frozen RGB-probe 300-step STAR gate:
  pass=true, total loss 1.399375 -> 1.032446,
  feature loss 0.999935 -> 0.811652,
  frozen-probe loss 0.039944 -> 0.022079,
  frozen-probe PSNR 13.985 -> 16.560,
  colorizer_grad_seen=false, zero overflow, fixedbin_eligible=true,
  mean step 1355.1ms, render forward 551.6ms, target/loss 18.5ms,
  probe loss 37.5ms, backward 680.6ms, target_grid_mib=1.0

STAR feature overfit checkpoint/resume smoke:
  config source: src/train_configs/star_uvt_feature_testvideo_8f_64_rgbpyramid_target_gradcache_reduce_vec4_10step.jsonc
  first run: steps=2, checkpoint=/tmp/star_uvt_checkpoint_resume_smoke/first.pt,
  pass=true, zero overflow, mean step 159.2ms
  resumed run: steps=2, resume_checkpoint_steps=2, resume_optimizer_loaded=true,
  checkpoint=/tmp/star_uvt_checkpoint_resume_smoke/resume.pt, pass=true,
  zero overflow, mean step 42.8ms

Frozen RGB-probe checkpointed 300+300 STAR gate:
  300-step checkpoint/no-media: pass=true, feature loss 0.999935 -> 0.811652,
  frozen-probe PSNR 13.985 -> 16.560,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step.pt,
  mean step 1268.0ms, render 530.2ms, probe loss 31.0ms, backward 632.8ms
  resume300-from300: pass=true, resume_loaded=true, resume_optimizer_loaded=true,
  feature loss 0.810827 -> 0.655366, frozen-probe PSNR 16.576 -> 19.884,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_600step_after_resume.pt,
  mean step 1439.5ms, render 569.3ms, probe loss 41.1ms, backward 733.7ms
  probe-emphasis resume200-from600: pass=true, global steps 600 -> 800,
  objective feature=0.25/rgb_probe=40, feature loss 0.655132 -> 0.703820,
  frozen-probe PSNR 19.888 -> 21.425, zero overflow,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_800step_after_resume.pt,
  mean step 1512.4ms, render 594.5ms, probe loss 42.4ms, backward 773.8ms
  scheduled balance resume200-from800: pass=false, global steps 800 -> 1000,
  schedule 800-900 feature=1/probe=10; 900-1000 feature=0.5/probe=20,
  feature loss 0.703862 -> 0.643852, frozen-probe PSNR 21.428 -> 21.382,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_1000step_after_resume.pt,
  mean step 1308.1ms, render 543.5ms, probe loss 27.7ms, backward 667.6ms
  feature0.5/probe40 resume100-from1000: pass=true, global steps 1000 -> 1100,
  objective feature=0.5/rgb_probe=40, feature loss 0.643823 -> 0.656728,
  frozen-probe PSNR 21.384 -> 21.789, zero overflow,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_1100step_after_resume.pt,
  mean step 1461.3ms, render 571.5ms, probe loss 37.3ms, backward 766.2ms
  recover schedule resume100-from1100: pass=false, global steps 1100 -> 1200,
  schedule 1100-1150 feature=1/probe=20; 1150-1200 feature=0.75/probe=30,
  feature loss 0.656765 -> 0.635093, frozen-probe PSNR 21.792 -> 21.738,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_1200step_after_resume.pt,
  mean step 1520.9ms, render 588.3ms, probe loss 41.4ms, backward 795.4ms
  feature0.75/probe40 resume50-from1200: pass=true, global steps 1200 -> 1250,
  objective feature=0.75/rgb_probe=40, feature loss 0.635066 -> 0.638799,
  frozen-probe PSNR 21.740 -> 21.929, zero overflow,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_1250step_after_resume.pt,
  mean step 1523.5ms, render 580.7ms, probe loss 41.4ms, backward 807.1ms
  feature1/probe40 resume50-from1250: pass=true, global steps 1250 -> 1300,
  objective feature=1.0/rgb_probe=40, feature loss 0.638803 -> 0.632192,
  frozen-probe PSNR 21.933 -> 21.963, zero overflow,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1300step_after_resume.pt,
  mean step 1285.0ms, render 538.3ms, probe loss 18.0ms, backward 677.6ms
  feature1/probe40 resume100-from1300: pass=true, global steps 1300 -> 1400,
  objective feature=1.0/rgb_probe=40, feature loss 0.632124 -> 0.627129,
  frozen-probe PSNR 21.965 -> 21.979, zero overflow,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1400step_after_resume.pt,
  mean step 1690.2ms, render 616.7ms, probe loss 48.9ms, backward 909.6ms
  feature1/probe40 effective-lr001 resume100-from1300: pass=true, global steps 1300 -> 1400,
  optimizer loaded/effective LRs [0.005] -> [0.001], feature loss 0.632124 -> 0.630549,
  frozen-probe PSNR 21.965 -> 22.034, zero overflow,
  checkpoint=outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_1400step_after_resume.pt,
  mean step 1463.8ms, render 571.8ms, probe loss 34.3ms, backward 778.4ms
  feature1/probe40 lr001-to-lr00025 resume100-from1300: pass=true, global steps 1300 -> 1400,
  optimizer schedule 0.001 until 1375 then 0.00025 until 1400, feature loss 0.632124 -> 0.630803,
  frozen-probe PSNR 21.965 -> 22.027, zero overflow,
  no checkpoint/media, mean step 1506.9ms, render 576.3ms, probe loss 36.8ms, backward 807.2ms
```

The first 512px attempt failed before the renderer with a 48 GiB trilinear
interpolation temporary because the adapter upsampled 768 V-JEPA channels and
only then truncated to F32. Channel adaptation now runs before dense grid
upsampling, which preserves channel-independent interpolation. Chunked
materialization avoids keeping the full dense target resident, but it rebuilds
the adapted target every step. `cached_chunks` proves that repeated target
interpolation was the major removable bucket, at the cost of a resident 2GiB
target cache. `target_grid` proves the lower-memory route and is faster, but it
is a different objective. The 20-step media row only proves feature-loss
overfit plus media plumbing because RGB loss is disabled. The RGB-aux probes
are the first visual controls, but aux10 barely beats aux1 on RGB and slightly
hurts feature loss at 20 steps. The 100-step aux10 row shows schedule length
matters, but it remains far below RGB STAR. The matched RGB-warm20 schedule is
faster but worse than constant aux10, so skipping feature loss early is not the
bridge. The frozen feature-to-RGB probe row is now wired into STAR and its
100-step follow-up moves probe PSNR to `14.641` at `1.268s/step`; the 300-step
extension reaches `16.560` at `1.355s/step`, but still trails the standalone
full-video number. Checkpoint/resume now removes the restart tax for longer
probes, and the resumed 300-step continuation reaches `19.884` at `1.440s/step`.
The probe-emphasis 600->800 row reaches `21.425` at `1.512s/step`, which passes
the standalone full-video upsample number, but it also drifts feature loss from
`0.655132` to `0.703820`. The scheduled 800->1000 balance row recovers feature
loss to `0.643852` but gives back a little probe quality (`21.428 -> 21.382`).
The feature0.5/probe40 1000->1100 row is passing and raises probe PSNR to
`21.789`, but drifts feature loss back to `0.656728`.
The 1100->1200 recover schedule lowers feature loss to `0.635093`, but gives
back a little probe PSNR to `21.738`.
The 1200->1250 feature0.75/probe40 row restores probe PSNR to `21.929`, but
raises feature loss to `0.638799`.
The 1250->1300 feature1/probe40 row improves both feature loss and probe PSNR,
ending at `0.632192` and `21.963`, with zero overflow.
The 1300->1400 feature1/probe40 extension keeps both improving, ending at
`0.627129` and `21.979`, with zero overflow, but is slower at `1.690s/step` on
the older dense target-grid path. The matched timing repeat is `1.711s/step`
with the same zero-overflow `68/45/128` max/p95/cap tile count. The
sparse-forward batched-VJP helper preserves that 100-step objective movement at
`0.400s/step`
mean and `0.263s/step` last-20, with valid but still-blurry probe media.
The whole-graph profile shows renderer backward is `81.3-81.4%` of manual
backward for the current target-grid/frozen-probe objective, but the isolated
manual split does not reproduce the trainer slowdown (`1565.9ms` at step 1250
vs `1504.0ms` at step 1300).
The end-to-end trainer trace adds per-step samples and does reproduce the
slowdown after dropping first-step optimizer/warmup (`1850.7ms` from step 1300
vs `1705.3ms` from step 1250), with a late objective spike at global step
`1318`. The chunk trace says this is distributed, not a single bad chunk:
`27/32` chunks worsen, frames `0-15` contribute `44.5%` of the weighted-loss
jump, and the elevated loss persists at `1319`.
The chain/profile/trace reports are
`outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md` and
`outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`
and
`outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`
and
`outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`.
The optimizer/LR checkpoint gate confirms the spike can be avoided without
changing the shader: after fixing the trainer to re-apply config LR after
optimizer state load, retained optimizer moments with effective `lr=0.001`
record loaded/effective LRs `[0.005] -> [0.001]`, pass with end loss
`0.884576`, feature loss `0.631648`, probe PSNR `21.991`, and no-first
`1384.4ms/step` / `748.9ms` backward. Resetting optimizer state at `lr=0.001`
also passes (`0.884902`, `0.631614`, `21.984`) but is slower here. Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`.
The 100-step effective-lr001 continuation passes but is mixed: it improves probe
PSNR and speed while lr005 keeps better feature/weighted loss. The matched
effective-lr001 sparse-forward rerun preserves that endpoint at `0.372s/step`
mean and `0.159s` backward, so the next bridge is no longer simple caching or
trace plumbing. It is either an LR schedule/checkpoint-selection gate while
closing the same-grid `23.401` oracle, or native-VJP/dataset-scale work before
larger 512px or 300-set runs.

The first checkpoint-selection gate is now done. Continuing from the lr005-sparse
1400 checkpoint for 50 effective-lr001 steps passes and improves feature loss to
`0.625976` plus probe PSNR to `22.010` at `262.7ms/step` mean. Continuing from
the lr001-sparse 1400 checkpoint under the same settings fails after a
`1444 -> 1445` jump and ends at feature loss `0.631770` / probe PSNR `21.843`.
The selected quality/media follow-up from the lr005-sparse 1450 checkpoint also
passes: 50 more effective-lr001 steps reach feature loss `0.625428`, probe PSNR
`22.027`, mean `315.8ms/step`, last-20 `254.0ms/step`, and zero overflow while
writing probe media/checkpoint. The contact sheet is still blurry, so this is a
stability/continuation pass rather than a visual-quality promotion. Continue
from the lr005-sparse 1500 checkpoint if staying on this lineage.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr005sparse_1450_to1500_media.md`.

The full-resolution autograd RGB-aux probe-init bridge from that sparse 1500
checkpoint is also complete and is a negative quality result. It loads the STAR
model, intentionally skips checkpoint colorizer/optimizer state, initializes the
trainable hidden64 colorizer from the target-grid RGB probe, and runs 20 RGB-aux
steps. RGB loss improves (`0.272626 -> 0.259968`; RGB PSNR
`5.644 -> 5.851`), but feature loss worsens (`0.625418 -> 0.626799`),
frozen-probe PSNR drops (`22.028 -> 21.879`), trainable-colorizer media
artifacts appear, and mean step time is `5206.6ms` (`16.5x` slower than sparse
1500). Do not promote this bridge; the next quality bridge needs to train on the
rendered feature-image distribution or use a native visual/probe VJP that keeps
the sparse-forward speed surface.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_autograd_rgbaux_probeinit_from1500_negative.md`.

The rendered-feature sparse-pixel RGB probe from the same sparse 1500 checkpoint
is now done as the distribution-matched follow-up. It freezes STAR, samples the
target-grid source lattice from actual rendered feature pixels (`65,536`
pixels/step, `0.390625%` dense), and trains only hidden64 `FeatureToColor`.
The sampled gate passes quickly (`0.168261 -> 0.099014`, sparse PSNR
`7.740 -> 10.043`, `241.4ms/step`), but dense full-video PSNR is only `6.096`
and the media remains sparse-streaked. This is diagnostic, not a quality bridge:
the rendered feature/alpha field itself is not a clean full-res visual basis for
a per-pixel colorizer under this sparse RGB supervision.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels.md`.

The denser stratified64 rendered-feature RGB probe is now done as the sampling
bias follow-up. It freezes the same STAR checkpoint, samples a deterministic
full-resolution `64x64` lattice on every frame (`262,144` pixels/step,
`1.5625%` dense, `4x` the prior rendered-feature probe), and trains only the
hidden64 colorizer. The sampled gate passes (`0.277860 -> 0.242981`,
stratified PSNR `5.562 -> 6.144`) at `331.5ms/step`, but dense full-video PSNR
is still only `6.132` and media remains sparse-streaked. This rules out
target-grid sampling bias as the explanation. The next quality bridge should
move visual/probe loss into STAR feature optimization with sparse/native VJP,
not continue training only a downstream colorizer on frozen rendered features.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64.md`.

The first native sparse visual VJP gate is also done. It uses the same
stratified64 full-resolution lattice, freezes the trained target-grid
FeatureToColor probe, computes local sparse RGB gradients with autograd, then
pushes those gradients into STAR with
`direct_atomic_feature_sparse_pixels_backward_cached_bins`. This proves the
missing bridge (`model_grad_seen=true`, `colorizer_grad_seen=false`) and runs at
`336.8ms/step`, with mean local/native backward `55.9/94.8ms`. It is quality
negative: sampled PSNR only moves `5.656 -> 5.779` and dense full-video PSNR is
`5.739`, worse than the colorizer-only stratified gate. Next quality work should
train STAR+colorizer jointly on sparse visual pixels or combine this native
visual VJP with the target-grid feature/probe objective.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe.md`.

The joint sparse visual VJP follow-up is now done. It keeps the same sparse
visual VJP path and stratified64 lattice but trains both STAR and hidden64
`FeatureToColor` from the target-grid probe initialization. The sampled loss
falls `0.271902 -> 0.247613`, sparse sample PSNR moves `5.656 -> 6.062`, and
dense full-video PSNR improves from the frozen sparse-VJP gate's `5.739` to
`6.025`. Both gradient paths are live (`model_grad_seen=true`,
`colorizer_grad_seen=true`), but the row costs `729.4ms/step` with
`365.9ms` backward and still trails the colorizer-only stratified diagnostic
(`6.132` full-video PSNR at `331.5ms/step`). This proves the joint gradient
path, not visual quality.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe.md`.

The mixed target-grid/probe plus sparse visual VJP follow-up is now done too. It
keeps the selected target-grid feature/probe objective live, adds the same
stratified64 sparse visual RGB VJP path, and trains both STAR and hidden64
`FeatureToColor`. Feature loss moves `0.625418 -> 0.625363`, probe PSNR moves
`22.028 -> 22.045`, and sparse visual sample PSNR moves `5.656 -> 6.036`, but
dense full-video PSNR remains `6.024` while step time rises to `964.0ms`.
That rules out the simple mix as the quality fix; the next quality gate should
change the visual support/basis rather than remix sparse RGB with the
target-grid objective again.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500.md`.

The first support-basis follow-up is also done and is negative. It keeps the
same `262,144` sparse visual pixels/step but samples contiguous `2x2` patches on
a `32x32` grid (`pixel_source=stratified_patch_grid`, `patch_shape=[2,2]`).
Sparse visual sample PSNR improves to `6.179` and mean step is faster
(`619.5ms`), but feature-target loss worsens `0.625418 -> 0.625532` and dense
full-video PSNR drops to `6.000`. Rearranging sparse RGB support is not enough;
the next gate should use a denser visual basis, such as downsampled dense
support or compact visibility/prefix tape.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500.md`.

The first denser visual-basis follow-up is now done too. It adds
`sparse_visual.loss_basis=patch_mean`, samples `1,048,576` sparse visual
pixels/step from a `64x64` patch grid, and pools them into `262,144` local-mean
visual cells before the RGB loss. This restores the selected token/probe motion:
feature loss improves `0.625418 -> 0.625345`, probe PSNR moves
`22.028 -> 22.045`, sparse visual PSNR moves `5.659 -> 6.043`, and dense
full-video PSNR returns to `6.023`. But the run costs `1124.6ms/step`, with
`446.5ms` in sparse visual loss construction, and media still shows sparse
high-frequency structure. Patch-mean pooling is therefore a mechanics-positive
gate, not a quality promotion. The next visual fork should carry denser visual
gradients more directly, via compact dense visual gradient support or a
visibility/prefix tape, instead of another sparse support pattern.
Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patchmean64_from1500.md`.

The target-area64 follow-up keeps the same `1,048,576` sparse rendered visual
pixels and `262,144` loss cells, but changes the target to true
area-downsampled dense RGB cells. It passes and is slightly faster than
patch-mean64 (`1103.1ms/step` vs `1124.6ms/step`) with slightly better sparse
visual PSNR (`6.064` vs `6.043`), but dense full RGB PSNR remains `6.023` and
media is unchanged. This rules out selected-patch target bias as the main
visual-quality blocker.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500.md`.

The phased target-area64 follow-up adds
`sparse_visual.pixel_source=stratified_patch_grid_phase` and records a
`patch_phase_shape=[4,4]` schedule. It keeps the same `1,048,576` sparse visual
pixels/step and `262,144` loss cells, but cycles the compact `2x2` patch through
all non-overlapping positions inside each `8x8` target-area cell over 16 steps.
It passes and raises sparse visual PSNR to `6.077`, but dense full RGB PSNR
falls to `6.019` at `1169.2ms/step`. Fixed support position is not the visual
quality blocker either.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500.md`.

The full-cell8 target-area follow-up sends gradients through every pixel in
each `8x8` target-area cell (`16,777,216` visual pixels/step, `262,144` loss
cells). It is nonpassing: sparse visual PSNR rises to `5.822`, but weighted
loss worsens, feature loss worsens, probe PSNR falls to `21.860`, dense full
RGB PSNR falls to `5.722`, and mean step is `7526.7ms` with `5702.6ms` in sparse
visual loss construction. This rejects Python-side full dense support as the
port-forward path; the next move is a fused visibility/prefix tape or fused
RGB/loss/gradient route.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500.md`.

The manual hidden64 VJP follow-up adds
`sparse_visual.loss_vjp_mode=manual_hidden64` for the same full-cell8 support.
It matches the autograd endpoint while cutting sparse visual loss construction
from `5702.6ms` to `3803.6ms` and mean step from `7526.7ms` to `6414.0ms`.
The row is still nonpassing and still lands at `5.722` dense full RGB PSNR, so
this is a parity scaffold for a native fused loss/visibility path, not a
promotion.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500.md`.

The star-only manual hidden64 follow-up adds
`sparse_visual.loss_vjp_mode=manual_hidden64_star_only`, which keeps the same
feature/alpha VJP but skips colorizer parameter gradients. It cuts the row
further to `5801.7ms/step` and `3405.1ms` sparse visual loss construction, but
dense full RGB PSNR drops to `5.648`; this is a lower-bound diagnostic, not a
promotion.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500.md`.

The fast-GELU derivative follow-up adds
`sparse_visual.loss_vjp_mode=manual_hidden64_fastgelu`, keeping colorizer
gradients but using the derivative of `x * sigmoid(1.702x)` for the hidden64
VJP. It is rejected: mean step is `6252.1ms`, sparse visual loss construction
is `3416.7ms`, dense full RGB stays at `5.722`, and the full-step profile has a
worse loss-side total than exact manual hidden64.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_fastgelu_from1500.md`.

The compact manual-linear VJP follow-up adds
`sparse_visual.loss_vjp_mode=manual_linear` with a no-hidden-layer
`FeatureToColor` checkpoint. The standalone linear target-grid RGB probe passes
at `3.21ms/step`, but reaches only `16.980` full-video PSNR, well below the
hidden64 oracle. The full-cell8 trainer gate passes mechanically and cuts the
row to `2064.4ms/step`, `1230.2ms` backward, and `383.3ms` sparse visual loss
construction; dense RGB is still only `5.668`, and feature loss slightly
worsens. This is a useful lower-complexity VJP diagnostic, not a quality route.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500.md`.

The hidden32 manual VJP follow-up adds the generic
`sparse_visual.loss_vjp_mode=manual_hidden` alias and uses
`colorize.hidden_dim=32`. The standalone hidden32 target-grid RGB probe reaches
`21.520` grid PSNR and `19.704` full-video PSNR at `2.288ms/step`, keeping most
of hidden64's `20.073` full-video oracle. In the full-cell8 trainer it is still
only a mechanics/Pareto diagnostic: weighted loss improves `1.143623 ->
1.140212`, but feature loss slightly worsens `0.625418 -> 0.625438`, dense
RGB remains only `5.678`, and mean step/backward/sparse visual loss
construction are `4298.4/3210.5/2136.1ms`. The subphase profile shows a
`3043.6ms` full-step loss VJP extrapolation, led by GELU backward
(`725.6ms`), fc1 (`666.5ms`), fc2 (`386.2ms`), and conv1 param grad
(`310.6ms`). This rejects shrinking the hidden decoder in Python as the next
route; the native/fused boundary still needs to avoid materializing dense
hidden/colorizer intermediates.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500.md`.

The matched native handoff rerun fills the scale gap between early 256px
direct-kernel handoff evidence and the current `64f/512px/8192t/F32` V-JEPA
route. All rows pass tiny F4/F32 parity and zero-overflow timing. At this
scale, `fused_first3_sigmoid_mse` is still fast (`494.09ms` backward,
`1152.58ms` total), `linear_sigmoid_mse` is rejected as a generalized in-kernel
path (`918.09ms` backward versus `522.02ms` gradcache), and
`logit_handoff_reduce_vec4` has the best native backward (`386.26ms`) but pays
`421.89ms` of Torch image-space prep. The next native gate should fuse
logit/RGB/loss prep or remove dense prep, not only call the current logit
handoff.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_handoff_matched_512_gate.md`.

The split manual-VJP subphase profiles cover all `32` full-cell8 chunks for
both joint and star-only modes. Target-area reduction is not the big cost
(`~0.12-0.13s` full-step extrapolated), and colorizer parameter accumulation is
not enough to explain the row. The largest loss-side phases are exact GELU
backward (`~1.34-1.44s`) plus the first hidden-layer matmul (`~0.75-0.89s`).
The hidden32 follow-up lowers the hidden-layer cost but not enough to make dense
Python-side full-cell8 visual supervision viable. The next native gate should
fuse or simplify hidden RGB/loss VJP into the sparse-pixel/visibility boundary,
avoiding Python-side dense hidden tensors and `grad_feature_values`
materialization.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_split_fullstep.md`
and
`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_staronly_subphase_profile_fullstep.md`.

The first fused hidden sigmoid-MSE native gate is now a correctness scaffold, not
a speed promotion. It passes F4/F32 parity with max STAR-gradient errors below
`3.8e-08`, but H32 scalar is `317.54/610.90/2549.39ms` total at
128/256/512px, H64 256px is `817.27ms`, and vec4 reduce is slower than scalar.
This confirms that dense per-pixel hidden work inside STAR traversal is still
too expensive; the next native design should reduce dense support through
compact visual gradients or a visibility/prefix tape.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_hidden_sigmoid_mse_native_gate.md`.

The sparse hidden sigmoid-MSE cached-bin gate is the first positive native port
of that lesson. It computes hidden RGB/loss VJP only over selected sparse pixels
and reuses cached tile bins. Tiny F4/F32 parity passes for H32/H64. At
`64f/512px/8192t/F32`, H32 sparse64 drops `29.66 -> 18.47ms` total, H32
sparse128 drops `111.17 -> 64.17ms`, and H64 sparse64 drops `45.09 -> 28.40ms`;
all timing rows have zero overflow. The 128px sparse64 row loses
(`100.14 -> 105.27ms`) because the same selected-pixel count and 8192 tubes are
packed into fewer tiles, so occupancy still matters. This is a sparse visual
boundary candidate, not dense full-frame hidden parity or a trainer promotion.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_sigmoid_mse_native_gate.md`.

The first trainer-wired version of that sparse native path is now done too, and
it is deliberately narrower than the full-cell visual basis. It adds
`sparse_visual.loss_vjp_mode=native_hidden64_star_only`, plumbs full-step loss
normalization into the native kernel for chunked training, and compares against
the manual hidden64 star-only pixel64 path on the same 64f/512px/8192t sparse
1500 checkpoint. The endpoint is correct (`3.26e-08` final sparse-loss
difference), but timing is neutral: warm sparse loss+backward is `113.25ms`
manual versus `116.27ms` native, and warm step time is `405.97ms` versus
`403.83ms`. This means pixel64 trainer promotion is not justified; the next
native port should target the actually expensive full-cell/target-area hidden
VJP with compact visual gradients or visibility/prefix tape.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_nativehidden_trainer_gate.md`.

The native target-area full-cell follow-up now covers the expensive visual basis
directly. It adds bin-only `bin_uvt_feature_tubes`, cached-bin native hidden64
target-area forward sums, and native STAR backward from target-area cell
gradients under `sparse_visual.loss_vjp_mode=native_hidden64_target_area_star_only`.
Tiny F32/H64 parity passes. Synthetic full-support timing loses at
`8f/64px/1024t` (`28.12ms` native versus `25.74ms` baseline), wins at
`64f/128px/8192t` (`386.55ms` versus `509.50ms`) and
`64f/256px/8192t` (`620.70ms` versus `1405.69ms`), and survives
`64f/512px/8192t` native-only at `1874.41ms` where the all-at-once Torch hidden
VJP baseline OOMs. The first-class trainer row cuts the matched manual
star-only full-cell8 path `5801.7 -> 3496.0ms/step` and last step
`6007.7 -> 3206.3ms`, with zero overflow and the same `5.648` dense RGB
endpoint. This is now the full-support visual-VJP speed/memory baseline, not a
quality promotion.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_native_gate.md`.

The hidden32 native target-area follow-up adds the generic
`native_hidden_target_area_star_only` alias and checks the first recompute
reduction candidate. Kernel parity passes, and native-only 512px total drops
from hidden64's `1874.41ms` to `1331.00ms`. The trainer row is much faster than
hidden64 native (`3496.0 -> 2464.6ms/step`) and manual hidden32
(`4298.4 -> 2464.6ms/step`), but it is rejected because it changes the endpoint:
`pass=false`, feature target loss ends worse, sparse visual loss ends worse,
probe PSNR is only `19.481`, and dense full RGB is `5.632`. Keep hidden64 native
target-area as the baseline; shrinking decoder capacity is not the recompute fix.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden32_gate.md`.

The feature/geometry split follow-up adds a benchmark-only
`target_area_feature_grad_only` mode and compares it against full star-only plus
`target_area_skip_feature_grad`. Tiny parity passes for the relevant gradient
subsets. At 256px, full/feature-only/geometry-only backward is
`581.3/548.2/547.3ms`; at 512px it is `1919.7/2106.7/2174.0ms`. The partial
modes still pay target-area traversal and hidden64 VJP, so simple output-gradient
masking is not a speed lever for the native target-area path.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_geometrysplit_gate.md`.

The recompute-floor follow-up exposes the mode-bit combination as
`target_area_recompute_only`, disabling both feature-gradient and geometry/
opacity-gradient output writes. Tiny loss parity passes, gradients are
intentionally zero, and timing is still `571.3ms` backward at 256px and
`2101.7ms` at 512px versus full `581.3ms` and `1919.7ms`. That makes the shared
target-area replay plus hidden64 VJP envelope the native bottleneck, not output
gradient atomics.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_recompute_floor_gate.md`.

The traversal-floor follow-up exposes `target_area_traversal_only`, which also
skips hidden64 forward/VJP. Tiny loss parity passes and timing drops to
`194.9ms` backward at 256px and `742.2ms` at 512px. Compared with recompute-only,
that isolates hidden64 forward/VJP at roughly `376.5ms` and `1359.6ms`.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_traversal_floor_gate.md`.

The hidden-forward/backward split follow-up exposes
`target_area_hidden_forward_only`, keeping hidden64 forward/logits/sigmoid while
skipping hidden backprop. Backward timing is `345.5ms` at 256px and `1192.8ms`
at 512px, splitting the hidden slice into forward `150.6/450.6ms` and backward
`225.8/909.0ms`. The hidden backward W^T/GELU feature VJP is the larger
subtarget.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_forward_backward_split_gate.md`.

The hidden-preact/W^T split follow-up exposes `target_area_hidden_preact_only`,
which computes output-weight backprop plus GELU derivative but skips
`hidden_weight^T @ grad_hidden_pre`. Backward timing is `400.3ms` at 256px and
`1254.4ms` at 512px. That splits hidden backward into output+GELU
`54.8/61.7ms` and F32 W^T feature-gradient reconstruction `171.0/847.3ms`.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_preact_wt_split_gate.md`.

The row-major W^T follow-up exposes an exact full-gradient mode
`target_area_star_only_rowmajor_wt` and a recompute-floor mode
`target_area_recompute_only_rowmajor_wt`. Full-gradient parity passes, but the
full trainable path slows: canonical vs row-major backward is
`647.4 -> 711.5ms` at 256px and `2040.5 -> 2161.6ms` at 512px. The isolated
recompute-only floor improves only slightly (`572.1 -> 555.8ms` and
`1993.0 -> 1983.4ms`). Reject simple W^T loop reordering; the next gate has to
reduce/avoid dense F32 W^T reconstruction or change objective/support.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_rowmajor_wt_gate.md`.

The vec4 W^T follow-up exposes exact `target_area_star_only_vec4_wt` and
benchmark-floor `target_area_recompute_only_vec4_wt`. Full-gradient F4/F32
parity passes. Same-build full backward improves `675.9 -> 642.2ms` at 256px
and `2408.1 -> 1804.7ms` at 512px, with a 512px repeat at `1832.8ms`;
recompute-only improves `586.6 -> 518.3ms` and `2305.2 -> 1635.8ms`. The
trainer opt-in `native_hidden64_target_area_star_only_vec4_wt` passes with the
same endpoint class. The current-build trainer A/B then promotes it as the
preferred full-support native target-area star-only mode: canonical vs vec4
mean step is `4262.1 -> 4071.0ms`, mean backward `3700.2 -> 3152.6ms`, and
mean sparse visual backward `2546.7 -> 1963.5ms`, with matched full RGB
`5.648`. The 50-step promoted-mode gate passes and warms to `3359.2ms` mean /
`3072.1ms` last step with full RGB `5.732` and zero overflow, but it still
trails the compact target-area64 helper route on the fresh current-build gate
(`930.6ms`, `6.023` full RGB). The compact native star-only diagnostic passes
mechanically but is rejected: it freezes colorizer gradients and costs
`2265.0ms` mean step, slower than compact autograd. The compact manual-hidden64
colorizer-gradient diagnostic is rejected too: it records colorizer grad
required/seen `true/true`, but costs `2007.4ms` mean / `1899.2ms` no-first
step, worsens feature/probe quality, and trails compact autograd's first-five
`991.9ms` mean / `787.7ms` no-first comparison. The native
colorizer-gradient vec4 W^T implementation closes the missing returned-gradient
ABI and passes tiny parity for STAR plus hidden/output colorizer parameters,
but the compact trainer gate rejects it too: `2738.7ms` mean step,
`1474.2ms` backward, colorizer grads seen, zero overflow, and the same
feature/probe regression.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_gate.md`.
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_trainer_ab_gate.md`.
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_50step_gate.md`.
`outputs/benchmarks/2026-05-20_star_uvt_compact_target_area_visual_route_gate.md`.
`outputs/benchmarks/2026-05-20_star_uvt_compact_native_staronly_diagnostic.md`.
`outputs/benchmarks/2026-05-20_star_uvt_compact_visual_vjp_gate.md`.
The same-pass SIMD-reduced colorizer follow-up fixes the direct native atomic
envelope (`297.2ms` compact native total versus `312.1ms` sparse-pixel baseline
in the matched direct run), but is still rejected by the 5-step trainer
(`2908.9ms` mean step, `604.0ms` sparse visual backward, same feature/probe
regression). Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_simdreduce_gate.md`.
The fresh dynamic-gsplat fixed-512 smoke closes the immediate matched-comparator
gap at smoke level: `64f/512px/8192` active Gaussians records step-5
`8.019s` total / `5.638s` backward, with raster only `0.362s`. It is not a
quality baseline, but it is enough to keep the fast-route plan on STAR UVT
visual quality rather than dynamic-gsplat promotion. Report:
`outputs/benchmarks/2026-05-20_dynamic_gsplat_512_matched_probe.md`.
The selected visual-quality gate then blocks scale-up: compact target-area
mechanics and speed pass, but dense full RGB is only `6.023` PSNR, the media
stays sparse/streaked or blurry, and RGB STAR reaches `12.444` PSNR on the
same-clip bracket. Report:
`outputs/benchmarks/2026-05-20_star_uvt_selected_visual_quality_gate.md`.
The next bridge gate added a trainable low-frequency RGB-grid loss on the fast
target-grid sparse VJP path. It works mechanically and is cheap after warmup
(`353.1ms` mean step, `289.9ms` no-first; colorizer gradients seen), but it is a
negative visual result: RGB-grid PSNR improves `22.028 -> 22.248` while feature
loss worsens `0.625418 -> 0.630230` and dense full RGB falls to `5.657` PSNR.
Report:
`outputs/benchmarks/2026-05-20_star_uvt_rgb_grid_lowfreq_bridge_gate.md`.
The combined compact target-area plus RGB-grid gate is also negative:
grid/probe/sparse metrics improve, but the run is slower (`1.648s` mean step),
feature loss worsens to `0.630296`, and dense full RGB is only `5.720` PSNR.
Report:
`outputs/benchmarks/2026-05-20_star_uvt_compact_rgbgrid40_visual_bridge_gate.md`.
The dense alpha diagnostic localizes these visual failures to
coverage/visibility/composition rather than only colorizer content: forced alpha
raises compact/rgbgrid/compact_rgbgrid to `11.450/14.548/14.616` PSNR and
target-background oracle composition reaches `20.149/25.562/25.505`, while
alpha `>0.1` covers only `43.5/41.5/43.1%` of pixels. Report:
`outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_failure_diagnostic.md`.
The direct sampled alpha-to-one follow-up is also negative: sampled alpha loss
improves `0.752440 -> 0.738210`, but feature loss worsens `0.625418 ->
0.627071`, RGB-probe PSNR drops `22.028 -> 21.900`, dense full RGB stays
`6.018`, and dense alpha `>0.1` stays `43.1%`. Reports:
`outputs/benchmarks/2026-05-20_star_uvt_compact_alpha1_coverage_gate.md` and
`outputs/benchmarks/2026-05-20_star_uvt_alpha1_dense_alpha_diagnostic.md`.
The phase-covered alpha follow-up is negative too: cycling `2x2` support phases
with `patch_phase_shape=[4,4]` improves sampled alpha loss `0.751768 ->
0.739891` and sparse visual PSNR `5.694 -> 6.072`, but feature loss worsens
`0.625418 -> 0.626961`, RGB-probe PSNR drops `22.028 -> 21.904`, dense full RGB
falls to `6.014`, and dense alpha `>0.1` falls to `43.0%`. Reports:
`outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_coverage_gate.md` and
`outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_dense_alpha_diagnostic.md`.
The target-aware black-hole follow-up is negative as well: `black_hole_loss`
improves `0.262537 -> 0.256889` and sparse visual PSNR improves `5.678 ->
6.059`, but feature loss worsens `0.625418 -> 0.627272`, RGB-probe PSNR drops
`22.028 -> 21.890`, dense full RGB stays `6.014`, and dense alpha `>0.1` stays
`43.0%`. Same-support target-energy alpha pressure is not the missing dense
coverage bridge. Reports:
`outputs/benchmarks/2026-05-20_star_uvt_blackhole4_coverage_gate.md` and
`outputs/benchmarks/2026-05-20_star_uvt_blackhole4_dense_alpha_diagnostic.md`.
The target-background composition gate is the first useful composition split
but still not a visual route. `sparse_visual.composition=target_background`
mechanically passes and improves feature/probe plus sparse visual PSNR, while
the dense diagnostic shows forced-alpha PSNR `14.891` and target-background
oracle `27.443`; black-background dense RGB is only `5.666` because alpha
`>0.1` falls to `40.8%`. Adding sampled alpha-to-one restores alpha `>0.1` to
`43.1%`, but feature/probe regress and dense RGB is only `5.748`. Report:
`outputs/benchmarks/2026-05-20_star_uvt_target_background_composition_gate.md`.
The alpha-sweep and patch4 support follow-up is negative. Post-render alpha
gain shows scalar opacity amplification is not enough: even `16x` gain reaches
only `8.337-8.592` PSNR on target-background checkpoints, while alpha floors
recover the forced-alpha result. The `4x4` support pilot raises sparse visual
support to `25%` and improves sparse visual PSNR `26.319 -> 27.251`, but it
fails total loss `1.631071 -> 1.637982`, worsens feature loss `0.625418 ->
0.626858`, drops frozen-probe PSNR `22.028 -> 21.878`, and ends at only
`5.698` dense RGB. Reports:
`outputs/benchmarks/2026-05-20_star_uvt_patch4_support_alpha_sweep_gate.md` and
`outputs/benchmarks/2026-05-20_star_uvt_patch4_alpha_sweep_dense_diagnostic.md`.
The raw-opacity bias follow-up is negative too. Adding logit-space opacity
bias before rasterization tests whether a simple opacity schedule can expand
useful support. Bias `+4` is best among `[-2,-1,0,1,2,3,4]`, but only reaches
`6.194/5.926/5.871` PSNR on compact/targetbg-alpha/patch4 and barely moves
alpha `>0.1` coverage (`43.5 -> 46.5%`, `43.1 -> 45.8%`, `41.5 -> 44.2%`).
Reports:
`outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_gate.md` and
`outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_dense_diagnostic.md`.
The dense alpha-only support gate rejects the most direct remaining
same-support visibility retry. The trainer now has opt-in `dense_alpha`
support, which renders dense alpha per chunk and sends `grad_alpha` through
`direct_atomic_feature_backward` with `gradcache_skip_feature_grad`. At
64f/512px/8192t from the selected sparse 1500 checkpoint, the 5-step pilot
fails strict loss decrease: weighted loss `1.271702 -> 1.284505`, dense alpha
loss `0.395507 -> 0.397107`, feature loss `0.625418 -> 0.626814`,
RGB-probe PSNR `22.028 -> 21.861`, dense full RGB `5.647`, and mean
dense-alpha render/loss/backward `834.5/124.6/858.9ms`. The diagnostic shows
better forced-alpha/oracle potential (`14.556` / `25.809` PSNR) but lower
actual alpha coverage (`40.7%` alpha `>0.1`) and raw-opacity bias still only
reaches `5.816`. Report:
`outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_support_gate.md`.

The alpha-only visibility profile answers the implementation-detail follow-up:
the current dense-alpha path should not render dense F32 features just to get
alpha. `render_uvt_feature_alpha_all_pixels_with_bins` reuses the existing
sparse-pixel Metal path with a dummy F1 feature, reshapes all-pixel alpha, and
returns cached bins for an F1 alpha-only backward. On the same 64f/512px/8192t
case across all 32 frame chunks, alpha parity is exact, geometry/opacity
gradient parity against dense cached F32 backward is within `4.7e-7`, overflow
is zero, and render+backward drops `1100.8 -> 634.6ms` (`1.735x`, with per-chunk
dense feature-image bytes `67.1MiB` versus F1 feature-values plus pixel ids
`4.2MiB`). Reports:
`outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_full.md`
and
`outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_1chunk.md`.
This is a benchmark-only speed gate. It makes future alpha-only diagnostics
cheaper, but it does not rescue the failed dense-alpha objective or unblock the
300-video scale lane.

The trainer follow-up wires the same path as
`dense_alpha.render_mode=sparse_f1`. It reproduces the dense F32 endpoint and
stays quality-negative (`1.271702 -> 1.284505` weighted loss,
`0.625418 -> 0.626814` feature loss, `22.028 -> 21.861` probe PSNR, `5.647`
dense RGB), but cuts mean step/backward `2558.6/1114.2 -> 873.3/370.0ms` and
dense-alpha render/loss/backward `834.5/124.6/858.9 -> 276.0/22.0/303.7ms`.
Report:
`outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_sparsef1_trainer_gate.md`.

The next missing implementation-detail gate is now a CPU support-changing
visibility proxy. `visibility_support_bridge_prototype.py` starts from a tiny
miss scene where target alpha `>0.10` coverage is `0.0`; same-support dense
alpha optimization still ends at `0.0`, but a soft target-pixel to projected
tube coverage proxy sends center/velocity gradients, lowers proxy loss
`45.109 -> 0.296`, raises target alpha mean `0.0 -> 0.092`, and reaches target
alpha `>0.10` coverage `0.324`. Report:
`outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.md`.
This is not a trainer or Metal promotion. It is the first positive evidence
that the next bridge needs support-changing geometry gradients rather than more
same-support alpha/grid pressure.

The trainer port now exists as a first-class `visibility_proxy` config path.
The 5-step gate from sparse step 1500 passes mechanically: weighted loss
`0.871986 -> 0.871864`, feature target loss `0.625418 -> 0.625379`,
RGB-probe PSNR `22.0277 -> 22.0291`, and visibility-proxy loss
`-4.20957 -> -4.20992`, with center/velocity gradients seen and 4096 target
points. Mean step/backward/proxy timing is `541.1/306.6/237.0ms`; last-step
step/backward/proxy is `397.5/286.1/211.6ms`. Report:
`outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_trainer_gate.md`.
This closes the trainer-plumbing gap, but not the visual-quality gap: dense
full RGB PSNR is still `5.640`, so the next experiment must prove actual
support/alpha movement or reduce the proxy overhead before scale-up.

That support check is now negative for the center-only proxy. The dense-support
gate compares the selected sparse step-1500 checkpoint, the 5-step proxy, and a
10x/20-step proxy follow-up:
`outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_support_gate.md`.
The 5-step proxy improves forced-alpha PSNR `11.722 -> 14.552` and
target-background oracle `20.140 -> 25.834`, but alpha `>0.1` falls
`41.1% -> 40.5%`. The 10x/20-step proxy improves proxy loss
`-4.20957 -> -4.21215`, but fails trainer loss `0.834100 -> 0.844115`, worsens
feature/probe losses, and only reaches `40.6%` alpha `>0.1`. This should not be
scaled as-is; the next STAR UVT support bridge needs explicit opacity/support or
support-density movement, not only center/velocity attraction to target points.

The opacity/precision support-aware trainer port closes that exact missing
implementation detail but still fails as a scale bridge:
`outputs/benchmarks/2026-05-20_star_uvt_visibility_support_proxy_gate.md`.
`visibility_proxy` now accepts `center_weight`, `support_weight`, and
`support_epsilon`; the support term estimates target-point coverage from tube
opacity and precision and focused tests prove raw opacity/precision gradients.
The 5-step support-only gate passes mechanically (`0.910498 -> 0.909964`
weighted loss, `3.4303 -> 3.3821` support proxy loss, `22.0277 -> 22.0289`
RGB-probe PSNR), but feature loss slightly worsens `0.625418 -> 0.625436` and
proxy work costs `693.7ms`/step. Dense support barely moves versus center-only:
normal PSNR `5.640 -> 5.643`, forced-alpha `14.552 -> 14.553`,
target-background oracle `25.834 -> 25.820`, and alpha `>0.1`
`40.5% -> 40.6%`. Treat this as plumbing proof, not a 300-video scale route.
The next experiment needs cheaper/fused support density, a better
opacity/support parameterization, or explicit support birth/split.

The first support birth/split CPU gate is now positive:
`outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.md`.
It keeps a fixed budget of `16` tubes, reallocates `8` dead/miss tubes onto a
fitted target support trajectory, and compares against the same-support alpha
and center-proxy controls. Same-support alpha remains at `0.0000` target alpha
`>0.10`; the center proxy reaches `0.5784`; birth/split reaches `1.0000`
immediately and keeps `1.0000` after dense-alpha refinement while background
alpha falls `0.0479 -> 0.0072`. This is not a Metal or trainer promotion, but
it changes the next port target: implement first-class dead/low-contribution
tube reallocation from uncovered target pixels, then run the sparse-1500 dense
support diagnostic again.

That trainer port now exists:
`outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_trainer_gate.md`.
`support_birth_split.enabled` samples target points, fits a screen-space
trajectory, reallocates a fixed number of existing tubes, and logs the exact
birth/split state. The 512px/64f sparse-1500 gate reallocates `32/8192`
low-opacity tubes, raises selected opacity `0.3418 -> 0.8000`, preserves zero
overflow (`100/71/128` max/p95/cap), and passes 5 steps with weighted loss
`0.910290 -> 0.909536`, feature target `0.635579 -> 0.635530`, RGB-probe loss
`0.006868 -> 0.006850`, and mean step/backward/render `189.4/55.6/70.1ms`.
It is a real support-changing trainer primitive, not a quality promotion. The
next gate should measure dense support/alpha coverage from this checkpoint and
then sweep radius/tube count; if support did not move, replace top-brightness
target sampling with uncovered/low-alpha target pixels.

The dense support diagnostic now says birth/split moved some support but not
enough:
`outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_dense_support_diagnostic.md`.
Birth32 beats center/support proxy rows on normal PSNR (`5.708`), forced-alpha
PSNR (`14.606`), alpha mean (`0.1737`), and alpha `>0.5` (`0.117`), but alpha
`>0.1` is only `0.411` and the target-background oracle falls to `25.234`
versus `25.834/25.820`. This changes the next gate from "prove dense support"
to "sweep amount/radius and fix target selection toward uncovered pixels."

Current RGB kernel boundary:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/rasterize.py
  _check_inputs requires color.shape == [N,3]
  direct backward wrappers require grad_image == [T,H,W,3]

third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal
  forward loads/stores float3 color
  direct/compact backward use float3 grad_rgb, float3 color, atomic_add3
  reducers allocate and reduce grad_color as 3 channels

third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/bindings.cpp
  registered ops expose RGB color/grad_color signatures only
```

Implication: the feature port is a real feature-specific kernel/API path, not a
one-line Python shape relaxation. The first direct feature functions are now
distinct from the RGB calls:

```text
torch_gsplat_bridge_star_uvt.feature_rasterize.render_uvt_feature_tubes
torch_gsplat_bridge_star_uvt.feature_rasterize.direct_atomic_feature_backward
```

## Shader Lessons To Port

### 1. Keep Feature Forks Separate

Do not mutate `star_uvt_v0` RGB kernels directly. Make a separate feature fork
or namespace with a hard `feature_dim` contract.

Reason:

The fast-mac history showed that stable RGB/F3 and F32 feature needs diverge.
The feature forks need different backward buffers, feature caps, and sometimes
different no-overflow assumptions.

### 2. Direct Path First

Start with direct atomic/index-add semantics for feature tubes.

Reason:

The current STAR UVT screen-space table says direct modes are the only reliable
fast rows at 64f/32768. Compact keyed sample-table paths are still too slow or
memory-sensitive. Determinism can be revisited after a trainable fast feature
path exists.

### 3. Avoid Per-Pixel Per-Feature Atomics Where Possible

The stable F32 feature path pays heavily for dense `g_colors` atomics. The
feature-fork wins came from replacing/reducing those atomics with:

```text
SIMD/threadgroup reductions
thread-local F<=32 accumulation
full-vector grad-feature caching
fixed/no-overflow binning
```

STAR UVT feature backward should not blindly emit one atomic per
pixel-feature-channel if it can accumulate the feature vector per tube/sample
or cache the pixel gradient vector once.

### 4. No-Overflow Fixedbin Is Powerful But Conditional

Fixedbin/fixed-cap removes cumsum/cat/clone/item overhead and can dominate at
512px, but it is only valid when tile occupancy stays under the runtime cap.

Observed:

```text
128px/B64/G32768/F32: fixedbin/v11 fail on tile overflow
256px/B64/G32768/F32: v11 fixedbin works and wins
512px/B64/G32768/F32: f32_fixedbin works and wins
```

Port rule:

Implement fixedbin as an opt-in fast path with an explicit overflow fallback.
Do not make it the only path.

### 5. Tile Support, Not Pixel Count Alone, Decides Speed

The 128px dynamic rows were slower than 256px because 32,768 splats packed into
fewer tiles created overflow/fallback pressure. Use support/tile-load metrics,
not only pixel count, to choose a shader path.

STAR UVT feature diagnostics should log:

```text
projected tube count
mean/max/p95 tile samples
overflow tile count
valid sample count
fixedbin eligibility
F and feature cap
```

### 6. Dense Output Tensor Still Matters

Even when shader backward is faster, the final `[T,H,W,F]` feature image and
FeatureToColor graph can dominate memory. The port needs render/loss
microbatching from the start, with parity gates against batched backward.

## Port Plan

### Gate 0: Lock The Contract

Write a minimal spec for feature UVT tensors:

```text
tube params:
  ma/q_uvt/depth/opacity as current RGB STAR UVT
  feature: [N,F], float32, initially F=32

render output:
  feature_image: [T,H,W,F]
  alpha: [T,H,W]

loss path:
  feature_image -> FeatureToColor -> RGB composition/loss
```

Acceptance:

- Dense CPU/Torch prototype matches expected shapes.
- `dense_feature_tube_prototype.py --smoke` still passes.
- One tiny overfit proves gradients reach tube features and geometry.
- Full-frame loss/backward and frame-chunked loss/backward match so
  render/loss microbatching has a parity gate before the Metal port.

Recorded result on 2026-05-18:

```text
CPU JSON:
  outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract.json
  pass: true
  shapes: feature [5,32,16,16], alpha [5,16,16], rgb [5,3,16,16]
  grad seen: raw_feature, center_uv, velocity_uv, colorizer
  chunked parity: loss diff 7.45e-09, max grad diff 3.73e-09
  tiny overfit: 0.20710 -> 0.11964 in 8 steps
  full dense timing: render 3.33ms, colorize+compose 0.44ms, backward 17.55ms

MPS JSON:
  outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json
  pass: true
  chunked parity: loss diff 3.73e-09, max grad diff 1.21e-08
  tiny overfit: 0.18395 -> 0.11739 in 8 steps
  warmed full parity timing: render 4.87ms, colorize+compose 0.82ms, backward 11.00ms
```

Timing caveat:

The dense prototype is intentionally tiny and launch-overhead dominated on MPS.
Use the timings only to prove the diagnostic split exists; the acceptance value
is the contract and gradient parity.

### Gate 1: Direct Feature Metal Kernel

Create a feature-specific STAR UVT fork or module, not a patch to RGB
`star_uvt_v0`.

Initial implementation:

```text
forward:
  render feature vector + alpha for F=32

backward:
  direct_atomic/index_add equivalent
  one path for geometry/opacity
  one path for feature gradient
```

Acceptance:

- Forward parity vs dense reference on tiny F=4/F=32.
- Backward gradient parity within MPS tolerance on tiny scenes.
- 64f/32768/256px timing row exists.

Recorded result on 2026-05-18:

```text
Tiny parity artifact:
  outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_tiny_parity.json
  F=4 forward feature err 2.98e-08, alpha err 1.19e-07
  F=4 backward max errors: feature 7.15e-07, ma 2.38e-07,
    opacity 1.19e-06, q 4.17e-07
  F=32 forward feature err 2.98e-08, alpha err 1.19e-07
  F=32 backward max errors: feature 1.43e-06, ma 7.15e-07,
    opacity 9.54e-06, q 1.91e-06

64f/32768/F32 timing:
  128px: total 259.9ms, forward 75.5ms, backward 184.4ms,
    overflow tiles 8093 -> speed row is not an accurate full-support row
  256px: total 757.9ms, forward 190.1ms, backward 567.8ms,
    overflow tiles 0 -> first usable direct feature stress row
```

Interpretation:

- The direct feature Metal path exists and passes tiny forward/backward parity.
- At 256px/64f/32768/F32 it is already far faster than projected F32 feature
  raster stable (`3642.2ms`) and faster than v11/fixedbin (`1582.2ms`) for the
  synthetic support pattern used here.
- It is still about `4.16x` slower than STAR RGB direct_atomic at 256px
  (`182.4ms`), mostly because the naive feature path emits and backprops a dense
  `[T,H,W,F]` tensor and uses per-channel feature accumulation.
- The 128px stress overflows the tile cap, matching the earlier lesson that
  support/tile load, not resolution alone, decides validity.
- 512px/64f/F32 allocates about 2GB for the feature image alone
  (`64*512*512*32*4` bytes) before gradients, so it should be run only after
  render/loss microbatching or a chunked benchmark harness is in place.

### Gate 2: Port Fast Feature Tricks

Implement candidates as isolated modes:

```text
feature_direct_atomic
feature_direct_gradcache
feature_direct_accum_f32
feature_direct_fixedbin
feature_direct_colorize_vjp_or_handoff
```

Each mode must report fixedbin eligibility and overflow counts.

Acceptance:

- `256px/64f/32768/F32` beats stable direct feature baseline.
- `512px/64f/32768/F32` does not regress into tens-of-seconds backward.
- Overflow path is explicit at 128px stress instead of crashing silently.
- First-class timing separates image-space colorizer/loss backward from Metal
  renderer backward, so the selected fast path is not chosen from synthetic
  renderer timing alone.

### Gate 3: Trainer Integration

Wire the feature STAR UVT renderer into the first-class trainer path with a
small config, not only a benchmark harness.

Acceptance:

- 1-step smoke exercises train and validation.
- 20-step single-video overfit is finite.
- Direct RGB STAR UVT remains unchanged.

Current partial result on 2026-05-18:

```text
Autograd wrapper:
  torch_gsplat_bridge_star_uvt.feature_rasterize.render_uvt_feature_tubes_autograd
  returns feature_image [T,F,H,W] and alpha [T,H,W]
  backward routes FeatureToColor/loss gradients through direct_atomic_feature_backward
  depth0/depth_beta remain order-only/non-differentiated, matching current RGB direct-backward scope

Synthetic FeatureToColor overfit:
  outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32.json
  pass: true
  autograd-vs-manual max errors: feature 2.98e-07, ma 0, opacity 0, q 1.79e-07
  4f/32px/64t/F32: loss 0.22818 -> 0.10965 in 12 steps
  mean step 24.45ms, last step 5.18ms, overflow 0

Real-video mini overfit:
  outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_video_overfit_8f_64px_512t_f32_20step.json
  source: test_data/test_video_small_128_4fps.mp4
  pass: true
  autograd-vs-manual max errors: feature 1.19e-07, ma 1.86e-08,
    opacity 5.96e-08, q 1.79e-07
  8f/64px/512t/F32: loss 0.18671 -> 0.04197 in 20 steps
  PSNR 7.29 -> 13.77
  mean step 24.70ms, last step 14.82ms, overflow 0

Frame-chunked autograd parity:
  outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32_chunkparity.json
  pass: true
  chunk size 2 vs full autograd max errors: feature 8.35e-07,
    ma 1.64e-07, opacity 2.24e-07, q 2.98e-07

First-class trainer/config path:
  src/train/train.py arch=star_uvt_feature_overfit
  src/train/train_star_uvt_feature_overfit.py
  src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc

First-class full-frame smoke:
  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_20step.json
  pass: true
  8f/64px/512t/F32: loss 0.18602 -> 0.04167 in 20 steps
  PSNR 7.30 -> 13.80
  mean step 43.32ms, last step 22.29ms, overflow 0

First-class frame-chunked smoke:
  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_20step.json
  pass: true
  frame_chunk_size: 2
  8f/64px/512t/F32: loss 0.18602 -> 0.04167 in 20 steps
  PSNR 7.30 -> 13.80
  mean step 76.79ms, last step 59.51ms, overflow 0
  tile max 76, p95 74, fixedbin eligible true
  media render 15.23ms
  contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_contact.png
  side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_side_by_side.mp4

First-class scale probes:
  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_8192t_f32_chunk4_3step.json
  pass: true
  source: test_data/test_video_384_128_6fps.mp4
  64f/256px/8192t/F32, frame_chunk_size 4: loss 0.32612 -> 0.31141
  mean step 964.66ms, forward 120.89ms, colorize/loss 71.80ms,
    backward 736.03ms, optimizer 18.32ms, overflow 0
  tile max 80, p95 63, fixedbin eligible true

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_2step.json
  pass: false due tile overflow
  64f/256px/16384t/F32, frame_chunk_size 4: loss 0.32039 -> 0.30803
  mean step 1075.03ms, forward 133.41ms, colorize/loss 76.67ms,
    backward 815.62ms, overflow 736
  tile max 151, p95 123, overflow excess refs 4528, fixedbin eligible false

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_cap256_2step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/16384t/F32, cap 256, frame_chunk_size 4: loss 0.32038 -> 0.30802
  mean step 1215.41ms, forward 143.18ms, colorize/loss 92.05ms,
    backward 921.74ms, overflow 0
  tile max 152, p95 123, fixedbin eligible true

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_2step.json
  pass: false due tile overflow
  64f/256px/32768t/F32, frame_chunk_size 4: loss 0.31908 -> 0.30823
  mean step 1142.94ms, forward 142.82ms, colorize/loss 87.67ms,
    backward 863.23ms, overflow 8160
  tile max 274, p95 238, overflow excess refs 753104, fixedbin eligible false

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_cap256_2step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: false due tile overflow
  64f/256px/32768t/F32, cap 256, frame_chunk_size 4: loss 0.31651 -> 0.30624
  mean step 1325.92ms, forward 156.13ms, colorize/loss 80.68ms,
    backward 1035.58ms, overflow 216
  tile max 275, p95 238, overflow excess refs 1496, fixedbin eligible false

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_2step.json
  pass: false due tile overflow
  64f/256px/32768t/F32, alpha >= 1/64, frame_chunk_size 4:
    loss 0.32037 -> 0.31078
  mean step 1778.87ms, forward 403.40ms, colorize/loss 122.31ms,
    backward 1058.12ms, overflow 5814
  tile max 230, p95 191, overflow excess refs 317382, fixedbin eligible false

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/32768t/F32, alpha >= 1/64, cap 256, frame_chunk_size 4:
    loss 0.31921 -> 0.29350 in 20 steps
  PSNR 4.96 -> 5.32
  mean step 1159.18ms, forward 143.32ms, colorize/loss 69.41ms,
    backward 926.09ms, overflow 0
  tile max 248, p95 204, fixedbin eligible true
  media render 222.36ms
  contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step_contact.png
  side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step_side_by_side.mp4

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_2step.json
  pass: false due tile overflow
  64f/256px/32768t/F32, alpha >= 1/32, frame_chunk_size 4:
    loss 0.32264 -> 0.31420
  mean step 1309.78ms, forward 161.80ms, colorize/loss 106.00ms,
    backward 921.20ms, overflow 5538
  tile max 205, p95 168, overflow excess refs 188460, fixedbin eligible false

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_2step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/32768t/F32, alpha >= 1/32, cap 256, frame_chunk_size 4:
    loss 0.32217 -> 0.31399
  mean step 1149.60ms, forward 137.80ms, colorize/loss 77.39ms,
    backward 886.12ms, overflow 0
  tile max 206, p95 168, fixedbin eligible true

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/32768t/F32, alpha >= 1/32, cap 256, frame_chunk_size 4:
    loss 0.32217 -> 0.29861 in 20 steps
  PSNR 4.92 -> 5.25
  mean step 1174.10ms, forward 152.10ms, colorize/loss 81.21ms,
    backward 915.24ms, overflow 0
  tile max 213, p95 175, fixedbin eligible true
  media render 235.17ms
  contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step_contact.png
  side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step_side_by_side.mp4

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_2step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/32768t/F32, alpha >= 1/96, cap 256, frame_chunk_size 4:
    loss 0.31830 -> 0.30875
  mean step 1255.92ms, forward 150.76ms, colorize/loss 89.06ms,
    backward 960.53ms, overflow 0
  tile max 238, p95 204, fixedbin eligible true

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_2step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/32768t/F32, alpha >= 1/80, cap 256, frame_chunk_size 4:
    loss 0.31865 -> 0.30922
  mean step 1213.18ms, forward 144.97ms, colorize/loss 77.73ms,
    backward 941.09ms, overflow 0
  tile max 236, p95 199, fixedbin eligible true

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_2step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/32768t/F32, alpha >= 1/72, cap 256, frame_chunk_size 4:
    loss 0.31889 -> 0.30955
  mean step 1621.19ms, forward 181.78ms, colorize/loss 106.71ms,
    backward 1082.21ms, overflow 0
  tile max 232, p95 195, fixedbin eligible true

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: false due tile overflow
  64f/256px/32768t/F32, alpha >= 1/96, cap 256, frame_chunk_size 4:
    loss 0.31830 -> 0.29150 in 20 steps
  PSNR 4.97 -> 5.35
  mean step 1182.75ms, forward 147.17ms, colorize/loss 71.29ms,
    backward 944.77ms, overflow 12
  tile max 269, p95 220, fixedbin eligible false
  contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step_contact.png
  side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step_side_by_side.mp4

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: false due tile overflow
  64f/256px/32768t/F32, alpha >= 1/80, cap 256, frame_chunk_size 4:
    loss 0.31865 -> 0.29237 in 20 steps
  PSNR 4.97 -> 5.34
  mean step 1173.14ms, forward 156.56ms, colorize/loss 69.47ms,
    backward 931.32ms, overflow 6
  tile max 261, p95 213, fixedbin eligible false
  contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step_contact.png
  side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step_side_by_side.mp4

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step.json
  env: STAR_UVT_TILE_CAPACITY=256
  pass: true
  64f/256px/32768t/F32, alpha >= 1/72, cap 256, frame_chunk_size 4:
    loss 0.31889 -> 0.29290 in 20 steps
  PSNR 4.96 -> 5.33
  mean step 1320.92ms, forward 189.33ms, colorize/loss 87.14ms,
    backward 1021.20ms, overflow 0
  tile max 252, p95 209, fixedbin eligible true
  media render 243.35ms
  contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_contact.png
  side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_side_by_side.mp4

  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_2048t_f32_chunk2_2step.json
  pass: true
  source: test_data/test_video_384_128_6fps.mp4
  64f/512px/2048t/F32, frame_chunk_size 2: loss 0.34517 -> 0.34406
  mean step 4020.73ms, forward 586.93ms, colorize/loss 281.59ms,
    backward 3070.12ms, optimizer 38.66ms, overflow 0
  tile max 11, p95 5, fixedbin eligible true

Scale report:
  outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md
  outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json
```

This proves a trainable autograd path and a finite small real-video overfit
through `FeatureToColor`, plus the first-class `src/train/train.py` bridge. The
chunked run is slower at 8f/64px because it pays four launches per step, but it
is the memory valve needed before attempting full 64f/512px/F32. The 64f scale
probes show that backward is the main cost once chunking is active, and that
real-video support overflows well before the synthetic 32768-tube row: 8192 is
the current zero-overflow 256px bracket under cap 128, while cap 256 makes the
16384 row valid and makes 32768 valid when paired with support pruning.
Unpruned 32768 still overflows even at cap 256 (`216` tiles, max `275`). The
best current passing 20-step validity candidate is `alpha>=1/72 + cap256`:
it beats `alpha>=1/64 + cap256` and `alpha>=1/32 + cap256` on loss/PSNR while
staying zero-overflow, but it is tight against the cap (`252/256` max tile).
`alpha>=1/80` and `alpha>=1/96` improve loss slightly but overflow late, so
they are useful diagnostics, not fixed-bin candidates. The low absolute PSNR
means this is still a speed/validity candidate, not a quality replacement for
RGB STAR. The 512px probes now prove feasibility through 8192 tubes with zero
overflow, but not practical speed: 4096t is `6.46s/step`, and 8192t is
`7.94s/step` with `4.88s` in backward and `1.22s` in colorize/loss. That keeps
512px support safe at modest tube counts, while 32768t should wait for a real
feature-backward/colorize speed improvement.

First-class backward split:

```text
Diagnostic script:
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py

256px/64f/32768t/alpha>=1/72/cap256, STAR_UVT_TILE_CAPACITY=256:
  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.md
  gradcache: manual total 1994.7ms, colorize/loss backward 987.8ms,
    renderer backward 553.0ms, renderer 35.9% of backward
  scalar reduce: manual total 1764.8ms, colorize/loss backward 887.8ms,
    renderer backward 526.6ms, renderer 37.2% of backward
  vec4 reduce: manual total 1727.7ms, colorize/loss backward 867.4ms,
    renderer backward 494.7ms, renderer 36.3% of backward

512px/64f gradcache:
  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.md
  4096t: manual total 6566.8ms, colorize/loss backward 3775.0ms,
    renderer backward 1071.7ms, renderer 22.1% of backward
  8192t: manual total 5372.8ms, colorize/loss backward 3430.1ms,
    renderer backward 700.0ms, renderer 16.9% of backward

512px/64f gradcache, no pre-norm A/B:
  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.md
  4096t: manual total 2018.9ms, colorize/loss backward 317.1ms,
    renderer backward 674.8ms, renderer 68.0% of backward
  8192t: manual total 2403.5ms, colorize/loss backward 400.6ms,
    renderer backward 751.5ms, renderer 65.2% of backward

512px/8192t first-class no pre-norm trainer:
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_2step.jsonc
  outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_2step.json
  pass: true, zero overflow
  loss: 0.33817 -> 0.33764 in 2 steps
  mean step: 3715.4ms
  mean forward: 1268.3ms
  mean colorize/loss: 440.9ms
  mean backward: 1585.6ms
```

Interpretation:

The first-class 512px bottleneck is now mostly the dense image-space
`FeatureToColor`/loss VJP, not the STAR UVT Metal renderer. Renderer work still
matters, especially for feature-gradient atomics and fixedbin promotion, but a
pure renderer-only improvement has a hard ceiling on end-to-end speed unless
the colorizer/loss VJP is also simplified or fused at a better boundary.
The no-pre-norm A/B proves the current LayerNorm-based colorizer is a major
part of that VJP cost and gives a plausible fast-overfit setting, but it still
needed a longer quality/media comparison before promotion. The 2026-05-19
20-step media comparison passed for both rows and kept the speed direction, but
did not promote no-pre-norm on quality: no-pre-norm is `7.37s/step` with
`3.37s` backward and ends at `0.32053` loss / `4.941` PSNR, while default
pre-norm is `11.10s/step` with `7.07s` backward and ends at `0.31742` loss /
`4.984` PSNR.
The same-session 2026-05-19 rerun after the vec4 reducer work is materially
faster than the older no-pre-norm artifact and selects
`feature_direct_gradcache_reduce_vec4` for the current fast diagnostic:
`2.491s/step`, `1.184s` backward, media render `1.360s`, end loss/PSNR
`0.32053` / `4.941`, zero overflow. The matched gradcache rerun is
`2.858s/step`, `1.327s` backward, media render `1.660s`, same loss/PSNR.
That makes vec4 reduce the speed choice when no-pre-norm is already in use,
but the default pre-norm A/B only improves about two percent, so the broader
lesson is still that renderer changes and colorizer/loss VJP changes must move
together.
The identity/no-pre-norm follow-up is faster again (`2.54s/step`, `1.17s`
backward), but it ends worse at `0.32446` loss / `4.888` PSNR. That closes the
easy "sigmoid/pre-norm is the whole quality problem" hypothesis; the next
quality step needs feature initialization/objective/decoder-capacity evidence,
not another clamp removal.
The hidden-64 decoder-capacity follow-up gives only a microscopic best-feature
PSNR change (`4.984 -> 4.987`) while slowing to `19.18s/step`, so increasing
dense per-pixel decoder capacity is also a negative practical route.
The gain-2 pre-norm init row gives a similarly tiny PSNR (`4.987`) while
slowing to `14.12s/step`, so the current colorizer gain is not the main quality
blocker either.
The Gate 4 same-clip bracket adds the missing RGB STAR comparator: RGB STAR
direct-atomic reaches `12.444` PSNR after 20 steps on the same 64f/512px/8192t
test-video row, so feature STAR is not yet a source-overfit quality
replacement. Dynamic RGB/F32 rows in the bracket remain speed-only synthetic
references.

### Gate 4: Quality And Stability Bracket

Compare:

```text
RGB STAR UVT direct_atomic
feature STAR UVT direct_atomic
feature STAR UVT best fast mode
dynamic F32 fast-mac baseline
```

Use the same clip, 64 frames, same source crop, and 256 -> 512 multires if
512px full steps are too expensive.

Acceptance:

- no nonfinite loss/grad
- quality at least matches RGB STAR UVT on source overfit before claiming a
  feature win
- timing report separates renderer, FeatureToColor, loss, model, and optimizer

## Immediate Next Command-Level Work

1. Treat `STAR_UVT_TILE_CAPACITY=256` as the current validity specialization:
   it clears `16384t` and clears `32768t` only with support pruning. Use
   `alpha>=1/72` as the current quality-max passing row, and `alpha>=1/64` as
   the conservative fallback because `alpha>=1/72` has only four refs of tile
   headroom. Unpruned `32768t` still overflows at cap 256, so do not pretend
   cap alone solved support.
2. Keep 512px on frame chunks. The `4096/8192` tube probes are now complete and
   zero-overflow, but they are too slow (`6.46s` and `7.94s` per step) to
   justify 32768t before a real feature-backward/colorize speed improvement.
3. Extend the generated scale report whenever a bracket is run, and add media
   rows for promoted quality candidates rather than every timing smoke.
4. Use `direct_feature_mode_matrix.py` for future direct-feature reruns. It now
   records `feature_dim`, mode, `fixedbin_eligible`, overflow, and JSON/log
   paths per mode/resolution. The first sequential matrix lives at
   `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_mode_matrix_128_256_64f_32768t_f32/summary.md`.
5. `firstclass_backward_breakdown.py` is now the gate for bottleneck claims.
   On real first-class rows, the dense `FeatureToColor`/loss backward dominates
   the 512px cost (`77.9-83.1%` of backward), while the renderer is only
   `16.9-22.1%`; at 256px/32768t/cap256 the renderer is about `36%`. Any next
   "fast STAR UVT feature" claim must either reduce that image-space VJP
   directly or use a handoff that avoids constructing/backpropagating dense
   F32 image gradients through the current pre-norm colorizer.
6. Keep no-pre-norm as a speed candidate, not a promoted quality default. The
   512px/8192t 2-step no-pre-norm config is `3.72s/step` versus `7.94s/step`
   for the default pre-norm row, and the 20-step media A/B still speeds up the
   row (`7.37s/step` versus `11.10s/step`), but default pre-norm has slightly
   better 20-step loss/PSNR. The broader Gate 4 bracket now proves the current
   feature decoder is far below RGB STAR source-overfit quality, so the next
   quality gate must improve feature decoding/objective before treating STAR
   feature tubes as a replacement path. The identity/no-pre-norm run is a
   useful fast diagnostic (`2.54s/step`) but worse quality (`4.888` PSNR), so
   do not spend the next pass on more simple activation/norm removals. The
   hidden-64 pre-norm row barely improves quality and is much slower, so do not
   scale dense decoder capacity as the next default either. The gain-2 pre-norm
   init row is also a near-noop on quality and slower than the gain-4 baseline.
7. `feature_direct_gradcache` has landed as the first real fast-backward mode,
   but it only trims a few percent. The skip-feature-gradient diagnostic shows
   feature-gradient atomics are a large target; the first barrier-heavy
   `feature_direct_gradcache_reduce` attempt is trainable but slower, and the
   vectorized `feature_direct_gradcache_reduce_vec4` follow-up is positive only
   in the synthetic direct-kernel cap128 row and in the fresh 512px
   no-pre-norm first-class diagnostic, not in the older first-class cap256 row
   or default-pre-norm 512px gate.
   The cached-bin sidecar is also diagnostic only: it wins isolated synthetic
   renderer backward, but does not improve the 512px first-class trainer.
   The next real shader should not be another per-contributor threadgroup
   barrier path.
   The narrow `fused_first3_sigmoid_mse` gate says an RGB-grad handoff can be
   fast, but the first fully generalized in-tile linear-colorizer handoff and
   the image-space-prep logit handoff are both slower than gradcache. Do not
   wire either version into the trainer. The next attempt should build an
   optimized fixedbin/tile-slot feature backward that reduces per-channel
   feature-gradient atomics without duplicating STAR traversal; another handoff
   should only be tried if it avoids both full in-tile colorizer reduction and
   per-pixel `W^T` over all F channels.
   The first combined logit-handoff tile-slot reducer gate keeps this direction
   alive but does not close it: `logit_handoff_reduce_vec4` passes direct
   256/512 parity and improves synthetic backward, while scalar reduce regresses
   512px backward. Treat vec4 as the sidecar candidate for a trainer-compatible
   native-VJP/tile-slot bridge, not a default mode.
   The real-video linear RGB-VJP profile now verifies the same handoff boundary
   against the 64f/512px 1300-step checkpoint with zero loss error,
   `9.43e-09` max gradient error, zero overflow, and a small total timing win
   (`1691.0 -> 1587.4ms`). It is useful proof for linear RGB reconstruction,
   but it is still not the V-JEPA target-grid/hidden64 frozen-probe VJP.
   The target-grid/frozen-probe VJP bridge profile now checks that current
   objective directly: it matches autograd with zero loss error and `2.57e-08`
   max gradient error, but repeat timing is a slight negative (`1545.5ms`
   autograd versus `1594.3ms` bridge). The analytic target-grid/probe VJP mode
   then removes the autograd image-VJP graph for the target-grid MSE plus
   hidden64 probe, keeps parity (`3.07e-08` max gradient error), and gives a
	   small repeat-5 win (`1510.6 -> 1477.2ms`, `1.023x`). Treat this as the next
	   trainer-gate candidate for native/analytic target-grid VJP. That gate is now
	   wired as `feature_target.image_vjp_mode=analytic` and passes a matched
	   5-step 64f/512 smoke, but it ties end-to-end step timing (`1303.6ms`
	   autograd versus `1304.6ms` warm analytic rerun; no-first `1264.1ms` versus
	   `1259.2ms`). The backward bucket improves by `103.3ms`, but manual VJP work
   moves into the loss bucket. Keep it as an opt-in diagnostic; scalar
   fixedbin/tile-slot renderer work remains separate.
   The current `feature_direct_fixedbin` surface is only the eligibility/fallback
   contract; keep using it to prevent accidental fixedbin promotion when
   overflow appears.
   The matched target-grid trainer render-mode matrix now checks that same
   keeper objective across current renderer modes. It passes for
   direct-atomic, gradcache, cached-bins, scalar reduce, vec4 reduce, and
   fixedbin-request, all from the same 1300-step checkpoint with the same final
	   loss/probe PSNR. The repeat-top check does not promote vec4/reduce
	   (`direct_atomic` no-first `1249.0ms`, cached-bins `1410.9ms`, vec4
	   `1509.6ms`, fixedbin-request `1422.6ms`), and fixedbin-request reports
	   `kernel_backward_mode=direct_atomic`.
	   The sparse-grid follow-up supersedes this older analytic-VJP/matrix state:
	   sparse-grid VJP reaches `795.3ms` no-first (`730.5ms` in the sparse-grid
	   matrix), and sparse-forward plus sparse-grid VJP is now the selected
	   diagnostic with repeat-sensitive timing (`492.3ms` best isolated no-first,
	   `598.2ms` isolated repeat, `973.0ms` sequential 512px row, repeat-3
	   no-first mean `504.9ms` with `411.0-626.4ms` range) and identical
	   loss/probe movement. The batched target/probe VJP trainer now reaches
	   repeat-3 no-first `179.3ms` mean with `159.7-215.6ms` range and zero
	   overflow, so it is the selected speed surface before native fixedbin or
	   target/probe VJP work.
8. Use `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
	   star-feature-512-fast` as the checked single-video V-JEPA feature-tube
   speed diagnostic. It runs
   `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc`.
   Do not present it as the source-view quality baseline; the current RGB STAR
   source-view row still wins quality by a wide margin and the probe media is
   still blurry. Use `star-feature-512-rgbfast` for the old RGB-target feature
   speed row.
9. For selected-shader scale claims, cite
   `outputs/benchmarks/2026-05-19_star_uvt_feature_selected_shader_scale_128_256_512.md`.
   It proves the vec4 decision is strongest at 512px and that 128px needs
   cap/pruning validity checks before timing comparisons.
10. For cached V-JEPA STAR claims, cite
   `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_512_scale_gate.md`.
   The chunked 64f/512px/8192t row now passes at `3.74s/step`, but this is a
   scale/trainability gate. The next implementation task is reducing target
   chunk/loss cost, because chunking removed the resident full dense target but
   makes target generation the largest step bucket.
11. For longer frozen-probe STAR gates, use `output.checkpoint` and
   `train.resume_checkpoint` in `src/train/train_star_uvt_feature_overfit.py`.
   This is a warm-start local-step resume with explicit `train.global_step_offset`
   for schedule/reporting semantics; the 2026-05-19 8f/64px smoke proves
   model/colorizer/optimizer save/load, and the 64f/512px resume300-from300 plus
   probe-emphasis resume200-from600 gates prove the same contract at keeper scale.
   The scheduled balance row proves `rgb_probe_loss_weight` can now be staged too,
   but its endpoint tradeoff is not a quality promotion.
12. For compact native target-area visual VJP, do not spend another iteration
   only rearranging STAR W^T loops. The 2026-05-20 colorizer-gradient-only split
   exposes `target_area_colorizer_grad_only=144` and shows compact-support
   direct native backward at `64f/512px/8192t`, `6.25%` support is `88.9ms`
   star-only, `536.6ms` colorizer-grad-only, and `531.4ms` full colorizer. The
   compact-native blocker is the colorizer parameter-gradient reduction shape;
   use a reducer/separate matrix-style reduction or stay with compact autograd
   for visual gradients until a native route beats it.
13. The first reducer prototype answers the sidecar version negatively. The
   Torch/MPS sparse-render + colorizer-matrix-reduce + native star-only backward
   path is correct (`1.26e-08` colorizer max error, `2.62e-10` STAR max error)
   and beats native atomics in the same-window direct gate (`390.9ms` versus
   `752.8ms`), but it loses to the sparse-pixel baseline (`276.6ms`) because it
   duplicates sparse feature rendering and hidden replay. A compact native
   reducer is only worth doing if it emits colorizer partials inside the same
   native pass; otherwise move back to visual objective quality.
14. The same-pass SIMD-reduced native colorizer prototype answers the first
   native partial-reduction version: the direct compact kernel is finally
   competitive (`297.2ms` native total versus `312.1ms` sparse-pixel baseline),
   but the trainer still rejects it (`2908.9ms` mean step, `604.0ms` sparse
   visual backward, same feature/probe regression). Porting more colorizer
   atomic reduction is not enough unless the same design also removes
   target-area forward/loss overhead or changes support/objective.
15. The matched dynamic-gsplat fixed-512 smoke answers the "maybe use dynamic
   gsplat instead" speed question at smoke level: at `64f/512px/8192` active
   Gaussians, step 5 is `8.019s` total and `5.638s` backward, with raster only
   `0.362s`. That path is backward-bound and slower than the selected STAR UVT
   helpers, so a future dynamic-gsplat run should be a separate quality/baseline
   exercise, not the fast local STAR replacement.
16. The selected compact visual route is fast but not scale-ready: the explicit
   visual gate reads dense full RGB `6.023` PSNR against RGB STAR `12.444`, with
   sparse/streaked dense media and blurry probe media. Do not launch the
   300-video scale lane until the visual objective/support/model bridge changes.
17. If another alpha-only visibility/support diagnostic is needed, use the
   sparse-pixel F1 alpha wrapper rather than dense F32 feature rendering:
   `render_uvt_feature_alpha_all_pixels_with_bins` plus cached F1 backward keeps
   exact alpha and gradient parity while cutting the measured all-chunk
   alpha-only envelope `1100.8 -> 634.6ms`. This is a diagnostic speed path,
   not evidence that same-support dense alpha is a good objective.
18. For trainer-integrated dense alpha diagnostics, set
   `dense_alpha.render_mode=sparse_f1`; it preserves the dense F32 quality
   endpoint but cuts the measured mean step/backward
   `2558.6/1114.2 -> 873.3/370.0ms`. Treat it as the default diagnostic route,
   not as a reason to scale the rejected dense-alpha objective.
19. The fixed-budget birth/split trainer primitive is now real, but still only
   a primitive: `support_birth_split.enabled` reallocates existing tubes before
   training, and the first 512px sparse-1500 gate passes with `32/8192`
   low-opacity tubes, zero overflow, `189.4ms` mean step, and dense RGB PSNR
   `5.708`. Dense support improves versus center/support rows on forced-alpha
   and high-alpha support but not enough on alpha `>0.1` coverage, so next work
   should not declare visual quality solved. The uncovered-brightness target
   sampler follow-up is also now run: it selects low-alpha bright samples
   (`selected_alpha_mean=0.0209`), passes at `187.4ms` mean step, and nudges
   dense RGB to `5.713`, but alpha `>0.1` stays `0.411`. The next useful gate is
   now partly answered: cap `128` overflows at `64+` births, cap `256` clears
   `64/128`, and radius `96px` raises alpha coverage (`0.420-0.422`) more than
   target source or tube count. Best safe cap-128 row is
   `low_alpha_n32_r96_cap128` (`0.420` alpha `>0.1`, `5.825` normal,
   `14.591` forced-alpha, `24.226` oracle, max tile `100/128`). This is a
   support-coverage nudge with oracle loss. The intermediate-radius follow-up
   finds no hidden sweet spot: uncovered `r64/r72/r80/r88` raises alpha `>0.1`
   `0.411/0.413/0.415/0.417` while oracle falls
   `25.319/25.187/25.015/24.802`, and low-alpha `r80/r88` fails loss decrease.
   Scalar born opacity is also not the missing bridge: at `r80`, uncovered
   opacity `0.4/0.6/0.8/0.9` moves alpha only `0.414->0.415` while oracle
   falls `25.177->24.987`; at `r88`, opacity `0.2/0.4/0.6/0.8` moves alpha
   `0.414->0.417` while oracle falls `25.242->24.802`. Next change support
   shape, not just radius, opacity, or another same-support objective. The
   first support-shape gate is also negative: `trajectory_ellipse` birth support
   with along `88px`, across `24/32px`, precision `48px`, opacity `0.4/0.6`,
   `32` births, and cap `128` passes all eight rows with zero overflow, but
   alpha `>0.1` stays `0.408-0.409`, below the isotropic uncovered `0.411`.
   This points away from one global fitted line and toward multi-center or
   stratified birth/split. The first multi-center gate confirms that direction:
   `farthest_xy` with `K=8`, `32` births, `r64`, and cap `128` reaches alpha
   `>0.1` `0.4309` and alpha `>0.5` `0.1550` with zero overflow
   (`101/71/128`) and forced-alpha PSNR `14.608`, but oracle drops to
   `23.965`. The multi-center radius/opacity frontier now selects `r64/o0.4`
   as the balanced row: alpha `>0.1` `0.4298`, alpha `>0.5` `0.1385`,
   forced-alpha `14.620`, oracle `24.805`, zero overflow, and
   `167.9/58.1ms` step/backward. Best raw coverage is `r72/o0.8`
   (`0.4318`) but oracle falls to `23.670`; next gate should be a 20-step media
   run on `K=8/r64/o0.4`. That gate now passes and holds the support gain:
   loss `0.903197->0.897231`, feature loss `0.631571->0.631083`, probe PSNR
   `21.681->21.769`, full RGB PSNR `5.794`, zero overflow, last step/backward
   `147.3/54.3ms`, dense alpha `>0.1` `0.431158`, forced-alpha `14.631`, and
   oracle `24.851`. The matched `K=8/r72/o0.4` 20-step sibling passes too:
   loss `0.910099->0.903088`, probe PSNR `21.601->21.703`, full RGB PSNR
   `5.820`, zero overflow, mean step/backward `157.9/61.1ms`, dense alpha
   `>0.1` `0.432454`, and oracle `24.668`. That is not a replacement for
   `r64/o0.4`: it buys tiny coverage/normal-PSNR improvement while giving back
   feature/probe loss and target-background oracle. This is positive support
   progress, not a visual-quality closeout.
