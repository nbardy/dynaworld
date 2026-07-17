# Dynamic Gsplat 512px Matched Probe

Date: 2026-05-20 01:26 +07

## Goal

Close the "fresh matched dynamic-gsplat timing" gap from the STAR UVT
fast-shader goal audit with the smallest existing repo path.

## Command

```bash
GSPLAT_CONFIG=src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step.jsonc \
PROBE_STEPS=5 \
WANDB_MODE=disabled \
rtk ./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh gsplat-smoke
```

The helper's default dynamic-gsplat config is multires and would remain at
256px for a short probe, so this run explicitly used the fixed 512px overfit
config.

Log:
`outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_probe5step_20260520_012448.log`

## Result

The probe completed on MPS. It used cached V-JEPA conditioning hits and the
intended `64f/512px/8192` active Gaussian setup.

Step 5 timing:

- step total `8.0188s`
- backward `5.6376s` (`70.3%`)
- forward decode `1.0617s` (`13.2%`)
- sample/load clip `0.6385s` (`8.0%`)
- render/rasterize `0.3622s` (`4.5%`)
- reconstruction loss `0.2732s` (`3.4%`)
- V-JEPA feature loss `0.0022s`
- loss moved `0.5999 -> 0.5930`

Comparison:

- STAR UVT compact target-area visual helper: `930.6ms` mean step,
  `581.3ms` mean backward, `6.023` full RGB endpoint.
- STAR UVT sparse-forward batched target-grid/probe helper: `399.9ms` mean
  step, `176.9ms` mean backward, but blurry media.

## Decision

This is a smoke-level comparator, not a full dynamic-gsplat baseline. Still, it
answers the immediate planning question: at the selected local 512px/64f/8192
scale, dynamic gsplat is much slower than the current STAR UVT helper routes and
is backward-dominated, not data-loader- or rasterizer-dominated.

Next work should stay on STAR UVT visual quality and then scale-up, not on
switching the fast route to dynamic gsplat without a separate longer quality
baseline.

Report:
`outputs/benchmarks/2026-05-20_dynamic_gsplat_512_matched_probe.md`
