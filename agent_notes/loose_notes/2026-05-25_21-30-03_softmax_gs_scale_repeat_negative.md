# 2026-05-25 21:30:03 Softmax-GS scale repeat negative

We continued the Softmax-GS dynamic-GS integration after the first positive
64px/4f/128-splat multicam heldout row. The question was whether that K=16
heldout PSNR jump survives a less toy repeat before spending STAR UVT or
WorldFoam integration time.

What happened:

- Added a practical 64px/4f/512-splat matched pair:
  - `src/train_configs/local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_512splats_20step.jsonc`
  - `src/train_configs/local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_512splats_20step.jsonc`
- Tried to scale along resolution/frame axes first, but the precomputed-model
  forward path aborts before rasterization with the MPS
  `MPSNDArrayDescriptor sliceDimension` assertion:
  - existing 128px/16f/512 train configs crash in `forward_clip`;
  - 128px/8f no-op train config crashes the same way;
  - inline 64px/8f forward probe crashes the same way;
  - inline 128px/4f forward probe crashes the same way;
  - AMP bf16 did not fix it because `mps_flash_attn` is absent.
- Inline 64px/4f/512 forward probe succeeded, so primitive count was the only
  scale axis we could test in this local MPS route.

Runnable repeat results:

```text
no-op 64px/4f/512:
    config local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_512splats_20step.jsonc
    initial/final train loss 0.5817 -> 0.2511
    train PSNR/SSIM view0 11.8441/0.1112
    train PSNR/SSIM view1 12.0649/0.1218
    heldout camera_0040 PSNR/SSIM 12.5002/0.0817
    step-20 total/backward/raster 707/155/97ms
    offline run wandb/offline-run-20260525_212845-8rj3swm6

enabled K=16 64px/4f/512:
    config local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_512splats_20step.jsonc
    command used GSP_TAPE_CAP=16
    initial/final train loss 0.5818 -> 0.2378
    train PSNR/SSIM view0 12.8191/0.0917
    train PSNR/SSIM view1 12.0651/0.1221
    heldout camera_0040 PSNR/SSIM 11.8847/0.0950
    step-20 total/backward/raster 554/140/102ms
    offline run wandb/offline-run-20260525_212923-wbr8y46t
```

Interpretation:

The tiny 128-splat heldout PSNR jump is not repeated at 512 splats. K=16
improves source/train loss and slightly improves heldout SSIM, but loses
heldout PSNR by `0.6155dB`. This keeps Softmax-GS as an active dynamic-GS
renderer probe, not a STAR/WorldFoam integration priority. Next useful work is
either fixing/bypassing the MPS model-forward blocker for 128px/8f rows, or
adding tape residual/coverage diagnostics around the runnable 64px/4f path.
