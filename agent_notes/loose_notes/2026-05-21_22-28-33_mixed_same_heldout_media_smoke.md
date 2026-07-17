# Mixed Same-View/Heldout Media Smoke

## Goal

Convert the mixed same-view plus heldout-view trainer bridge from scalar-only
smoke evidence into a W&B trace that also exercises final-step preview and
multicam validation media. This is still trainer-interface evidence, not a
baseline or quality claim.

## Config Change

Updated:

`src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`

Change:

- `logging.always_log_last_step=false -> true`

The config keeps `image_log_every=1000`, `video_log_every=1000`, and
`log_initial_media=false`, so it remains cheap: only the final step logs media.

## Run

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

Result:

- Passed on MPS.
- Offline W&B run:
  `wandb/offline-run-20260521_222750-9yvznqiq`
- Loaded the DeepView RGB-pyramid multicam feature cache.
- Loaded the same-view lazy manifest path.
- Selected dense renderer for `64` effective Gaussians at `32px`.
- Completed 10 optimizer steps.
- Final eval printed:
  - `TrainView0/Eval/PSNR = 3.8886`
  - `TrainView1/Eval/PSNR = 3.7170`
  - `Heldout0_camera_0040/Eval/PSNR = 3.5569`

## W&B Evidence

`strings wandb/offline-run-20260521_222750-9yvznqiq/run-9yvznqiq.wandb` contains:

- `Loss/same_view_recon`
- `Loss/same_view_recon_weighted`
- `Loss/heldout_view_recon`
- `Loss/heldout_view_recon_weighted`
- `Render_GT_vs_Pred`
- `TrainView0_Rendered_Video`
- `Heldout0_camera_0040_Rendered_Video`

Media files written:

- `files/media/images/Render_GT_vs_Pred_9_0f848e5b862d5472b6b9.png`
- `files/media/videos/TrainView0_Rendered_Video_9_26b9e376cb90716bfd61.mp4`
- `files/media/videos/TrainView0_GT_Video_9_bcf1551c86f9cb2896d2.mp4`
- `files/media/videos/TrainView1_Rendered_Video_9_b8107cf720a624d7cce6.mp4`
- `files/media/videos/TrainView1_GT_Video_9_74a6b6143360617cf2a3.mp4`
- `files/media/videos/Heldout0_camera_0040_Rendered_Video_9_afe4f7d49357b371dd35.mp4`
- `files/media/videos/Heldout0_camera_0040_GT_Video_9_d55e781ff8f82a382b74.mp4`

## Interpretation

The mixed trainer now has stronger interface evidence:

- the `src/train/train.py` registry route works,
- alternating same-view/heldout steps execute,
- separate loss names reach W&B,
- final-step preview/media logging works,
- multicam validation media runs through the shared media helper.

This does not prove convergence, novel-view quality, or a baseline. The next
promotion step is a longer W&B-enabled benchmark with a source/camera-disjoint
eval plan and explicit result rows before touching `BASELINES.md`.
