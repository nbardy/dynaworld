# Full-Clip All-Enabled Feature PowerFoam

## Trigger

The 512px geometry-only dynamic PowerFoam render looked like it was working but
was visibly coarse. The next hypothesis was to train on the full clip with all
dynamic channels enabled instead of freezing appearance/material.

## Full Clip

Materialized the full raw high-motion segment as a center crop at 8fps:

```text
source: data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4
derived: data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4
shape: 640x640
fps: 8
frames: 56
duration: 7.0s
codec: h264 / avc1 / yuv420p
```

The first extracted frame is the expected car/person/mountain view, not a crop
or color-channel failure.

## 1024-Cell All-Enabled F32 Feature Foam

Config:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step.jsonc
```

Important settings:

- `dynamic_mode: token_rbf_features`
- `feature_dim: 32`
- `num_texel_sites: 8`
- all dynamic switches true: centers, radii, densities, features, normals, texel sites
- 56 frames, 512px render, 120 train steps
- W&B enabled, run launched offline:
  `wandb/offline-run-20260506_160432-fisc6jqy`

Final artifacts:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step/render_step_0120.mp4
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step/side_by_side_step_0120.mp4
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step/checkpoint_final.pt
```

Metrics:

```text
step0  mean/min PSNR 17.2776 / 10.5138
step120 mean/min PSNR 18.0796 / 11.1518
step0  L1/MSE 0.09850 / 0.02296
step120 L1/MSE 0.08914 / 0.01959
train loop elapsed_s 90.73
```

Motion/state:

```text
state_mean_center_delta 0.02728
state_mean_density_delta 0.09014
state_mean_feature_delta 0.10550
state_mean_normal_delta 0.12548
state_mean_texel_site_delta 0.15625
state_mean_temporal_screen_delta_px 0.4027
state_p95_temporal_screen_delta_px 0.7849
state_mean_temporal_feature_abs_delta 0.03063
eval_mean_temporal_alpha_delta 0.000425
```

MP4 verifier passed on final render and side-by-side:

```text
render var_min 0.05113, unique RGB 6632, green fraction 0.000320, h264/avc1/yuv420p
side-by-side var_min 0.05089, unique RGB 7200, green fraction 0.000160, h264/avc1/yuv420p
```

Visual read: valid render, no green export issue, and the 1024-cell all-enabled
F32 run is the nice-looking baseline from this pass. It is still visibly
blocky/foamy, so this improved metrics and enabled appearance/material dynamics
but did not remove the coarse support footprint.

## 2048-Cell Follow-Up

Because the 1024-cell all-enabled result was still coarse, ran a shorter
2048-cell follow-up to test whether spatial support count is the real bottleneck.

Config:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_2048_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_60step.jsonc
```

Final artifacts:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_2048_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_60step/render_step_0060.mp4
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_2048_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_60step/side_by_side_step_0060.mp4
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_2048_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_60step/checkpoint_final.pt
```

Metrics:

```text
step0 mean/min PSNR 16.3705 / 10.0788
step60 mean/min PSNR 17.7607 / 10.6739
step0 L1/MSE 0.11315 / 0.02706
step60 L1/MSE 0.09222 / 0.02109
train loop elapsed_s 46.72
```

Motion/state:

```text
state_mean_center_delta 0.01880
state_mean_density_delta 0.08515
state_mean_feature_delta 0.09081
state_mean_normal_delta 0.15834
state_mean_texel_site_delta 0.09568
state_mean_temporal_screen_delta_px 0.2291
state_p95_temporal_screen_delta_px 0.4937
```

MP4 verifier passed on final render and side-by-side:

```text
render var_min 0.05140, unique RGB 6919, green fraction 0.000233, h264/avc1/yuv420p
side-by-side var_min 0.05103, unique RGB 7221, green fraction 0.000130, h264/avc1/yuv420p
```

Visual read: 2048 cells looks a bit finer in places but is still strongly
cellular, and at 60 steps it is below the 1024-cell 120-step run on mean PSNR.
The denser init started with worse alpha/coverage (`alpha_mean 0.878` vs
`0.953` for 1024), so increasing cells is not a free win under the current init
and schedule.

## Current Takeaway

The all-enabled feature-foam path works on the full 7s/56f clip and logs valid
H.264 MP4s, but the visible coarseness is primarily support/representation
quality, not just frozen appearance. The next useful experiment is not another
appearance switch; it is either:

- better density/radius/coverage init for 2048+ cells
- a longer 2048 run with coverage warmup after alpha is fixed
- multires/hierarchical foam support
- smaller initial radii / more cells with an alpha coverage loss so extra cells
  do not start undercovered
