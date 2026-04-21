# 128 Failure vs Stable Baseline

User asked what changed, what broke, and whether the stable baseline was in git.

The stable committed baseline I can identify is commit `be87e96`:

```text
be87e96 Record dynamic train matrix
```

At that commit, `src/train_configs/local_mac_overfit_prebaked_camera.jsonc`
was the only prebaked-camera local overfit config. It used:

- `sequence_dir`: `test_data/dust3r_outputs/test_video_small_all_frames`
- video/data: `test_video_small.mp4`, 23 frames, 2fps
- model size: `32`
- tokens x splats: `128 x 4 = 512`
- `frames_per_step`: `0`, meaning all 23 frames per step
- `lr`: `0.005`
- dense renderer, framewise render loop

Today we added new 64px/128px 4fps variants and config routing:

- `small_64_4fps` -> `test_data/dust3r_outputs/test_video_small_64_4fps_all_frames`
- `small_128_4fps` -> `test_data/dust3r_outputs/test_video_small_128_4fps_all_frames`
- both are 46-frame, 4fps bakes from the original 30fps source clip
- the trainer now uses batched dense rendering, but batch-vs-loop checks showed
  equivalent outputs/gradients on real cameras

What broke is not the old 32px baseline. The new 128px DUSt3R camera solve has
a very different relative pose scale:

- old 32px/2fps bake raw median fx: `396.68`, median FoV at 224: `31.53 deg`
- corrected 64px/4fps bake raw median fx: `602.65`, median FoV at 224:
  `21.06 deg`
- corrected 128px/4fps bake raw median fx: `662.79`, median FoV at 224:
  `19.18 deg`
- corrected 64px relative camera z translation stays about `[-0.28, 0.11]`
- corrected 128px relative camera z translation reaches about `2.56`, with
  frames `37-45` clustered near `2.4-2.56`

The model's current Gaussian initialization/head constrains predicted Gaussian
depth to `[0.5, 2.5]` in the first-camera coordinate system:

```python
z = sigmoid(raw_z) * 2.0 + 0.5
```

Therefore the 128px late-frame cameras can sit at or past the initialized
Gaussian volume. In those frame windows, many Gaussians are behind or nearly on
the camera plane before any optimizer step.

Actual failing run:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh src/train_configs/local_mac_overfit_prebaked_camera_128_4fps.jsonc
```

W&B: `https://wandb.ai/nbardy/dynaworld/runs/jn8q1e6s`

Failure at step 1, before optimizer update, on frames `[35, 36, 37, 38]`:

- `RenderDiag/RenderNonfiniteCount=9.83e4`
- `RenderDiag/CameraZMin=-0.7978`
- `RenderDiag/CameraZMax=1.008`
- `RenderDiag/FrontGaussiansMin=106`
- `RenderDiag/NearOrBehindGaussiansMean=195.8`
- `RenderDiag/PowerMax=1.014e31`
- `RenderDiag/PowerGt80Count=7.035e5`
- `RenderDiag/AlphaPreNonfiniteCount=7.035e5`
- `RenderDiag/RawDetMin=-2.535e30`
- `RenderDiag/RawDetNegativeCount=26`

Interpretation:

- The failure is renderer/projection numerical failure caused by camera/scene
  scale mismatch, not optimizer instability.
- Decoded Gaussian tensors are finite at failure.
- The old committed baseline did not cover this 128px camera trajectory.
- The renderer still has a robustness issue because it computes projection
  exponents for near/behind Gaussians even after opacity masking. That converts
  `0 * exp(huge)` into `NaN`.
- The deeper modeling issue is that DUSt3R pose scale is arbitrary relative to
  the TokenGS fixed output depth volume. The camera trajectory should probably
  be normalized into a model-compatible coordinate range before training.

No fix was applied in this note.
