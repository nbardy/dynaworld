# STAR UVT Frozen RGB-Probe Integration Gate

## Goal

After the standalone target-grid feature-to-RGB probe proved that cached V-JEPA
target-grid features are decodable, the next gate was to put that trained
decoder into the STAR feature trainer itself. This should answer whether the
bridge is only a separate oracle, or whether STAR-rendered features can receive
a cheap visual gradient through the frozen decoder at the V-JEPA token grid.

## Implementation

Updated `src/train/train_star_uvt_feature_overfit.py` with opt-in fields under
`feature_target`:

- `rgb_probe_checkpoint`
- `rgb_probe_loss_weight`
- `rgb_probe_target_rgb_adapter`

The trainer loads the checkpointed `FeatureToColor`, freezes it, downsamples
the source RGB to the target grid, downsamples rendered STAR features to the
same grid, and adds a frozen RGB-probe loss. Optional output keys write probe
contact-sheet/video media without using the normal untrained colorizer path:

- `output.rgb_probe_contact_sheet`
- `output.rgb_probe_side_by_side_video`
- `output.rgb_probe_side_by_side_fps`

Added config:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.jsonc
```

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.jsonc
```

Offline W&B id: `f7v5bs0r`.

## Result

Artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.json
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_20step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_20step_sbs.mp4
```

Measured:

- pass: `true`
- total loss: `1.399375 -> 1.391043`
- feature loss: `0.999935 -> 0.998357`
- frozen RGB-probe loss: `0.039944 -> 0.039269`
- frozen RGB-probe PSNR: `13.985 -> 14.060`
- mean step: `1219.98ms`
- render forward: `547.80ms`
- feature-target adapter/loss: `15.79ms`
- frozen RGB-probe loss: `34.27ms`
- backward: `571.85ms`
- zero tile overflow; all model gradients present

## Interpretation

The trained decoder can be loaded into STAR and backpropagates to STAR tube
features cheaply. This is a real integration gate, not just a standalone probe.
It is not yet a visual-quality promotion: 20 steps only nudge probe PSNR by
`0.074dB`, and final feature loss is worse than the pure 20-step target-grid
media row. The next useful branch is a longer frozen-probe run or a schedule
that uses the probe loss after the feature target has moved, not more plain RGB
aux weight.

## Validation

- `py_compile` passed for the trainer and focused target-adapter tests.
- `pytest tests/test_star_uvt_feature_target_adapter.py -q` passed.
- The JSON pass and media artifacts were checked during the session.
