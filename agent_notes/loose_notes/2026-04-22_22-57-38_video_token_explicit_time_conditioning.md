# Video-Token Explicit Time Conditioning

User corrected the intended contract: time should be explicit frame-index time scaled to `[0, 1]`, not inferred from the clip, and the video encoder should know the time span of the clip it is encoding.

## Code changes

Changed `src/train/train_video_token_implicit_dynamic.py`:

- `prepare_clip(...)` now builds `clip_times` directly from selected frame indices:
  `clip_indices / max(sequence_data.frame_count - 1, 1)`.
- This avoids relying on source timestamps and makes the video-token trainer use absolute normalized frame-index time.

Changed `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`:

- Added time projectors for video encoder stage-1 and stage-2 tokens.
- `VideoEncoder.forward(video, frame_times=...)` now conditions tokens on:
  - per-tubelet mean frame time,
  - clip span `[first_time, last_time]`.
- `DynamicVideoTokenGSImplicitCamera.forward(...)` now accepts `input_times`; defaults to `decode_times` for reconstruction clips.
- Query decoding still receives `decode_time` before cross-attention.
- Added `head_time_proj` so `decode_time` is also injected directly into path and Gaussian tokens immediately before camera/path heads and `GaussianParameterHeads`.

This makes the contract explicit:

- input frames carry absolute normalized frame-index times into the encoder,
- decode requests carry absolute normalized frame-index times into query attention,
- the final head MLP input is explicitly time-conditioned.

## Verification

Compile:

```text
uv run python -m py_compile src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/train_video_token_implicit_dynamic.py
```

Forward smoke:

```text
DynamicVideoTokenGSImplicitCamera(clip_length=4, image_size=32, ...)
video [1,4,3,32,32], times [0, 1/3, 2/3, 1]
output xyz/scales [4,4,3], 4 cameras, rotation_delta [4,3]
```

## 200-step rerun

Run:

- W&B id: `febn4gq6`
- URL: https://wandb.ai/nbardy/dynaworld/runs/febn4gq6
- Config base: `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`
- Override: 200 steps, log/image/video every 200.

Final summary:

- `Loss`: `0.14756226539611816`
- `Loss/Reconstruction`: `0.1474999189376831`
- `Eval/Loss`: `0.15407316386699677`
- `Eval/L1`: `0.10153083503246307`
- `Eval/MSE`: `0.022670362144708633`
- `Eval/SSIM`: `0.27151501178741455`
- `Eval/PSNR`: `16.44541542238176`
- `Eval/TemporalAdjacentL1Ratio`: `0.017286796122789383`
- `Eval/TemporalToFirstL1Ratio`: `0.26961952447891235`
- Camera eval: FOV `58.5468`, radius `2.8596`, mean rotation delta `1.0231 deg`, mean translation delta `0.1368`.

## Interpretation

The explicit time contract is now wired correctly, but this alone did not recover the video-token implicit baseline. The full-video eval is close to prior 200-step patched-init runs and temporal adjacent motion is still too small. That argues the remaining failure is not just "the model lacked `t`"; it is more likely the video-token clip-memory architecture/training contract, capacity allocation, or camera/splat coupling.
