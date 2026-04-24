# Camera Feed-In vs Image-Implicit 32x2 Rerun

User asked whether implicit camera is intrinsically worse than camera-feed-in, and whether both pass time into the MLP/token decode path.

## Why this comparison matters

The bad full-capacity 128px video-token implicit render looked far worse than the known-camera baseline. That could mean:

- implicit camera is broken,
- the full video-token architecture/training contract is broken,
- the recent time/init changes regressed something,
- or the high-capacity 128px/4fps setup is not comparable to the small image-encoder baseline.

The old small config is a cleaner check because both runs use the same 23-frame 2fps DUSt3R sequence at 32px with 128 tokens x 4 splats/token, dense renderer, all frames per step, LR 0.005, and a 100-step override.

## Paired runs

Known camera feed-in:

- Config: `src/train_configs/local_mac_overfit_prebaked_camera.jsonc`
- W&B: `w3a14cnp`
- URL: https://wandb.ai/nbardy/dynaworld/runs/w3a14cnp
- Final `Eval/Loss`: `0.06592795997858047`
- Final `Eval/L1`: `0.06392080336809158`
- Final `Eval/MSE`: `0.010035772807896137`
- Final `Eval/SSIM`: `0.5125110149383545`
- Final `Eval/PSNR`: `19.98449178903249`
- Final train `Loss`: `0.06584010273218155`

Image implicit camera:

- Config: `src/train_configs/local_mac_overfit_image_implicit_camera.jsonc`
- W&B: `mv4ggjq8`
- URL: https://wandb.ai/nbardy/dynaworld/runs/mv4ggjq8
- Final `Eval/Loss`: `0.06029919907450676`
- Final `Eval/L1`: `0.05853332206606865`
- Final `Eval/MSE`: `0.008829416707158089`
- Final `Eval/SSIM`: `0.5913111567497253`
- Final `Eval/PSNR`: `20.540679860294404`
- Final train `Loss`: `0.058711227029561996`
- Camera eval: FOV `59.67697`, radius `2.94718`, mean rotation delta `1.55063 deg`, mean translation delta `0.31763`

## Interpretation

This does not reproduce the 128px video-token degeneration. On the small 32px overfit task, image-implicit camera is at least competitive with the known-camera feed-in baseline after the same 100-step smoke run. That supports the user's intuition: camera conditioning should not be the dominant issue for this overfit setup, because the image-implicit path still learns the sequence.

The bad 128px run is therefore more likely specific to the video-token implicit baseline and/or the 128px/4fps/full-scene capacity contract, not a generic failure of implicit camera heads.

## Time conditioning audit

Known camera feed-in `DynamicTokenGS`:

- Builds Plucker rays from the known camera.
- Converts `frame_times` to `[B, 1]`.
- Adds `time_proj(frame_times)` to token queries before token attention and before `GaussianParameterHeads`.

Image implicit camera `DynamicTokenGSImplicitCamera`:

- Converts `frame_times` to `[B, 1]`.
- Adds `time_proj(frame_times)` to all non-global tokens before token attention.
- The Gaussian MLP decodes from the refined splat tokens.

Separated image implicit camera:

- Adds time to path features and splat-token queries.
- Global camera feature remains sequence/global.

Video-token implicit camera after today's fix:

- Preserves absolute sequence frame times in `prepare_clip`.
- Adds `time_proj(decode_time)` to non-global query tokens before query-to-video cross-attention.
- Decodes each requested time separately from the shared video-token memory.

None of these paths concatenate raw scalar `t` directly into the final Gaussian MLP. The practical contract is "time-conditioned token/query features are fed to the MLP."

## Remaining hypothesis

The full video-token baseline is not apples-to-apples with the working 32px image-implicit baseline:

- The image baseline sees each target frame as the encoder input for that frame.
- The video-token baseline compresses a clip into shared tubelet/video memory and decodes requested times from that memory.
- The first frame is not as trivial in the video-token path because it is decoded from clip memory and query tokens, not directly from first-frame image features.
- The bad zero-init issue was fixed, but 200-step probes still plateaued around `Eval/Loss ~0.149-0.150`; the small 32px implicit result says this is not explained by "implicit camera cannot learn."
