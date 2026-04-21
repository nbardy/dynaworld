# Test Data Catalogue

This folder holds small local test assets and generated outputs for the `dynaworld` experiments.

## Files

- `test_video_small.mp4`
  - Center-cropped `256x256` test video
  - Current version is downsampled to `2 fps`
  - Duration: about `11.5s`
  - Frame count: `23`

- `test_video_384_3fps.mp4`
  - Center-cropped `384x384` test video generated from the original source clip
  - Downsampled to `3 fps`
  - Duration: about `11.67s`
  - Frame count: `35`

- `test_video_small_64_4fps.mp4`
  - `64x64` variant generated from the original `14951282_2160_3840_30fps.mp4` source clip
  - Center-cropped to square, sampled at `4 fps`, then scaled to `64x64`
  - Duration: about `11.5s`
  - Frame count: `46`

- `test_video_small_128_4fps.mp4`
  - `128x128` variant generated from the original `14951282_2160_3840_30fps.mp4` source clip
  - Center-cropped to square, sampled at `4 fps`, then scaled to `128x128`
  - Duration: about `11.5s`
  - Frame count: `46`

- `test_video_384_64_6fps.mp4`
  - `64x64` variant generated from `test_video_384_3fps.mp4`
  - Doubles the source clip rate from `3 fps` to `6 fps`
  - Duration: about `11.67s`
  - Frame count: `70`

- `test_video_384_128_6fps.mp4`
  - `128x128` variant generated from `test_video_384_3fps.mp4`
  - Doubles the source clip rate from `3 fps` to `6 fps`
  - Duration: about `11.67s`
  - Frame count: `70`

- `ffmpeg_command.txt`
  - The most recent ffmpeg commands used to regenerate local test videos

## Generated Outputs

- `dust3r_outputs/smoke_test/`
  - Small DUSt3R smoke-test reconstruction output
  - Includes:
    - `scene.glb`
    - `rgb_images.npy`
    - `depthmaps.npy`
    - `pts3d.npy`
    - `poses_c2w.npy`
    - `intrinsics.npy`
    - `focals.npy`
    - `confidences.npy`
    - `confidence_masks.npy`
    - `summary.json`

- `dust3r_outputs/test_video_small_64_4fps_all_frames/`
  - DUSt3R all-frame reconstruction output for `test_video_small_64_4fps.mp4`
  - Uses `frame-stride 1`, `max-frames 0`, and `swin-5-noncyclic`
  - Frame count: `46`
  - Pair count: `430`
  - Alignment loss: about `0.0336`
  - Includes the same camera, point, preview, and summary files as the all-frame outputs

- `dust3r_outputs/test_video_small_128_4fps_all_frames/`
  - DUSt3R all-frame reconstruction output for `test_video_small_128_4fps.mp4`
  - Uses `frame-stride 1`, `max-frames 0`, and `swin-5-noncyclic`
  - Frame count: `46`
  - Pair count: `430`
  - Alignment loss: about `0.0633`
  - Includes the same camera, point, preview, and summary files as the all-frame outputs

## Main Camera Extraction Command

From `dynaworld/`:

```bash
./src/train_scripts/get_camera.sh
```

This uses `test_video_small.mp4` by default and writes outputs under:

```text
test_data/dust3r_outputs/test_video_small_all_frames
```
