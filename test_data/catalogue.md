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

## Main Camera Extraction Command

From `dynaworld/`:

```bash
./get_camera.sh
```

This uses `test_video_small.mp4` by default and writes outputs under:

```text
test_data/dust3r_outputs/test_video_small_all_frames
```
