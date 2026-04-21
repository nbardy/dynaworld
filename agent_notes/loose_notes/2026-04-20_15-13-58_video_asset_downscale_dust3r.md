# Video Asset Downscale and DUSt3R Bake

Worker 1 data-pipeline task: inspect existing `test_data` videos, generate `64x64` and `128x128` MP4 variants at double source FPS, and run DUSt3R bakes where requested.

Existing inputs:

- `test_data/test_video_small.mp4`: `256x256`, `2 fps`, `11.5s`, `23` frames.
- `test_data/test_video_384_3fps.mp4`: `384x384`, `3 fps`, `11.67s`, `35` frames.
- `src/train_scripts/get_camera.sh` calls `src/train/run_dust3r_video.py` with `--frame-stride 1 --max-frames 0 --scene-graph swin --winsize 5 --noncyclic`, so the new bakes followed that all-frame pattern.

Generated MP4s:

- `test_data/test_video_small_64_4fps.mp4`: `64x64`, `4 fps`, `46` frames.
- `test_data/test_video_small_128_4fps.mp4`: `128x128`, `4 fps`, `46` frames.
- `test_data/test_video_384_64_6fps.mp4`: `64x64`, `6 fps`, `70` frames.
- `test_data/test_video_384_128_6fps.mp4`: `128x128`, `6 fps`, `70` frames.

FFmpeg commands:

```bash
ffmpeg -y -i test_data/test_video_small.mp4 -vf "fps=4,scale=64:64:flags=lanczos" -an -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart test_data/test_video_small_64_4fps.mp4
ffmpeg -y -i test_data/test_video_small.mp4 -vf "fps=4,scale=128:128:flags=lanczos" -an -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart test_data/test_video_small_128_4fps.mp4
ffmpeg -y -i test_data/test_video_384_3fps.mp4 -vf "fps=6,scale=64:64:flags=lanczos" -an -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart test_data/test_video_384_64_6fps.mp4
ffmpeg -y -i test_data/test_video_384_3fps.mp4 -vf "fps=6,scale=128:128:flags=lanczos" -an -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart test_data/test_video_384_128_6fps.mp4
```

DUSt3R commands completed:

```bash
uv run python src/train/run_dust3r_video.py --video test_data/test_video_small_64_4fps.mp4 --frame-stride 1 --max-frames 0 --scene-graph swin --winsize 5 --noncyclic --output-dir test_data/dust3r_outputs/test_video_small_64_4fps_all_frames
uv run python src/train/run_dust3r_video.py --video test_data/test_video_small_128_4fps.mp4 --frame-stride 1 --max-frames 0 --scene-graph swin --winsize 5 --noncyclic --output-dir test_data/dust3r_outputs/test_video_small_128_4fps_all_frames
```

DUSt3R results:

- `test_video_small_64_4fps_all_frames`: `46` sampled frames, `430` pairs, MPS device, final alignment loss about `0.0446`, runtime about `4m46s`.
- `test_video_small_128_4fps_all_frames`: `46` sampled frames, `430` pairs, MPS device, final alignment loss about `0.0648`, runtime about `5m43s`.

The 384-derived MP4s were generated and verified, but their DUSt3R bakes were not started after the user requested that no additional 384-derived DUSt3R runs be started during the status check.
