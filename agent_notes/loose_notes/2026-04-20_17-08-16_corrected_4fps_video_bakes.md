# Corrected 4fps Video Bakes

The first `test_video_small_64_4fps.mp4` and `test_video_small_128_4fps.mp4`
assets were generated from `test_data/test_video_small.mp4`, which is already
2fps. That was wrong for the camera bake because it doubled frame count from a
low-FPS asset instead of sampling real source frames from the original 30fps
clip.

Corrected both videos by generating directly from:

```text
/Users/nicholasbardy/Downloads/14951282_2160_3840_30fps.mp4
```

The corrected ffmpeg path center-crops the portrait source to a square, samples
at 4fps, and scales to the target size in one pass:

```bash
ffmpeg -y -i /Users/nicholasbardy/Downloads/14951282_2160_3840_30fps.mp4 -vf "crop='min(iw,ih):min(iw,ih):(iw-min(iw,ih))/2:(ih-min(iw,ih))/2',fps=4,scale=64:64:flags=lanczos" -an -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart test_data/test_video_small_64_4fps.mp4
ffmpeg -y -i /Users/nicholasbardy/Downloads/14951282_2160_3840_30fps.mp4 -vf "crop='min(iw,ih):min(iw,ih):(iw-min(iw,ih))/2:(ih-min(iw,ih))/2',fps=4,scale=128:128:flags=lanczos" -an -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart test_data/test_video_small_128_4fps.mp4
```

Validation after correction:

- `test_video_small_64_4fps.mp4`: `64x64`, `4fps`, `46` frames, `11.5s`
- `test_video_small_128_4fps.mp4`: `128x128`, `4fps`, `46` frames, `11.5s`
- Adjacent frame absolute-diff medians were `0.0771` for 64px and `0.0890`
  for 128px, with no near-zero adjacent diffs.

Reran DUSt3R with:

```bash
uv run python src/train/run_dust3r_video.py --video test_data/test_video_small_64_4fps.mp4 --frame-stride 1 --max-frames 0 --scene-graph swin --winsize 5 --noncyclic --output-dir test_data/dust3r_outputs/test_video_small_64_4fps_all_frames
uv run python src/train/run_dust3r_video.py --video test_data/test_video_small_128_4fps.mp4 --frame-stride 1 --max-frames 0 --scene-graph swin --winsize 5 --noncyclic --output-dir test_data/dust3r_outputs/test_video_small_128_4fps_all_frames
```

Corrected DUSt3R stats:

- 64px bake: `46` frames, `430` pairs, alignment loss `0.03363766893744469`,
  median raw DUSt3R fx/fy `602.65`, median FoV at 224 input `21.06` degrees.
- 128px bake: `46` frames, `430` pairs, alignment loss `0.06328976154327393`,
  median raw DUSt3R fx/fy `662.79`, median FoV at 224 input `19.18` degrees.

The duplicate-frame mistake is fixed, but the corrected 64/128 DUSt3R solves
still have narrower FoV than the old 32px/2fps bake, whose median raw fx is
`396.68` and median FoV is `31.53` degrees. That now looks like a DUSt3R
behavior on these low-resolution/upscaled inputs rather than a config/render
viewport mismatch.
