# PowerFoam MP4 Pixel Gate And Backend Audit

## Trigger

The 512px dynamic PowerFoam MP4 paths opened as solid green in the local
player even though the training runs had completed. The concrete request was
to stop trusting saved paths blindly, check the first decoded frame for pixel
variance, and use two subagents in parallel on:

- the green-video artifact path
- the larger Metal/CUDA script unification / line-count question

## Green MP4 Findings

The exact four user-facing MP4s now pass a decoded first-frame artifact gate:

- `outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_center_crop_8fps_512_16f_40step/render_step_0040.mp4`
- `outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_center_crop_8fps_512_16f_40step/side_by_side_step_0040.mp4`
- `outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_center_crop_8fps_512_16f_40step/render_step_0040.mp4`
- `outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_center_crop_8fps_512_16f_40step/side_by_side_step_0040.mp4`

Verifier command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_video_artifact_pixels.py --require-h264 \
  outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_center_crop_8fps_512_16f_40step/render_step_0040.mp4 \
  outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_center_crop_8fps_512_16f_40step/side_by_side_step_0040.mp4 \
  outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_center_crop_8fps_512_16f_40step/render_step_0040.mp4 \
  outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_center_crop_8fps_512_16f_40step/side_by_side_step_0040.mp4
```

Results:

- all four: `codec_name=h264`, `codec_tag_string=avc1`, `pix_fmt=yuv420p`
- geometry render: first-frame `var_min=0.0537067`, sampled unique RGB `3278`, green-dominance fraction `0.0`
- geometry side-by-side: `var_min=0.0551725`, unique `5430`, green fraction `0.000225`
- color render: `var_min=0.0654688`, unique `5218`, green fraction `0.0000343`
- color side-by-side: `var_min=0.0624400`, unique `6547`, green fraction `0.000210`

I also extracted frame 0 from the exact geometry render:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -i outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_center_crop_8fps_512_16f_40step/render_step_0040.mp4 \
  -frames:v 1 /tmp/dynaworld_geom_exact_frame0.png
```

The decoded PNG is visually a sky/cloud/ground PowerFoam render, not green.

## Code Changes

Created `src/train/video_io.py` as the single artifact media helper:

- `save_mp4(...)` now writes ffmpeg/libx264 H.264 with `avc1`, `yuv420p`, BT.709 tags, and `+faststart`
- no OpenCV `mp4v` fallback; if ffmpeg is missing, artifact writing fails loudly
- `save_png(...)` and tensor/video uint8 conversion moved with it

Updated local trainers to import the shared helper:

- `src/train/train_powerfoam_metal.py`
- `src/train/train_powerfoam_direct.py`
- `src/train/train_dynamic_gauge_foam.py`
- `src/train/train_dynamic_powerfoam_metal.py`

Added/kept artifact guard:

- `research_experiments/dynamic_foam/verify_video_artifact_pixels.py`
- `tests/test_powerfoam_direct.py::test_powerfoam_metal_save_mp4_uses_quicktime_compatible_h264`

The test now ffprobes the output and extracts frame 0 to assert the video is
not color-channel corrupted.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/video_io.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_dynamic_powerfoam_metal.py \
  research_experiments/dynamic_foam/verify_video_artifact_pixels.py
```

passed.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_save_mp4_uses_quicktime_compatible_h264 -q -rs
```

passed: `1 passed in 3.20s`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train .venv/bin/python - <<'PY'
import train_dynamic_gauge_foam
import train_powerfoam_direct
import video_io
print('imports ok')
PY
```

passed.

`git diff --check` passed for the edited files.

## Backend Unification Audit

Subagent audit conclusion: do not merge Metal and CUDA by `device.type` yet.
The local Metal path is an in-repo trainer with MPS autograd extensions, while
the CUDA path is a Modal/upstream `PowerfoamScene` smoke harness that clones
official PowerFoam and applies scene-side patches. The CUDA patches intentionally
do not modify upstream CUDA/Warp raster kernels.

Current safe direction:

- share media/logging/data/loss helpers now
- introduce an explicit `render.backend` / backend protocol later
- implement Metal and Torch-reference backends first
- keep the Modal CUDA smoke runner separate until CUDA exposes the same local
  tensor-state render contract

Line-count snapshot after the first media extraction:

```text
3358 src/train/train_powerfoam_metal.py
 694 src/train/train_powerfoam_direct.py
 375 src/train/train_dynamic_gauge_foam.py
1730 src/train/train_dynamic_powerfoam_metal.py
  77 src/train/video_io.py
1052 research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py
 273 research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py
```

Next clean reduction pass should extract neutral PowerFoam data/loading,
adjacency, and loss/logging helpers before attempting any backend protocol.
