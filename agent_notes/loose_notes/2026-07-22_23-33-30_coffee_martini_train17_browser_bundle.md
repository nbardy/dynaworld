# Coffee Martini train17/holdout1 browser bundle

## Scope

Added a separate demo-tier breadth artifact without changing the existing
`train2_holdout1` browser demo or any SPA/WGSL runtime. The canonical manifest
is `src/dataset_configs/neural3d_coffee_martini_train17_holdout1_full_300f_manifest.jsonl`.
It declares all 18 local calibrated cameras exactly once: 17 train cameras and
`cam06` as the only heldout, validation-only camera. `cam04` remains the anchor
and condition camera.

The browser bundle is a thin export through `load_multicam_video_bundle`; it
does not duplicate calibration, relative-pose, or split logic. The exporter
display name now derives from the selected canonical split instead of being
hard-coded to `train2/holdout1`.

## Intentional demo-tier sampling

The manifest preserves the full synchronized source contract: 300 frames at
30 fps over 10 seconds and native 2704x2028 source imagery. The generated
browser artifact intentionally samples only eight frame indices
`[0, 43, 85, 128, 171, 214, 256, 299]` and decodes them to 96x72. This is the
only divergence from the full source row and is explicit in the manifest's
`browser_demo_tier` metadata and bundle payload.

The output is
`web/dynaworld_browser_trainer/coffee_martini_train17_holdout1.json` plus one
PNG atlas per camera. Each atlas is exactly 768x72 pixels (eight 96x72 frames
placed horizontally), below portable WebGPU texture limits. `cam06` appears
only with role `heldout`; the point-cloud visibility filter uses only the 17
train cameras.

## Exact artifact sizes

- JSON: 99,261 bytes
- 18 PNG atlases: 1,647,953 bytes total
- JSON plus atlases: 1,747,214 bytes total
- Decoded RGBA atlas payload: 3,981,312 bytes total
- Seed payload: 768 SfM/COLMAP XYZRGB points in the `cam04_opencv` frame

Per-camera PNG bytes:

| Camera | Role | Bytes |
| --- | --- | ---: |
| cam00 | train | 90,175 |
| cam01 | train | 89,659 |
| cam02 | train | 89,563 |
| cam04 | train | 88,583 |
| cam05 | train | 90,078 |
| cam07 | train | 92,115 |
| cam08 | train | 92,719 |
| cam09 | train | 95,395 |
| cam10 | train | 94,068 |
| cam11 | train | 92,398 |
| cam12 | train | 89,857 |
| cam13 | train | 93,899 |
| cam14 | train | 90,832 |
| cam16 | train | 87,832 |
| cam18 | train | 89,723 |
| cam19 | train | 94,107 |
| cam20 | train | 95,512 |
| cam06 | heldout | 91,438 |

## Verification

```text
PYTHONPATH=src/train .venv/bin/python -m pytest \
  tests/test_browser_multicam_export_adapter.py tests/test_multicam_video_data.py -q
15 passed in 1.12s
```

The artifact regression verifies the exact split and sampled indices, 96x72
decode geometry, 768x72 dimensions for all 18 atlases, 768 seeds, and
validation-only heldout metadata.
