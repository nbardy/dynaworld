# AIST++ camera unification + multicam loader audit trail

Date: 2026-04-28
Topic: pulling in the AIST++ camera bundle so multicam V-JEPA training can use AIST as a pinhole-style alternative to DeepView, and writing the math/audit story down so it stays interpretable later.

## What changed

1. `src/dataset_pipeline/multicam_val.py`
   - Added `download_aist_cameras` stage. It pulls `cameras.zip` from
     `https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip`,
     extracts it into `data/external/aist_dance_db/cameras/extracted/cameras/`,
     parses `mapping.txt` (which maps each AIST `cAll`-form sequence name to its
     environment setting JSON), and writes a small per-sequence inventory at
     `data/multicam_val/metadata/aist/camera_inventory.jsonl`.
   - Extended `aist_pairs` so every AIST manifest record now carries:
     - `aist_seq_name` (full `cAll`-form key, e.g. `gBR_sBM_cAll_d04_mBR0_ch01`)
     - `aist_setting_name` (e.g. `setting1`)
     - `aist_setting_path` (absolute path to the setting JSON)
     - `aist_cameras_dir` (extracted `cameras/` directory)
     - `aist_raw_dir` (where AIST mp4s live, used to look up arbitrary cameras
       beyond the source/target pair encoded in the manifest row)
     - `camera_model = "opencv_pinhole_radial"`
   - New stage wired into argparse and into `src/dataset_scripts/multicam_val_v1_seed.sh` (no shell change needed; just call `download-aist-cameras`).

2. `src/dataset_configs/multicam_val_v1_128_4fps_16f.jsonc`
   - Added `cameras_zip_url` and `cameras_dir` under `aist`, with comment
     pointing to the AIST++ docs.

3. `src/train/multicam_video_data.py`
   - Added `aist_camera_from_setting(record, name, *, H, W, device, translation_scale)` that reads the setting JSON, builds a scaled `K`, and converts OpenCV `(rotation, translation)` into our `c2w`.
   - Added `aist_video_path_for_camera(record, name)` that resolves arbitrary AIST cameras (`c01..c09`) by substituting into `aist_seq_name`'s `cAll` slot and looking up `aist_raw_dir`.
   - Added `make_aist_multiview_cameras(...)` that does the same anchor-relative pose construction we use for DeepView, plus a `translation_scale` knob.
   - Routed `load_camera_video` and `load_multicam_video_bundle`'s `rig_init` dispatch to handle `aist`, with explicit cross-dataset failure messages.

## Camera math, audit-grade

This block is the contract every reader should keep open while reading the
adapter code. Independent verification on `gBR_sBM_d04_mBR0_ch01 / setting1`:

```
sample_id        = aist_gBR_sBM_d04_mBR0_ch01_c01_to_c05
aist_seq_name    = gBR_sBM_cAll_d04_mBR0_ch01
aist_setting     = setting1
available_cams   = c01..c09 (AIST++ provides 9 cameras / setting)
```

### AIST++ raw schema (per camera)

```
name        : str            e.g. "c05"
size        : [W, H]         [1920, 1080] for our sequence
matrix      : 3x3 list       OpenCV K (fx, fy, cx, cy on diagonal/principal)
rotation    : [rx, ry, rz]   OpenCV Rodrigues vector (rvec)
translation : [tx, ty, tz]   OpenCV tvec (in millimeters in this dataset)
distortions : [k1,k2,p1,p2,k3]  OpenCV radial+tangential coefficients
```

### Convention math

OpenCV (and therefore AIST++) defines `(R, t)` so that

```
x_camera = R @ x_world + t                      # world-to-camera
```

with `R = Rodrigues(rvec)`. Concretely, for a homogeneous 4x4 rigid
transform:

```
w2c = | R  t |          c2w = inv(w2c) = | R^T   -R^T @ t |
      | 0  1 |                            |  0      1     |
```

The adapter constructs `w2c` directly from `(rotation, translation)`, then
inverts to get `c2w`. We do **not** mirror the GL-axis flip we use for
DeepView (where `models.json` rotation lives in OpenGL convention with
camera +Z pointing back); AIST++ is OpenCV from the start, so no flip.

### Intrinsic scaling under non-uniform resize

The multicam validation pipeline force-resizes 1920x1080 frames to 128x128.
That stretches the horizontal and vertical principal lengths by different
factors:

```
sx = 128 / 1920 = 0.0667
sy = 128 / 1080 = 0.1185
fx_new = fx * sx,  fy_new = fy * sy,  cx_new = cx * sx,  cy_new = cy * sy
```

Audit (c01, native fx=fy=1310.486):
```
fx_new ≈ 87.37,  fy_new ≈ 155.32,  (cx_new, cy_new) = (64, 64)
```

This non-square focal length is correct for square-resized fisheye-of-the-
mind frames; the K matches the visual squash users have seen in W&B. If
you ever want a clean undistorted pinhole render, switch the dataset prep to
center-crop+pad rather than force-resize, and the K will become isotropic.

### Anchor-relative pose

We follow the same pattern as DeepView:

```
anchor_c2w  = c2w of the anchor / condition camera (e.g. c01)
rel_w2c[v]  = inv(c2w[v]) @ anchor_c2w
```

so that the anchor camera becomes identity in the relative frame and the
splat world is anchored in front of the input view.

Audit on c01 (anchor):
```
anchor_w2c[0] - I   L2-norm = 9.6e-7    # round-trip numeric error only
det(c2w[c01..c09]) = +1                  # proper SO(3)
```

### Translation units (gotcha)

AIST++ tvec is in **millimeters** (typical for an OpenCV-calibrated dance
rig). Cameras live ~450mm from origin and the dancer's torso is ~1m from
each lens. Camera baselines on this scene:

```
c01 <-> c05  =  987.4 mm
c01 <-> c09  =   73.2 mm
c05 <-> c09  =  977.9 mm
```

`c01` and `c09` are basically a stereo pair; `c05` is on the opposite arc.

For the splat renderer, only relative geometry matters because we feed
`rel_w2c`. But the absolute scene scale still has to roughly match what
the model decodes (e.g. `rig_radius=3.0` defaults are unitless world
units). To put AIST into "meter-ish" world scale to match DeepView, set:

```
"camera": { "rig_init": "aist", "aist_translation_scale": 0.001 }
```

We deliberately default `aist_translation_scale=1.0` (= AIST native mm)
rather than silently rescaling: the audit comment in the adapter spells
this out and a regression catches the wrong-scale case if the rig
explodes.

### Lens / distortion approximation

AIST cameras carry a small radial-only distortion (`k1 ~ -0.114`, all other
terms 0). At 1920x1080 the residual after pinhole approximation is a few
pixels at the corners; at 128x128 (our square-resized target) that scales
to <0.2 px. We keep `CameraSpec(lens_model="pinhole")` for now, in parity
with the DeepView path, and preserve `distortions` in metadata so a later
renderer pass can switch to the full OpenCV radial-tangential model
without touching ingestion. This is documented in the comment block above
`aist_camera_from_setting` so the choice does not get rediscovered.

## Audit script

`/tmp/audit_aist_bundle.py` is the standalone script I used to walk the
math end-to-end on CPU. It loads the manifest, reconstructs each c2w from
the raw JSON via `aist_camera_from_setting`, prints determinants and
positions, builds the full multicam bundle, and verifies that the anchor
camera is identity. Keep this around as a smoke-test pattern for future
adapters.

## What still defers

- AIST records currently bypass the lens distortion model. Edge content of
  c01/c05/c09 will be a touch off; not a blocker for getting a real-world
  pinhole-ish multicam baseline with calibrated cameras.
- We have not yet added a multicam_v1 sample-id selector for AIST in any
  train config; that is a follow-up in `src/train_configs/`.
- `aist_translation_scale` is exposed but not yet defaulted per-config.
  The current convention is "AIST native mm unless you opt in to 0.001".
