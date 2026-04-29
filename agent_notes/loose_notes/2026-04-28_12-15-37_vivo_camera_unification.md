# ViVo camera unification + multicam loader audit trail

Date: 2026-04-28
Topic: bringing ViVo (Bristol VICR multicam human body capture) up to the same
multicam-loader polish as AIST++. Adds a `vivo` rig adapter alongside DeepView,
AIST, and Neural 3D, plus a cleanup-staging stage on the dataset pipeline so we
can reclaim derived MP4 space without ever touching the gated raw bundle.

## Context

ViVo is the only dataset in this repo whose access flow is **manual**. Raw
download requires an MS Form submission (https://forms.office.com/e/gtKpYriSMJ)
and a per-recipient Google Drive link, NOT a public release URL. So the
working assumption for this slice is: don't try to redownload anything; verify
what's locally rehydratable, polish the camera math, and make cleanup
non-destructive by default.

Local data state at start: `data/external/vivo/` (5.8 GB total). Single scene
on disk, `athlete_rows`, with 10 train cams + 4 test cams, ~501 frames per
camera, 2560x1440 colour at 30 fps, calibration.json present, no
rotation_correction.json.

## What changed

1. `src/train/multicam_video_data.py`
   - Added a full ViVo camera adapter:
     - `vivo_camera_from_calibration(record, name, *, H, W, device, translation_scale)`
       reads `calibration.json`, builds a scaled colour-camera `K`, composes
       the rig pose chain `R_w2c = R_cd @ R_d`, `t_w2c = R_cd @ t_d + t_cd`,
       and inverts to `c2w`.
     - `vivo_video_path_for_camera(record, name)` resolves arbitrary cameras
       under `<rgb_mp4_root>/<scene>/{train,test}/<serial>.mp4`.
     - `make_vivo_multiview_cameras(...)` does anchor-relative pose
       construction in parity with the AIST/DeepView/Neural-3D paths.
   - Routed `load_camera_video` and `load_multicam_video_bundle`'s `rig_init`
     dispatch to handle `vivo`, with explicit cross-dataset failure messages
     and a documented `vivo_translation_scale` knob (default 1.0 = ViVo native
     meters).
   - Documented an audit-grade comment block above the adapter explaining the
     calibration schema, the rig-pose composition math, the IMU-vs-rig
     distinction in the per-frame meta JSONs, and the deferral of
     `rotation_correction.json` (with a clear `NotImplementedError` if a record
     ever carries one).

2. `src/dataset_pipeline/vivo.py`
   - Added `cleanup-staging` stage. It enumerates derived staging artifacts
     (`rgb_mp4/`, `metadata/rgb_mp4_frames/`, `metadata/rgb_mp4_manifest.jsonl`,
     `metadata/scene_inventory.json`, `logs/`), prints a dry-run plan, and only
     actually deletes when `--execute` is passed. The classification table
     above the helper makes the recoverability contract explicit.
   - Important guarantee: `extracted/` and `raw/` are **never** marked
     deletable, because they require the MS-Form upstream access flow to
     recreate. The cleanup-staging plan also prints them as PROTECTED so the
     human reading the dry-run can see exactly what is and isn't reclaimable.

3. `src/dataset_scripts/cleanup_vivo_staging.sh`
   - Thin wrapper that invokes the python `cleanup-staging` stage with
     `--execute`. Non-interactive. The dry-run path is reachable via the
     python entrypoint directly.

4. `data/REHYDRATE.md` (ViVo section only)
   - Loud manual-access flow at the top of the section.
   - Explicit "what's recoverable vs. NOT" table.
   - Validator command (`uv run python /tmp/audit_vivo_bundle.py`).
   - New cleanup commands (dry-run + execute).
   - Final reminder that `extracted/` is the irreplaceable layer.

5. `/tmp/audit_vivo_bundle.py`
   - CPU-only validator that walks `data/external/vivo/` and reports per-area
     disk usage, scenes present, per-camera frame and metadata coverage,
     calibration / rotation_correction presence, and an mp4 sample dimension
     via ffprobe. Works even if compaction has not run yet. Mirrors the
     pattern of `/tmp/audit_aist_bundle.py` from the AIST work.

## Camera math, audit-grade

### ViVo calibration schema (per camera serial in `calibration.json`)

```
depth_extrinsics:           {orientation: 9 floats row-major R_d, translation: 3-vec t_d}
                            # rig WORLD -> depth camera
depth_intrinsics:           {fx, fy, ppx, ppy, width, height, ...}     # not used here
colour_intrinsics:          {fx, fy, ppx, ppy, width, height,
                             coefficients: [], distortion_mode: 0}     # rectified pinhole
colour_to_depth_extrinsics: {orientation: 9 floats R_cd, translation: 3-vec t_cd}
                            # depth -> colour camera
```

### Convention math

For each camera C:

```
x_depth(C)  = R_d(C)  @ x_world + t_d(C)            # depth_extrinsics
x_colour(C) = R_cd(C) @ x_depth(C) + t_cd(C)        # colour_to_depth_extrinsics
```

Compose to get world -> colour:

```
R_w2c(C) = R_cd(C) @ R_d(C)
t_w2c(C) = R_cd(C) @ t_d(C) + t_cd(C)
c2w      = inv([[R_w2c, t_w2c], [0, 1]])
```

Orientation packing is **row-major** (verified against the upstream
ViVo-DataProcessing repo). Use directly as `R`; do NOT transpose.

### Audit on athlete_rows

Independent verification on `athlete_rows / setting <none>`:

```
000236320812: det(R)=+1.000000  pos=[+0.001, -0.031, +0.006]   # rig origin reference
000404613112: det(R)=+1.000000  pos=[-2.341, +0.021, -1.554]
000409113112: det(R)=+1.000000  pos=[+1.862, +0.337, -1.848]
000454921912: det(R)=+1.000000  pos=[-0.874, -0.088, -1.828]
000497113112: det(R)=+1.000000  pos=[+3.047, -0.092, -0.305]   # test cam
000516213112: det(R)=+1.000000  pos=[-1.738, -0.691, +2.818]   # test cam
```

Rotation determinants are all `+1` to numerical precision (proper SO(3)).
Camera 000236320812's `depth_extrinsics` is identity, which is why its
position is at the origin offset only by the small colour-to-depth lever arm
(~3 cm). The other train cameras live at 2-3 m radius around the subject;
test cameras are placed at the back of the rig (positive z) and at +x.

Sanity-check baselines (all in meters):

```
000236320812 <-> 000404613112  =  2.81 m
000404613112 <-> 000454921912  =  1.50 m   # adjacent train cams
000454921912 <-> 000516213112  =  4.76 m
000404613112 <-> 000497113112  =  5.53 m   # widest baseline
```

These are right for a body-capture rig (subject is ~1.5-2 m tall, cameras
encircle them).

Anchor self-identity check:

```
inv(c2w[anchor]) @ c2w[anchor] - I    L2-norm = 1.34e-7
```

i.e. round-trip numerical error only.

### Intrinsic scaling

Colour intrinsics live at native 2560x1440 with `fx ~ 1237`, `fy ~ 1237`,
`ppx ~ 700`, `ppy ~ 1280`. After force-resize to 128x128, the principal
lengths and centers stretch by different factors:

```
sx = 128 / 2560 = 0.05
sy = 128 / 1440 ~ 0.0889
fx_new ~  61.85,  fy_new ~ 110.0,  cx_new ~ 35,  cy_new ~ 113
```

This non-square focal length is correct for the square-resized 2560x1440
frames and matches the same pattern we documented for AIST. If a future
caller wants an isotropic K, switch ingestion to center-crop+pad rather than
force-resize.

### Translation units (gotcha)

ViVo translations are in **meters** (Femto Bolt sensor calibration). No
rescaling is needed to match Neural 3D / DeepView's meter-ish world scale,
unlike AIST's millimeter convention. We still expose
`vivo_translation_scale=1.0` for parity with the other adapters, in case a
future rig wants a different absolute scale.

### Per-frame meta.json: NOT for rig pose

The per-frame `imageMetadata.extrinsics` blocks in `*.meta.json` are the
**IMU-anchored sensor pose** in the device body frame, not the rig pose.
Empirical audit on `000236320812`: orientation varies by <0.05 rad across
all 501 frames -- a stationary camera with IMU jitter. The translation is
~3 cm, consistent with the lens-IMU lever arm. **Use calibration.json for
multicam geometry; ignore the per-frame extrinsics for pose work.**

### Distortion: rectified, treat as pinhole

`colour_intrinsics` in calibration.json carries `distortion_mode=0` and
`coefficients=[]` -- the upstream pipeline rectifies the colour camera and
exposes a clean pinhole. The per-frame meta JSONs DO carry an OPENCV_8VAL
8-coefficient set, but those are RAW sensor intrinsics (with a slightly
different principal point), not the rectified ones used at training time.
We use calibration.json and treat the colour camera as pinhole, in parity
with the AIST/DeepView paths.

## What is deferred and why

- **`rotation_correction.json` (cross-recording rig alignment).** Some ViVo
  scenes carry a 4x4 world-frame rigid transform that aligns this scene's rig
  coordinate frame to a canonical capture-room frame across recordings. The
  athlete_rows scene we have locally does **not** carry one. The adapter
  raises a clear `NotImplementedError` if a record ever specifies
  `vivo_rotation_correction_path`, asking the integrator to extend the math
  at exactly that site rather than silently producing rig-frame-misaligned
  poses. The right time to wire it up is when the first such record lands in
  a manifest.
- **OpenCV_8VAL distortion model.** We treat colour as pinhole because
  calibration.json publishes rectified intrinsics. If a later renderer pass
  ever wants the full 8-coefficient model, it can read the per-frame
  meta.json blocks; the raw distortion data is still on disk.
- **No ViVo entry in any multicam_val manifest yet.** This work makes the
  loader ready; an upcoming follow-up would add a `dataset: "vivo"` row with
  `vivo_calibration_path`, `vivo_rgb_mp4_root`, `vivo_scene`, `train/test`
  camera lists, etc., and a `camera.rig_init = "vivo"` train config.

## Audit script

`/tmp/audit_vivo_bundle.py` walks the local bundle CPU-only and reports
disk usage per area, scene/camera/frame counts, calibration presence, and an
mp4 sample dimension. Mirrors the pattern of `/tmp/audit_aist_bundle.py`.
The validator does not depend on the camera adapter -- its job is to verify
what is locally rehydratable, even if downstream training is deferred.

## Cleanup contract (recoverability is a hard line)

```
PROTECTED   raw/                                MS-Form upstream access required
PROTECTED   extracted/<scene>/...                MS-Form upstream access required

DERIVED     rgb_mp4/<scene>/...                  rebuilt by compact-rgb
DERIVED     metadata/rgb_mp4_frames/             rebuilt by compact-rgb
DERIVED     metadata/rgb_mp4_manifest.jsonl      rebuilt by compact-rgb
DERIVED     metadata/scene_inventory.json        rebuilt by inspect
DERIVED     logs/                                ffmpeg/inspect transient
```

The cleanup-staging stage NEVER auto-deletes anything in PROTECTED. The
`compact-rgb --delete-heavy` flag (already existed pre-this-session) only
deletes DEPTH/POINTCLOUD/MASK directories the upstream pipeline considers
heavy, and even then it's opt-in. We deliberately did NOT extend that flag's
scope.
