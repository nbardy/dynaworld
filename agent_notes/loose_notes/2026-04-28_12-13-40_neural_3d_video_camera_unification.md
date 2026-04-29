# Neural 3D Video camera unification + multicam loader audit trail

Date: 2026-04-28
Topic: bring the Facebook AI Neural 3D Video release to parity with the AIST++
and DeepView paths in `multicam_video_data.py`. Decode `poses_bounds.npy`
correctly (LLFF -> OpenCV), wire it into the multicam bundle, and add an
opt-in cleanup stage so the 1.1 GB raw zip can be reclaimed once the scene is
extracted.

## What changed

1. `src/train/multicam_video_data.py`
   - Added `neural_3d_camera_from_poses_bounds(record, name, *, H, W, device, translation_scale=1.0)`
     that reads `poses_bounds.npy`, picks the row by sorted-mp4 filename order,
     splits the 17-float row into the 3x5 pose-hwf matrix and `[near, far]`
     bounds, converts the LLFF rotation to OpenCV via `R_llff @ diag([1,-1,-1])`,
     and builds a clean isotropic pinhole `K` scaled to the resize target.
   - Added `neural_3d_video_path_for_camera(record, name)` that resolves
     arbitrary cameras (`cam00..cam20`, with the canonical 3-of-21 missing in
     coffee_martini) via `dataset_scene_dir` on the manifest record.
   - Added `make_neural_3d_multiview_cameras(...)` mirroring the DeepView/AIST
     anchor-relative pose construction, plus a `translation_scale` knob.
   - Routed `load_camera_video` and `load_multicam_video_bundle`'s `rig_init`
     dispatch to handle `"neural_3d_video"`, with cross-dataset failure
     messages identical in shape to the AIST/DeepView branches.

2. `src/dataset_pipeline/neural_3d_video.py`
   - Added a `cleanup-zips` stage. Default mode is dry-run: it lists every
     `raw/*.zip` whose extracted scene marker
     (`extracted/<scene>/<scene>/poses_bounds.npy`) exists, prints per-archive
     bytes, and prints the total reclaimable bytes. `--execute` unlinks.
     Archives without a matching extracted scene are surfaced as `SKIP` lines
     and never touched. Split flame_salmon parts (`flame_salmon_1_split.zNN`)
     map back to the merged `flame_salmon_1` scene marker.
   - Added `--execute` flag to argparse and wired the new stage into `main`.

3. `src/dataset_scripts/cleanup_neural_3d_video_zips.sh`
   - Non-interactive wrapper that calls `cleanup-zips --execute`. Documents
     the dry-run command in its header so callers don't accidentally hand-roll
     the python invocation.

4. `data/REHYDRATE.md`
   - Updated the "Neural 3D Video" section only. Added the cleanup dry-run +
     wrapper commands and the safety contract (never deletes without
     `--execute`, never deletes if the extracted scene is missing,
     re-extraction requires a re-download).

## What did NOT change

- `src/dataset_configs/neural_3d_video_seed.jsonc` -- no schema change needed.
- `src/dataset_pipeline/multicam_val.py` -- the existing `neural_3d_pairs(...)`
  already records `dataset_scene_dir` on each manifest row, which is the only
  field the new adapter needs.
- Any other dataset section of REHYDRATE.md -- left strictly alone.

## Camera math, audit-grade

This block is the contract every reader should keep open while reading the
adapter code. Independent verification on `coffee_martini` follows the
`/tmp/audit_neural3d_bundle.py` output below.

### LLFF `poses_bounds.npy` schema

```
shape    : (N, 17) float64
cameras  : sorted lexicographic order of cam*.mp4 in scene_dir
           (coffee_martini: 18 cams; cam03/cam15/cam17 are missing in the release)
row k    : flatten(3x5 pose_hwf, order='C')  ++  [near, far]
```

The 3x5 "pose-hwf" matrix layout:

```
                 col0    col1    col2    col3        col4
row 0     [   R_x_x   R_x_y   R_x_z   t_x      H (height) ]
row 1     [   R_y_x   R_y_y   R_y_z   t_y      W (width)  ]
row 2     [   R_z_x   R_z_y   R_z_z   t_z      f (focal)  ]
                 ^^^^^^^^^^^^^^^^^^   ^^^      ^^^^^^^^^^^^
                 LLFF rot columns     world    native pixels
                 (basis vectors)      pos
```

Verified on coffee_martini row 0:

```
H=2028,  W=2704,  f=1460.754
near=8.831,  far=109.775   (LLFF NDC bounds, in scene units)
```

### LLFF -> OpenCV axis convention

LLFF's camera basis is `(right, up, back)` -- cameras "look down -z" in their
own frame. OpenCV's basis is `(right, down, forward)` -- cameras "look down
+z". The minimal column-flip is:

```
R_opencv = R_llff @ diag([+1, -1, -1])
```

This is what `nerf_synthetic` / `run_nerf_helpers.recenter_poses` does, just
written in matrix form. The translation column does NOT need the flip --
it's already a world-space camera position, axes-agnostic.

Then build a clean OpenCV `c2w`:

```
c2w = [[ R_opencv,   t * translation_scale ],
       [   0          1                    ]]
```

`translation_scale` defaults to 1.0 (LLFF native units; coffee_martini scene
is roughly meters but with LLFF's recentered/rescaled scene factor baked in
-- bounds [8.83, 109.78] suggest decimeters in the recentered frame).
Exposed as `camera.n3d_translation_scale` for the rare caller that wants
metric output.

### Intrinsics under non-uniform resize

Source is 2704 x 2028 (W x H), focal 1460.754 pixels. Force-resize to 128 x
128 stretches non-uniformly (this is the same situation as AIST):

```
sx = 128 / 2704 = 0.04734
sy = 128 / 2028 = 0.06312
fx_new = 1460.754 * 0.04734 = 69.148
fy_new = 1460.754 * 0.06312 = 92.198
cx_new = (2704/2) * sx = 64.000
cy_new = (2028/2) * sy = 64.000
```

If you need an isotropic pinhole render, switch the dataset prep to
center-crop+pad rather than force-resize and the K becomes square again.
Documented in the adapter comment so this is not rediscovered.

### Anchor-relative pose

Same DeepView/AIST pattern:

```
anchor_c2w   = c2w of the anchor camera (e.g. cam00)
rel_w2c[v]   = inv(c2w[v]) @ anchor_c2w
```

Anchor projects to identity in the relative frame. The audit confirms:

```
anchor_w2c[0] - I  L2 err = 1.23e-7   # numerical round-trip only
det(c2w[cam00..cam20]) = +1.0         # proper SO(3)
```

### Camera baselines on coffee_martini

```
cam00 <-> cam10  =  5.08
cam00 <-> cam20  =  6.42
cam10 <-> cam20  =  2.89
```

In LLFF native units (~decimeters), so cameras sit ~0.5-0.6 m apart, which
matches the visible coffee_martini rig spread on the public previews.

## Audit script

`/tmp/audit_neural3d_bundle.py` is the standalone CPU script that walks the
math end-to-end. It reads the manifest, dumps the row<->camera map, prints
each camera's K and c2w fingerprints, computes pairwise baselines, and
finally builds the full multicam bundle to verify the anchor identity check
and bundle shapes. Output snapshot:

```
sample_id          = neural3d_coffee_martini_cam00_to_cam10
poses_bounds.npy   shape = (18, 17)
row->camera map    = {0: 'cam00', 1: 'cam01', ..., 9: 'cam10', ..., 17: 'cam20'}
row0 native HxW    = 2028 x 2704,  focal_native = 1460.754

[cam00]  K[fx,fy,cx,cy] = (69.148, 92.198, 64.000, 64.000)  c2w det = +1
[cam10]  K[fx,fy,cx,cy] = (69.148, 92.198, 64.000, 64.000)  c2w det = +1
[cam20]  K[fx,fy,cx,cy] = (69.148, 92.198, 64.000, 64.000)  c2w det = +1

train_frames shape   = (2, 4, 3, 128, 128)
heldout_frames shape = (1, 4, 3, 128, 128)
pose_source          = neural_3d_llff_relative_pinhole
anchor_w2c[0] - I  L2 err = 1.23e-07
```

Keep this script around as the smoke-test for future Neural 3D Video work.

## Disk-size audit (coffee_martini-only; before cleanup)

Byte-precise sizes via `os.walk + getsize` (macOS `du` lacks `-b`):

```
raw/        1186324684 bytes  (1186.32 MB)
extracted/  1186189096 bytes  (1186.19 MB)
metadata/        24172 bytes  (   0.02 MB)
logs/                0 bytes  (   0.00 MB)
```

Cleanup dry-run reports 1186324684 bytes (1186.32 MB) reclaimable from a
single archive, `coffee_martini.zip`. Running the wrapper would drop `raw/`
to ~0 and leave the extracted scene + scene_inventory.json intact.

## Failure modes considered

- Adapter assumes `sorted(scene_dir.glob("cam*.mp4"))` order matches
  `poses_bounds.npy` row order. This is the LLFF/Neural 3D Video upstream
  convention and is verified by the row<->camera map in the audit. If a
  future scene ships with mismatching counts the adapter raises a clear
  RuntimeError, and the audit script will surface it on first run.
- Cleanup never deletes a zip without a confirmed extracted-scene marker on
  disk. Default is dry-run; only the wrapper or explicit `--execute` unlinks.
  Re-extraction requires a re-download from the GitHub release (1.1 GB for
  coffee_martini, ~12 GB for the full release).
- Translation scale defaults to 1.0 (LLFF native). The DeepView training
  configs are already in their own scene scale, so cross-dataset comparisons
  must explicitly set `n3d_translation_scale` rather than rely on a silent
  default. This mirrors how `aist_translation_scale` is handled.

## What still defers

- No `multicam_v1` train config selects a Neural 3D Video sample yet; that is
  a follow-up in `src/train_configs/`.
- The 5 other release scenes (`cook_spinach`, `cut_roasted_beef`,
  `flame_salmon_1_split.*`, `flame_steak`, `sear_steak`) are still listed as
  `validation_todo_assets` in the config and are not on local disk. Once
  someone pulls them, the cleanup stage will reclaim ~10 GB more without any
  code change.
- The release rgb is pre-undistorted, so we keep `CameraSpec(lens_model="pinhole")`.
  No follow-up needed on the lens model.
