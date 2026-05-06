# DeepView 03_Dog Overlap Camera Selection

Goal: replace the bad heldout choice `camera_0040` for the DeepView
`03_Dog` two-train-camera setup. The train views are `camera_0001` and
`camera_0015`.

The current learned-residual baseline config used:

```text
train cameras: camera_0001, camera_0015
heldout: camera_0040
```

That split is disjoint, but `camera_0040` is too far around the rig to have
enough overlap with either train camera.

## Geometry Rank

Using DeepView `models.json` poses, camera forward-angle separation from the
two train cameras:

| Candidate | angle to 0001 | angle to 0015 | baseline to 0001 | baseline to 0015 |
|---|---:|---:|---:|---:|
| camera_0006 | 13.66 deg | 19.96 deg | 0.159 | 0.179 |
| camera_0005 | 17.42 deg | 21.66 deg | 0.163 | 0.180 |
| camera_0014 | 36.12 deg | 21.65 deg | 0.328 | 0.186 |
| camera_0016 | 38.51 deg | 21.90 deg | 0.330 | 0.185 |
| camera_0040 | 71.33 deg | 87.70 deg | 0.547 | 0.673 |

## First-Frame Feature Match Check

First sampled frames were extracted at the manifest start time `0.5s` to:

```text
outputs/multicam_first_frames/deepview_03_Dog_candidate_overlap/
```

ORB + RANSAC inlier counts against both train views:

| Candidate | inliers vs 0001 | inliers vs 0015 | min inliers | total inliers |
|---|---:|---:|---:|---:|
| camera_0006 | 420 | 449 | 420 | 869 |
| camera_0005 | 259 | 437 | 259 | 696 |
| camera_0014 | 174 | 440 | 174 | 614 |
| camera_0016 | 95 | 245 | 95 | 340 |
| camera_0040 | 7 | 4 | 4 | 11 |

Conclusion: use `camera_0006` as the better-overlap heldout camera for the
next DeepView `03_Dog` train2/test1 runs. Keep `camera_0040` only as a known
hard/extrapolative heldout row, not as the default overlap validation camera.

Also added a loader guard in `src/train/multicam_video_data.py` so train and
heldout camera names cannot overlap accidentally.

## Better Three-Camera Set

After visual inspection, the preferred local-overlap trio is:

```text
camera_0005, camera_0006, camera_0014
```

Pairwise geometry inside this trio:

| Pair | forward angle | baseline |
|---|---:|---:|
| camera_0005 vs camera_0006 | 18.63 deg | 0.185 |
| camera_0005 vs camera_0014 | 18.84 deg | 0.169 |
| camera_0006 vs camera_0014 | 33.12 deg | 0.309 |

For a train2/test1 interpolation split, use:

```text
train: camera_0006, camera_0014
heldout: camera_0005
anchor/condition: camera_0006
```

`camera_0005` is the balanced heldout view: about `18.6/18.8 deg` from the
two train cameras.

Config entrypoints:

```text
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_learned_residual_relpose_small_bf16_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats_oracle_relative_camera_goodset_train0006_0014_holdout0005.jsonc
```
